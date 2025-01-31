"""Module to evaluate the achievable entanglement in circuits."""

import itertools
import typing
from qiskit.quantum_info import partial_trace
from scipy.special import comb
import numpy as np
import torch
import os
import pandas as pd

class EntanglementCapability():
    """Calculates entangling capability of a parameterized quantum circuit"""

    def __init__(self, circuit, device, num_wires, num_samples, inputs: np.ndarray, params: np.ndarray, input_is_bin=False):
        """Constructor for entanglement capability plotter
        :param circuit: input circuit as a CircuitDescriptor object
        :param noise_model:  (dict, NoiseModel) initialization noise-model dictionary for
            generating noise model
        :param samples: number of samples for the experiment
        :returns Entanglement object instance
        :raises ValueError: If circuit and noise model does not correspond to same framework
        """
        self.circuit = circuit
        self.device = device
        self.num_wires = num_wires
        self.num_samples = num_samples
        self.inputs = inputs
        self.input_is_bin = input_is_bin
        self.params = params

    # TODO: added also inputs, and not just params
    def gen_params(self) -> typing.Tuple[typing.List, typing.List]:
        """Generate parameters for the calculation of expressibility

        Returns:
            inputs_theta (np.array): first list of inputs for the circuit
            inputs_phi (np.array): second list of inputs for the circuit
            params_theta (np.array): first list of parameters for the circuit
            params_phi (np.array): second list of parameters for the circuit

        Note:
            Compared with the qleet librairy, here we are adding the input values because we want the entanglement of the whole circuit and not just the variational part.
        """
        if self.inputs is not None:
            if self.input_is_bin == True:
                # generate random binary inputs
                inputs_theta = [torch.from_numpy(np.random.randint(2, size=len(self.inputs)))
                                for _ in range(self.num_samples)
                                ]
                inputs_phi = [torch.from_numpy(np.random.randint(2, size=len(self.inputs)))
                              for _ in range(self.num_samples)
                              ]
            else:
                # generate inputs between -1 and 1
                inputs_theta = [torch.from_numpy(np.random.uniform(-1, 1, self.inputs.shape))
                                for _ in range(self.num_samples)
                                ]
                inputs_phi = [torch.from_numpy(np.random.uniform(-1, 1, self.inputs.shape))
                              for _ in range(self.num_samples)
                              ]
        else:
            inputs_theta, inputs_phi = None, None

        params_theta = [torch.from_numpy(2 * np.random.random(self.params.shape) * np.pi)
                        for _ in range(self.num_samples)
                        ]
        params_phi = [torch.from_numpy(2 * np.random.random(self.params.shape) * np.pi)
                      for _ in range(self.num_samples)
                      ]

        return inputs_theta, inputs_phi, params_theta, params_phi

    @staticmethod
    def scott_helper(state, perms):
        """Helper function for entanglement measure. It gives trace of the output state"""
        dems = np.linalg.matrix_power(
            [partial_trace(state, list(qb)).data for qb in perms], 2
        )
        trace = np.trace(dems, axis1=1, axis2=2)
        return np.sum(trace).real

    def meyer_wallach_measure(self, states, num_qubits):
        r"""Returns the meyer-wallach entanglement measure for the given circuit.
        .. math::
            Q = \frac{2}{|\vec{\theta}|}\sum_{\theta_{i}\in \vec{\theta}}
            \Bigg(1-\frac{1}{n}\sum_{k=1}^{n}Tr(\rho_{k}^{2}(\theta_{i}))\Bigg)
        """
        permutations = list(itertools.combinations(
            range(num_qubits), num_qubits - 1))
        ns = 2 * sum(
            [
                1 - 1 / num_qubits * self.scott_helper(state, permutations)
                for state in states
            ]
        )
        return ns.real

    def scott_measure(self, states, num_qubits):
        r"""Returns the scott entanglement measure for the given circuit.
        .. math::
            Q_{m} = \frac{2^{m}}{(2^{m}-1) |\vec{\theta}|}\sum_{\theta_i \in \vec{\theta}}\
            \bigg(1 - \frac{m! (n-m)!)}{n!}\sum_{|S|=m} \text{Tr} (\rho_{S}^2 (\theta_i)) \bigg)\
            \quad m= 1, \ldots, \lfloor n/2 \rfloor
        """
        m = range(1, num_qubits // 2 + 1)
        permutations = [
            list(itertools.combinations(range(num_qubits), num_qubits - idx))
            for idx in m
        ]
        combinations = [1 / comb(num_qubits, idx) for idx in m]
        contributions = [2 ** idx / (2 ** idx - 1) for idx in m]
        ns = []

        for ind, perm in enumerate(permutations):
            ns.append(
                contributions[ind]
                * sum(
                    [
                        1 - combinations[ind] * self.scott_helper(state, perm)
                        for state in states
                    ]
                )
            )

        return np.array(ns)

    def entanglement_capability(
        self, measure: str = "meyer-wallach", shots: int = 1024
    ) -> float:
        """Returns entanglement measure for the given circuit
        :param measure: specification for the measure used in the entangling capability
        :param shots: number of shots for circuit execution
        :returns pqc_entangling_capability (float): entanglement measure value
        :raises ValueError: if invalid measure is specified
        """
        inputs_theta, inputs_phi, params_theta, params_phi = self.gen_params()

        theta_circuits = []
        if inputs_theta is not None:
            for input_theta, param_theta in zip(inputs_theta, params_theta):
                self.circuit(input_theta, param_theta)
                theta_circuit = self.device._state
                theta_circuits.append(theta_circuit)
        else:
            for param_theta in params_theta:
                self.circuit(param_theta)
                theta_circuit = self.device._state
                theta_circuits.append(theta_circuit)

        phi_circuits = []
        if inputs_phi is not None:
            for input_phi, param_phi in zip(inputs_phi, params_phi):
                self.circuit(input_phi, param_phi)
                phi_circuit = self.device._state
                phi_circuits.append(phi_circuit)
        else:
            for param_phi in params_phi:
                self.circuit(param_phi)
                phi_circuit = self.device._state
                phi_circuits.append(phi_circuit)

        num_qubits = self.num_wires

        if measure == "meyer-wallach":
            pqc_entanglement_capability = self.meyer_wallach_measure(
                theta_circuits + phi_circuits, num_qubits
            ) / (2 * self.num_samples)
        elif measure == "scott":
            pqc_entanglement_capability = self.scott_measure(
                theta_circuits + phi_circuits, num_qubits
            ) / (2 * self.num_samples)
        else:
            raise ValueError(
                "Invalid measure provided, choose from 'meyer-wallach' or 'scott'"
            )

        return pqc_entanglement_capability

    def entanglement_capability_var(self, save_dir, measure: str = "meyer-wallach", shots: int = 1024):
        ent_values = []
        for i in range(3):
            ent_value = self.entanglement_capability(measure, shots)
            ent_values.append(ent_value)
        ent_df = pd.DataFrame({"ent_mean": [np.mean(ent_values)], "ent_std": [np.std(ent_values)]})
        ent_df.to_csv(os.path.join(save_dir, "entanglement" + ".csv"))
    
    def save(self, ent, save_dir, save_name="entanglement"):
        """Returns plot for expressibility visualization"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        ent_df = pd.DataFrame({"entanglement": [ent]})
        ent_df.to_csv(os.path.join(save_dir, save_name + ".csv"))
        
        
        

