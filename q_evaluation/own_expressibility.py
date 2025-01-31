# %%
import itertools
import typing
import numpy as np
from matplotlib import pyplot as plt
import torch
import pennylane as qml
from qiskit.quantum_info import state_fidelity, Statevector
import os
import pandas as pd

# Inspired from the qleet librairy https://github.com/QLemma/qleet


class Expressibility():
    def __init__(self, circuit, device, num_wires, num_samples, inputs: np.ndarray, params: np.ndarray, input_is_bin=False):
        """Calculates expressibility of a parameterized quantum circuit

        Args:
            circuit: pennylane quantum circuit
            dev: circuit device
            num_wires: number of qubits in the circuit
            inputs: inputs of the circuit
            params: (trainable) parameters of the circuit
            num_samples: number of samples used in the generation of the parameters to calculate state fidelity
        """
        self.circuit = circuit
        self.device = device
        self.num_wires = num_wires
        self.num_samples = num_samples
        self.inputs = inputs
        self.input_is_bin = input_is_bin
        self.params = params

    def gen_params(self):
        """Generates parameters for the calculation of expressibility

        Returns:
            inputs_theta (np.array): first list of inputs for the circuit
            inputs_phi (np.array): second list of inputs for the circuit
            params_theta (np.array): first list of parameters for the circuit
            params_phi (np.array): second list of parameters for the circuit

        Note:
            Compared with the qleet librairy, here we are adding the input values because we want the expressibility of the whole circuit and not just the variational part.
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

    def prob_pqc(self) -> np.ndarray:
        """Returns probability density function of fidelities for PQC

        Returns:
            fidelities (np.array): np.array of fidelities
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

        fidelity = np.array(
            [
                state_fidelity(rho_a, rho_b)
                for rho_a, rho_b in itertools.product(theta_circuits, phi_circuits)
            ]
        )
        return np.array(fidelity)

    def kl_divergence(self, prob_a: np.ndarray, prob_b: np.ndarray) -> float:
        """Returns KL divergence between two probabilities"""
        prob_a[prob_a == 0] = 1e-10
        kl_div = np.sum(np.where(prob_a != 0, prob_a *
                        np.log(prob_a / prob_b), 0))
        return typing.cast(float, kl_div)

    def prob_haar(self) -> np.ndarray:
        """Returns probability density function of fidelities for Haar Random States"""
        fidelity = np.linspace(0, 1, self.num_samples)
        return (2 ** self.num_wires - 1) * (1 - fidelity + 1e-8) ** (2 ** self.num_wires - 2)

    def calculate_expressibility(self, measure: str = "kld") -> float:
        """Returns expressibility for the circuit
            Expr = D_{KL}(\hat{P}_{PQC}(F; \theta) | P_{Haar}(F))\\
            Expr = D_{\sqrt{JSD}}(\hat{P}_{PQC}(F; \theta) | P_{Haar}(F))

        Args:
            measure: specification for the measure used in the expressibility calculation (only kld is supported as of now)
            shots: number of shots for circuit execution

        Returns:
            pqc_expressibility: float, expressibility value
        """
        haar = self.prob_haar()
        haar_prob: np.ndarray = haar / float(haar.sum())
        haar_prob[haar_prob == 0] = 1.0e-300

        if len(self.params) > 0:
            fidelity = self.prob_pqc()
        else:
            fidelity = np.ones(self.num_samples ** 2)

        bin_edges: np.ndarray
        pqc_hist, bin_edges = np.histogram(
            fidelity, self.num_samples, range=(0, 1), density=True
        )
        pqc_prob: np.ndarray = pqc_hist / float(pqc_hist.sum())

        if measure == "kld":
            pqc_expressibility = self.kl_divergence(pqc_prob, haar_prob)
        # elif measure == "jsd":
        #    pqc_expressibility = jensenshannon(pqc_prob, haar_prob, 2.0)
        else:
            raise ValueError(
                "Invalid measure provided, choose 'kld'")
        plot_data = [haar_prob, pqc_prob, bin_edges]
        #self.expr = pqc_expressibility

        return pqc_expressibility, plot_data
    
    def calculate_expressibility_var(self, save_dir, measure: str = "kld"):
        expr_values = []
        for i in range(3):
            expr_value, plot_data = self.calculate_expressibility(measure)
            expr_values.append(expr_value)
        expr_df = pd.DataFrame({"expr_mean": [np.mean(expr_values)], "expr_std": [np.std(expr_values)]})
        expr_df.to_csv(os.path.join(save_dir, "expressibility" + ".csv"))
        return expr_value, plot_data

    def plot(self, plot_data, expr, save_dir, save_name="expressibility", figsize=(6, 4), dpi=300, **kwargs):
        """Returns plot for expressibility visualization"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        haar_prob, pqc_prob, bin_edges = plot_data

        bin_middles = (bin_edges[1:] + bin_edges[:-1]) / 2.0
        bin_width = bin_edges[1] - bin_edges[0]

        fig = plt.figure(figsize=figsize, dpi=dpi, **kwargs)
        plt.bar(bin_middles, haar_prob, width=bin_width, label="Haar")
        plt.bar(bin_middles, pqc_prob, width=bin_width, label="PQC", alpha=0.6)
        plt.xlim((-0.05, 1.05))
        plt.ylim(bottom=0.0, top=max(max(pqc_prob), max(haar_prob)) + 0.01)
        plt.grid(True)
        plt.title(f"Expressibility: {np.round(expr,5)}")
        plt.xlabel("Fidelity")
        plt.ylabel("Probability")
        plt.legend()
        plt.show()
        plt.savefig(os.path.join(save_dir, save_name + ".png"))

        return fig


def run_baseline():
    num_wires = 1
    dev = qml.device('qiskit.aer', wires=num_wires, shots=1024,
                     backend='statevector_simulator')

    @qml.qnode(dev)
    def circuit(params):
        qml.Hadamard(wires=0)
        qml.RZ(params[0], wires=0)
        qml.RX(params[1], wires=0)
        return qml.expval(qml.PauliZ(0))

    params = np.array([0.5, 0.7])
    expr = Expressibility(circuit, dev, num_wires, None, params, 1000)
    expr_value, plot_data = expr.calculate_expressibility()
    expr.plot(plot_data, expr_value)

# %%
