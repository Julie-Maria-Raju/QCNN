from utils.pennylane_qiskit import pennylane_to_qiskit_ansatz, op_build_linear
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import Estimator
from qiskit.algorithms.gradients import SPSAEstimatorGradient
#from qiskit_machine_learning.neural_networks import EstimatorQNN
import pennylane.numpy as np
from qiskit.opflow import PauliSumOp
from qiskit.utils import algorithm_globals
#from qiskit_machine_learning.neural_networks import EffectiveDimension


class EffectiveDimensionQiskit:
    """
    This class computes the global effective dimension for a Pennylane circuit
    following the definition used in [1].

        **References**
        [1]: Abbas et al., The power of quantum neural networks.
        `The power of QNNs <https://arxiv.org/pdf/2011.00027.pdf>`__.
    """

    def __init__(
        self,
        encoding,
        ansatz,
        num_qubits,
        num_inputs,
        num_weights,
        dataset_size,
        circuit_layers,
        data_reuploading
    ) -> None:
        """
        Args:
            encoding: The PQC encoding.
            ansatz: The PQC architecture.
            num_qubits: The number of qubits in the PQC.
            num_inputs: The number of input values in the PQC.
            num_weights: The number of (trainable) weights in the PQC.
            dataset_size: The number of training data samples in the dataset.
            circuit_layers: Number of variational circuit layers
            data_reuploading: Reuploads the same input data sequentially

        """
        self.encoding = encoding
        self.ansatz = ansatz
        self.num_qubits = num_qubits
        self.num_inputs = num_inputs
        self.num_weights = num_weights
        self.dataset_size = dataset_size
        self.circuit_layers = circuit_layers
        self.data_reuploading = data_reuploading


    def _compose_circuit(self):
        wires_to_act_on = list(range(self.num_qubits))
        encoding, _ = pennylane_to_qiskit_ansatz("x", self.encoding, self.num_inputs)
        ansatz, _ = pennylane_to_qiskit_ansatz(
                    "theta", self.ansatz, self.num_weights
                )
        circuit = QuantumCircuit(len(wires_to_act_on))
        circuit.compose(encoding.to_instruction(), inplace=True)
        circuit.compose(ansatz.to_instruction(), inplace=True)
        if self.circuit_layers > 1:
            if self.data_reuploading:
                for i in range(self.circuit_layers-1):
                    _ans, _ = pennylane_to_qiskit_ansatz(
                        "theta" + str(i), self.ansatz, self.num_weights
                    ) 
                    circuit.compose(encoding.to_instruction(), inplace=True)
                    circuit.compose(_ans.to_instruction(), inplace=True)
                    ansatz.compose(_ans.to_instruction(), inplace=True)
            else:
                for i in range(self.circuit_layers-1):
                    _ans, _ = pennylane_to_qiskit_ansatz(
                        "theta" + str(i), self.ansatz, self.num_weights
                    )
                    circuit.compose(_ans.to_instruction(), inplace=True)  
                    ansatz.compose(_ans.to_instruction(), inplace=True)

        return circuit, encoding, ansatz

    def get_effective_dimension(self):
        
        circuit, encoding, ansatz = self._compose_circuit()
        op_list = op_build_linear()
        estimator = Estimator()
        
        qnn = EstimatorQNN(
            circuit=circuit,
            estimator=estimator,
            observables=[PauliSumOp.from_list([op]) for op in op_list],
            input_params=encoding.parameters,
            weight_params=ansatz.parameters,
            input_gradients=True,
        )

        input_samples = algorithm_globals.random.uniform(
            -1, 1, size=(10, qnn.num_inputs)
        )
        weight_samples = algorithm_globals.random.uniform(
            -np.pi, np.pi, size=(10, qnn.num_weights)
        )

        global_ed = EffectiveDimension(
            qnn=qnn, weight_samples=weight_samples, input_samples=input_samples
        )

        # Fisher Information Matrix data generation.
        d = qnn.num_weights
        grads, output = global_ed.run_monte_carlo()
        fishers = global_ed.get_fisher_information(gradients=grads, model_outputs=output)
        fisher_trace = np.trace(np.average(fishers, axis=0)) #compute the trace with all fishers
        # average the fishers over the num_inputs to get the empirical fishers
        fisher = np.average(np.reshape(fishers, (10, 10, d, d)), axis=1)  # 
        f_hat = d * fisher / fisher_trace

        effdim = global_ed.get_effective_dimension(self.dataset_size)

        return effdim / qnn.num_weights, f_hat

