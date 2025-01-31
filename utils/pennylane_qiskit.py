import inspect
from collections import OrderedDict
import pennylane as qml
import pennylane.numpy as np
from pennylane_qiskit.qiskit_device import QiskitDevice
from qiskit.circuit import ParameterVector, QuantumCircuit, QuantumRegister
from qiskit.converters import circuit_to_dag, dag_to_circuit

def pennylane_to_qiskit_ansatz(name_param, ansatz, num_params):
    """Convert an ansatz from PennyLane to a circuit in Qiskit.

    Args:
        name_param (str) : Name of parameter
        ansatz (PQC): The PQC encoding or circuit
        num_params (int): Number of parameters.

    Returns:
        circuit_ansatz (QuantumCircuit): Translated qiskit quantum circuit
        trainable_params (ParameterVector): List of trainable/input parameters
    """

    if isinstance(ansatz.circuit, (qml.QNode, qml.tape.QuantumTape)):
        raise qml.QuantumFunctionError(
            "The ansatz must be a callable quantum function."
        )
    
    if callable(ansatz.circuit):
        #if len(inspect.getfullargspec(ansatz).args) != 1:
        #    raise qml.QuantumFunctionError("Param should be a single vector.")

        tape = qml.transforms.make_tape(ansatz.circuit)(np.zeros(num_params)).expand(
            depth=5, stop_at=lambda obj: obj.name in QiskitDevice._operation_map
        )

        # Raise exception if there are no operations
        if len(tape.operations) == 0:
            raise qml.QuantumFunctionError("Function contains no quantum operations.")

        params = tape.get_parameters()
        trainable_params = []

        for p in params:
            if qml.math.requires_grad(p):
                trainable_params.append(p)

        num_trainable_params = len(trainable_params)

        all_wires = tape.wires

        # Set the number of qubits
        num_qubits = len(tape.wires)

        circuit_ansatz = qiskit_ansatz(
            name_param, num_trainable_params, num_qubits, all_wires, tape
        )   
        #This method cannot generalize to this type of encoding. We need to build it manually
        if ansatz.name == 'Higher_Order_Encoder' or ansatz.name == 'Custom_Higher_Order_Encoder': 
            
            trainable_params = circuit_ansatz.parameters[:num_params]
            circuit_ansatz =  QuantumCircuit(num_qubits)
            for i in range(num_qubits):circuit_ansatz.h(i);circuit_ansatz.rz(trainable_params[i],i)
            for q1 in range(num_params): 
                for q2 in range(num_params): 
                    if q1 != q2 : circuit_ansatz.rzz((np.pi*trainable_params[q1]*trainable_params[q2]), q1, q2)

    else:
        raise ValueError("Input ansatz is not a quantum function")

    return circuit_ansatz, trainable_params


def qiskit_ansatz(name_param, num_params, num_qubits, wires, tape):
    """Transform a quantum tape from PennyLane to a Qiskit circuit.

    Args:
        name_param (str) : Name of parameter
        num_params (int): Number of parameters.
        num_qubits (int): Number of qubits.
        wires (qml.wire.Wires): Wires used in the tape
        tape (qml.tape.QuantumTape): The quantum tape of the circuit ansatz in PennyLane.

    Returns:
        QuantumCircuit: Qiskit quantum circuit.

    """
    consecutive_wires = qml.wires.Wires(range(num_qubits))
    wires_map = OrderedDict(zip(wires, consecutive_wires))
    # From here: Create the Qiskit ansatz circuit
    params_vector = ParameterVector(name_param, num_params)

    reg = QuantumRegister(num_qubits, "q")
    circuit_ansatz = QuantumCircuit(reg, name="vqe")

    circuits = []

    j = 0
    for operation in tape.operations:
        wires = operation.wires.map(wires_map)
        par = operation.parameters
        operation = operation.name
        mapped_operation = QiskitDevice._operation_map[operation]

        qregs = [reg[i] for i in wires.labels]

        if operation.split(".inv")[0] in ("QubitUnitary", "QubitStateVector"):
            # Need to revert the order of the quantum registers used in
            # Qiskit such that it matches the PennyLane ordering
            qregs = list(reversed(qregs))

        dag = circuit_to_dag(QuantumCircuit(reg, name=""))

        if operation in ("QubitUnitary", "QubitStateVector"):
            # Parameters are matrices
            gate = mapped_operation(par[0])
        else:
            # Parameters for the operation
            if par and qml.math.requires_grad(par[0]):
                op_num_params = len(par)
                par = []
                for num in range(op_num_params):
                    par.append(params_vector[j + num])
                j += op_num_params

            gate = mapped_operation(*par)

        if operation.endswith(".inv"):
            gate = gate.inverse()

        dag.apply_operation_back(gate, qargs=qregs)
        circuit = dag_to_circuit(dag)
        circuits.append(circuit)

    for circuit in circuits:
        circuit_ansatz &= circuit

    return circuit_ansatz

def op_build_linear(num_op=4):
    return [(("I" * i) + "Z" + ("I" * (num_op - 1 - i)), 1) for i in range(num_op)]
