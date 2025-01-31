#%% do the qiskit implementation of mid-circuit measurements:
from utils.pennylane_qiskit import pennylane_to_qiskit_ansatz, op_build_linear
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import Estimator
from qiskit.algorithms.gradients import SPSAEstimatorGradient
from qiskit_machine_learning.neural_networks import EstimatorQNN
import pennylane.numpy as np
from qiskit.opflow import PauliSumOp
from qiskit.utils import algorithm_globals
from qiskit_machine_learning.neural_networks import EffectiveDimension

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import Parameter
#https://quantumcomputing.stackexchange.com/questions/9372/how-to-implement-if-statement-based-on-measurement-results-in-qiskit
#def tryout():
    #encoding, _ = pennylane_to_qiskit_ansatz("x", encoding, num_inputs)

theta1 = Parameter('θ1')
theta2 = Parameter('θ2')
theta3 = Parameter('θ3')
theta4 = Parameter('θ4')
theta5 = Parameter('θ5')
theta6 = Parameter('θ6')
circuit = QuantumCircuit(4, 1)
#circuit.compose(encoding.to_instruction(), inplace=True)

circuit.measure(0, 0)
circuit.rx(theta1, 1).c_if(0, 1)
circuit.rx(theta2, 2).c_if(0, 1)
circuit.rx(theta3, 3).c_if(0, 1)
circuit.measure(1, 0)
circuit.rx(theta4, 2).c_if(0, 1)
circuit.rx(theta5, 3).c_if(0, 1)
circuit.measure(2, 0)
circuit.rx(theta6, 3).c_if(0, 1)



circuit.draw(output="mpl", idle_wires=False)

#%%
from qiskit.circuit import QuantumCircuit, Qubit, Clbit
bits = [Qubit(), Qubit(), Qubit(), Clbit(), Clbit()]
qc = QuantumCircuit(bits)

qc.h(0)
qc.cx(0, 1)
qc.measure(0, 0)
qc.h(0)
qc.cx(0, 1)
qc.measure(0, 1)

with qc.if_test((bits[3], 0)) as else_:
    qc.x(2)
with else_:
    qc.h(2)
    qc.z(2)

qc.draw(output="mpl", idle_wires=False)
# %%
