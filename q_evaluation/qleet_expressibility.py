import qiskit
import qleet


def run_qleet_baseline():
    params = [qiskit.circuit.Parameter(r"$θ_1$"),
              qiskit.circuit.Parameter(r"$θ_2$")]

    qiskit_circuit = qiskit.QuantumCircuit(1)
    qiskit_circuit.h(0)
    qiskit_circuit.rz(params[0], 0)
    qiskit_circuit.rx(params[1], 0)
    qiskit_circuit.draw("mpl")

    qiskit_descriptor = qleet.interface.circuit.CircuitDescriptor(
        circuit=qiskit_circuit, params=params, cost_function=None
    )
    qiskit_expressibility = qleet.analyzers.expressibility.Expressibility(
        qiskit_descriptor, samples=100
    )

    expr_jsd = qiskit_expressibility.expressibility("jsd")
    qiskit_expressibility.plot()


run_qleet_baseline()
