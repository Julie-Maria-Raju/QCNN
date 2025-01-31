class CircuitComponents:
    """ Parent class of the Components passed to generate_circuit. """

    def __init__(self):
        self.required_qubits = None
        self.name = None
        self.available_encoders = ["Threshold_Encoder", "NEQR", "FRQI_for_2x2", "Amplitude_Encoder", "Angle_Encoder", "Higher_Order_Encoder", "Custom_Higher_Order_Encoder", "Mottonen_Encoder", "Custom_Amplitude_Encoder"]
        self.available_calculations = ["Randomlayer", "StronglyEntanglingLayer", "Nothing", "Sequence_CNOTs", "Sequence_RX_CNOTs", "strongEnt_Rot_CRY", "quantumTemplate","Sequence_ancilla_RX_CNOTs","Sequence_mid_measure"]
        self.available_measurements = ["Uniform_gate_measurements", "Prob_measurement"]

    def circuit(self, weights):
        pass
