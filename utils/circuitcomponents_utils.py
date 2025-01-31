
from encoders.amplitude_encoder import Amplitude_Encoder
from encoders.custom_amplitude_encoder import Custom_Amplitude_Encoder
from encoders.higher_order_encoder import Higher_Order_Encoder
from encoders.custom_higher_order_encoder import Custom_Higher_Order_Encoder
from encoders.angle_encoder import Angle_Encoder
from calculations.random_layer import RandomLayer
from calculations.strongly_entangling_layer import StronglyEntanglingLayer
from calculations.custom_entanglements import Sequence_CNOTs, Sequence_RX_CNOTs, Sequence_ancilla_RX_CNOTs, strongEnt_Rot_CRY, quantumTemplate, Sequence_ancilla_RX_CZs, Sequence_ancilla_RX_CYs
from calculations.midcircuit_measurement import Sequence_mid_measure
from calculations.nothing import Nothing
from calculations.pooling_layer import PoolingLayer, ConvPoolingLayer, ConvPoolingLayerHad, ConvPoolingLayerBounded
from calculations.characterized_circuits import Circuit_2, Circuit_3, Circuit_4, Circuit_5, Circuit_6, Circuit_10, Circuit_14, Circuit_19
from measurements.uniform_gate import UniformGateMeasurements
from q_evaluation.effective_dimension_qiskit import EffectiveDimensionQiskit
from noise.noise_models import make_noiseModel

import pennylane as qml
import pandas as pd
import os
import numpy as np
import math

circuit_dict = {
    "Amplitude_Encoder": Amplitude_Encoder,
    "Custom_Amplitude_Encoder": Custom_Amplitude_Encoder,
    "Higher_Order_Encoder": Higher_Order_Encoder,
    "Custom_Higher_Order_Encoder": Custom_Higher_Order_Encoder,
    "Angle_Encoder":Angle_Encoder,
    "RandomLayer": RandomLayer,
    "StronglyEntanglingLayer": StronglyEntanglingLayer,
    "Sequence_ancilla_RX_CNOTs":Sequence_ancilla_RX_CNOTs,
    "Sequence_ancilla_RX_CYs":Sequence_ancilla_RX_CYs,
    "Sequence_ancilla_RX_CZs":Sequence_ancilla_RX_CZs,
    "Sequence_CNOTs": Sequence_CNOTs,
    "Sequence_RX_CNOTs": Sequence_RX_CNOTs,
    "strongEnt_Rot_CRY": strongEnt_Rot_CRY,
    "quantumTemplate": quantumTemplate,
    "Nothing": Nothing,
    "PoolingLayer": PoolingLayer,
    "ConvPoolingLayer": ConvPoolingLayer,
    "ConvPoolingLayerHad": ConvPoolingLayerHad,
    "ConvPoolingLayerBounded": ConvPoolingLayerBounded,
    "Circuit_2": Circuit_2,
    "Circuit_3": Circuit_3,
    "Circuit_4": Circuit_4,
    "Circuit_5": Circuit_5,
    "Circuit_6": Circuit_6,
    "Circuit_10": Circuit_10,
    "Circuit_14": Circuit_14,
    "Circuit_19": Circuit_19,
    "UniformGateMeasurements": UniformGateMeasurements,
    "Sequence_mid_measure":Sequence_mid_measure,
    "noise_model": make_noiseModel,
}



def generate_circuit(encoding, calculation, measurement=None, circuit_layers=1, data_reupload=False):
    """ Combine the three steps encoding, calculation and measurement into a singular pennylane circuit.
        Allows for plug in testing of different options in our QNN.
        All inputs must be sequences of pennylane gates. Measurements can be included directly in calculation
        or explicitly in measurement.
    """
    if measurement is not None:
        if circuit_layers == 1:
            def func(inputs, weights):
                encoding(inputs)
                calculation(weights[0])
                result = measurement()
                return result
        elif circuit_layers > 1:
            if data_reupload == False:
                def func(inputs, weights):
                    encoding(inputs)
                    for i in range(circuit_layers):
                        calculation(weights[i])
                    result = measurement()
                    return result
            elif data_reupload == True:
                def func(inputs, weights):
                    for i in range(circuit_layers):
                        encoding(inputs)
                        calculation(weights[i])
                    result = measurement()
                    return result
    else: # only one layer here because some qubits are traced out
        def func(inputs, weights):
            encoding(inputs)
            result = calculation(weights[0])
            return result
    return func




def generate_circuit_pre_encoded_input(calculation, q_bits, measurement=None):
    if measurement is not None:
        def calc(weights):
            calculation(weights)
            return measurement()
    else:
        calc = calculation

    def pre_encoded_calc(inputs, weights):
        qml.QubitStateVector(inputs, wires=list(range(q_bits)))
        inputs = inputs.float()
        weights = weights.float()  # added that here... maybe not correct
        results = calc(weights)
        return results

    return pre_encoded_calc


def generate_corresponding_circuit(hyperparams, weights_initialized=None, encoding=True, data_3D=True, rotation_factor=math.pi):
    """ Wrap the entire circuit creation in here."""
    try:
        encoder = circuit_dict[hyperparams["encoder"]](filter_length=hyperparams["filter_length"], data_3D=data_3D, rotation_factor=rotation_factor,
                                                       **hyperparams["encoder_args"])
        calculation = circuit_dict[hyperparams["calculation"]](encoder.required_qubits,
                                                               **hyperparams["calculation_args"])
        if hyperparams["measurement"] == "None":
            measurement_circuit = None
        else:
            measurement = circuit_dict[hyperparams["measurement"]](encoder.required_qubits, **hyperparams["measurement_args"]) #change 
            measurement_circuit = measurement.circuit


        if not hyperparams["trainable"]:
            calculation.circuit = make_untrainable(
                calculation.circuit, weights_initialized)


        if encoding:
            return generate_circuit(encoder.circuit, calculation.circuit, measurement_circuit, hyperparams["circuit_layers"], hyperparams["data_reuploading"])
        else:
            return generate_circuit_pre_encoded_input(calculation.circuit, encoder.required_qubits, measurement_circuit)

    except KeyError:
        raise Exception(
            f"Most likely a circuit specified could not be found. Available are {circuit_dict.keys()}")



def make_untrainable(circuit, weights_initialized):
    """ Render a circuit untrainable, by ignoring the passed parameters called weights, since pytorch differentiates wrt
    those only. """

    def circuit_var(weights):
        circuit(weights_initialized)

    return circuit_var


def get_wires_number(hyperparams, data_3D=False):
    encoder = circuit_dict[hyperparams["encoder"]](filter_length=hyperparams["filter_length"], data_3D=data_3D,
                                                   **hyperparams["encoder_args"])
    return encoder.required_qubits


def generate_status_encoding_circuit(hyperparams):
    try:
        encoding = circuit_dict[hyperparams["encoder"]](filter_length=hyperparams["filter_length"],
                                                        **hyperparams["encoder_args"])

        def get_encoded_state(inputs):
            encoding.circuit(inputs)
            return qml.state()

        return get_encoded_state
    except KeyError:
        raise Exception(
            f"Most likely a circuit specified could not be found. Available are {circuit_dict.keys()}")


def get_circuit_weights_shape(hyperparams):
    if hyperparams["calculation"] == "quantumTemplate":                     # Empty quantumTemplate doesn't need weights.
        if len(hyperparams["calculation_args"]["wires_to_act_on"]) == 0:    # 'calculation_args': {'wires_to_act_on': []}
            if hyperparams["encoder"] == "Threshold_Encoder" or hyperparams["encoder"] == "Higher_Order_Encoder":
                weights_shape = (0, 4)
        else:
            weights_shape = (0, len(hyperparams["calculation_args"]["wires_to_act_on"])) 

    elif hyperparams["calculation"] == "Sequence_mid_measure":
        weights_shape = (1, 6)

    elif hyperparams["calculation"] == "PoolingLayer":
        weights_shape = (1, 3, 2)
    
    elif hyperparams["calculation"] == "ConvPoolingLayer":
        weights_shape = (1, 3, 4)
    
    elif hyperparams["calculation"] == "ConvPoolingLayerHad":
        weights_shape = (1, 3, 4)
    
    elif hyperparams["calculation"] == "ConvPoolLayerBounded":
        weights_shape = (1, 4, 4)

    elif hyperparams["calculation"] == "strongEnt_Rot_CRY":                 # Needs 4 weights per wire.
        if len(hyperparams["calculation_args"]["wires_to_act_on"]) == 0:    # 'calculation_args': {'wires_to_act_on': []}
            if hyperparams["encoder"] == "Threshold_Encoder" or hyperparams["encoder"] == "Higher_Order_Encoder":
                weights_shape = (4, 4) 
        else:
            weights_shape = (4, len(hyperparams["calculation_args"]["wires_to_act_on"])) 

    elif hyperparams["calculation"] == "RandomLayer":
        weights_shape = (hyperparams['circuit_layers'],
                         hyperparams['n_rotations'])

    elif hyperparams["calculation"] == "StronglyEntanglingLayer":
        weights_shape = (qml.StronglyEntanglingLayers.shape(n_layers=hyperparams['circuit_layers'],
                         n_wires=hyperparams['n_rotations']))

    elif hyperparams["calculation"] == "Sequence_CNOTs":
        weights_shape = (hyperparams['circuit_layers'], 1)
    elif hyperparams["calculation"] == "Sequence_ancilla_RX_CNOTs" or hyperparams["calculation"] == "Sequence_ancilla_RX_CYs" or hyperparams["calculation"] == "Sequence_ancilla_RX_CZs":
        weights_shape = (1, 4)
    elif hyperparams["calculation"].startswith("Sequence_"):
        if not hyperparams["calculation_args"]["wires_to_act_on"]:
            encoder = circuit_dict[hyperparams["encoder"]](filter_length=hyperparams["filter_length"],
                                                           data_3D=hyperparams["do_3D_conv"],
                                                           rotation_factor=hyperparams["rotation_factor"],
                                                           **hyperparams["encoder_args"])
            weights_shape = (hyperparams['circuit_layers'], encoder.required_qubits)
        else:
            if hyperparams["calculation"] == "Sequence_RX_CNOTs":
                weights_shape = (hyperparams['circuit_layers'], 
                                 len(hyperparams["calculation_args"]["wires_to_act_on"]))
            elif hyperparams["calculation"] == "Sequence_RX_RY_CNOTs":
                weights_shape = (hyperparams['circuit_layers'],
                                 2*len(hyperparams["calculation_args"]["wires_to_act_on"]))
            else:
                print("ERROR: You did not choose a supported calculation! Exiting...")
                exit()

    elif hyperparams["calculation"].startswith("Circuit_"):
        encoder = circuit_dict[hyperparams["encoder"]](filter_length=hyperparams["filter_length"],
                                                           data_3D=hyperparams["do_3D_conv"],
                                                           rotation_factor=hyperparams["rotation_factor"],
                                                           **hyperparams["encoder_args"])
        if hyperparams["calculation"] == "Circuit_2":
            weights_shape = (hyperparams['circuit_layers'], 2 * encoder.required_qubits)
        elif hyperparams["calculation"] == "Circuit_3":
            weights_shape = (hyperparams['circuit_layers'], 3 * encoder.required_qubits - 1)
        elif hyperparams["calculation"] == "Circuit_4":
            weights_shape = (hyperparams['circuit_layers'], 3 * encoder.required_qubits - 1)
        elif hyperparams["calculation"] == "Circuit_5":
            weights_shape = (hyperparams['circuit_layers'], 4 * encoder.required_qubits + 4 * (encoder.required_qubits - 1))
        elif hyperparams["calculation"] == "Circuit_6":
            weights_shape = (hyperparams['circuit_layers'], 4 * encoder.required_qubits + 4 * (encoder.required_qubits - 1))
        elif hyperparams["calculation"] == "Circuit_10":
            weights_shape = (hyperparams['circuit_layers'], 2 * encoder.required_qubits)
        elif hyperparams["calculation"] == "Circuit_14":
            weights_shape = (hyperparams['circuit_layers'], 4 * encoder.required_qubits)
        elif hyperparams["calculation"] == "Circuit_19":
            weights_shape = (hyperparams['circuit_layers'], 3 * encoder.required_qubits)

    else:
        print("ERROR: You did not choose a supported calculation! Exiting...")
        exit()
    return weights_shape

def calculate_effective_dimension(hyperparams, num_inputs, dataset_size, save_dir, data_3D=True):
    encoder = circuit_dict[hyperparams["encoder"]](filter_length=hyperparams["filter_length"], data_3D=data_3D,
                                                    **hyperparams["encoder_args"])
    calculation = circuit_dict[hyperparams["calculation"]](encoder.required_qubits,
                                                            **hyperparams["calculation_args"])
    ed = EffectiveDimensionQiskit(encoding=encoder, ansatz=calculation, num_qubits=encoder.required_qubits, num_inputs=num_inputs, num_weights=np.prod(get_circuit_weights_shape(hyperparams)), dataset_size=dataset_size,circuit_layers = hyperparams["circuit_layers"], data_reuploading  = hyperparams["data_reuploading"])
    ed_value, f_hat = ed.get_effective_dimension()
    ed_df = pd.DataFrame({"effective_dim": [ed_value]})
    ed_df.to_csv(os.path.join(save_dir, "effective_dim" + ".csv"))
    np.save(os.path.join(save_dir, "fisher_information" + ".npy"), f_hat)

def calculate_effective_dimension_var(hyperparams, num_inputs, dataset_size, save_dir, data_3D=True):
    encoder = circuit_dict[hyperparams["encoder"]](filter_length=hyperparams["filter_length"], data_3D=data_3D,
                                                    **hyperparams["encoder_args"])
    calculation = circuit_dict[hyperparams["calculation"]](encoder.required_qubits,
                                                            **hyperparams["calculation_args"])
    ed = EffectiveDimensionQiskit(encoding=encoder, ansatz=calculation, num_qubits=encoder.required_qubits, num_inputs=num_inputs, num_weights=np.prod(get_circuit_weights_shape(hyperparams)), dataset_size=dataset_size,circuit_layers = hyperparams["circuit_layers"], data_reuploading  = hyperparams["data_reuploading"])

    ed_values = []
    for i in range(3):
        ed_value, f_hat = ed.get_effective_dimension()
        ed_values.append(ed_value)
        
    ed_df = pd.DataFrame({"effective_dim_mean": [np.mean(ed_values)], "effective_dim_std": [np.std(ed_values)]})
    ed_df.to_csv(os.path.join(save_dir, "effective_dim" + ".csv"))
    np.save(os.path.join(save_dir, "fisher_information" + ".npy"), f_hat)
    
    
