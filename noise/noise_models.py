
import qiskit.providers.aer.noise as noise
from qiskit.providers.aer.noise.errors.readout_error import ReadoutError
import pennylane as qml
from pennylane import qchem
from pennylane import numpy as np
import matplotlib.pyplot as plt
import time

#hyperparams["noise_name"]="depolarizing"
#hyperparams["magnitude"]=0.1 # error rate
#hyperparams["shotnum"]=1024
devicename="qiskit.aer"

## setup noise model
def make_noiseModel(hyperparams):
    #hyperparams["noise_name"]: str: which type of noise: "none", "Readout" or "depolarizing"
    #magnitude: any for "none",float for "Readout",tupel (single-qubit error, two-qubit error) for depolarizing 
    

    my_noise_model = noise.NoiseModel()
    
    if hyperparams["noise_name"] =="none":
        magnitude_name=str(hyperparams["shotnum"])+' shots'
    
    if hyperparams["noise_name"]=="Readout":
        #apply same readout errors to each qubit
        magnitude=[hyperparams["magnitude"] for i in range(4)]
        magnitude_name="all"+str(hyperparams["magnitude"])
        for i in range(4):
        #for i in range(hyperparams["calculation_args"]["wires_to_act_on"]):
            my_noise_model.add_readout_error(ReadoutError([[1-magnitude[i],magnitude[i]],[magnitude[i],1-magnitude[i]]]),[i])
    
    if hyperparams["noise_name"]=="depolarizing": 
        magnitude_single=[hyperparams["magnitude"][0] for i in range (hyperparams["calculation_args"]["wires_to_act_on"])] #probability for single qubit gate on each qubit
        magnitude_two=hyperparams["magnitude"][1] #probability for 2-qubit gate, namely CNOT #2-qubit gate error typically 10*1-qubit gate error
        magnitude_name="allSingle"+str(magnitude_single[0])+"Cnot"+str(magnitude_two)
        for i in range(hyperparams["calculation_args"]["wires_to_act_on"]):
            error_single=noise.depolarizing_error(magnitude_single[i],1)
            my_noise_model.add_quantum_error(error_single, ['u1', 'u2', 'u3'], [i]) #local
        error_two=noise.depolarizing_error(magnitude_two,2)
        my_noise_model.add_all_qubit_quantum_error(error_two, ['cx']) #applied to all qubits
    
    if hyperparams["noise_name"]=="dephasing":
        magnitude_name='all'+str(hyperparams["magnitude"])
        for i in range(hyperparams["calculation_args"]["wires_to_act_on"]):
            error_dephasing=noise.phase_damping_error(hyperparams["magnitude"])
            my_noise_model.add_quantum_error(error_dephasing, ['u1','u2','u3'],[i])
    
    if hyperparams["noise_name"]=="amplitudeDamping":
        magnitude_name='all'+str(hyperparams["magnitude"])
        for i in range(hyperparams["calculation_args"]["wires_to_act_on"]):
            error_damping=noise.amplitude_damping_error(hyperparams["magnitude"])
            my_noise_model.add_quantum_error(error_damping, ['u1','u2','u3'],[i])
    
    return my_noise_model, magnitude_name