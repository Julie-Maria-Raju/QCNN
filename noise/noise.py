import qiskit.providers.aer.noise as noise
from qiskit.providers.aer.noise.errors.readout_error import ReadoutError
import pennylane as qml
from pennylane import qchem
from pennylane import numpy as np
import matplotlib.pyplot as plt
import time

## settings
noise_name="depolarizing"
magnitude=0.5
shotnum=1024




## setup noise model
def make_noiseModel(noise_name, magnitude, shotnum):
    #noise_name: str: which type of noise: "none", "Readout" or "depolarizing"
    #magnitude: any for "none",float for "Readout",tupel (single-qubit error, two-qubit error) for depolarizing
    my_noise_model = noise.NoiseModel()
    if noise_name=="none":
        magnitude_name=str(shotnum)+' shots'
    if noise_name=="Readout":
        #apply same readout errors to each qubit
        magnitude=[magnitude for i in range(4)]
        magnitude_name="all"+str(magnitude[0])
        for i in range(4):
            my_noise_model.add_readout_error(ReadoutError([[1-magnitude[i],magnitude[i]],[magnitude[i],1-magnitude[i]]]),[i])
    if noise_name=="depolarizing":
        #magnitude_single=[magnitude[0] for i in range(4)] #probability for single qubit gate on each qubit
        magnitude_two= magnitude #magnitude[1] #probability for 2-qubit gate, namely CNOT #2-qubit gate error typically 10*1-qubit gate error
        magnitude_name= "Cnot"+str(magnitude_two) #"allSingle"+str(magnitude_single[0])+"Cnot"+str(magnitude_two)
        #for i in range(4):
            #error_single=noise.depolarizing_error(magnitude_single[i],1)
            #my_noise_model.add_quantum_error(error_single, ['u1', 'u2', 'u3'], [i]) #local
        error_two=noise.depolarizing_error(magnitude_two,2)
        my_noise_model.add_all_qubit_quantum_error(error_two, ['cx']) #applied to all qubits
    if noise_name=="dephasing":
        magnitude_name='all'+str(magnitude)
        for i in range(4):
            error_dephasing=noise.phase_damping_error(magnitude)
            my_noise_model.add_quantum_error(error_dephasing, ['u1','u2','u3'],[i])
    if noise_name=="amplitudeDamping":
        magnitude_name='all'+str(magnitude)
        for i in range(4):
            error_damping=noise.amplitude_damping_error(magnitude)
            my_noise_model.add_quantum_error(error_damping, ['u1','u2','u3'],[i])
    return my_noise_model, magnitude_name