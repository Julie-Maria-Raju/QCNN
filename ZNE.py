from mitiq import benchmarks
from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors.standard_errors import (depolarizing_error,)
from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.providers.aer import AerSimulator
#from pennylane_qiskit import AerDevice
import pennylane as qml
from pennylane import broadcast
import numpy as np
import csv
import os
import qiskit 
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import Gate
from mitiq import Observable
from mitiq.observable.pauli import PauliString
from qiskit.quantum_info import Pauli
from mitiq import pec
from mitiq.interface import convert_to_mitiq
from mitiq.pec.representations.depolarizing import represent_operations_in_circuit_with_local_depolarizing_noise
from mitiq.pec.representations.damping import _represent_operation_with_amplitude_damping_noise
from mitiq.pec.representations.depolarizing import(represent_operation_with_local_depolarizing_noise)
from mitiq.pec.representations import local_depolarizing_kraus, amplitude_damping_kraus
from mitiq.pec.representations import find_optimal_representation
from mitiq.pec.channels import kraus_to_super
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
import qiskit_qasm2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from medmnist import OrganAMNIST,BreastMNIST
from mitiq import Executor
from mitiq.interface.mitiq_qiskit import execute_with_noise, initialized_depolarizing_noise
import pandas as pd
import ast
from timer import Timer
from zennit.composites import EpsilonGammaBox
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.attribution import Gradient
from zennit.image import imgify
import matplotlib.pyplot as plt
from mitiq.zne.scaling import fold_gates_at_random
from mitiq import zne
from functools import partial
from mitiq.zne.inference import RichardsonFactory
from mitiq.zne import mitigate_executor

batch_size =16#8
wires = 4
num_epochs =1
input_size = 784 # 28x28
num_classes = 11 # 0,1,...,9
learning_rate = 0.001
n_train = 16#546
n_test = 100#300#78
n_layers = 1
timer = Timer()



#load MNIST dataset (60,000 training_images and 10,000 test_images)


#CustomNormalize(mean=[0.5], std=[0.5])
transform =transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5], std=[0.5])])#, ttransforms.Normalize(mean=[0.5], std=[0.5])
# Load the dataset
train_data = OrganAMNIST(split='train', transform=transform, download=True)
#OrganAMNIST(split='train', transform=transform, download=True)
test_data = OrganAMNIST(split='val', transform=transform, download=False)
#OrganAMNIST(split='val', transform=transform, download=False)

# Extract labels for train and test datasets
train_labels = [train_data.labels[i][0] for i in range(len(train_data.labels))]
test_labels = [test_data.labels[i][0] for i in range(len(test_data.labels))]

# Perform stratified split for train and test subsets
train_indices = np.arange(len(train_data))
test_indices = np.arange(len(test_data))

# Create a stratified split for the train dataset
train_subset_indices, _ = train_test_split(
    train_indices, train_size=n_train, stratify=train_labels, random_state=56)

# Create a stratified split for the test dataset
test_subset_indices, _ = train_test_split(
    test_indices, train_size=n_test, stratify=test_labels, random_state=56)

# Create subsets using the stratified indices
train_dataset = Subset(train_data, train_subset_indices)
test_dataset = Subset(test_data, test_subset_indices)

# Create class weights based on the train subset label distribution
"""subset_train_labels = [train_data.labels[i][0] for i in train_subset_indices]
n_train_samples = [subset_train_labels.count(value) for value in set(subset_train_labels)]

label_counts={label:subset_train_labels.count(label) for label in set(subset_train_labels)}
max_count=max(label_counts.values())

class_weights = [max_count/ label_counts[label] for label in sorted(label_counts.keys())]"""

# Create DataLoader for both training and testing
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# Example of getting a batch
dataset_iter = iter(train_dataloader)
samples, labels = next(dataset_iter)
print(samples.shape)
"""train_data =OrganAMNIST(split = "train", transform=transform, download = True)
test_data = OrganAMNIST(split= "val", transform=transform, download=False)
#print('Min Pixel Value: {} \nMax Pixel Value: {}'.format(train_data.data.min(), train_data.data.max()))
# Reducing size ( 500 train_images and 300 test_images)
train_dataset = torch.utils.data.Subset(train_data,range(n_train))
test_dataset =torch.utils.data.Subset(test_data,range(n_test))
first_image,first_label = train_dataset[0]
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
dataset_iter = iter(train_dataloader)
samples, labels = next(dataset_iter)
print(samples.shape)"""


#for i in range(num_epochs):
    #weights = torch.load(f"/data/julie.maria.raju/qc-diag/results/OrganAMNIST_20_seeds/Sequence_RX_CNOTs_ Custom_Higher_Order_Encoder_layers_ 1_ 0.785_seed_ 1/weights_epoch_19.pt")

class QLayer:
    def __init__(self):
        wires=4
        self.rotation_factor=0.785
        self.wires_to_act_on=list(range(wires))
    def apply(self,inputs,weights):
        
        flat_image= inputs #inputs.flatten()
        #print(flat_image)
        for i in range(len(self.wires_to_act_on)):
            qml.Hadamard(i)
            qml.RZ(flat_image[i]*self.rotation_factor,wires=i)
        for i in range(len(self.wires_to_act_on)- 1):
            for j in range(i+1,len(self.wires_to_act_on)):
                feature_product = flat_image[i]*flat_image[j]*self.rotation_factor
                qml.CNOT(wires=[i,j])
                qml.RZ(feature_product,wires=j)
                qml.CNOT(wires=[i,j])
        #qml.BasicEntanglerLayers(weights=[[0.5226, 1.2727, -1.2735, 1.2391]], wires=[0,1,2,3], rotation=qml.RX) #np.random.random((n_layers,wires))
        for i in range(len(self.wires_to_act_on)):
            qml.RX(weights[0][i], wires=self.wires_to_act_on[i]) #change item() to run in cpu() also use weights[0][i]
        for i in range(len(self.wires_to_act_on)- 1):
            qml.CNOT(wires=[self.wires_to_act_on[i], self.wires_to_act_on[i+1]])
        if len(self.wires_to_act_on) > 2:
            qml.CNOT(wires=[self.wires_to_act_on[-1],0])
        




"""q_layer=QLayer()
inputs= torch.tensor([0.0431,-0.4902,-0.2627,-0.4196])  
weights=[[0.05226,1.2727,-1.2735,1.2391]]  
print(qml.draw(q_layer.apply)(inputs,weights))
"""
class Hybrid:
    def __init__(self, noise_model=None):
        super().__init__()
        #Initialize the circuit with a given noise model.
        self.dev = qml.device("qiskit.aer", wires=4) #noise-model
        
        self.q_layer =QLayer()
        self.qnode=qml.QNode(self.circuit,self.dev)
        
    def circuit(self,inputs, weights):
        #Define the quantum circuit using the QLayer.
        #print("cir",inputs)
        result=self.q_layer.apply(inputs,weights)
        #measurement = [qml.expval(qml.PauliZ(wires=i)) for i in range(4)]
        return result #measurement  #
    def run(self,inputs,weights):
        #Execute the circuit and return the result.
        return self.qnode(inputs,weights) 
    def get_operations(self):
        #Get operations from the tape of the QNode.
        return self.qnode.tape.operations
    
    def get_qasm(self,inputs,weights):
        #print("gasm",inputs)
        # Create a QuantumTape and apply the circuit
        with qml.tape.QuantumTape() as tape:
            self.circuit(inputs,weights)
        # Convert the tape to OpenQASM
        return tape.to_openqasm()
   







pec=="True"
class Q_Conv_Layer(nn.Module):

    def __init__(self):
        super(Q_Conv_Layer,self).__init__()

        #self.params = np.random.random([6], requires_grad=True)
        self.counter=0
        self.hybrid = Hybrid()
        self.quanv_circ= self.hybrid.circuit
        #weight_shapes = {'weights': [4]}
        #self.torch_circ=qml.qnn.TorchLayer(self.quanv_circ,weight_shapes)
        weights_dict= torch.load('/Users/juliemariaraju/Desktop/Master Thesis/qc-diag-julie-qcnn/results/OrganAMNIST_20_seeds/Sequence_RX_CNOTs_ Custom_Higher_Order_Encoder_layers_ 1_ 0.785_seed_ 1/weights_epoch_9.pt', map_location=torch.device('cpu'))
        #torch.load(f"/data/julie.maria.raju/qc-diag/results/OrganAMNIST_20_seeds/Sequence_RX_CNOTs_ Custom_Higher_Order_Encoder_layers_ 1_ 0.785_seed_ 1/weights_epoch_9.pt")
        #state_dict = weights_dict['state_dict']
        self.weights=weights_dict['qcnn.0.torch_qlayer.weights'].cpu().numpy()
        #self.weights = [v.cpu().numpy() for k, v in state_dict.items() if 'qcnn' in k]
        #print(self.weights)
        #weights_dict['qcnn.0.torch_qlayer.weights'].cpu().numpy() #.flatten()#
        #print(self.weights.shape)
        #self.weights=[[1,1,1,1]]
        #weights =weights_dict
        
        #self.torch_layer = qml.qnn.TorchLayer(self.hybrid.qnode,weight_shapes=weight_shapes)

    timer.start_section("Forward")
    def forward(self,image):
        out = np.zeros((image.size(dim=0),4,14, 14)) #(4,14, 14)
        
        if image.size(dim=2)==28: #change 20 to 28

        # Loop over the coordinates of the top-left pixel of 2X2 squares
            """def parse_list_str(list_str):
                    try:
                        # Try parsing normally
                        return np.array(ast.literal_eval(list_str))
                    except (ValueError, SyntaxError):
                        # If that fails, handle unquoted lists by splitting on spaces or commas
                        list_str = list_str.replace('[', '').replace(']', '')
                        return np.array([float(x) for x in list_str.split()])

            # Read CSV file
            df = pd.read_csv('/Users/juliemariaraju/Desktop/Master Thesis/qc-diag-julie-qcnn/results/1_ZNE_100_RF_[1,3,5,7].csv', header=None, names=['epsilon', 'expectation_values'])
            df['parsed_values'] = df['expectation_values'].apply(parse_list_str)
            zne_results = df[df['epsilon'] == 0.0495]['parsed_values'].values"""
            #print(pec_values.shape)
        
            #print(pec_values[:4])
            
            for b in range(image.size(dim=0)):
                for j in range(0, 28, 2):
                    for k in range(0, 28, 2):
                    
                    # Process a squared 2x2 region of the image with a quantum circuit
                        """q_results = self.quanv_circ(inputs = torch.tensor(

                            [
                                image[b,0,j, k],
                                image[b,0,j, k + 1],
                                image[b,0, j + 1, k],
                                image[b,0, j + 1, k + 1]
                            ]), weights=weights)"""
                        timer.start_section("qasm code")
                        #print(image[0])
                        ##stop
                        qasm_code = self.hybrid.get_qasm(inputs = (                     #self.hybrid.get_qasm

                        [
                            image[b,0,j, k].item(),
                            image[b,0,j, k + 1].item(),
                            image[b,0, j + 1, k].item(),
                            image[b,0, j + 1, k + 1].item()
                        ]),weights=self.weights)
                        
                        timer.end_section()
                        timer.start_section("Qiskit")
                        qc =qiskit_qasm2.loads(qasm_code, include_path=('.',), custom_instructions=(), custom_classical=(), strict=False)
                        timer.end_section()
                        #print(qc)
                        #stop
                        Z = np.array([[1, 0], [0, -1]], dtype=np.complex64)
                        I = np.eye(2, dtype=np.complex64)

                        # Construct observables for each qubit in a 4-qubit system
                        observable_0 = np.kron(Z, np.kron(I, np.kron(I, I)))  # Z for qubit 0
                        observable_1 = np.kron(I, np.kron(Z, np.kron(I, I)))  # Z for qubit 1
                        observable_2 = np.kron(I, np.kron(I, np.kron(Z, I)))  # Z for qubit 2
                        observable_3 = np.kron(I, np.kron(I, np.kron(I, Z)))  # Z for qubit 3

                        # Combine observables into a list
                        observable = [observable_3, observable_2, observable_1, observable_0]
                        
                        def execute_qc(qc,epsilon=0.05):
                            #0.049
                            qc_copy=qc.copy()
                            
                            noise_model = NoiseModel()
                            #noise_model.add_all_qubit_quantum_error(depolarizing_error(epsilon, 1), ["rz"])
                            noise_model.add_all_qubit_quantum_error(depolarizing_error(epsilon, 1), ["rz"])
                            return execute_with_noise(qc_copy, observable,noise_model=noise_model)#
                        # def execute(circuit, noise_level=0.01):
                        #     Returns Tr[ρ |0⟩⟨0|] where ρ is the state prepared by the circuit
                        #     executed with depolarizing noise.
                        #     
                        #     noisy_circuit = circuit.with_noise(depolarize(p=noise_level))
                        #     rho = DensityMatrixSimulator().simulate(noisy_circuit).final_density_matrix
                        #     return rho[0, 0].real
                        
                        def save_to_csv(filename, epsilon, result):
                            # Ensure the directory exists
                            directory = os.path.dirname(filename)
                            if directory:
                                os.makedirs(directory, exist_ok=True)
                            
                            # Append results to CSV
                            with open(filename, mode="a", newline="") as file:  # Open in append mode
                                writer = csv.writer(file)
                                writer.writerow([epsilon, result])

                        # Filename with directory structure
                        filename = "1_ZNE_100_RF_[1,3,5,7]_fold_all.csv"
                        filepath = os.path.join("results",filename)
                    
                        # Add a header only if the file doesn't exist
                        if not os.path.exists(filepath):
                            directory = os.path.dirname(filepath)
                            if directory:
                                os.makedirs(directory, exist_ok=True)
                            with open(filename, mode="a", newline="") as file:
                                writer = csv.writer(file)
                                writer.writerow(["Noise Epsilon", "Result"])
                        
                        timer.start_section("noiseless_result")
                        noiseless_result = execute_qc(qc, epsilon=0)
                        save_to_csv(filepath, 0, noiseless_result)
                        timer.end_section()
                        
                        # scale_factors=[1, 3,5]
                        # for scale_factor in scale_factors:
                        #     scaled_circuit = zne.scaling.fold_gates_at_random(qc, scale_factor)
                            
                        # expectation_values = [execute_qc(circ) for circ in scaled_circuit]
                        # fac = RichardsonFactory(scale_factors=[1, 3, 5])
                        # for s, e in zip(scale_factors, expectation_values):
                        #     fac.push({"scale_factor": s}, e)

                        # # Evaluate the extrapolation
                        # fac.reduce()"""
                        #####
                        # random_state = np.random.RandomState(0)
                        # noise_scaling_function= partial(
                        #     zne.scaling.fold_gates_at_random,
                        #     fidelities = {"double": 1.0},  # Avoid folding single-qubit gates
                        #     random_state=random_state, # Useful to get reproducible results
                        # )
                        
                        # richardson_factory = zne.inference.RichardsonFactory(scale_factors=[1,3,5,7])
                        
                        # zne_results = []
                        # for i, obs in enumerate(observable):
                        #     #def observable_executor(qc, i=i):  # Capture `i` in the function scope
                        #             # Executor for the specific observable
                        #             # results = execute_qc(qc, epsilon=0.0)  # Array of expectation values
                        #             # print(f"Results for all qubits (before indexing): {results}")
                        #             # print(f"Result for qubit {i}: {results[i]}")
                        #             #return results[i][i]  # Extract the result for the current qubit
                        #     def execute_qc(qc,epsilon=0.05):
                        #         #0.049
                        #         qc_copy=qc.copy()
                                
                        #         noise_model = NoiseModel()
                        #         #noise_model.add_all_qubit_quantum_error(depolarizing_error(epsilon, 1), ["rz"])
                        #         noise_model.add_all_qubit_quantum_error(depolarizing_error(epsilon, 1), ["rz"])
                        #         return execute_with_noise(qc_copy, [obs],noise_model=noise_model)
                        #     #print(execute_qc(qc,0))
                        #     executor=Executor(execute_qc)   
                                    
                        #     try:
                        #         zne_value = zne.execute_with_zne(
                        #             circuit=qc,  # Your quantum circuit
                        #             executor=executor,
                        #             observable=None,
                        #             factory=richardson_factory,
                        #             scale_noise=noise_scaling_function,
                        #             num_to_average=3,
                        #         )
                        #         zne_results.append(zne_value)
                            
                        #         plot= richardson_factory.plot_fit()
                        #         plt.show()
                        #     except Exception as e:
                        #         print(f"Error processing qubit {i}: {e}")
                        # save_to_csv(filepath, 0.05, zne_results)
                        #####
                        #print(noiseless_result)
                        scale_factors = [1.0, 3.0, 5.0, 7.0]
                        # Generate a list of folded circuits
                        noise_scaled_circuits = [zne.scaling.fold_all(qc, s) for s in scale_factors]
                        expectation_values = [execute_qc(circ) for circ in noise_scaled_circuits]
                        #print(expectation_values)
                        save_to_csv(filepath, 0.05, expectation_values[0])
                        fac=zne.inference.RichardsonFactory(scale_factors=scale_factors)
                        for s, e in zip(scale_factors, expectation_values):
                            fac.push({"scale_factor": s}, e)

                        # Evaluate the extrapolation
                        zne_results=fac.reduce()
                        #print(zne_results)
                        save_to_csv(filepath, 0.0495, zne_results)
                        #_ = fac.plot_fit()
                        #plt.show()
                        
                        for c in range(4):
                            out[b,c,j // 2, k // 2] = zne_results[c] #execute_qc(qc, epsilon=0)[c] # pec_values[iter][c]#
                        #self.counter +=1
                            #print("out",out[b,c,j // 2, k // 2])
                        
                        #print(f"During the previous PEC process, {len(executor.executed_circuits)} ", "circuits have been executed.")
  
                        #print(f"The corresponding noisy expectation values are:")  for c in executor.quantum_results[:5]: print(c)
            return torch.tensor(out,dtype=torch.float32)
                #print(pec_values)
        else:
            print('hi')
            for j in range(0, 28, 2):
                for k in range(0, 28, 2):

                    # Process a squared 2x2 region of the image with a quantum circuit
                    q_results = self.torch_circ(inputs = torch.tensor(

                        [
                            image[0,j, k],
                            image[0,j, k + 1],
                            image[0, j + 1, k],
                            image[0, j + 1, k + 1]
                        ]),weights=self.weights)

                    # Assign expectation values to different channels of the output pixel (j/2, k/2)
                    
                    #print("QASM Code:\n", qasm_code)
                    for c in range(4):
                        out[c,0,j // 2, k // 2] = q_results[c] 
                    #print(q_results)   # change for sign and tanh post processing
            return torch.tensor(out,dtype=torch.float32)

cnn = torch.nn.Sequential(
        Q_Conv_Layer(),
        #torch.nn.Conv2d(in_channels=1, out_channels=4,
                    #  kernel_size=2, stride=2),
        #torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.LazyLinear( out_features=11)
        )
# Load the checkpoint (saved model weights)
"""checkpoint = torch.load('/data/julie.maria.raju/qc-diag/results/var-quantum-exp/Sequence_RX_CNOTs_ Custom_Higher_Order_Encoder_layers_ 1_ 0.785_seed_ 4/checkpoints/epoch=9-step=690.ckpt')

new_state_dict = {}

# Manually map the layers from the checkpoint to the model
new_state_dict['0.circuit.weights'] = checkpoint['state_dict']['model.qcnn.0.torch_qlayer.weights']

new_state_dict['2.weight'] = checkpoint['state_dict']['model.fc.weight']
new_state_dict['2.bias'] = checkpoint['state_dict']['model.fc.bias']

# Attempt to load the new state_dict into the model
try:
    cnn.load_state_dict(new_state_dict, strict=False)
    print("Weights loaded successfully.")
except Exception as e:
    print(f"Error loading state_dict: {e}")"""


# Load the previous state_dict

previous_state_dict = torch.load('/Users/juliemariaraju/Desktop/Master Thesis/qc-diag-julie-qcnn/results/OrganAMNIST_20_seeds/Sequence_RX_CNOTs_ Custom_Higher_Order_Encoder_layers_ 1_ 0.785_seed_ 1/weights_epoch_9.pt', map_location=torch.device('cpu'))
#, map_location=torch.device('cpu'))
# Initialize a new state dictionary for the new model
new_state_dict = {}
# Map keys from the previous model to the new model
new_state_dict['0.circuit.weights'] = previous_state_dict['qcnn.0.torch_qlayer.weights']
new_state_dict['2.weight'] = previous_state_dict['fc.weight']
new_state_dict['2.bias'] = previous_state_dict['fc.bias']

# Attempt to load the new state_dict into the new model
try:
    cnn.load_state_dict(new_state_dict, strict=False)  # Use strict=False to allow for missing keys
    print("Weights loaded successfully.")
except Exception as e:
    print(f"Error loading state_dict: {e}")
    print(cnn[0].weight)

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#cnn = cnn.to(device)

# Define a loss function and an optimizer
Loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)

def train():
    cnn.train()
    train_loss = 0
    train_correct = 0
    train_samples = 0

    for i, (images,labels) in enumerate(train_dataloader,0):              #loop over batches
        # forward
        #images = images.view(-1, 28, 28 )
        #images,labels=images.to(device), labels.to(device)
        outputs = cnn(images)
        labels = labels.squeeze().long()
        loss = Loss(outputs,labels)
        train_loss += loss.item() 

        # backward
        optimizer.zero_grad()   # clearing gradients for this training step
        loss.backward()         # dL/dw
        optimizer.step()        # single optimisation step

        # Accuracy
        _, predictions = torch.max(outputs,1)
        train_samples += images.shape[0]
        train_correct += (predictions ==labels).sum().item()
    train_accuracy = 100* train_correct/ train_samples

    print("epoch {}/ {},Train_loss ={:.4f},Train_accuracy = {:.3f}"
            . format(epoch +1, num_epochs,train_loss/train_samples,train_accuracy))


def test():
    cnn.eval()
    with torch.no_grad():
        """weights = torch.load("/data/julie.maria.raju/qc-diag/results/OrganAMNIST_20_seeds/Sequence_RX_CNOTs_ Custom_Higher_Order_Encoder_layers_ 1_ 0.785_seed_ 1/weights_epoch_9.pt")
        #cnn[1].weight= torch.nn.Parameter(weights['classic.0.weight'].cpu())
        #cnn[1].bias= torch.nn.Parameter(weights['classic.0.bias'].cpu())
        cnn[2].weight= torch.nn.Parameter(weights['fc.weight'].cpu())
        cnn[2].bias= torch.nn.Parameter(weights['fc.bias'].cpu())"""
        #print(cnn[2].weight.shape)
        
        test_loss = 0
        test_correct = 0
        test_samples = 0
        i = 0
        for  (images, labels) in test_dataloader:
            #images, labels = images.to(device), labels.to(device)
            # images = images.reshape(-1, 28*28) #.to(device)
            # labels = labels #.to(device)
            labels = labels.squeeze().long()
            #print("images",images[0])
            """if images.dim() == 3:  # Shape (batch_size, height, width)
                 images = images.unsqueeze(1) """
            #images = images.view(-1, 28, 28 )
            print(images.shape)
            if i==7:
                images=images[:4]
                labels=labels[:4]
            outputs = cnn(images)
            test_loss += Loss(outputs,labels).item()

            #print(f'Test_loss ={test_loss.item():.4f}')
            #value, index
            _, predictions = torch.max(outputs,1)
            test_samples += labels.shape[0]
            test_correct += (predictions ==labels).sum().item()
            print("labels",labels)
            print("pred",predictions)
            if i == 6:
                break
            i+=1
    test_accuracy = 100 * test_correct/test_samples
    print("Test_loss ={:.4f},Test_accuracy = {:.3f}". format(test_loss/test_samples,test_accuracy))          
    test_accuracy = test_correct / test_samples  # Proportion
    test_accuracy_percent = 100 * test_accuracy 
    # Calculate error band (SEM)
    sem = (test_accuracy * (1 - test_accuracy) / test_samples) ** 0.5  # Standard error
    error_band = 1.96 * sem  # For 95% confidence interval

    # Convert SEM and error band to percentage
    sem_percent = 100 * sem
    error_band_percent = 100 * error_band

    # Print results
    print(f"Test Loss: {test_loss / test_samples:.4f}")
    print(f"Test Accuracy: {test_accuracy_percent:.3f}% ± {error_band_percent:.3f}% (95% CI)")         

for epoch in range(num_epochs):
   #train()
   test()            
timer.report()

# batch_size = 1

# canonizers = [SequentialMergeBatchNorm()]
# composite = EpsilonGammaBox(
#     low=-1,
#     high=1)




    
# for i in range(8): #18
#     with Gradient(model=cnn,composite=composite) as attributor:
#         #output, attribution = attributor(samples[i],torch.eye(11)[[labels[i].item()]])
#         class_index = labels[i].item() 
#         output, attribution = attributor(samples[i], torch.eye(11)[[class_index]])
#     print(labels[i].item())
#     prediction = output.argmax(1)[0].item()
#     print(f'Prediction: {output.argmax(1)[0].item()}')

#     # absolute sum over the channels
#     relevance = attribution
#     #relevance = attribution
#     # create an image of the visualize attribution the relevance is only
#     # positive, so we use symmetric=False and an unsigned color-map

#     img = imgify(relevance, symmetric=True, cmap='coldnhot')
#     img0 = imgify(relevance, symmetric=True, cmap='wred')
#     img1 = imgify(relevance, symmetric=True, cmap='bwr')
#     img2 = imgify(relevance, symmetric=True, cmap='0ff,444,f0f')
#     # show the image
#     fig, ax = plt.subplots(figsize=(3,3)) 
#     ax.imshow(img)
#     #plt.savefig("digit_{}.png".format(prediction))
#     fig0, ax0 = plt.subplots(figsize=(3,3))
#     ax0.imshow(img0)
#     #plt.savefig("digit0_{}.png".format(prediction))
#     fig1, ax1 = plt.subplots(figsize=(3,3))
#     ax1.imshow(img1)
#     #plt.savefig("digit1_{}.png".format(prediction))
#     fig2, ax2 = plt.subplots(figsize=(3,3))
#     ax2.imshow(img2)
#     #plt.savefig("digit2_{}.png".format(prediction))

#     # img0.show()
#     # img1.show()
#     # img2.show()

#     plt.show()
#     plt.imshow(samples[i].reshape(14,14), cmap="gray")
#     #plt.savefig('ori_img_{}.png'.format(prediction))
#     plt.show()