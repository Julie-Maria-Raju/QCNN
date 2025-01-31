import random
import torch
from torch import nn
from utils.circuitcomponents_utils import generate_corresponding_circuit, get_wires_number, get_circuit_weights_shape, calculate_effective_dimension_var

from models.quonv_layer import QuonvLayer
import numpy as np
from utils import forward_print_img
import os


class HybridModel(nn.Module):
    def __init__(self, config):
        self.params = config["PARAMS"]
        self.root = os.path.join(config["SAVE_DIR"], config["SAVE_NAME"], config["EXP_NAME"])
        self.dataset = config["DATASET"]
        # set seeds
        torch.manual_seed(self.params['all_seeds'])
        np.random.seed(self.params['all_seeds'])
        random.seed(self.params['all_seeds'])
        torch.backends.cudnn.deterministic = True  # tested - needed for reproducibility
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(self.params['all_seeds'])
        torch.cuda.manual_seed_all(self.params['all_seeds'])
        # set qcnn
        if self.params["preencoded"] == True:
            self.stride = 1
            self.filter_size = 1
        else:
            self.stride = self.params['stride']
            self.filter_size = self.params['filter_length']
        self.ancilla_qbit = self.params["ancilla_qbit"]

            
class HybridModel_MNIST_one_conv(nn.Module):
    def __init__(self, config, quantum, input_dim=(1, 28, 28), out_classes=10):
        super(HybridModel_MNIST_one_conv, self).__init__()
        HybridModel.__init__(self, config)
        self.quantum = quantum
        self.input_dim = input_dim
        self.out_classes = out_classes
        # define quantum circuit
        # number of input channels for quantum layer. Corresponds to the number of output channels of the last classical
        # convolutional layer before the quantum layer
        self.in_channels = 1
        self.out_channels = get_wires_number(self.params)
        self.logits= None
        self.weights_list = []
        self.circuit_list = []
        weights_shape = get_circuit_weights_shape(self.params)

        for i in range(self.in_channels):
            # initialize quantum layer weights
            self.weights_list.append(
                torch.tensor(np.random.default_rng(self.params['all_seeds'] + i).uniform(-1, 1, weights_shape), device=torch.device("cuda")))
            self.circuit_list.append(generate_corresponding_circuit(self.params, weights_initialized=self.weights_list[i],
                                                                    encoding=not self.params["preencoded"], data_3D=False, rotation_factor=self.params["rotation_factor"]))
    
        quonv_layers = [QuonvLayer(data_3D=False,
                                              stride=self.stride, hyperparams= self.params,
                                              circuit=self.circuit_list[i],
                                              weights=self.weights_list[i],
                                              wires=get_wires_number(self.params),
                                              out_channels=self.out_channels,
                                              filter_size=self.filter_size,
                                              device=self.params["device"],
                                              diff_method=self.params["diff_method"],
                                              ancilla_qbit=self.ancilla_qbit) for i in range(self.in_channels)]
        self.quonv_layers=quonv_layers
        
        self.qcnn = nn.ModuleList(quonv_layers)
        
        # calculate expressibility
        if self.params["calculate_expressibility"] == True:
            quonv_layers[0].calculate_expressibility(save_dir=self.root)

        # calculate expressibility
        if self.params["calculate_entanglement"] == True:
            quonv_layers[0].calculate_entanglement(save_dir=self.root)

        # calculate effective dimension
        if self.params["calculate_effective_dim"] == True:
            dataset_size = 600 if self.dataset == "medmnist" else 1000 #medmnist
            calculate_effective_dimension_var(self.params, 4, dataset_size, self.root, False)

        # make quantum layer untrainable if specified in the settings
        if self.params["trainable"] == False:
            self.qcnn.torch_qlayer.weights.requires_grad = False

        self.classic = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4,
                      kernel_size=self.filter_size, stride=self.stride),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.LazyLinear(out_features=self.out_classes)

    def forward(self, x, y=None, batch_idx=None, img_idx=None, epoch=None):
        # save validation images
        if batch_idx is not None:
            forward_print_img.save_tensor(x, root=self.root, epoch=epoch, batch_idx=batch_idx, layer="before_first_conv", label=y)

        if self.quantum == True:
            #x.requires_grad_(True)
            x = x.permute(0, 2, 3, 1)
            bs, out_dim = self.qcnn[0].calc_out_dim(x[:, :, :, 0])
            x_out_quantum = torch.empty(
                bs, out_dim[0], out_dim[1], self.in_channels * self.out_channels, device=torch.device('cuda'))
            for j in range(self.in_channels):
                x_channel = x[:, :, :, j]
                print("Circuit channel", str(j))
                x_channel = self.qcnn[j](x_channel)
                for q in range(self.out_channels):
                    x_out_quantum[:, :, :, j *
                                           self.out_channels + q] = x_channel[:, :, :, q]
                print("---------------------------------------")
            x = x_out_quantum.permute(0, 3, 2, 1)  # permute back in place
            x = x.cuda()
            # save validation images
            if batch_idx is not None:
                forward_print_img.save_tensor(x, root=self.root, epoch=epoch, batch_idx=batch_idx, layer="after_first_conv",
                                              label=y)

        elif self.quantum == False:
            x = self.classic(x)
            # save validation images
            if batch_idx is not None:
                forward_print_img.save_tensor(x, root=self.root, epoch=epoch, batch_idx=batch_idx, layer="after_classic_conv",
                                              label=y)

        x = self.flatten(x)
        x = self.fc(x)
        return x

class HybridModel_MNIST_one_conv_multiOut_multipass(nn.Module):
    def __init__(self, config, quantum, input_dim=(1, 28, 28), out_classes=10,quantum_run=2):
        super(HybridModel_MNIST_one_conv_multiOut_multipass, self).__init__()
        HybridModel.__init__(self, config)
        self.quantum = quantum
        self.input_dim = input_dim
        self.out_classes = out_classes
        self.quantum_run=quantum_run
        # define quantum circuit
        # number of input channels for quantum layer. Corresponds to the number of output channels of the last classical
        # convolutional layer before the quantum layer
        self.in_channels = 1
        
        # Set number of out_channels to value defined in settings_file
        self.out_channels = self.params['n_out_channels'] ############
        # Check if we want to sum over the qubits measurements
        self.sum_over_kernel = self.params['sum_over_kernel']

        self.weights_list = []
        self.circuit_list = []
        self.bias = nn.Parameter(torch.rand(1,1))
        weights_shape = get_circuit_weights_shape(self.params)


        for i in range(self.in_channels * self.out_channels):
            # initialize quantum layer weights
            self.weights_list.append(
                torch.tensor(np.random.default_rng(self.params['all_seeds'] + i).uniform(-1, 1, weights_shape), device=torch.device("cuda")))
            self.circuit_list.append(generate_corresponding_circuit(self.params, weights_initialized=self.weights_list[i],
                                                                    encoding=not self.params["preencoded"], data_3D=False))
        
        # Determine number of single qubit measurements in quantum circuit by applying the kernel to an arbitrary input (--> np.zeros(4))
        wires_number = get_wires_number(self.params)
        self.kernel_out_dim = len(self.circuit_list[0](np.zeros(wires_number), self.weights_list[0])) if self.sum_over_kernel == False else 1 ############

        # Create ModuleList of kernels with indivually initialized weights of length (in_channels * out_channels)
        self.qcnn = nn.ModuleList([QuonvLayer(data_3D=False,
                                              stride=self.stride,
                                              circuit=self.circuit_list[i],
                                              weights=self.weights_list[i],
                                              wires=get_wires_number(self.params),
                                              out_channels=self.kernel_out_dim, ############
                                              sum_over_kernel = self.sum_over_kernel,
                                              filter_size=self.filter_size,
                                              device=self.params["device"],
                                              diff_method=self.params["diff_method"],
                                              ancilla_qbit=self.ancilla_qbit) for i in range(self.in_channels * self.out_channels)]) ############

        # make quantum layer untrainable if specified in the settings
                # calculate effective dimension
        if self.params["calculate_effective_dim"] == True:
            dataset_size = 600 if self.dataset == "breast_mnist" else 1000 #medmnist
            calculate_effective_dimension_var(self.params, 4, dataset_size, self.root, False)

        if self.params["trainable"] == False:
            self.qcnn.torch_qlayer.weights.requires_grad = False

        self.classic = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4,
                      kernel_size=self.filter_size, stride=self.stride),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.LazyLinear(out_features=self.out_classes)

    def forward(self, x, y=None, batch_idx=None, img_idx=None, epoch=None):
        # save validation images
        if batch_idx is not None:
            forward_print_img.save_tensor(x, root=self.root, epoch=epoch, batch_idx=batch_idx, layer="before_first_conv", label=y)

        if self.quantum == True:
            
            x = x.permute(0, 2, 3, 1)
            bs, out_dim = self.qcnn[0].calc_out_dim(x[:, :, :, 0])

            x_out_quantum = torch.empty(
                bs, out_dim[0], out_dim[1], self.in_channels * self.out_channels * self.kernel_out_dim, device=torch.device('cuda'))
            # Run several time the quantum circuit
            for j in range(self.in_channels):

                print("Circuit channel", str(j))

                for q in range(self.out_channels):
                    # Extract in_channel j and process it for every out_channel q separately
                    x_channel_in = x[:, :, :, j]
                    x_channel = self.qcnn[j * self.in_channels + q](x_channel_in)

                    for l in range(self.kernel_out_dim):
                        # Loop over the single qubit measurement in one kernel and write the results in a unique position in x_out_quantum
                        # (j * self.out_channels * self.kernel_out_dim + q * self.kernel_out_dim + l) assigns every permutation of 
                        # (out_channel x in_channel x kernel_out_dim) a unique integer number
                        x_out_quantum[:, :, :, j * self.out_channels * self.kernel_out_dim + q * self.kernel_out_dim + l] += (x_channel[:, :, :, l]+self.bias)/self.quantum_run
                print("---------------------------------------")
            x_prob = x_out_quantum.permute(0, 3, 2, 1)  # permute back in place
            x = torch.tanh(x_prob) #or torch.sign(x) #TODO: what happened here?
            x = x.cuda()
            # save validation images
            if batch_idx is not None:
                forward_print_img.save_tensor(x, root=self.root, epoch=epoch, batch_idx=batch_idx, layer="after_first_conv",
                                              label=y)

        elif self.quantum == False:
            x = self.classic(x)
            # save validation images
            if batch_idx is not None:
                forward_print_img.save_tensor(x, root=self.root, epoch=epoch, batch_idx=batch_idx, layer="after_classic_conv",
                                              label=y)

        x = self.flatten(x)
        x = self.fc(x)
        return x

class HybridModel_MNIST_one_conv_multiOut(nn.Module):
    def __init__(self, config, quantum, input_dim=(1, 28, 28), out_classes=10):
        super(HybridModel_MNIST_one_conv_multiOut, self).__init__()
        HybridModel.__init__(self, config)
        self.quantum = quantum
        self.input_dim = input_dim
        self.out_classes = out_classes
        # define quantum circuit
        # number of input channels for quantum layer. Corresponds to the number of output channels of the last classical
        # convolutional layer before the quantum layer
        self.in_channels = 1
        
        # Set number of out_channels to value defined in settings_file
        self.out_channels = self.params['n_out_channels'] ############
        # Check if we want to sum over the qubits measurements
        self.sum_over_kernel = self.params['sum_over_kernel']

        self.weights_list = []
        self.circuit_list = []
        weights_shape = get_circuit_weights_shape(self.params)


        for i in range(self.in_channels * self.out_channels):
            # initialize quantum layer weights
            self.weights_list.append(
                torch.tensor(np.random.default_rng(self.params['all_seeds'] + i).uniform(-1, 1, weights_shape), device=torch.device("cuda")))
            self.circuit_list.append(generate_corresponding_circuit(self.params, weights_initialized=self.weights_list[i],
                                                                    encoding=not self.params["preencoded"], data_3D=False))
        
        # Determine number of single qubit measurements in quantum circuit by applying the kernel to an arbitrary input (--> np.zeros(4))
        wires_number = get_wires_number(self.params)
        self.kernel_out_dim = len(self.circuit_list[0](np.zeros(wires_number), self.weights_list[0])) if self.sum_over_kernel == False else 1 ############

        # Create ModuleList of kernels with indivually initialized weights of length (in_channels * out_channels)
        self.qcnn = nn.ModuleList([QuonvLayer(data_3D=False,
                                              stride=self.stride, hyperparams= self.params,
                                              circuit=self.circuit_list[i],
                                              weights=self.weights_list[i],
                                              wires=get_wires_number(self.params),
                                              out_channels=self.kernel_out_dim, ############
                                              sum_over_kernel = self.sum_over_kernel,
                                              filter_size=self.filter_size,
                                              device=self.params["device"],
                                              diff_method=self.params["diff_method"],
                                              ancilla_qbit=self.ancilla_qbit) for i in range(self.in_channels * self.out_channels)]) ############
        
        # calculate effective dimension
        if self.params["calculate_effective_dim"] == True:
            dataset_size = 600 if self.dataset == "breast_mnist" else 1000 #medmnist
            calculate_effective_dimension_var(self.params, 4, dataset_size, self.root, False)

        # make quantum layer untrainable if specified in the settings
        if self.params["trainable"] == False:
            self.qcnn.torch_qlayer.weights.requires_grad = False

        # calculate effective dimension
        if self.params["calculate_effective_dim"] == True:
            dataset_size = 600 if self.dataset == "breast_mnist" else 1000 #medmnist
            calculate_effective_dimension_var(self.params, 4, dataset_size, self.root, False)

        self.classic = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4,
                      kernel_size=self.filter_size, stride=self.stride),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.LazyLinear(out_features=self.out_classes)

    def forward(self, x, y=None, batch_idx=None, img_idx=None, epoch=None):
        # save validation images
        if batch_idx is not None:
            forward_print_img.save_tensor(x, root=self.root, epoch=epoch, batch_idx=batch_idx, layer="before_first_conv", label=y)

        if self.quantum == True:
            
            x = x.permute(0, 2, 3, 1)
            bs, out_dim = self.qcnn[0].calc_out_dim(x[:, :, :, 0])

            x_out_quantum = torch.empty(
                bs, out_dim[0], out_dim[1], self.in_channels * self.out_channels * self.kernel_out_dim, device=torch.device('cuda'))
            for j in range(self.in_channels):

                print("Circuit channel", str(j))

                for q in range(self.out_channels):
                    # Extract in_channel j and process it for every out_channel q separately
                    x_channel_in = x[:, :, :, j]
                    x_channel = self.qcnn[j * self.in_channels + q](x_channel_in)

                    for l in range(self.kernel_out_dim):
                        # Loop over the single qubit measurement in one kernel and write the results in a unique position in x_out_quantum
                        # (j * self.out_channels * self.kernel_out_dim + q * self.kernel_out_dim + l) assigns every permutation of 
                        # (out_channel x in_channel x kernel_out_dim) a unique integer number
                        x_out_quantum[:, :, :, j * self.out_channels * self.kernel_out_dim + q * self.kernel_out_dim + l] = x_channel[:, :, :, l]
                print("---------------------------------------")
            x = x_out_quantum.permute(0, 3, 2, 1)  # permute back in place
            x = x.cuda()
            # save validation images
            if batch_idx is not None:
                forward_print_img.save_tensor(x, root=self.root, epoch=epoch, batch_idx=batch_idx, layer="after_first_conv",
                                              label=y)

        elif self.quantum == False:
            x = self.classic(x)
            # save validation images
            if batch_idx is not None:
                forward_print_img.save_tensor(x, root=self.root, epoch=epoch, batch_idx=batch_idx, layer="after_classic_conv",
                                              label=y)

        x = self.flatten(x)
        x = self.fc(x)
        return x


class HybridModel_MNIST_two_conv(nn.Module):
    def __init__(self, config, quantum=False, out_classes=10):
        super(HybridModel_MNIST_two_conv, self).__init__()
        HybridModel.__init__(self, config)
        self.quantum = quantum
        self.input_dim = (1, 28, 28)
        self.out_classes = out_classes
        # define quantum circuit
        # number of input channels for quantum layer. Corresponds to the number of output channels of the last classical
        # convolutional layer before the quantum layer
        self.in_channels = 1
        self.out_channels = get_wires_number(self.params)

        self.weights_list = []
        self.circuit_list = []
        weights_shape = get_circuit_weights_shape(self.params)

        for i in range(self.in_channels):
            # initialize quantum layer weights
            self.weights_list.append(
                torch.tensor(np.random.default_rng(self.params['all_seeds'] + i).uniform(-1, 1, weights_shape), device=torch.device("cuda")))
            self.circuit_list.append(generate_corresponding_circuit(self.params, weights_initialized=self.weights_list[i],
                                                                    encoding=not self.params["preencoded"], data_3D=False))

        self.qcnn = nn.ModuleList([QuonvLayer(data_3D=False,
                                              stride=self.stride,
                                              circuit=self.circuit_list[i],
                                              weights=self.weights_list[i],
                                              wires=get_wires_number(self.params),
                                              out_channels=self.out_channels,
                                              filter_size=self.filter_size,
                                              device=self.params["device"],
                                              diff_method=self.params["diff_method"],
                                              ancilla_qbit=self.ancilla_qbit) for i in range(self.in_channels)])

        # CLASSICAL params
        self.classic_first_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=self.filter_size, stride=self.stride),
            nn.ReLU(),
        )
        self.classic_second_conv = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten(start_dim=1)

        self.fc = nn.LazyLinear(out_features=self.out_classes)


    def forward(self, x, batch_idx=None):
        if self.quantum == True:
            x = x.permute(0, 2, 3, 1)
            bs, out_dim = self.qcnn[0].calc_out_dim(x[:, :, :, 0])
            x_out_quantum = torch.empty(
                bs, out_dim[0], out_dim[1], self.in_channels * self.out_channels, device=torch.device("cuda"))
            for j in range(self.in_channels):
                x_channel = x[:, :, :, j]
                print("Circuit channel", str(j))
                x_channel = self.qcnn[j](x_channel)
                for q in range(self.out_channels):
                    x_out_quantum[:, :, :, j *
                                           self.out_channels + q] = x_channel[:, :, :, q]
                print("---------------------------------------")
            x = x_out_quantum.permute(0, 3, 2, 1).cuda()

        elif self.quantum == False:
            x = self.classic_first_conv(x)

        x = self.classic_second_conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class HybridModel_MNIST_two_conv_middle(nn.Module):
    def __init__(self, config, quantum=False, out_classes=10):
        super(HybridModel_MNIST_two_conv_middle, self).__init__()
        HybridModel.__init__(self, config)
        self.quantum = quantum
        self.input_dim = (1, 28, 28)
        self.out_classes = out_classes
        # define quantum circuit
        # number of input channels for quantum layer. Corresponds to the number of output channels of the last classical
        # convolutional layer before the quantum layer
        self.in_channels = 4
        self.out_channels = get_wires_number(self.params)

        self.weights_list = []
        self.circuit_list = []
        weights_shape = get_circuit_weights_shape(self.params)

        for i in range(self.in_channels):
            # initialize quantum layer weights
            self.weights_list.append(
                torch.tensor(np.random.default_rng(self.params['all_seeds'] + i).uniform(-1, 1, weights_shape), device=torch.device("cuda")))
            self.circuit_list.append(generate_corresponding_circuit(self.params, weights_initialized=self.weights_list[i],
                                                                    encoding=not self.params["preencoded"], data_3D=False))

        self.qcnn = nn.ModuleList([QuonvLayer(data_3D=False,
                                              stride=self.stride,
                                              circuit=self.circuit_list[i],
                                              weights=self.weights_list[i],
                                              wires=get_wires_number(self.params),
                                              out_channels=self.out_channels,
                                              filter_size=self.filter_size,
                                              device=self.params["device"],
                                              diff_method=self.params["diff_method"],
                                              ancilla_qbit=self.ancilla_qbit) for i in range(self.in_channels)])

        # CLASSICAL params
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.second_conv = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(in_features=576,
                            out_features=self.out_classes)

    def forward(self, x):
        print("Before first conv", x.shape)
        x = self.first_conv(x)
        print("After first conv", x.shape)

        if self.quantum == True:
            x = x.permute(0, 2, 3, 1)
            bs, out_dim = self.qcnn[0].calc_out_dim(x[:, :, :, 0])
            x_out_quantum = torch.empty(
                bs, out_dim[0], out_dim[1], self.in_channels * self.out_channels, device=torch.device("cuda"))
            for j in range(self.in_channels):
                x_channel = x[:, :, :, j]
                print("Circuit channel", str(j))
                x_channel = self.qcnn[j](x_channel)
                for q in range(self.out_channels):
                    x_out_quantum[:, :, :, j *
                                           self.out_channels + q] = x_channel[:, :, :, q]
                print("---------------------------------------")
            x = x_out_quantum.permute(0, 3, 2, 1)  # permute back in place
            x = x.cuda()
        else:
            x = self.second_conv(x)

        x = self.flatten(x)
        x = self.fc(x)
        return x


class HybridModel_medical_2D(nn.Module):
    def __init__(self, config, quantum=False):
        super(HybridModel_medical_2D, self).__init__()
        HybridModel.__init__(self, config)
        self.quantum = quantum

        # define quantum circuit
        # number of input channels for quantum layer. Corresponds to the number of output channels of the last classical
        # convolutional layer before the quantum layer
        self.in_channels = 1
        self.out_channels = get_wires_number(self.params)

        # initialize quantum layer weights
        weights_shape = get_circuit_weights_shape(self.params)

        self.weights_list = []
        self.circuit_list = []
        for i in range(self.in_channels):
            self.weights_list.append(
                torch.tensor(np.random.default_rng(self.params['all_seeds'] + i).uniform(-1, 1, weights_shape), device=torch.device("cuda")))
            self.circuit_list.append(generate_corresponding_circuit(self.params, weights_initialized=self.weights_list[i],
                                                                    encoding=not self.params["preencoded"], data_3D=False))

        self.qcnn = nn.ModuleList([QuonvLayer(data_3D=False,
                                              stride=self.stride,
                                              circuit=self.circuit_list[i],
                                              weights=self.weights_list[i],
                                              wires=self.out_channels,
                                              out_channels=self.out_channels,
                                              filter_size=self.filter_size,
                                              device=self.params["device"],
                                              diff_method=self.params["diff_method"],
                                              ancilla_qbit=self.ancilla_qbit
                                              ) for i in range(self.in_channels)])

        # set model
        if quantum == True:
            first_channel_nb = self.out_channels
            linear_features = 30 * 30 * 16
        else:
            first_channel_nb = 1
            linear_features = 31 * 31 * 16

        # CLASSICAL params
        self.classic = nn.Sequential(
            nn.Conv2d(in_channels=first_channel_nb,
                      out_channels=16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(in_features=linear_features, out_features=2)

    def forward(self, x):
        if (self.params["preencoded"] == False) and (self.quantum == True):
            # batch size not at the usual place in the quantum layer
            x = x.permute(0, 2, 3, 1)
        if self.quantum == True:
            bs, out_dim = self.qcnn[0].calc_out_dim(x[:, :, :, 0])
            x_out_quantum = torch.empty(
                bs, out_dim[0], out_dim[1], self.in_channels * self.out_channels, device=torch.device("cuda"))
            for j in range(self.in_channels):
                x_channel = x[:, :, :, j]
                print("Circuit channel", str(j))
                x_channel = self.qcnn[j](x_channel)
                for q in range(self.out_channels):
                    x_out_quantum[:, :, :, j *
                                           self.out_channels + q] = x_channel[:, :, :, q]
                print("---------------------------------------")
            x = x_out_quantum.cuda()
        if (self.params["preencoded"] == False) and (self.quantum == True):
            x = x.permute(0, 3, 2, 1)  # permute back in place
        x = self.classic(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class HybridModel_medical_3D(nn.Module):
    def __init__(self, config, quantum=False):
        super(HybridModel_medical_3D, self).__init__()
        HybridModel.__init__(self, config)
        self.quantum = quantum
        # define quantum circuit
        # number of input channels for quantum layer. Corresponds to the number of output channels of the last classical
        # convolutional layer before the quantum layer
        self.in_channels = 8
        self.out_channels = get_wires_number(self.params, data_3D=True)

        # initialize quantum layer weights
        weights_shape = get_circuit_weights_shape(self.params)

        self.weights_list = []
        self.circuit_list = []
        for i in range(self.in_channels):
            self.weights_list.append(
                torch.tensor(np.random.default_rng(self.params['all_seeds'] + i).uniform(-1, 1, weights_shape), device=torch.device("cuda")))
            self.circuit_list.append(generate_corresponding_circuit(self.params, weights_initialized=self.weights_list[i],
                                                                    encoding=not self.params["preencoded"], data_3D=True))

        self.qcnn = nn.ModuleList([QuonvLayer(data_3D=True,
                                              stride=self.stride,
                                              circuit=self.circuit_list[i],
                                              weights=self.weights_list[i],
                                              wires=self.out_channels,
                                              out_channels=self.out_channels,
                                              filter_size=self.filter_size,
                                              device=self.params["device"],
                                              diff_method=self.params["diff_method"]) for i in range(self.in_channels)])

        # set model
        if quantum == True:
            first_channel_nb = self.out_channels
            linear_features = 30 * 30 * 16
        else:
            first_channel_nb = 1
            linear_features = 1968624

        # CLASSICAL params
        self.classic = nn.Sequential(
            nn.Conv3d(in_channels=first_channel_nb,
                      out_channels=16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2))
        self.flatten = nn.Flatten(start_dim=1)
        self.f