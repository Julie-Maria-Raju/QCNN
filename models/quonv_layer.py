import logging
from qiskit import IBMQ
import torch
from torch import nn
import pennylane as qml
import numpy as np
from noise.noise import make_noiseModel
from q_evaluation.own_expressibility import Expressibility
from q_evaluation.entanglement import EntanglementCapability

#backend = 'ibmq_manila' # 5 qubits
#ibmqx_token = 'XXX'
#IBMQ.save_account(ibmqx_token, overwrite=True)
#IBMQ.load_account()


class QuonvLayer(nn.Module):
    def __init__(self, hyperparams,data_3D, weights, stride=1, device="default.qubit", diff_method="best", wires=4,
                 number_of_filters=1, circuit=None, filter_size=2, out_channels=4, sum_over_kernel=False, dtype=torch.float32, ancilla_qbit=0):

        super(QuonvLayer, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.data_3D = data_3D
        self.stride = stride
        self.wires = wires
        self.circuit = circuit
        self.weights = weights
        self.filter_size = filter_size
        self.sum_over_kernel = sum_over_kernel
        self.my_noise_model, self.magnitude_name = make_noiseModel(hyperparams["noise_name"], hyperparams["magnitude"], hyperparams["shotnum"])
        # setup device
        if device == "qulacs.simulator":
            if(ancilla_qbit!=0):
                self.device = qml.device(device, wires=self.wires+ancilla_qbit, gpu=True)
            else:
                self.device = qml.device(device, wires=self.wires, gpu=True)
        elif device == "qulacs.simulator-cpu":
            if(ancilla_qbit!=0):
                self.device = qml.device("qulacs.simulator", wires=self.wires+ancilla_qbit, gpu=False)
            else:
                self.device = qml.device("qulacs.simulator", wires=self.wires, gpu=False)
        elif device == "qiskit.aer":
            if(ancilla_qbit!=0):
                self.device = qml.device(device, wires=self.wires+ancilla_qbit, gpu=False)
            else:
                self.device = qml.device(device,wires=self.wires,noise_model = self.my_noise_model) # noise_model = self.my_noise_model
        elif device == "qiskit.ibmq":
            # IBM quantum computer
            # define your credentials at top of this file
            # and uncomment the IBMQ account saving/loading
            if(ancilla_qbit!=0):
                self.device = qml.device('qiskit.ibmq', wires=self.wires+ancilla_qbit, backend=backend)
            else:
                self.device = qml.device('qiskit.ibmq', wires=self.wires, backend=backend)
        else:
            # default simulator
            if(ancilla_qbit!=0):
                self.device = qml.device(device, wires=self.wires+ancilla_qbit)
            else:
                self.device = qml.device(device, wires=self.wires)

        self.number_of_filters = number_of_filters
        self.filter_size = filter_size
        self.out_channels = out_channels
        self.dtype = dtype

        self.qlayer = qml.QNode(circuit, self.device, interface="torch", diff_method=diff_method)
        if weights is not None:
            self.torch_qlayer = qml.qnn.TorchLayer(self.qlayer, weight_shapes={"weights": weights.shape})
            self.torch_qlayer.weights.data = weights
            
        else:
            self.torch_qlayer = self.qlayer

        self.draw_object = None

    def convolve(self, img):
        bs, dim = self.get_img_dim(img)
        for b in range(bs):
            for i in range(0, dim[0] - self.filter_size + 1, self.stride):
                for j in range(0, dim[1] - self.filter_size + 1, self.stride):
                    if self.data_3D:
                        for k in range(0, dim[2] - self.filter_size + 1, self.stride):
                            # Process a squared nxnxn region of the image with a quantum circuit
                            yield img[b, i: i + self.filter_size, j: j + self.filter_size, k: k + self.filter_size].flatten(), b, i, j, k
                    else:
                        yield img[b, i: i + self.filter_size, j: j + self.filter_size].flatten(), b, i, j

    def get_img_dim(self, img):
        img_size = img.size()
        bs = img_size[0]
        if self.data_3D:
            dim = torch.tensor(img_size[1:4])
        # in case of 2D convolutions of 3D images: Third image dimension is treated as channel number and
        # therefore not included in the image dimension
        else:
            dim = torch.tensor(img_size[1:3])
        return bs, dim

    def calc_out_dim(self, img):
        bs, dim = self.get_img_dim(img)
        dim_out = dim.clone().detach()
        for i, d in enumerate(dim_out):
            dim_out[i] = (int(d) - self.filter_size) // self.stride + 1
        return bs, dim_out

    def forward(self, img):
        bs, dim_out = self.calc_out_dim(img)
        if self.data_3D:
            out = torch.empty((bs, dim_out[0], dim_out[1], dim_out[2], self.out_channels), dtype=self.dtype, device=torch.device("cuda"))
            # Loop over the coordinates of the top-left pixel of 2X2X2 squares
            for qnode_inputs, b, i, j, k in self.convolve(img):
                q_results = self.torch_qlayer(qnode_inputs)
                # Assign expectation values to different channels of the output pixel (i/2, j/2, k/2)
                for q in range(self.out_channels):
                    out[b, i // self.stride, j // self.stride, k // self.stride, q] = q_results[q]
        else:
            out = torch.empty((bs, dim_out[0], dim_out[1], self.out_channels), dtype=self.dtype, device=torch.device("cuda"))

            ''' Alternative approach, which is closer to a classical convolution:
            Sum over all output qubits instead of writing each measurement into a separate channel'''
            # TODO: Implement this approach into a separate function / flag and fix this for new quonv_layer implementation using nn.ModuleList and without channels in quonv_layer.py
            # out_interm = torch.empty((bs, dim_out[0], dim_out[1], self.out_channels, self.in_channels), dtype=self.dtype).cuda()

            # Loop over the coordinates of the top-left pixel of 2X2 squares
            for qnode_inputs, b, i, j in self.convolve(img):
                q_results = self.torch_qlayer(qnode_inputs)

                if self.sum_over_kernel == False:
                # Assign expectation values to different channels of the output pixel (i/2, j/2)
                    for q in range(self.out_channels):
                        out[b, i // self.stride, j // self.stride, q] = q_results[q]

                elif self.sum_over_kernel == True:
                # Sum over all output qubits instead of writing each measurement into a separate channel
                    out[b, i // self.stride, j // self.stride, :] = sum(q_results)

                ''' the two lines below refer again to the alternative approach'''
                # out_interm[b, i // self.stride, j // self.stride, :, c] = q_results
            # out = torch.sum(out_interm, dim=4)

        self.draw_object = qml.draw(self.qlayer, expansion_strategy="device")(qnode_inputs, self.torch_qlayer.weights.data)
        print(self.draw_object)
        self.qnode_inputs = qnode_inputs
        return out

    def get_out_template(self, img):
        dim = img.size()
        for i, d in enumerate(dim):
            dim[i] = (d - self.filter_size) // self.stride + 1
        return torch.zeros(dim.size())

    def calculate_expressibility(self, save_dir):
        device_eval = qml.device('qiskit.aer', wires=self.wires, shots=1024,
                 backend='statevector_simulator')
        qlayer_eval = qml.QNode(self.circuit, device_eval, interface="torch", diff_method="best")

        if self.weights is not None:
            print("Calculating circuit expressibility...")
            if self.data_3D:
                expr = Expressibility(qlayer_eval, device_eval, self.wires, 100, np.empty([self.filter_size **3]), self.weights)
            else:
                expr = Expressibility(qlayer_eval, device_eval, self.wires, 100, np.empty([self.filter_size **2]), self.weights)
            expr_value, plot_data = expr.calculate_expressibility_var(save_dir)
            expr.plot(plot_data, expr_value, save_dir)

    def calculate_entanglement(self, save_dir):
        device_eval = qml.device('qiskit.aer', wires=self.wires, shots=1024,
                 backend='statevector_simulator')
        qlayer_eval = qml.QNode(self.circuit, device_eval, interface="torch", diff_method="best")

        if self.weights is not None:
            print("Calculating circuit entanglement...")
            if self.data_3D:
                ent = EntanglementCapability(qlayer_eval, device_eval, self.wires, 100, np.empty([self.filter_size **3]), self.weights)
            else:
                ent = EntanglementCapability(qlayer_eval, device_eval, self.wires, 1000, np.empty([self.filter_size **2]), self.weights)
            ent_value = ent.entanglement_capability_var(save_dir)


class ExtractStatesQuonvLayer(QuonvLayer):

    def __init__(self, weights, stride=1, device="default.qubit", wires=6,
                 number_of_filters=1, circuit=None, filter_size=2, out_channels=4, seed=None, dtype=torch.complex64):
        super().__init__(weights, stride, device, wires, number_of_filters, circuit, filter_size, out_channels, seed, dtype)

    def calc_out_dim(self, img):
        bs, dim = self.get_img_dim(img)
        for i, d in enumerate(dim):
            dim[i] = (d - self.filter_size) // self.stride + 1
        if self.data_3D:
            return bs, dim[0], dim[1], dim[2], 2**self.wires
        else:
            return bs, dim[0], dim[1], 2**self.wires