import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import pennylane as qml
import matplotlib.pyplot as plt
from utils.model_utils import count_parameters
import os

### Pytorch lightning module for model wrapping ###


class ModelWrapper(pl.LightningModule):

    def __init__(self, data, model, config):
        super(ModelWrapper, self).__init__()
        self.config = config
        self.data = data
        self.analysis = {"train_loss": [], "train_accuracy": [],
                         "val_loss": [], "val_accuracy": []}
        self.model = self.init_model(model)
        count_parameters(self.model)
        self.all_weights = []    

    def init_model(self, model):
        '''
        Dry run to initialize the LazyModules.
        '''
        model = model.cuda()
        sample_tensor = torch.randn(
            (self.config["PARAMS"]["batch_size"], ) + model.input_dim, device=torch.device("cuda"))
        model(sample_tensor)
        return model

    def forward(self, x, batch_idx=None):
        return self.model.forward(x, batch_idx)

    def cross_entropy_loss(self, logits, labels):
        loss = torch.nn.CrossEntropyLoss()
        return loss(logits, labels)

    def accuracy(self, logits, labels):
        _, predicted = torch.max(logits.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / len(labels)
        return torch.tensor(accuracy)

    def training_step(self, batch, batch_idx):
        x, y = batch
        self.input_shape = x.shape
        logits = self.forward(x, batch_idx)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)

        # save weigths
        if self.config["QUANTUM"] == True:
            print("weight data", self.model.qcnn[0].torch_qlayer.weights.data)
            if self.config["save_weights"] == True:
                weights_copy = torch.clone(self.model.qcnn[0].torch_qlayer.weights.data)
                self.all_weights.append(weights_copy)

        return {"loss": loss, "accuracy": accuracy}           

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)
        return {"val_loss": loss, "val_accuracy": accuracy}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        proba = F.softmax(logits)
        accuracy = self.accuracy(logits, y)
        return {"test_loss": loss, "test_accuracy": accuracy, "proba": proba, "y": y}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["accuracy"] for x in outputs]).mean()
        self.log("ptl/train_loss", avg_loss)
        self.log("ptl/train_accuracy", avg_acc)

        self.logger.experiment.add_scalar(
            "Loss/Train", avg_loss, global_step=self.current_epoch)
        self.logger.experiment.add_scalar(
            "Accuracy/Train", avg_acc, global_step=self.current_epoch)
        self.analysis["train_loss"].append(avg_loss.cpu())
        self.analysis["train_accuracy"].append(avg_acc.cpu())

        if (self.current_epoch == 1):
            self.logger.experiment.add_text(
                "Params", self.log_without_markdown(self.config["PARAMS"]))
            self.logger.experiment.add_text(
                "Model architecture", self.log_without_markdown(self.model))
            self.logger.experiment.add_text(
                "Model architecture", "Quantum is " + str(self.config["QUANTUM"]))

            # add graph model, is not supported for quantum layer
            # if settings.QUANTUM == False:
            #    sampleImg = torch.rand(self.input_shape).cuda()
            #    self.logger.experiment.add_graph(self.model, sampleImg)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_accuracy", avg_acc)

        self.logger.experiment.add_scalar(
            "Loss/Val", avg_loss, global_step=self.current_epoch)
        self.logger.experiment.add_scalar(
            "Accuracy/Val", avg_acc, global_step=self.current_epoch)
        self.analysis["val_loss"].append(avg_loss.cpu())
        self.analysis["val_accuracy"].append(avg_acc.cpu())

        # save circuit
        if self.config["QUANTUM"]:
            fig, ax = qml.draw_mpl(self.model.qcnn[0].qlayer, expansion_strategy="device")(
                self.model.qcnn[0].qnode_inputs, self.model.qcnn[0].torch_qlayer.weights.data)
            plt.savefig(os.path.join(self.config["SAVE_DIR"], self.config["SAVE_NAME"], self.config["EXP_NAME"],
                                     "circuit_epoch_{}.png".format(self.current_epoch)))

    def test_epoch_end(self, outputs):
        avg_acc = torch.stack([x["test_accuracy"] for x in outputs]).mean()
        self.stacked_y = torch.cat([x["y"] for x in outputs], dim=0)
        logs = {'test_acc': avg_acc}
        return {'avg_test_acc': avg_acc, 'progress_bar': logs}

    def on_after_backward(self):
        if self.config["QUANTUM"]:
            pass
            # self.logger.experiment.add_histogram("qlayer", self.model.qcnn.torch_qlayer.weights, self.current_epoch)
            # self.logger.experiment.add_histogram("qlayer_grad", self.model.qcnn.torch_qlayer.weights.grad, self.current_epoch)

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return DataLoader(self.data.train_set, batch_size=self.config["PARAMS"]["batch_size"], shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.data.valid_set, batch_size=self.config["PARAMS"]["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config["PARAMS"]["lr"])
        return optimizer

    def log_without_markdown(self, text):
        text = str(text)
        text = text.replace("\n", "  \n    ")
        text = "    " + text
        return text