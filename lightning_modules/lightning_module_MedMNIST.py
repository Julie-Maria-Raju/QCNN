import pytorch_lightning as pl
import torch
from torch.nn import functional as F
import os
from sklearn.metrics import confusion_matrix
from utils.model_utils import count_parameters
import pennylane as qml
import matplotlib.pyplot as plt
import pandas as pd
import csv
#from q_evaluation.effective_dimension import EffectiveDimension

### Pytorch lightning module for model wrapping ###


class ModelWrapper(pl.LightningModule):

    def __init__(self, data, model, config, analysis="None"):
        super(ModelWrapper, self).__init__()
        self.config = config
        self.data = data
        self.model = self.init_model(model)
        if (analysis == str(None)):
            self.analysis = {"train_loss": [], "train_accuracy": [],
                            "val_loss": [], "val_accuracy": []}
        else:
            self.analysis = analysis
        count_parameters(self.model)
        self.all_weights = []
        self.logits = []
        self.val_labels = os.path.join(config["SAVE_DIR"], config["SAVE_NAME"], config["EXP_NAME"], f"Labels_{config['PARAMS']['magnitude']}.csv")
        self.val_logits = os.path.join(config["SAVE_DIR"], config["SAVE_NAME"], config["EXP_NAME"], f"Logits_{config['PARAMS']['magnitude']}.csv")
        #self.val_logits_data = os.path.join(config["SAVE_DIR"], config["SAVE_NAME"], config["EXP_NAME"], 'Logits_data.csv')
        # Initialize the CSV file with headers
        """if not os.path.exists(self.train_output_file):
            df = pd.DataFrame(columns=["epoch", "train_loss", "train_accuracy"])
            df.to_csv(self.train_output_file, index=False)

        if not os.path.exists(self.val_output_file):
            df = pd.DataFrame(columns=["epoch", "val_loss", "val_accuracy"])
            df.to_csv(self.val_output_file, index=False)"""

    def init_model(self, model):
        '''
        Dry run to initialize the LazyModules.
        '''
        model = model.cuda()
        sample_tensor = torch.randn(
            (self.config["PARAMS"]["batch_size"], ) + model.input_dim, device=torch.device("cuda"))
        
        model(sample_tensor)
        return model

    def forward(self, x, y=None, batch_idx=None, img_idx=None):
        # option to freeze the classical or the quantum layers 
        # self.model.classic.requires_grad_(False)
        # self.model.flatten.requires_grad_(False)
        # self.model.fc.requires_grad_(False)
        # self.model.qcnn.requires_grad_(True)
        return self.model.forward(x, y, batch_idx, img_idx, epoch=self.current_epoch)

    def cross_entropy_loss(self, logits, labels):
        #weights = self.data.class_weights#.detach().clone()
        #weights = weights.cuda()
        loss = torch.nn.CrossEntropyLoss()
        return loss(logits, labels)

    def accuracy(self, logits, labels):
        
        #for logit in logits:
         #   self.logits.append(logit.cpu().numpy())
        _, predicted = torch.max(logits.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / len(labels)
        print(predicted,labels)
        """with open(self.val_labels, mode='a', newline='') as file:
             writer = csv.writer(file)
             writer.writerow([predicted.cpu().numpy()])
        with open(self.val_logits, mode='a', newline='') as file:
             writer = csv.writer(file)
             writer.writerow([logits.cpu().numpy()])"""
        
        return torch.tensor(accuracy)

    def get_predicted(self, logits):
        _, predicted = torch.max(logits.data, 1)
        return predicted

    def training_step(self, batch, batch_idx):
        x, y = batch
        self.input_shape = x.shape
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)

        # save weigths
        if self.config["QUANTUM"] == True:
            print("weight data", self.model.qcnn[0].torch_qlayer.weights.data)
            if self.config["save_weights"] == True:
                weights_copy = torch.clone(self.model.qcnn[0].torch_qlayer.weights.data)
                self.all_weights.append(weights_copy)
            #torch.save(self.model.state_dict(), f'/data/julie.maria.raju/results/weights_epoch_{1}.pt')

        return {"loss": loss, "accuracy": accuracy}

    def validation_step(self, batch, batch_idx):
        x, y, img_idx = batch
        logits = self.forward(x, y, batch_idx, img_idx)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)
        return {"val_loss": loss, "val_accuracy": accuracy}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        proba = F.softmax(logits)
        accuracy = self.accuracy(logits, y)
        pred = self.get_predicted(logits)
        return {"test_loss": loss, "test_accuracy": accuracy, "proba": proba, "y": y, "pred": pred}

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

        """with open(self.train_output_file, mode='a', newline='') as file:
             writer = csv.writer(file)
             writer.writerow([self.current_epoch, avg_loss.cpu().item(), avg_acc.cpu().item()])"""
             
        if (self.current_epoch == 1):
            self.logger.experiment.add_text(
                "Params", self.log_without_markdown(self.config["PARAMS"]))
            self.logger.experiment.add_text(
                "Model architecture", self.log_without_markdown(self.model))
            self.logger.experiment.add_text(
                "Model architecture", "Quantum is " +
                str(self.config["QUANTUM"]))
            # add graph model, is not supported for quantum layer
            # if settings.QUANTUM == False:
            #    sampleImg = torch.rand(self.input_shape).cuda()
            #    self.logger.experiment.add_graph(self.model, sampleImg)
        # Save metrics to CSV
        
        

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

        # Save metrics to CSV
        
        """with open(self.val_output_file, mode='a', newline='') as file:
             writer = csv.writer(file)
             writer.writerow([self.current_epoch, avg_loss.cpu().item(), avg_acc.cpu().item()])"""
        
        # save weights
        torch.save(self.model.state_dict(), os.path.join(self.config["SAVE_DIR"], self.config["SAVE_NAME"], self.config["EXP_NAME"],
                                     "weights_epoch_{}.pt".format(self.current_epoch)))
        # save circuit
        if self.config["QUANTUM"]:
            fig, ax = qml.draw_mpl(self.model.qcnn[0].qlayer, expansion_strategy="device", decimals=3)(
                self.model.qcnn[0].qnode_inputs, self.model.qcnn[0].torch_qlayer.weights.data)
            plt.savefig(os.path.join(self.config["SAVE_DIR"], self.config["SAVE_NAME"], self.config["EXP_NAME"],
                                     "circuit_epoch_{}.png".format(self.current_epoch)))
    
    def test_epoch_end(self, outputs):
        avg_acc = torch.stack([x["test_accuracy"] for x in outputs]).mean()
        self.stacked_preds = torch.cat([x["pred"] for x in outputs], dim=0)
        self.stacked_y = torch.cat([x["y"] for x in outputs], dim=0)
        logs = {'test_acc': avg_acc}
        return {'avg_test_acc': avg_acc, 'progress_bar': logs}

    def on_after_backward(self):
        if self.config["QUANTUM"]:
            pass
            # self.logger.experiment.add_histogram("qlayer", self.model.qcnn.torch_qlayer.weights, self.current_epoch)
            # self.logger.experiment.add_histogram("qlayer_grad", self.model.qcnn.torch_qlayer.weights.grad, self.current_epoch)

    def train_dataloader(self):
        return self.data.train_dataloader()
        
    def val_dataloader(self):
        return self.data.val_dataloader()

    def test_dataloader(self):
        return self.data.test_dataloader()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config["PARAMS"]["lr"])      
            # option to have separate learning rates for the classical and the quantum layers 
            # [
            #     {"params": self.model.qcnn.parameters(), "lr" : self.config["PARAMS"]["lr_quantum"]},
            #     {"params": self.model.classic.parameters()},
            #     {"params": self.model.flatten.parameters()},
            #     {"params": self.model.fc.parameters()},
            # ],lr=self.config["PARAMS"]["lr_classical"],

            # option to change the learning rate depending on epoch
            # if(self.current_epoch<5):

        return optimizer


    def log_without_markdown(self, text):
        text = str(text)
        text = text.replace("\n", "  \n    ")
        text = "    " + text
        return text

    def print_conf_matrix(self, stacked_preds, stacked_y):
        conf = confusion_matrix(stacked_y, stacked_preds)
        self.logger.experiment.add_text("Validation confusion matrix", self.log_without_markdown(conf))
        return conf
