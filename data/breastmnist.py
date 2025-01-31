# %%
import medmnist
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
import torch
import numpy as np
import random
import math


class BreastMNISTDataModule(LightningDataModule):
    def __init__(self, params, download_dir=None):
        super().__init__()
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.params = params
        if download_dir is None:
            self.root = "./"
        else:
            self.root = download_dir

        BreastMNISTWithIndices = self.dataset_with_indices(medmnist.BreastMNIST)

        self.train_dataset = medmnist.BreastMNIST(
            split='train', transform=data_transform, download=True, root=self.root)
        self.val_dataset = BreastMNISTWithIndices(
            split='val', transform=data_transform, download=True, root=self.root)
        self.test_dataset = medmnist.BreastMNIST(
            split='test', transform=data_transform, download=True, root=self.root)
        self.train_dataset.labels = [self.train_dataset.labels[i][0]
                                     for i in range(len(self.train_dataset.labels))]
        self.val_dataset.labels = [self.val_dataset.labels[i][0]
                                   for i in range(len(self.val_dataset.labels))]
        self.test_dataset.labels = [self.test_dataset.labels[i][0]
                                   for i in range(len(self.test_dataset.labels))]
        self.get_weights_data()

    def get_weights_data(self):
        targets = self.train_dataset.labels
        n_train_samples = [targets.count(value) for value in set(targets)]
        self.class_weights = [max(n_train_samples) / n_train_samples[i] for i in set(targets)]
        #print("class_weights", self.class_weights)

    def seed_worker(self, worker_id):
        torch.manual_seed(self.params['all_seeds'])
        np.random.seed(self.params["all_seeds"])
        random.seed(self.params["all_seeds"])
        torch.backends.cudnn.deterministic = True  # tested - needed for reproducibility
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(self.params['all_seeds'])
        torch.cuda.manual_seed_all(self.params['all_seeds'])

    def train_dataloader(self):
        gen = torch.Generator()
        gen.manual_seed(0)
        return DataLoader(self.train_dataset, batch_size=self.params["batch_size"], shuffle=True,
                          num_workers=4, generator=gen, pin_memory=True)

    def val_dataloader(self):
        gen = torch.Generator()
        gen.manual_seed(0)
        return DataLoader(self.val_dataset, batch_size=self.params["batch_size"], shuffle=False,
                          num_workers=4, generator=gen, worker_init_fn=self.seed_worker, pin_memory=True)

    def test_dataloader(self):
        gen = torch.Generator()
        gen.manual_seed(0)
        return DataLoader(self.test_dataset, batch_size=self.params["batch_size"], shuffle=False,
                          num_workers=4, generator=gen, worker_init_fn=self.seed_worker, pin_memory=True)

    def dataset_with_indices(self, cls):
        """
        Modifies the given Dataset class to return a tuple data, target, index
        instead of just data, target.
        """
        def __getitem__(self, index):
            data, target = cls.__getitem__(self, index)
            return data, target, index
        return type(cls.__name__, (cls,), {
            '__getitem__': __getitem__,
        })


def breast_mnist_dataset(params, download_dir=None):
    data = BreastMNISTDataModule(params, download_dir)
    return data

def breast_mnist_dataload(params):
    data = data = BreastMNISTDataModule(params)
    return data.train_dataloader()
