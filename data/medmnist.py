# %%
import medmnist
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from pytorch_lightning import LightningDataModule
import torch
import numpy as np
import random
from sklearn.model_selection import train_test_split
import math

class MedMNISTDataModule(LightningDataModule):
    def __init__(self, params, train_size, val_size):
        super().__init__()
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]) #if commented, data is between 0 and 1
        ])
        self.params = params
        self.train_size = train_size
        self.val_size = val_size
        self.root = "./"

        OrganAMNISTWithIndices = self.dataset_with_indices(medmnist.OrganAMNIST)

        self.train_dataset = medmnist.OrganAMNIST(
            split='train', transform=data_transform, download=True, root=self.root)
        self.val_dataset = OrganAMNISTWithIndices(
            split='val', transform=data_transform, download=True, root=self.root)
        self.train_dataset.labels = [self.train_dataset.labels[i][0]
                                     for i in range(len(self.train_dataset.labels))]
        self.val_dataset.labels = [self.val_dataset.labels[i][0]
                                   for i in range(len(self.val_dataset.labels))]
        self.get_subset_data()

    def get_subset_data(self):
        train_indices, val_indices = np.arange(
            len(self.train_dataset)), np.arange(len(self.val_dataset))
        train_samples, _ = train_test_split(train_indices, train_size=self.train_size,
                                            stratify=self.train_dataset.labels, random_state=56)
        val_samples, _ = train_test_split(val_indices, train_size=self.val_size,
                                          stratify=self.val_dataset.labels, random_state=56)
        self.train_dataset_subset = Subset(self.train_dataset, train_samples)
        self.val_dataset_subset = Subset(self.val_dataset, val_samples)

        targets = self.train_dataset_subset.dataset.labels
        n_train_samples = [targets.count(value) for value in set(targets)]
        self.class_weights = [max(n_train_samples) / n_train_samples[i] for i in set(targets)]

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
        return DataLoader(self.train_dataset_subset, batch_size=self.params["batch_size"], shuffle=True,
                          num_workers=4, generator=gen, pin_memory=True)

    def val_dataloader(self):
        gen = torch.Generator()
        gen.manual_seed(0)
        return DataLoader(self.val_dataset_subset, batch_size=self.params["batch_size"], shuffle=False,
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
        
def medmnist_dataset(params, train_size=34500, val_size=6400):
    data = MedMNISTDataModule(params, train_size, val_size)
    return data
def medmnist_dataload(params, train_size=1000, val_size=300):  # val_size=600
    data = MedMNISTDataModule(params, train_size, val_size)
    return data.train_dataloader()