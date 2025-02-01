import torch
from torchquantum.dataset import CIFAR10
from torchquantum.dataset import MNIST

from .models import *



# Seed for reproducibility
seed = 151836

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Variable to decide whether or not to redo all experiments if scripts already found outputs
redo = False

# Datasets
datasets = [
    MNIST(
        root="./datasets",
        train_valid_split_ratio=[0.9, 0.1],
        digits_of_interest=[3, 6]
        ),
    MNIST(
        root="./datasets",
        train_valid_split_ratio=[0.9, 0.1],
        digits_of_interest=[3, 6],
        fashion=True
        ),
    CIFAR10(
        root="./datasets",
        train_valid_split_ratio=[0.9, 0.1],
        digits_of_interest=[3, 5],
        center_crop = 28,
        grayscale=True
    )
]

dataset_names = [
    'MNIST', 
    'FashionMNIST',
    'CIFAR10'
]

# Models
models = [
    QFCModel().to(device),
    CFCModel().to(device),
]

model_names = [
    'QFC',
    'CFC'
]

n_epochs = 10

generator = torch.Generator()
generator.manual_seed(seed)

# Attacks
epsilons = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0
]

attack_names = [
    'APGD',
    'BIM',
    'DIFGSM',
    'FAB',
    'EOTPGD',
    'FFGSM',
    'FGSM',
    'Jitter',
    'MIFGSM',
    'NIFGSM',
    'PGD',
    'PGDRS',
    'PGDL2',
    'PGDRSL2',
    'RFGSM',
    'SINIFGSM',
    'SPSA',
    'TPGD',
    'UPGD',
    'VMIFGSM',
    'VNIFGSM'
]