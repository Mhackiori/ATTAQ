import os
import sys
import warnings

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.helpers import *
from utils.models import *
from utils.params import *

warnings.filterwarnings("ignore")

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

setSeed()



for dataset, dataset_name in zip(datasets, dataset_names):
    print(f'[📚 DATASET] {dataset_name}')
    # Create the directory if it doesn't exist
    model_dir = f'./models/{dataset_name}'
    # If the directory does not exists or redo is True proceed
    if not os.path.exists(model_dir) or redo:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        # Defining dataflow
        dataflow = dict()
        # Train, validation, and test dataflow
        for split in dataset:
            sampler = torch.utils.data.RandomSampler(dataset[split])
            dataflow[split] = torch.utils.data.DataLoader(
                dataset[split],
                batch_size=256,
                sampler=sampler,
                num_workers=8,
                pin_memory=True,
            )

        for model, name in zip(models, model_names):
            print(f'\t[🤖 MODEL] {name}')
            # Defining parameters
            optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

            # Training
            for epoch in range(1, n_epochs + 1):
                train_loss = train(dataflow, model, device, optimizer)

                # Validation
                val_acc, val_loss = test(dataflow, "valid", model, device)
                scheduler.step()

                print(f'\t\t[#️⃣ EPOCH {epoch}/{n_epochs}] {val_acc:.3f}')

            # Testing
            test_acc, test_loss = test(dataflow, "test", model, device, qiskit=False)
            print(f'\t\t[🧪 TEST ACCURACY] {test_acc:.3f}')

            # Save the model
            model_path = os.path.join(model_dir, f'{name}.pt')
            torch.save(model.state_dict(), model_path)
            print(f'\t\t[💾 MODEL SAVED] {model_path}')