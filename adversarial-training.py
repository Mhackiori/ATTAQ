import os
import pandas as pd
import sys
import warnings

from sklearn.metrics import f1_score
import torch
import torch.optim as optim
import torchattacks
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

# Training each model on their own attack samples
for dataset, dataset_name in zip(datasets, dataset_names):
    print(f'[📚 DATASET] {dataset_name}')
    # Check if the dataset exists
    if not os.path.exists(f'./results/{dataset_name}/training.csv') or redo:
        # Defining save DataFrame
        results_df = []
        # Defining dataflow
        dataflow = dict()
        # Train, validation, and test dataflow
        for split in dataset:
            sampler = torch.utils.data.RandomSampler(dataset[split], generator=generator)
            dataflow[split] = torch.utils.data.DataLoader(
                dataset[split],
                batch_size=256,
                sampler=sampler,
                num_workers=8,
                pin_memory=True,
                generator=generator
            )

        print(f'\t[🤖 MODEL] {model_names[0]}')
        qfc = QFCModel().to(device)
        optimizer = optim.Adam(qfc.parameters(), lr=5e-3, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

        # Training
        for epoch in range(1, n_epochs + 1):
            for i, feed_dict in enumerate(dataflow["train"]):
                print(f'\t\t[#️⃣ EPOCH {epoch}/{n_epochs}] Batch: {i+1}/{len(dataflow["train"])}', end='\r')
                inputs = feed_dict["image"].to(device)
                targets = feed_dict["digit"].to(device)

                outputs = qfc(inputs)
                loss = F.nll_loss(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Attacks
                epoch_eps = (epoch/n_epochs)

                attacks = [
                    torchattacks.APGD(qfc, eps=epoch_eps),
                    torchattacks.BIM(qfc, eps=epoch_eps),
                    torchattacks.DIFGSM(qfc, eps=epoch_eps),
                    torchattacks.FAB(qfc, eps=epoch_eps),
                    torchattacks.EOTPGD(qfc, eps=epoch_eps),
                    torchattacks.FFGSM(qfc, eps=epoch_eps),
                    torchattacks.FGSM(qfc, eps=epoch_eps),
                    torchattacks.Jitter(qfc, eps=epoch_eps),
                    torchattacks.MIFGSM(qfc, eps=epoch_eps),
                    torchattacks.NIFGSM(qfc, eps=epoch_eps),
                    torchattacks.PGD(qfc, eps=epoch_eps),
                    torchattacks.PGDRS(qfc, eps=epoch_eps),
                    torchattacks.PGDL2(qfc, eps=epoch_eps),
                    torchattacks.PGDRSL2(qfc, eps=epoch_eps),
                    torchattacks.RFGSM(qfc, eps=epoch_eps),
                    torchattacks.SINIFGSM(qfc, eps=epoch_eps),
                    torchattacks.SPSA(qfc, eps=epoch_eps),
                    torchattacks.TPGD(qfc, eps=epoch_eps),
                    torchattacks.UPGD(qfc, eps=epoch_eps),
                    torchattacks.VMIFGSM(qfc, eps=epoch_eps),
                    torchattacks.VNIFGSM(qfc, eps=epoch_eps)
                ]

                for attack in attacks:
                    adv_inputs = attack(inputs, targets)
                    adv_outputs = qfc(adv_inputs)

                    adv_loss = F.nll_loss(adv_outputs, targets)
                    optimizer.zero_grad()
                    adv_loss.backward()
                    optimizer.step()

            # Validation
            val_acc, val_loss = test(dataflow, "valid", qfc, device)
            scheduler.step()

            print(f'\n\t\t[#️⃣ EPOCH {epoch}/{n_epochs}] {val_acc:.3f}')

        # Testing
        test_acc, test_loss, qfc_predicted = test(dataflow, "test", qfc, device, qiskit=False, extract_predicted=True)
        print(f'\t\t[🧪 TEST ACCURACY] {test_acc:.3f}')

        qfc_df = test_atks(qfc, 'QFC', dataflow, dataset_name, qfc_predicted)

        ###

        print(f'\t[🤖 MODEL] {model_names[1]}')
        cfc = CFCModel().to(device)
        optimizer = optim.Adam(cfc.parameters(), lr=5e-3, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

        # Training
        for epoch in range(1, n_epochs + 1):
            for i, feed_dict in enumerate(dataflow["train"]):
                print(f'\t\t[#️⃣ EPOCH {epoch}/{n_epochs}] Batch: {i+1}/{len(dataflow["train"])}', end='\r')
                inputs = feed_dict["image"].to(device)
                targets = feed_dict["digit"].to(device)

                outputs = cfc(inputs)
                loss = F.nll_loss(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Attacks
                epoch_eps = (epoch/n_epochs)

                attacks = [
                    torchattacks.APGD(cfc, eps=epoch_eps),
                    torchattacks.BIM(cfc, eps=epoch_eps),
                    torchattacks.DIFGSM(cfc, eps=epoch_eps),
                    torchattacks.FAB(cfc, eps=epoch_eps),
                    torchattacks.EOTPGD(cfc, eps=epoch_eps),
                    torchattacks.FFGSM(cfc, eps=epoch_eps),
                    torchattacks.FGSM(cfc, eps=epoch_eps),
                    torchattacks.Jitter(cfc, eps=epoch_eps),
                    torchattacks.MIFGSM(cfc, eps=epoch_eps),
                    torchattacks.NIFGSM(cfc, eps=epoch_eps),
                    torchattacks.PGD(cfc, eps=epoch_eps),
                    torchattacks.PGDRS(cfc, eps=epoch_eps),
                    torchattacks.PGDL2(cfc, eps=epoch_eps),
                    torchattacks.PGDRSL2(cfc, eps=epoch_eps),
                    torchattacks.RFGSM(cfc, eps=epoch_eps),
                    torchattacks.SINIFGSM(cfc, eps=epoch_eps),
                    torchattacks.SPSA(cfc, eps=epoch_eps),
                    torchattacks.TPGD(cfc, eps=epoch_eps),
                    torchattacks.UPGD(cfc, eps=epoch_eps),
                    torchattacks.VMIFGSM(cfc, eps=epoch_eps),
                    torchattacks.VNIFGSM(cfc, eps=epoch_eps)
                ]

                for attack in attacks:
                    adv_inputs = attack(inputs, targets)
                    adv_outputs = cfc(adv_inputs)

                    adv_loss = F.nll_loss(adv_outputs, targets)
                    optimizer.zero_grad()
                    adv_loss.backward()
                    optimizer.step()

            # Validation
            val_acc, val_loss = test(dataflow, "valid", cfc, device)
            scheduler.step()

            print(f'\n\t\t[#️⃣ EPOCH {epoch}/{n_epochs}] {val_acc:.3f}')

        # Testing
        test_acc, test_loss, cfc_predicted = test(dataflow, "test", cfc, device, qiskit=False, extract_predicted=True)
        print(f'\t\t[🧪 TEST ACCURACY] {test_acc:.3f}')

        cfc_df = test_atks(cfc, 'CFC', dataflow, dataset_name, cfc_predicted)

        # Concatenate results and save
        results_df = pd.concat([qfc_df, cfc_df], ignore_index=True)
        results_df.to_csv(f'./results/{dataset_name}/training.csv', index=False)

        #####################################################
        # Training both models together, with cross attacks #
        #####################################################

        print('\t[🤖 BOTH MODELS]')
        qfc = QFCModel().to(device)
        q_optimizer = optim.Adam(qfc.parameters(), lr=5e-3, weight_decay=1e-4)
        q_scheduler = CosineAnnealingLR(q_optimizer, T_max=n_epochs)

        cfc = CFCModel().to(device)
        c_optimizer = optim.Adam(cfc.parameters(), lr=5e-3, weight_decay=1e-4)
        c_scheduler = CosineAnnealingLR(c_optimizer, T_max=n_epochs)

        # Training
        for epoch in range(1, n_epochs + 1):
            for i, feed_dict in enumerate(dataflow["train"]):
                print(f'\t\t[#️⃣ EPOCH {epoch}/{n_epochs}] Batch: {i+1}/{len(dataflow["train"])}', end='\r')
                inputs = feed_dict["image"].to(device)
                targets = feed_dict["digit"].to(device)

                q_outputs = qfc(inputs)
                c_outputs = cfc(inputs)
                q_loss = F.nll_loss(q_outputs, targets)
                c_loss = F.nll_loss(c_outputs, targets)
                q_optimizer.zero_grad()
                c_optimizer.zero_grad()
                q_loss.backward()
                c_loss.backward()
                q_optimizer.step()
                c_optimizer.step()

                # Attacks
                epoch_eps = (epoch/n_epochs)

                attacks = [
                    torchattacks.APGD(qfc, eps=epoch_eps),
                    torchattacks.BIM(qfc, eps=epoch_eps),
                    torchattacks.DIFGSM(qfc, eps=epoch_eps),
                    torchattacks.FAB(qfc, eps=epoch_eps),
                    torchattacks.EOTPGD(qfc, eps=epoch_eps),
                    torchattacks.FFGSM(qfc, eps=epoch_eps),
                    torchattacks.FGSM(qfc, eps=epoch_eps),
                    torchattacks.Jitter(qfc, eps=epoch_eps),
                    torchattacks.MIFGSM(qfc, eps=epoch_eps),
                    torchattacks.NIFGSM(qfc, eps=epoch_eps),
                    torchattacks.PGD(qfc, eps=epoch_eps),
                    torchattacks.PGDRS(qfc, eps=epoch_eps),
                    torchattacks.PGDL2(qfc, eps=epoch_eps),
                    torchattacks.PGDRSL2(qfc, eps=epoch_eps),
                    torchattacks.RFGSM(qfc, eps=epoch_eps),
                    torchattacks.SINIFGSM(qfc, eps=epoch_eps),
                    torchattacks.SPSA(qfc, eps=epoch_eps),
                    torchattacks.TPGD(qfc, eps=epoch_eps),
                    torchattacks.UPGD(qfc, eps=epoch_eps),
                    torchattacks.VMIFGSM(qfc, eps=epoch_eps),
                    torchattacks.VNIFGSM(qfc, eps=epoch_eps),
                    torchattacks.APGD(cfc, eps=epoch_eps),
                    torchattacks.BIM(cfc, eps=epoch_eps),
                    torchattacks.DIFGSM(cfc, eps=epoch_eps),
                    torchattacks.FAB(cfc, eps=epoch_eps),
                    torchattacks.EOTPGD(cfc, eps=epoch_eps),
                    torchattacks.FFGSM(cfc, eps=epoch_eps),
                    torchattacks.FGSM(cfc, eps=epoch_eps),
                    torchattacks.Jitter(cfc, eps=epoch_eps),
                    torchattacks.MIFGSM(cfc, eps=epoch_eps),
                    torchattacks.NIFGSM(cfc, eps=epoch_eps),
                    torchattacks.PGD(cfc, eps=epoch_eps),
                    torchattacks.PGDRS(cfc, eps=epoch_eps),
                    torchattacks.PGDL2(cfc, eps=epoch_eps),
                    torchattacks.PGDRSL2(cfc, eps=epoch_eps),
                    torchattacks.RFGSM(cfc, eps=epoch_eps),
                    torchattacks.SINIFGSM(cfc, eps=epoch_eps),
                    torchattacks.SPSA(cfc, eps=epoch_eps),
                    torchattacks.TPGD(cfc, eps=epoch_eps),
                    torchattacks.UPGD(cfc, eps=epoch_eps),
                    torchattacks.VMIFGSM(cfc, eps=epoch_eps),
                    torchattacks.VNIFGSM(cfc, eps=epoch_eps)
                ]

                for attack in attacks:
                    adv_inputs = attack(inputs, targets)
                    q_adv_outputs = qfc(adv_inputs)
                    c_adv_outputs = cfc(adv_inputs)

                    q_adv_loss = F.nll_loss(q_adv_outputs, targets)
                    q_optimizer.zero_grad()
                    q_adv_loss.backward(retain_graph=True)
                    q_optimizer.step()
                    c_adv_loss = F.nll_loss(c_adv_outputs, targets)
                    c_optimizer.zero_grad()
                    c_adv_loss.backward(retain_graph=True)
                    c_optimizer.step()

            # Validation
            q_val_acc, q_val_loss = test(dataflow, "valid", qfc, device)
            q_scheduler.step()

            c_val_acc, c_val_loss = test(dataflow, "valid", cfc, device)
            c_scheduler.step()

            print(f'\n\t\t[#️⃣ Q EPOCH {epoch}/{n_epochs}] {q_val_acc:.3f}')
            print(f'\t\t[#️⃣ C EPOCH {epoch}/{n_epochs}] {c_val_acc:.3f}')

        # Testing
        q_test_acc, q_test_loss, q_predicted = test(dataflow, "test", qfc, device, qiskit=False, extract_predicted=True)
        c_test_acc, c_test_loss, c_predicted = test(dataflow, "test", cfc, device, qiskit=False, extract_predicted=True)
        print(f'\t\t[🧪 Q TEST ACCURACY] {q_test_acc:.3f}')
        print(f'\t\t[🧪 C TEST ACCURACY] {c_test_acc:.3f}')

        cross_qfc_df = test_atks(qfc, 'QFC', dataflow, dataset_name, q_predicted)
        cross_cfc_df = test_atks(cfc, 'CFC', dataflow, dataset_name, c_predicted)

        # Concatenate results and save
        cross_results_df = pd.concat([cross_qfc_df, cross_cfc_df], ignore_index=True)
        cross_results_df.to_csv(f'./results/{dataset_name}/cross-training.csv', index=False)