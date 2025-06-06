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



for dataset, dataset_name in zip(datasets, dataset_names):
    print(f'[📚 DATASET] {dataset_name}')
    # Check if the dataset exists
    if not os.path.exists(f'./attack/{dataset_name}/transferability.csv') or redo:
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
            train_loss = train(dataflow, qfc, device, optimizer)

            # Validation
            val_acc, val_loss = test(dataflow, "valid", qfc, device)
            scheduler.step()

            print(f'\t\t[#️⃣ EPOCH {epoch}/{n_epochs}] {val_acc:.3f}')

        # Testing
        test_acc, test_loss, qfc_predicted = test(dataflow, "test", qfc, device, qiskit=False, extract_predicted=True)
        print(f'\t\t[🧪 TEST ACCURACY] {test_acc:.3f}')

        ###

        print(f'\t[🤖 MODEL] {model_names[1]}')
        cfc = CFCModel().to(device)
        optimizer = optim.Adam(cfc.parameters(), lr=5e-3, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

        # Training
        for epoch in range(1, n_epochs + 1):
            train_loss = train(dataflow, cfc, device, optimizer)

            # Validation
            val_acc, val_loss = test(dataflow, "valid", cfc, device)
            scheduler.step()

            print(f'\t\t[#️⃣ EPOCH {epoch}/{n_epochs}] {val_acc:.3f}')

        # Testing
        test_acc, test_loss, cfc_predicted = test(dataflow, "test", cfc, device, qiskit=False, extract_predicted=True)
        print(f'\t\t[🧪 TEST ACCURACY] {test_acc:.3f}')

        ###

        print()

        # Generating attacks
        os.makedirs(f'./attack/{dataset_name}', exist_ok=True)
        for epsilon in epsilons:
            attacks = [
                torchattacks.APGD(qfc, eps=epsilon),
                torchattacks.BIM(qfc, eps=epsilon),
                torchattacks.DIFGSM(qfc, eps=epsilon),
                torchattacks.FAB(qfc, eps=epsilon),
                torchattacks.EOTPGD(qfc, eps=epsilon),
                torchattacks.FFGSM(qfc, eps=epsilon),
                torchattacks.FGSM(qfc, eps=epsilon),
                torchattacks.Jitter(qfc, eps=epsilon),
                torchattacks.MIFGSM(qfc, eps=epsilon),
                torchattacks.NIFGSM(qfc, eps=epsilon),
                torchattacks.PGD(qfc, eps=epsilon),
                torchattacks.PGDRS(qfc, eps=epsilon),
                torchattacks.PGDL2(qfc, eps=epsilon),
                torchattacks.PGDRSL2(qfc, eps=epsilon),
                torchattacks.RFGSM(qfc, eps=epsilon),
                torchattacks.SINIFGSM(qfc, eps=epsilon),
                torchattacks.SPSA(qfc, eps=epsilon),
                torchattacks.TPGD(qfc, eps=epsilon),
                torchattacks.UPGD(qfc, eps=epsilon),
                torchattacks.VMIFGSM(qfc, eps=epsilon),
                torchattacks.VNIFGSM(qfc, eps=epsilon)
            ]

            for attack, attack_name in zip(attacks, attack_names):

                atk_path = f'./attack/{dataset_name}/QFC/{attack_name}/{str(epsilon)}'
                os.makedirs(atk_path, exist_ok=True)

                for i, feed_dict in enumerate(dataflow['test']):
                    print(f'\t[⚔️ QFC {attack_name} @ {epsilon}] {i+1}/{len(dataflow["test"])}', end='\r')

                    images, labels = feed_dict['image'].to(device), feed_dict['digit'].to(device)

                    # Generate adversarial examples
                    adv_images = attack(images, labels)

                    # Save the adversarial images directly
                    torch.save(adv_images, f'{atk_path}/adv_images_{i}.pt')
                    # Save the labels
                    torch.save(labels, f'{atk_path}/labels_{i}.pt')
                print()

        ###
            
        print()

        # Generating attacks
        for epsilon in epsilons:
            attacks = [
                torchattacks.APGD(cfc, eps=epsilon),
                torchattacks.BIM(cfc, eps=epsilon),
                torchattacks.DIFGSM(cfc, eps=epsilon),
                torchattacks.FAB(cfc, eps=epsilon),
                torchattacks.EOTPGD(cfc, eps=epsilon),
                torchattacks.FFGSM(cfc, eps=epsilon),
                torchattacks.FGSM(cfc, eps=epsilon),
                torchattacks.Jitter(cfc, eps=epsilon),
                torchattacks.MIFGSM(cfc, eps=epsilon),
                torchattacks.NIFGSM(cfc, eps=epsilon),
                torchattacks.PGD(cfc, eps=epsilon),
                torchattacks.PGDRS(cfc, eps=epsilon),
                torchattacks.PGDL2(cfc, eps=epsilon),
                torchattacks.PGDRSL2(cfc, eps=epsilon),
                torchattacks.RFGSM(cfc, eps=epsilon),
                torchattacks.SINIFGSM(cfc, eps=epsilon),
                torchattacks.SPSA(cfc, eps=epsilon),
                torchattacks.TPGD(cfc, eps=epsilon),
                torchattacks.UPGD(cfc, eps=epsilon),
                torchattacks.VMIFGSM(cfc, eps=epsilon),
                torchattacks.VNIFGSM(cfc, eps=epsilon)
            ]

            for attack, attack_name in zip(attacks, attack_names):

                atk_path = f'./attack/{dataset_name}/CFC/{attack_name}/{str(epsilon)}'
                os.makedirs(atk_path, exist_ok=True)

                for i, feed_dict in enumerate(dataflow['test']):
                    print(f'\t[⚔️ CFC {attack_name} @ {epsilon}] {i+1}/{len(dataflow["test"])}', end='\r')

                    images, labels = feed_dict['image'].to(device), feed_dict['digit'].to(device)

                    # Generate adversarial examples
                    adv_images = attack(images, labels)

                    # Save the adversarial images directly
                    torch.save(adv_images, f'{atk_path}/adv_images_{i}.pt')
                    # Save the labels
                    torch.save(labels, f'{atk_path}/labels_{i}.pt')
                print()

        ######

        print()

        for target_model_name in model_names:
            for attack_name in attack_names:
                for epsilon in epsilons:
                    # Loading attack samples
                    all_adv_images = []
                    all_labels = []

                    atk_path = f'./attack/{dataset_name}/{target_model_name}/{attack_name}/{str(epsilon)}'

                    for i in range(len(dataflow['test'])):
                        adv_images = torch.load(f'{atk_path}/adv_images_{i}.pt')
                        labels = torch.load(f'{atk_path}/labels_{i}.pt')
                        all_adv_images.append(adv_images)
                        all_labels.append(labels)

                    # Concatenate all adversarial images and labels
                    all_adv_images = torch.cat(all_adv_images, dim=0)
                    all_labels = torch.cat(all_labels, dim=0)

                    # Evaluate the model on the adversarial examples
                    qfc.eval()
                    outputs = qfc(all_adv_images)
                    _, predicted = torch.max(outputs.data, 1)

                    # Calculate accuracy
                    correct = (predicted == all_labels).sum().item()
                    total = all_labels.size(0)
                    accuracy = correct / total
                    f1 = f1_score(all_labels.cpu(), predicted.cpu(), average='weighted')

                    # ASR
                    qfc_predictions_tensor = torch.cat(qfc_predicted, dim=0).to(device)
                    correctly_classified_clean = (qfc_predictions_tensor == all_labels)
                    misclassified_adv = (predicted != all_labels)
                    successful_attacks = correctly_classified_clean & misclassified_adv

                    num_successful_attacks = successful_attacks.sum().item()
                    num_correct_clean = correctly_classified_clean.sum().item()
                    asr = num_successful_attacks / num_correct_clean if num_correct_clean > 0 else 0


                    print(f'\t[⚔️ QFC VS. {target_model_name} {attack_name} @ {epsilon}] {accuracy:.3f} | ASR: {asr:.3f}')

                    results_df.append({
                        'Target_Model': target_model_name,
                        'Source_Model': 'QFC',
                        'Attack': attack_name,
                        'Epsilon': epsilon,
                        'Accuracy': accuracy,
                        'F1': f1,
                        'ASR': asr
                    })

        print()

        for target_model_name in model_names:
            for attack_name in attack_names:
                for epsilon in epsilons:
                    # Loading attack samples
                    all_adv_images = []
                    all_labels = []

                    atk_path = f'./attack/{dataset_name}/{target_model_name}/{attack_name}/{str(epsilon)}'

                    for i in range(len(dataflow['test'])): 
                        adv_images = torch.load(f'{atk_path}/adv_images_{i}.pt')
                        labels = torch.load(f'{atk_path}/labels_{i}.pt')
                        all_adv_images.append(adv_images)
                        all_labels.append(labels)

                    # Concatenate all adversarial images and labels
                    all_adv_images = torch.cat(all_adv_images, dim=0)
                    all_labels = torch.cat(all_labels, dim=0)

                    # Evaluate the model on the adversarial examples
                    cfc.eval()
                    outputs = cfc(all_adv_images)
                    _, predicted = torch.max(outputs.data, 1)

                    # Calculate accuracy
                    correct = (predicted == all_labels).sum().item()
                    total = all_labels.size(0)
                    accuracy = correct / total
                    f1 = f1_score(all_labels.cpu(), predicted.cpu(), average='weighted')

                    # ASR
                    cfc_predictions_tensor = torch.cat(cfc_predicted, dim=0).to(device)
                    correctly_classified_clean = (cfc_predictions_tensor == all_labels)
                    misclassified_adv = (predicted != all_labels)
                    successful_attacks = correctly_classified_clean & misclassified_adv

                    num_successful_attacks = successful_attacks.sum().item()
                    num_correct_clean = correctly_classified_clean.sum().item()
                    asr = num_successful_attacks / num_correct_clean if num_correct_clean > 0 else 0

                    print(f'\t[⚔️ CFC VS. {target_model_name} {attack_name} @ {epsilon}] {accuracy:.3f} | ASR: {asr:.3f}')

                    results_df.append({
                        'Target_Model': target_model_name,
                        'Source_Model': 'CFC',
                        'Attack': attack_name,
                        'Epsilon': epsilon,
                        'Accuracy': accuracy,
                        'F1': f1,
                        'ASR': asr
                    })

        results_path = f'./results/{dataset_name}'
        os.makedirs(results_path, exist_ok=True)
        results_df = pd.DataFrame(results_df)
        results_df.to_csv(f'./results/{dataset_name}/transferability.csv', index=False)