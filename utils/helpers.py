import numpy as np
import pandas as pd
import os
import random

from sklearn.metrics import f1_score
import torch
import torch.nn.functional as F

from .params import *



def setSeed(seed=seed):
    """
    Setting the seed for reproducibility
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

setSeed()


def train(dataflow, model, device, optimizer):
    for feed_dict in dataflow["train"]:
        inputs = feed_dict["image"].to(device)
        targets = feed_dict["digit"].to(device)

        outputs = model(inputs)
        loss = F.nll_loss(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss


def test(dataflow, split, model, device, qiskit=False):
    target_all = []
    output_all = []
    with torch.no_grad():
        for feed_dict in dataflow[split]:
            inputs = feed_dict["image"].to(device)
            targets = feed_dict["digit"].to(device)

            outputs = model(inputs)

            target_all.append(targets)
            output_all.append(outputs)
        target_all = torch.cat(target_all, dim=0)
        output_all = torch.cat(output_all, dim=0)

    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size
    loss = F.nll_loss(output_all, target_all).item()

    return accuracy, loss


def test_atks(model, model_name, dataflow, dataset_name):
    results_df = []
    for target_model_name in model_names:
        for attack_name in attack_names:
            for epsilon in epsilons:
                # Loading attack samples
                all_adv_images = []
                all_labels = []

                atk_path = f'./attack/{dataset_name}/{target_model_name}/{attack_name}/{str(epsilon)}'

                for i in range(len(dataflow['test'])):  # Assuming equal number of images and labels
                    adv_images = torch.load(f'{atk_path}/adv_images_{i}.pt')
                    labels = torch.load(f'{atk_path}/labels_{i}.pt')
                    all_adv_images.append(adv_images)
                    all_labels.append(labels)

                # Concatenate all adversarial images and labels
                all_adv_images = torch.cat(all_adv_images, dim=0)
                all_labels = torch.cat(all_labels, dim=0)

                # Evaluate the model on the adversarial examples
                model.eval()
                outputs = model(all_adv_images)
                _, predicted = torch.max(outputs.data, 1)

                # Calculate accuracy
                correct = (predicted == all_labels).sum().item()
                total = all_labels.size(0)
                accuracy = correct / total
                f1 = f1_score(all_labels.cpu(), predicted.cpu(), average='weighted')

                results_df.append({
                    'Target_Model': target_model_name,
                    'Source_Model': model_name,
                    'Attack': attack_name,
                    'Epsilon': epsilon,
                    'Accuracy': accuracy,
                    'F1': f1
                })

    return pd.DataFrame(results_df)