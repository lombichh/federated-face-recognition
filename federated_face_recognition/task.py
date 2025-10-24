from collections import OrderedDict

import torch
import torch.nn as nn
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch.optim as optim

dataset = None

"""--- Model ---"""
def get_weights(model):
    """
    Get weights of model.
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_weights(model, parameters):
    """
    Set weights on model.
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

"""--- Dataset ---"""
def extract_person(filename):
    """
    Extracts person name from image filename.
    """
    return filename.rsplit("_", 1)[0]

def load_data(partition_id: int, num_partitions: int):
    """
    Load dataset and create trainloader and testloader.
    """
    # only initialize FederatedDataset once
    global dataset
    if dataset is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        dataset = FederatedDataset(
            dataset="bitmind/lfw",
            partitioners={"train": partitioner},
        )

    partition = dataset.load_partition(partition_id)

    # divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    # mapping person name -> idx (label)
    all_persons = [extract_person(fname) for fname in partition_train_test["train"]["filename"] + partition_train_test["test"]["filename"]]
    unique_persons = sorted(set(all_persons))
    person_to_idx = {name: idx for idx, name in enumerate(unique_persons)}

    # transform data for compatibility with the model
    pytorch_transforms = Compose(
        [Resize((160, 160)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    def apply_transforms(batch):
        batch["image"] = torch.stack([pytorch_transforms(img) for img in batch["image"]])
        batch["label"] = torch.tensor([person_to_idx[extract_person(fname)] for fname in batch["filename"]], dtype=torch.long)
        return batch
    partition_train_test = partition_train_test.with_transform(apply_transforms)

    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)

    return trainloader, testloader

def load_test_data():
    """
    Load dataset and create trainloader and testloader.
    """
    dataset = FederatedDataset(
        dataset="bitmind/lfw",
        partitioners={"train": 1},
    )

    # carica la singola partizione (tutto il dataset)
    full_partition = dataset.load_partition(0)  # l'unica partizione disponibile

    # mapping person name -> idx (label)
    all_persons = [extract_person(fname) for fname in full_partition["filename"]]
    unique_persons = sorted(set(all_persons))
    person_to_idx = {name: idx for idx, name in enumerate(unique_persons)}

    # transform data for compatibility with the model
    pytorch_transforms = Compose(
        [Resize((160, 160)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        batch["image"] = torch.stack([pytorch_transforms(img) for img in batch["image"]])
        batch["label"] = torch.tensor(
            [person_to_idx[extract_person(fname)] for fname in batch["filename"]],
            dtype=torch.long,
        )
        return batch

    full_partition = full_partition.with_transform(apply_transforms)

    # unico testloader con tutti i dati
    testloader = DataLoader(full_partition, batch_size=32)

    return testloader

"""--- Train ---"""
def train(model, trainloader, epochs, device, lr=1e-4): # TODO: change
    """
    Train model on a dataset.
    """
    model.train()

    criterion = nn.CrossEntropyLoss() # loss function (calculate loss)
    optimizer = optim.Adam(model.parameters(), lr=lr) # optimizer (update model weights based on the loss)

    epoch_losses = []

    # progress
    num_batches = len(trainloader)
    thresholds = {25, 50, 75} # percentages to be printed
    printed = set() # percentages already printed

    for epoch in range(epochs):
        print(f"Training epoch {epoch+1}/{epochs} started")

        running_loss = 0.0 # loss sum
        total = 0 # num. total images

        for batch_idx, batch in enumerate(trainloader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad() # reset gradients from previous batch
            outputs = model(images) # generate logits
            loss = criterion(outputs, labels) # calculate loss
            loss.backward() # calculate gradients
            optimizer.step() # update model weights based on the loss

            running_loss += loss.item() * images.size(0)
            total += images.size(0)

            # print progress
            progress = int((batch_idx + 1) / num_batches * 100)
            for t in thresholds:
                if progress >= t and t not in printed:
                    print(f"Training epoch {epoch+1}/{epochs}: {t}% completed")
                    printed.add(t)

            # clean memory on MPS
            del images, labels, outputs, loss
            if device.type == "mps":
                torch.mps.empty_cache()

        epoch_loss = running_loss / total
        epoch_losses.append(epoch_loss)

        print(f"Training epoch {epoch+1}/{epochs} finished - Loss: {epoch_loss:.4f}")

    return sum(epoch_losses) / len(epoch_losses) # avg loss on all epochs

"""--- Test ---"""
def extract_embeddings(model, dataloader, device):
    """
    Extract embeddings for each image in a dataset.
    """
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        # extract embeddings in batch for better memory usage
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"]

            embeddings = model(images) # calculate embeddings for each image in batch

            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # clean memory on MPS
            del images, embeddings
            if device.type == "mps":
                torch.mps.empty_cache()

    return np.vstack(all_embeddings), np.array(all_labels)

def calculate_person_embeddings(embeddings, labels):
    """
    Calculate average embedding for each person.
    """
    unique_labels = np.unique(labels)
    person_embeddings = {}
    
    for person_id in unique_labels:
        person_mask = labels == person_id
        person_imgs = embeddings[person_mask]
        person_embeddings[person_id] = np.mean(person_imgs, axis=0)
    
    return person_embeddings

def calculate_recognition_accuracy_topk(embeddings, labels, person_embeddings, k=5):
    """
    Calculate recognition accuracy of the model for the single person and top k
    """
    correct_top1 = 0
    correct_topk = 0
    total = len(embeddings)

    person_ids = list(person_embeddings.keys())
    person_emb_matrix = np.array([person_embeddings[p] for p in person_ids])

    for test_embedding, true_label in zip(embeddings, labels):
        sims = cosine_similarity([test_embedding], person_emb_matrix)[0]  # calculate cosine similarity between test_embedding and every person embedding
        
        # top 1
        predicted_person_top1 = person_ids[np.argmax(sims)]
        if predicted_person_top1 == true_label:
            correct_top1 += 1
        
        # top k
        topk_indices = np.argsort(sims)[::-1][:k]
        if true_label in [person_ids[idx] for idx in topk_indices]:
            correct_topk += 1

    accuracy_top1 = correct_top1 / total
    accuracy_topk = correct_topk / total
    return accuracy_top1, accuracy_topk

def test(model, testloader, device, k):
    """
    Test model on a dataset and calculate accuracy.
    """
    model.eval()

    embeddings, labels = extract_embeddings(model, testloader, device) # extract embeddings for each image
    person_embeddings = calculate_person_embeddings(embeddings, labels) # average embedding for each person

    accuracy_top1, accuracy_topk = calculate_recognition_accuracy_topk(embeddings, labels, person_embeddings, k=k)

    return {
        "accuracy_top1": accuracy_top1,
        "accuracy_topk": accuracy_topk
    }