import argparse
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from torch.utils.data import DataLoader

from dataloader.pairwise_dataloader import PairwiseDataset
from dataloader.premise_pool_loader import load_premise_pool
from eval_pretrain_model import evaluate
from index import Index


def SCL(x1, x2, label, margin=0.1):
    # Supervised Contrastive Loss from https://gist.github.com/kongzii/0a108b115179cc17d58c158a94465a3c
    # Change to using cosine distance
    # Assume label is 1 for positive pairs and 0 for negative pairs
    dist = torch.nn.functional.cosine_similarity(x1, x2)
    loss = (label) * torch.pow(dist, 2) \
        + (1 - label) * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
    loss = torch.mean(loss)
    return loss


def train_one_epoch(model, train_dataloader, optimizer, index, device):
    model.train()
    train_loss = 0
    for i, batch in enumerate(train_dataloader):
        query, premise, target = batch
        premise_embedding = torch.from_numpy(index.get_embedding(premise)).to(device)
        optimizer.zero_grad()
        outputs = model(query)["sentence_embedding"]
        loss = SCL(outputs, premise_embedding, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= i + 1
    return train_loss


def val_one_epoch(model, val_dataloader, index, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            query, premise, target = batch
            premise_embedding = torch.from_numpy(index.get_embedding(premise)).to(device)
            outputs = model(query)["sentence_embedding"]
            loss = SCL(outputs, premise_embedding, target)
            val_loss += loss.item()
    val_loss /= i + 1
    return val_loss


def save_loss_curves(train_losses, val_losses, save_dir):
    plt.clf()
    plt.plot(np.arange(len(train_losses)), train_losses, label='Train Loss')
    plt.plot(np.arange(len(val_losses)), val_losses, label='Validation Loss')
    plt.legend()
    plt.title("Training and Validation Loss")
    file_name = "train_and_val_loss.png"
    plt.savefig(os.path.join(save_dir, file_name))


def save_model_if_better(best_val_loss, val_loss, best_epoch, epoch, model, save_dir):
    if val_loss > best_val_loss:
        return best_val_loss, best_epoch
    torch.save(model.state_dict(), os.path.join(save_dir, "model_{}.pt".format(epoch)))
    return val_loss, epoch


def fine_tune_model(model, train_dataloader, val_dataloader, device, save_dir, args):
    premise_pool = load_premise_pool()
    index = Index({}, premise_pool, model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_val_loss = float("inf")
    best_epoch = 0
    all_train_losses = [val_one_epoch(model, train_dataloader, index, device)]
    all_val_losses = [val_one_epoch(model, val_dataloader, index, device)]
    save_loss_curves(all_train_losses, all_val_losses, save_dir)
    for epoch in tqdm(range(args.epochs)):
        train_loss = train_one_epoch(model, train_dataloader, optimizer, index, device)
        val_loss = val_one_epoch(model, val_dataloader, index, device)
        best_val_loss, best_epoch = save_model_if_better(best_val_loss, val_loss, best_epoch, epoch, model, save_dir)
        all_train_losses.append(train_loss)
        all_val_losses.append(val_loss)
        save_loss_curves(all_train_losses, all_val_losses, save_dir)
    return model, best_epoch


def main(args):
    if not args.save_dir:
        save_dir = args.model
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(args.model).to(device)
    if args.debug:
        print(model)
    train_dataset = PairwiseDataset(args.train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = PairwiseDataset(args.val_data)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    model, best_epoch = fine_tune_model(model, train_dataloader, val_dataloader, device, save_dir, args)

    model.load_state_dict(torch.load(os.path.join(save_dir, "model_{}.pt".format(best_epoch))))
    premise_model = SentenceTransformer(args.model).to(device)
    evaluate(premise_model, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all-mpnet-base-v2",
                        help="pre-trained model \
                            (best general purpose model: all-mpnet-base-v2 (https://huggingface.co/sentence-transformers/all-mpnet-base-v2); \
                            best semantic search model: multi-qa-mpnet-base-dot-v1 (https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1))")
    parser.add_argument("--train-data", type=str, default="data/processed_pairwise/pairwise_data_train_triplet_root_leaf_cross_join.tsv")
    parser.add_argument("--val-data", type=str, default="data/processed_pairwise/pairwise_data_dev_triplet_root_leaf_cross_join.tsv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--save-dir", type=str, help="directory to save best model weights in and loss curves (default is what `--model` is set as)")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    main(args)