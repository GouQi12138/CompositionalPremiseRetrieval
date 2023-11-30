import os
import argparse
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from info_nce import InfoNCE, info_nce

from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.evaluation import RerankingEvaluator
from sentence_transformers import losses

from dataloader.pairwise_dataloader import PairwiseDataset, TripletDataset, TripletEvalDataset
from dataloader.premise_pool_loader import load_premise_pool
from eval_pretrain_model import evaluate



def train_one_epoch(model, train_dataloader, optimizer, index, device):
    model.train()
    train_loss = 0
    print("Training...")
    for i, batch in enumerate(tqdm(train_dataloader)):
        query, premise, target = batch
        premise_embedding = torch.from_numpy(index.get_embedding(premise)).to(device)
        query_features = gen_query_features(query, model, device)

        optimizer.zero_grad()  # zero-out gradients after query_features computed so tokenizer is not updated

        outputs = model(query_features)["sentence_embedding"]
        target = target.to(device)
        loss = SCL(outputs, premise_embedding, target)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_loss /= i + 1
    return train_loss


def val_one_epoch(model, val_dataloader, index, device):
    model.eval()
    val_loss = 0
    print("Validating...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_dataloader)):
            query, premise, target = batch
            premise_embedding = torch.from_numpy(index.get_embedding(premise)).to(device)
            query_features = gen_query_features(query, model, device)
            outputs = model(query_features)["sentence_embedding"]
            target = target.to(device)
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
    print("Building index...")
    if args.debug:
        check_missing_premises(train_dataloader, val_dataloader)
    index = build_index(train_dataloader, val_dataloader, model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_val_loss = float("inf")
    best_epoch = 0

    print("Initial evaluation:")
    all_train_losses = [val_one_epoch(model, train_dataloader, index, device)]
    all_val_losses = [val_one_epoch(model, val_dataloader, index, device)]
    save_loss_curves(all_train_losses, all_val_losses, save_dir)

    for epoch in range(args.epochs):
        print("Epoch: {}/{}".format(epoch + 1, args.epochs))
        train_loss = train_one_epoch(model, train_dataloader, optimizer, index, device)
        val_loss = val_one_epoch(model, val_dataloader, index, device)
        best_val_loss, best_epoch = save_model_if_better(best_val_loss, val_loss, best_epoch, epoch, model, save_dir)
        all_train_losses.append(train_loss)
        all_val_losses.append(val_loss)
        save_loss_curves(all_train_losses, all_val_losses, save_dir)
    return model, best_epoch


def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Model to train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    query_model = SentenceTransformer(args.model).to(device)
    if args.debug:
        print(query_model)

    # Process data format
    train_dataset = TripletDataset(os.path.join(os.getcwd(), args.train_data))
    val_dataset = TripletEvalDataset(os.path.join(os.getcwd(), args.val_data))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    #val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Train
    #query_model, best_epoch = fine_tune_model(query_model, train_dataloader, val_dataloader, device, args.save_dir, args)

    # info NCE loss
    #loss = InfoNCE(negative_mode='paired')
    # Triplet loss
    loss = losses.TripletLoss(model=query_model, distance_metric=losses.TripletDistanceMetric.COSINE, triplet_margin=0.001)
    # eucl or cos

    evaluator = RerankingEvaluator(val_dataset)
    query_model.fit(train_objectives=[(train_dataloader, loss)],
                    evaluator=evaluator,
                    evaluation_steps=5000,
                    epochs=args.epochs,
                    #optimizer_params={'lr': args.learning_rate},
                    output_path=args.save_dir)


    # Test
    #query_model.load_state_dict(torch.load(os.path.join(args.save_dir, "model_{}.pt".format(best_epoch))))
    #premise_model = SentenceTransformer(args.model).to(device)
    query_model = SentenceTransformer(args.save_dir).to(device)
    evaluate(query_model, query_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all-mpnet-base-v2",
                        help="pre-trained model \
                            (best general purpose model: all-mpnet-base-v2 (https://huggingface.co/sentence-transformers/all-mpnet-base-v2); \
                            best semantic search model: multi-qa-mpnet-base-dot-v1 (https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1))")
    parser.add_argument("--train-data", type=str, default="data/processed_pairwise/pairwise_data_train_triplet_root_leaf_cross_join.tsv")
    parser.add_argument("--val-data", type=str, default="data/processed_pairwise/pairwise_data_dev_triplet_root_leaf_cross_join.tsv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--save-dir", type=str, default="./checkpoints/", help="directory to save best model weights and loss curves in")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    main(args)
