import os
import argparse
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from info_nce import InfoNCE, info_nce

from sentence_transformers import SentenceTransformer, InputExample, models
from sentence_transformers.evaluation import RerankingEvaluator
from sentence_transformers import losses

from dataloader.pairwise_dataloader import PairwiseDataset, TripletDataset, TripletEvalDataset
from dataloader.compositional_dataloader import CompositionalDataset, CompositionalEvalDataset
from dataloader.premise_pool_loader import load_premise_pool
from evaluation.eval_baseline_model import evaluate
from util.compositional_loss_evaluator import CompositionalLossEvaluator
from util.loss_functions import CompositionalLoss, ContrastiveRegLoss



# ---------- Framework ready ----------

def gen_projection_model(model, device):
    fc1 = models.Dense(in_features=768, out_features=512, activation_function=torch.nn.ReLU())
    fc2 = models.Dense(in_features=512, out_features=256, activation_function=torch.nn.ReLU())
    return SentenceTransformer(modules=[model[0], model[1], fc1, fc2]).to(device)


def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(args.model).to(device)

    # Projection model
    model = gen_projection_model(model, device)

    if args.debug:
        print(model)


    # Process data format
    #train_dataset = CompositionalDataset(os.path.join(os.getcwd(), args.train_data), contrastive=False) TODO: replace with train, parallel dataloader, batch size
    train_dataset = CompositionalDataset(os.path.join(os.getcwd(), args.val_data), contrastive=False)
    reg_dataset = CompositionalDataset(contrastive=True, dict=train_dataset)
    val_dataset = CompositionalDataset(os.path.join(os.getcwd(), args.val_data), contrastive=True)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    reg_dataloader = DataLoader(reg_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    #val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)


    # info NCE loss
    #loss = InfoNCE(negative_mode='paired')
    # Triplet loss
    #loss = losses.TripletLoss(model=query_model, distance_metric=losses.TripletDistanceMetric.COSINE, triplet_margin=0.001)
    
    train_loss = CompositionalLoss(model=model, batch=args.batch_size)
    reg_loss = ContrastiveRegLoss(model=model, batch=args.batch_size)

    #val_loss = ContrastiveRegLoss(model=model, batch=args.batch_size, triplet_margin=0)

    # Evaluator
    # DP retrieval is too expensive; consider:
    # validation loss
    # regularization loss
    # DP retrieval on task2
    evaluator = CompositionalLossEvaluator(val_dataset)
    

    # Train
    model.fit(train_objectives=[(train_dataloader, train_loss), (reg_dataloader, reg_loss)],
                    evaluator=evaluator,
                    evaluation_steps=500000,
                    epochs=args.epochs,
                    #optimizer_params={'lr': args.learning_rate},
                    output_path=args.save_dir)
    

    raise Exception("Finish training.")


    # Test
    query_model = SentenceTransformer(args.save_dir).to(device)
    evaluateCompositional(query_model, query_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all-mpnet-base-v2",
                        help="pre-trained model, ./checkpoints/triplet_adjacent or root_leaf \
                            (best general purpose model: all-mpnet-base-v2 (https://huggingface.co/sentence-transformers/all-mpnet-base-v2); \
                            best semantic search model: multi-qa-mpnet-base-dot-v1 (https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1))")
    parser.add_argument("--train-data", type=str, default="data/processed_compositional/compositional_data_train_dict_adjacent_cross_join")
    parser.add_argument("--val-data", type=str, default="data/processed_compositional/compositional_data_dev_dict_adjacent_cross_join")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-6) # 2e-5
    parser.add_argument("--save-dir", type=str, default="./checkpoints/compositional/", help="directory to save best model weights and loss curves in")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    main(args)
