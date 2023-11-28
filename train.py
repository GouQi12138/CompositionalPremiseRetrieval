import argparse
import time

import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import average_precision_score, ndcg_score
from sentence_transformers import SentenceTransformer
import torch
from torch.utils.data import DataLoader

from dataloader.pairwise_dataloader import PairwiseDataset
from dataloader.premise_evaluation_loader import load_target_dict
from dataloader.premise_pool_loader import load_premise_pool
from index import *
import time


### DEBUG FUNCTIONS ###


def check_map(gt_labels, scores):
    print("GT labels:")
    print(gt_labels)
    print("Scores:")
    print(scores)
    print("Scores of gt_labels:")
    print(scores[0][np.nonzero(gt_labels[0].astype(int))])
    print(scores[1][np.nonzero(gt_labels[1].astype(int))])
    print("Ranked scores")
    print(np.sort(scores)[:,-40:])


### HELPER FUNCTIONS ###


def gen_labels_and_scores(hypo_to_premises, index, model):
    gt_labels = []
    scores = []
    for i, query in enumerate(hypo_to_premises):
        gt_premise_indices = index.get_gt_premises(query, one_hot=True)
        gt_labels.append(gt_premise_indices)
        with torch.no_grad():
            query_embedding = model.encode(query)
        similarity_scores = index.get_similarity_scores(query_embedding)
        scores.append(similarity_scores)
        if args.debug:
            if i == 1:
                break
    gt_labels = np.stack(gt_labels)
    scores = np.concatenate(scores)
    return gt_labels, scores


### EVALUATION FUNCTIONS ###


def k_at_recall(premises, index, scores, recall_level=1):
    # Recall is the number of relevant documents retrieved divided by the total number of relevant documents
    # Relevant documents are those that are in the ground truth
    # Retrieved documents are those that are in the top K samples
    k_at_recall_list = []
    for i, query in enumerate(hypo_to_premises):
        gt_premise_indices = index.get_gt_premises(query)
        query_score = scores[i]
        top_ind = np.argpartition(query_score, -50)[-50:]
        sorted_top_ind = top_ind[np.argsort(query_score[top_ind])]
        num_relevant_docs = np.count_nonzero(gt_premise_indices)
        num_retrieved_docs = 0
        for j in range(50):
            if sorted_top_ind[j] in gt_premise_indices:
                num_retrieved_docs += 1
            if num_retrieved_docs/num_relevant_docs >= recall_level:
                k_at_recall_list.append(j+1)
                break
    k_at_recall = sum(k_at_recall_list)/len(k_at_recall_list)
    return k_at_re


def compute_precision_at_full_recall(hypo_to_premises, gt_labels, scores):
    # Precision at full recall is the average of the precision values at each recall threshold
    # Recall threshold is the number of relevant documents
    # Precision is the number of relevant documents divided by the number of retrieved documents
    # Relevant documents are those that are in the ground truth
    # Retrieved documents are those that are in the top K samples
    precision_at_full_recall_list = []
    for i in range(gt_labels.shape[0]):
        num_relevant_docs = np.count_nonzero(gt_labels[i].astype(int))
        num_retrieved_docs = np.count_nonzero(scores[i])
        precision_at_full_recall_list.append(num_relevant_docs/num_retrieved_docs)
    precision_at_full_recall = sum(precision_at_full_recall_list)/len(precision_at_full_recall_list)
    return precision_at_full_recall
    # NOT DONE or tested


def compute_hit_at_k(hypo_to_premises, index, scores):
    # In top K samples, count how many are correct premises, divide by total number of correct premises
    hit_10_list = []
    hit_20_list = []
    hit_30_list = []
    hit_40_list = []
    hit_50_list = []
    for i, query in enumerate(hypo_to_premises):
        gt_premise_indices = index.get_gt_premises(query)
        query_score = scores[i]
        top_ind = np.argpartition(query_score, -50)[-50:]
        sorted_top_ind = top_ind[np.argsort(query_score[top_ind])]
        num_matches_10 = 0
        num_matches_20 = 0
        num_matches_30 = 0
        num_matches_40 = 0
        num_matches_50 = 0
        for j in range(50):
            if sorted_top_ind[j] in gt_premise_indices:
                num_matches_50 += 1
                if j > 9:
                    num_matches_40 += 1
                if j > 19:
                    num_matches_30 += 1
                if j > 29:
                    num_matches_20 += 1
                if j > 39:
                    num_matches_10 += 1
        hit_10_list.append(num_matches_10/len(gt_premise_indices))
        hit_20_list.append(num_matches_20/len(gt_premise_indices))
        hit_30_list.append(num_matches_30/len(gt_premise_indices))
        hit_40_list.append(num_matches_40/len(gt_premise_indices))
        hit_50_list.append(num_matches_50/len(gt_premise_indices))
        if args.debug:
            if i == 1:
                break
    hit_10 = sum(hit_10_list)/len(hit_10_list)
    hit_20 = sum(hit_20_list)/len(hit_20_list)
    hit_30 = sum(hit_30_list)/len(hit_30_list)
    hit_40 = sum(hit_40_list)/len(hit_40_list)
    hit_50 = sum(hit_50_list)/len(hit_50_list)
    return hit_10, hit_20, hit_30, hit_40, hit_50

def compute_hit_at_k_v2(hypo_to_premises, index, scores):
    # In top K samples, count how many are correct premises, divide by total number of correct premises
    hit_10_list = []
    hit_20_list = []
    hit_30_list = []
    hit_40_list = []
    hit_50_list = []
    prem_count_list = []
    for i, query in enumerate(hypo_to_premises):
        gt_premise_indices = index.get_gt_premises(query)
        query_score = scores[i]
        top_ind = np.argpartition(query_score, -50)[-50:]
        sorted_top_ind = top_ind[np.argsort(query_score[top_ind])]
        num_matches_10 = 0
        num_matches_20 = 0
        num_matches_30 = 0
        num_matches_40 = 0
        num_matches_50 = 0
        for j in range(50):
            if sorted_top_ind[j] in gt_premise_indices:
                num_matches_50 += 1
                if j > 9:
                    num_matches_40 += 1
                if j > 19:
                    num_matches_30 += 1
                if j > 29:
                    num_matches_20 += 1
                if j > 39:
                    num_matches_10 += 1
        hit_10_list.append(num_matches_10)
        hit_20_list.append(num_matches_20)
        hit_30_list.append(num_matches_30)
        hit_40_list.append(num_matches_40)
        hit_50_list.append(num_matches_50)
        prem_count_list.append(len(gt_premise_indices))
        if args.debug:
            if i == 1:
                break
    hit_10 = sum(hit_10_list)/sum(prem_count_list)
    hit_20 = sum(hit_20_list)/sum(prem_count_list)
    hit_30 = sum(hit_30_list)/sum(prem_count_list)
    hit_40 = sum(hit_40_list)/sum(prem_count_list)
    hit_50 = sum(hit_50_list)/sum(prem_count_list)
    return hit_10, hit_20, hit_30, hit_40, hit_50


def evaluate_pretrained_model(model, index, hypo_to_premises):

    model.eval()
    gt_labels, scores = gen_labels_and_scores(hypo_to_premises, index, model)

    map_ = average_precision_score(gt_labels, scores, average="samples")
    if args.debug:
        check_map(gt_labels, scores)

    ndcg = ndcg_score(gt_labels, scores)
    ndcg_10 = ndcg_score(gt_labels, scores, k=10)
    ndcg_20 = ndcg_score(gt_labels, scores, k=20)
    ndcg_30 = ndcg_score(gt_labels, scores, k=30)
    ndcg_40 = ndcg_score(gt_labels, scores, k=40)
    ndcg_50 = ndcg_score(gt_labels, scores, k=50)

    hit_10, hit_20, hit_30, hit_40, hit_50 = compute_hit_at_k_v2(hypo_to_premises, index, scores)

    return map_, ndcg, ndcg_10, ndcg_20, ndcg_30, ndcg_40, ndcg_50, hit_10, hit_20, hit_30, hit_40, hit_50


def train_one_epoch(model, train_dataloader, optimizer):
    model.train()
    train_loss = 0
    for i, batch in enumerate(train_dataloader):
        query, premise, target = batch
        optimizer.zero_grad()
        outputs = model(query)
        # use index to get premise embedding and use that with outputs embedding to compute loss along with target
        loss = SCL()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= i + 1
    return train_loss


def val_one_epoch(model, val_dataloader):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            query, premise, target = batch
            outputs = model(query)
            # use index to get premise embedding and use that with outputs embedding to compute loss along with target
            loss = SCL()
            val_loss += loss.item()
    val_loss /= i + 1
    return val_loss


def save_model_if_better(best_val_loss, val_loss, model, save_dir):
    if val_loss > best_val_loss:
        return best_val_loss
    torch.save(model.state_dict(), save_dir)
    return val_loss


def fine_tune_model(model, train_dataloader, val_dataloader, args):
    if not args.save_dir:
        save_dir = args.model
    # freeze half of the model's weights
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_val_loss = float("inf")
    for epoch in tqdm(range(args.epochs)):
        train_loss = train_one_epoch(model, train_dataloader, optimizer)
        val_loss = val_one_epoch(model, val_dataloader)
        best_val_loss = save_model_if_better(best_val_loss, val_loss, model, save_dir)
        save_loss_curves()
    return


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(args.model).to(device)
    train_dataset = PairwiseDataset(args.train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = PairwiseDataset(args.val_data)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    model = fine_tune_model(model, train_dataloader, val_dataloader, args)
    model.load_state_dict(torch.load(path_to_best_weights))
    evaluate(model)


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
    parser.add_argument("--save-dir", type=str, help="directory to save best model weights in (default is what `--model` is set as)")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    main(args)