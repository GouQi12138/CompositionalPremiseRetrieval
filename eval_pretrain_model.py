import argparse
import time

import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import average_precision_score, ndcg_score
from sentence_transformers import SentenceTransformer
import torch

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


def gen_labels_and_scores(hypo_to_premises, index, query_model):
    gt_labels = []
    scores = []
    for i, query in enumerate(hypo_to_premises):
        gt_premise_indices = index.get_gt_premises(query, one_hot=True)
        gt_labels.append(gt_premise_indices)
        with torch.no_grad():
            query_embedding = query_model.encode(query)
        similarity_scores = index.get_similarity_scores(query_embedding)
        scores.append(similarity_scores)
        if args.debug:
            if i == 1:
                break
    gt_labels = np.stack(gt_labels)
    scores = np.concatenate(scores)
    return gt_labels, scores


### EVALUATION FUNCTIONS ###


def k_at_recall(premises, predictions, recall_level=1):
    # Recall is the number of relevant documents retrieved divided by the total number of relevant documents
    # Relevant documents are those that are in the ground truth
    # Retrieved documents are those that are in the top K samples
    items_left = len(premises)
    k = 0
    for idx in predictions:
        k += 1
        if idx in premises:
            items_left -= 1
        if items_left == 0:
            return k
        
    raise Exeption("Not all premises are retrieved")
    return -1

def compute_precision_at_full_recall(hypo_to_premises, faissIndex):
    # Precision at full recall is the average of the precision values at each recall threshold
    # Recall threshold is the number of relevant documents
    # Precision is the number of relevant documents divided by the number of retrieved documents
    # Relevant documents are those that are in the ground truth
    # Retrieved documents are those that are in the top K samples
    precision_at_full_recall_list = []
    for h in hypo_to_premises:
        prem_label = faissIndex._hypo_to_indices[h]
        prem_pred = faissIndex.retrieve_index(h, k=len(faissIndex._premise_pool))

        num_relevant_docs = len(hypo_to_premises[h])
        num_retrieved_docs = k_at_recall(prem_label, prem_pred)
        precision_at_full_recall_list.append(num_relevant_docs/num_retrieved_docs)
    precision_at_full_recall = sum(precision_at_full_recall_list)/len(precision_at_full_recall_list)
    return precision_at_full_recall


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


def evaluate_pretrained_model(query_model, index, hypo_to_premises, faissIndex):

    query_model.eval()
    gt_labels, scores = gen_labels_and_scores(hypo_to_premises, index, query_model)

    prec_full_rec = compute_precision_at_full_recall(hypo_to_premises, faissIndex)

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

    return prec_full_rec, map_, ndcg, ndcg_10, ndcg_20, ndcg_30, ndcg_40, ndcg_50, hit_10, hit_20, hit_30, hit_40, hit_50


def evaluate(premise_model, query_model):
    hypo_to_premises = load_target_dict("test")
    premise_pool = load_premise_pool()

    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("Start Indexing...", current_time)
    index = Index(hypo_to_premises, premise_pool, premise_model)
    faissIndex = FaissIndex(hypo_to_premises, premise_pool, premise_model)
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("Indexing Done", current_time)
    
    res = evaluate_pretrained_model(query_model, index, hypo_to_premises, faissIndex)
    tab = PrettyTable()
    tab.field_names = ["Prec@Full Recall", "MAP", "NDCG", "NDCG@10", "NDCG@20", "NDCG@30", "NDCG@40", "NDCG@50", "Hit@10", "Hit@20", "Hit@30", "Hit@40", "Hit@50"]
    tab.add_row(["{0:0.3f}".format(i) for i in res])
    print(tab)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(args.model).to(device)
    evaluate(model, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all-mpnet-base-v2",
                        help="pre-trained model \
                            (best general purpose model: all-mpnet-base-v2 (https://huggingface.co/sentence-transformers/all-mpnet-base-v2); \
                            best semantic search model: multi-qa-mpnet-base-dot-v1 (https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1))")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    main(args)
