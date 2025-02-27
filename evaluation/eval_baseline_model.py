import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sklearn.metrics import average_precision_score, ndcg_score
from sentence_transformers import SentenceTransformer
import torch

from dataloader.premise_evaluation_loader import load_target_dict, load_pool_dict
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


def gen_labels_and_scores(hypo_to_premises, index, query_model, debug=False):
    gt_labels = []
    scores = []
    for i, query in enumerate(hypo_to_premises):
        gt_premise_indices = index.get_gt_premises(query, one_hot=True)
        gt_labels.append(gt_premise_indices)
        with torch.no_grad():
            query_embedding = query_model.encode(query)
        similarity_scores = index.get_similarity_scores(query_embedding)
        scores.append(similarity_scores)
        if debug:
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

def compute_precision_at_full_recall(hypo_to_premises, faissIndex, query_model=None):
    # Precision at full recall is the average of the precision values at each recall threshold
    # Recall threshold is the number of relevant documents
    # Precision is the number of relevant documents divided by the number of retrieved documents
    # Relevant documents are those that are in the ground truth
    # Retrieved documents are those that are in the top K samples
    precision_at_full_recall_list = []
    for h in hypo_to_premises:
        prem_label = faissIndex._hypo_to_indices[h]
        prem_pred = faissIndex.retrieve_index(h, k=len(faissIndex._premise_pool), model=query_model)

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

def compute_hit_at_k_v2(hypo_to_premises, index, scores, debug=False):
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
        n = min(50, len(query_score))
        top_ind = np.argpartition(query_score, -n)[-n:]
        sorted_top_ind = top_ind[np.argsort(query_score[top_ind])]
        num_matches_10 = 0
        num_matches_20 = 0
        num_matches_30 = 0
        num_matches_40 = 0
        num_matches_50 = 0
        for j in range(n):
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
        if debug:
            if i == 1:
                break
    hit_10 = sum(hit_10_list)#/sum(prem_count_list)
    hit_20 = sum(hit_20_list)#/sum(prem_count_list)
    hit_30 = sum(hit_30_list)#/sum(prem_count_list)
    hit_40 = sum(hit_40_list)#/sum(prem_count_list)
    hit_50 = sum(hit_50_list)#/sum(prem_count_list)
    return hit_10, hit_20, hit_30, hit_40, hit_50, sum(prem_count_list)


def evaluate_pretrained_model(query_model, index, hypo_to_premises, faissIndex, debug=False):

    query_model.eval()
    gt_labels, scores = gen_labels_and_scores(hypo_to_premises, index, query_model, debug=debug)

    prec_full_rec = compute_precision_at_full_recall(hypo_to_premises, faissIndex, query_model)

    map_ = average_precision_score(gt_labels, scores, average="samples")
    if debug:
        check_map(gt_labels, scores)

    ndcg = ndcg_score(gt_labels, scores)
    ndcg_10 = ndcg_score(gt_labels, scores, k=10)
    ndcg_20 = ndcg_score(gt_labels, scores, k=20)
    ndcg_30 = ndcg_score(gt_labels, scores, k=30)
    ndcg_40 = ndcg_score(gt_labels, scores, k=40)
    ndcg_50 = ndcg_score(gt_labels, scores, k=50)

    hit_10, hit_20, hit_30, hit_40, hit_50, count_sum = compute_hit_at_k_v2(hypo_to_premises, index, scores, debug=debug)

    return [prec_full_rec, map_, ndcg, ndcg_10, ndcg_20, ndcg_30, ndcg_40, ndcg_50, hit_10, hit_20, hit_30, hit_40, hit_50, count_sum]

"""

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

def compute_hit_at_k(labels, predictions, k=10):
    # in top-k results, calculate relevant items retrieved / total number of relevants
    hit = 0
    total = 0
    # totaled over all samples
    for label, prediction in zip(labels, predictions):
        # look in top-k predictions
        for i in range(min(k, len(prediction))):
            if prediction[i] in label:
                hit += 1
        total += len(label)
    return hit/total

def evaluate_metrics(labels, one_hot_labels, pred_scores):
    result = []

    # Rank the documents based on their similarity scores
    ranked_indices = []
    for scores in pred_scores:
        ranked_documents = [(score, idx) for score, idx in zip(scores, range(len(scores)))]
        ranked_documents = sorted(ranked_documents, reverse=True)

        ranked_index = []
        for score, idx in ranked_documents:
            ranked_index.append(idx)

        ranked_indices.append(ranked_index)

    # "Prec@Full Recall"
    prec_rec_list = []
    for label, ranked_index in zip(labels, ranked_indices):
        num_relevant_docs = len(label)
        num_retrieved_docs = k_at_recall(label, ranked_index)
        prec_rec_list.append(num_relevant_docs/num_retrieved_docs)
    precision_at_full_recall = sum(prec_rec_list)/len(prec_rec_list)
    result.append(precision_at_full_recall)

    # "MAP"
    #y_true = np.array([[0, 0, 1, 1],
    #                    [0, 1, 0, 0]])
    #y_scores = np.array([[0.1, 0.4, 0.35, 0.8],
    #                    [0.6, 0.9, 0.3, 0.1]])
    #result.append(average_precision_score(one_hot_labels, pred_scores, average='samples'))
    map_list = []
    for one_hot_label, pred_score in zip(one_hot_labels, pred_scores):
        map_list.append(average_precision_score(one_hot_label, pred_score, average='samples'))
    result.append(sum(map_list)/len(map_list))

    # "NDCG", "NDCG@10", "NDCG@20", "NDCG@30", "NDCG@40", "NDCG@50"
    #result.append(ndcg_score(one_hot_labels, pred_scores))
    ndcg_list = []
    for one_hot_label, pred_score in zip(one_hot_labels, pred_scores):
        ndcg_list.append(ndcg_score([one_hot_label], [pred_score]))
    result.append(sum(ndcg_list)/len(ndcg_list))
    for k in [10, 20, 30, 40, 50]:
        ndcg_list = []
        for one_hot_label, pred_score in zip(one_hot_labels, pred_scores):
            ndcg_list.append(ndcg_score([one_hot_label], [pred_score], k=k))
        result.append(sum(ndcg_list)/len(ndcg_list))
        #result.append(ndcg_score(one_hot_labels, pred_scores, k=k))

    # "Hit@10", "Hit@20", "Hit@30", "Hit@40", "Hit@50"
    for k in [10, 20, 30, 40, 50]:
        result.append(compute_hit_at_k(labels, ranked_indices, k=k))

    return result


def evaluate_model(model, full_corpus = True):
    # Evaluate Tf-idf and BM-25

    hypo_to_prem = load_target_dict("test")
    hypo_to_pool = load_pool_dict("test")
    premise_pool = load_premise_pool()

    # preprocessing
    for h in hypo_to_prem:
        for p in hypo_to_prem[h]:
            if p not in premise_pool:
                premise_pool.append(p)

    n = len(premise_pool)

    # build indexing
    index = FaissIndex(hypo_to_prem, premise_pool, model)

    label_indices = []
    one_hot_labels = []
    pred_scores = []

    for h in hypo_to_prem:

        # ------ limited pool ------
        if not full_corpus:
            premise_pool = list(hypo_to_pool[h])
            index = FaissIndex([{h:hypo_to_prem[h]}], premise_pool, model)
            n = len(premise_pool)
        # --------------------------

        label_index = []
        for p in hypo_to_prem[h]:
            label_index.append(premise_pool.index(p))
        one_hot = np.zeros(n, dtype=int)
        np.put(one_hot, label_index, 1)

        tfidf_score, ranked_index = tfidf_query(vectorizer, index, h)
        bm25_score = bm25_query(bm25, h)

        label_indices.append(label_index)
        one_hot_labels.append(one_hot)
        tfidf_scores.append(tfidf_score)
        bm25_scores.append(bm25_score)

    # run evaluations
    res = evaluate_metrics(label_indices, one_hot_labels, pred_scores)
    tab = PrettyTable()
    tab.field_names = ["Prec@Full Recall", "MAP", "NDCG", "NDCG@10", "NDCG@20", "NDCG@30", "NDCG@40", "NDCG@50", "Hit@10", "Hit@20", "Hit@30", "Hit@40", "Hit@50"]
    tab.add_row(["{0:0.3f}".format(i) for i in res])
    print(tab)

"""

def evaluate(premise_model, query_model, split="test", debug=False, full_corpus=True):
    hypo_to_premises = load_target_dict(split)
    hypo_to_pool = load_pool_dict(split)
    premise_pool = load_premise_pool()

    if full_corpus:
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print("Start Indexing...", current_time)
        index = Index(hypo_to_premises, premise_pool, premise_model)
        faissIndex = FaissIndex(hypo_to_premises, premise_pool, premise_model)
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print("Indexing Done", current_time)

    res = None
    count = 0
    # run evaluations
    for h in hypo_to_premises:
        hypo_to_prem = {h : hypo_to_premises[h]}
        # ------ limited pool ------
        if not full_corpus:
            premise_pool = list(hypo_to_pool[h])
            index = Index(hypo_to_prem, premise_pool, premise_model)
            faissIndex = FaissIndex(hypo_to_prem, premise_pool, premise_model)
        # --------------------------
        result = evaluate_pretrained_model(query_model, index, hypo_to_prem, faissIndex, debug=debug)
        if res is None:
            res = result
            count = 1
        else:
            for i in range(len(result)):
                res[i] += result[i]
            count += 1

    for i in range(8):
        res[i] /= count
    for i in range(8, 13):
        res[i] /= res[-1]

    res = res[:-1]

    tab = PrettyTable()
    tab.field_names = ["Prec@Full Recall", "MAP", "NDCG", "NDCG@10", "NDCG@20", "NDCG@30", "NDCG@40", "NDCG@50", "Hit@10", "Hit@20", "Hit@30", "Hit@40", "Hit@50"]
    tab.add_row(["{0:0.3f}".format(i) for i in res])
    print(tab)


def retrieve(premise_model):
    hypo_to_premises = load_target_dict("test")
    premise_pool = load_premise_pool()
    
    faissIndex = FaissIndex(hypo_to_premises, premise_pool, premise_model)

    count = 0
    for h in hypo_to_premises:
        true_premises = hypo_to_premises[h]
        pred_premises = faissIndex.retrieve_index(h, k=8000)

        print("\nHypothesis:", h)
        print("Ranking true premises:")
        """
        for i in true_premises:
            print(premise_pool.index(i), "\t", i)
        """
        print(len(true_premises))
        #print("Predicted premises:")
        for i in range(len(pred_premises)):
            if premise_pool[pred_premises[i]] in true_premises:
                # ranking and index
                print(i+1, "\t", pred_premises[i], "\t", premise_pool[pred_premises[i]])
        
        count += 1
        if count >= 20:
            break

def genL2report(model, fig_filename):
    hypo_to_premises = load_target_dict("test")
    premise_pool = load_premise_pool()

    # Encoding of premise pool
    prem_pool_emb = model.encode(premise_pool)
    prem_pool_norm = np.linalg.norm(prem_pool_emb, axis=1)

    hypo_list = []
    prem_list = []
    for h in hypo_to_premises:
        hypo_list.append(h)
        prem_list.extend(hypo_to_premises[h])

    # Encoding of hypothesis
    hypo_emb = model.encode(hypo_list)
    hypo_norm = np.linalg.norm(hypo_emb, axis=1)

    # Encoding of used premises
    prems_emb = model.encode(prem_list)
    prems_norm = np.linalg.norm(prems_emb, axis=1)


    print(prem_pool_norm.shape)
    print(hypo_norm.shape)
    print(prems_norm.shape)

    print(prem_pool_norm.mean())
    print(hypo_norm.mean())
    print(prems_norm.mean())
    print(hypo_norm[:6])

    bins = np.arange(0, 5, 0.01)

    # Generate histogram
    #https://stackoverflow.com/questions/26218704/matplotlib-histogram-with-collection-bin-for-high-values
    fig, axes = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(12, 5))
    plt.setp(axes, ylim=(0, 10))

    axes[0].hist(np.clip(prem_pool_norm, bins[0], bins[-1]), bins=bins, density=True, label='Premise Pool')
    axes[0].legend(loc='upper right')

    axes[1].hist(np.clip(hypo_norm, bins[0], bins[-1]), bins=bins, density=True, label='Hypothesis')
    axes[1].legend(loc='upper right')

    axes[2].hist(np.clip(prems_norm, bins[0], bins[-1]), bins=bins, density=True, label='Used Premises')
    axes[2].legend(loc='upper right')

    fig.suptitle("L2 Norm for embeddings on pretrained SBERT model")
    fig.supxlabel("L2 Norm")
    fig.supylabel("Frequency")
    fig.tight_layout()
    
    print("Saving figure...")
    plt.savefig(fig_filename)   #'l2_norm_finetune_adjacent.png' #finetune_root_leaf
    #plt.show()



def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #premise_model = SentenceTransformer(args.model).to(device)
    #./checkpoints/triplet_adjacent/
    query_model = SentenceTransformer(args.model).to(device)
    if args.model_path:
        print("Loading query model from checkpoint...")
        query_model = SentenceTransformer(args.model).to(device)
        query_model.load_state_dict(torch.load(args.model_path))
    evaluate(query_model, query_model, full_corpus=False)#, split=args.split, debug=args.debug)
    # retrieve(query_model)

    # remove the last Normalization layer
    #query_model = SentenceTransformer(modules=[query_model[0], query_model[1]]).to(device)

    # L2 norm of premise-pool, test data from 3 baseline models
    #genL2report(query_model, args.fig_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all-mpnet-base-v2",
                        help="pre-trained model \
                            (best general purpose model: all-mpnet-base-v2 (https://huggingface.co/sentence-transformers/all-mpnet-base-v2); \
                            best semantic search model: multi-qa-mpnet-base-dot-v1 (https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1))")
    parser.add_argument("--model-path", type=str, help="specify to evaluate a specific query model")
    parser.add_argument("--split", type=str, default="test", help="data split to evaluate on (train, dev, test)")
    parser.add_argument("--fig-filename", type=str)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    main(args)
