import sys
import time

sys.path.insert(0, '..')

import numpy as np

from rank_bm25 import BM25Okapi

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, ndcg_score
from sklearn.metrics.pairwise import cosine_similarity

from prettytable import PrettyTable

from dataloader.premise_evaluation_loader import load_target_dict
from dataloader.premise_pool_loader import load_premise_pool


# Build Indexing

def tfidf_index(premise_pool):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_index = tfidf_vectorizer.fit_transform(premise_pool)
    return tfidf_vectorizer, tfidf_index


def bm25_index(premise_pool):
    token_corpus = [doc.split() for doc in premise_pool]
    bm25 = BM25Okapi(token_corpus)
    return bm25


# Retreival system

def tfidf_query(vectorizer, index, query):
    # Compute the TF-IDF scores for the query
    query_tfidf = vectorizer.transform([query])

    # Compute the cosine similarity between the query and the documents
    similarity_scores = cosine_similarity(query_tfidf, index)[0]

    # Rank the documents based on their similarity scores
    ranked_documents = [(score, idx) for score, idx in zip(similarity_scores, range(index.shape[0]))]
    ranked_documents = sorted(ranked_documents, reverse=True)

    """
    # Print the ranked documents
    i = 0
    for score, document in ranked_documents:
        print(f"Score: {score:.2f} Document: {document}, {premise_pool[document]}")
        i += 1
        if i > 10:
            raise Exception()
    """
    ranked_index = []
    for score, idx in ranked_documents:
        ranked_index.append(idx)

    return similarity_scores, ranked_index


def bm25_query(bm25, query):
    token_query = query.split()
    scores = bm25.get_scores(token_query)
    return scores


# Evaluation

def calculate_map(ground_truth, retrieved_items):
    """
    Calculate the Mean Average Precision (MAP)

    Parameters:
    ground_truth (list): List of ground truth items
    retrieved_items (list): List of retrieved items

    Returns:
    float: The calculated MAP
    """
    precision_at_k = []
    relevant_items = 0

    for i, item in enumerate(retrieved_items):
        if item in ground_truth:
            relevant_items += 1
            precision_at_k.append(relevant_items / (i + 1))

    if precision_at_k:
        return sum(precision_at_k) / len(ground_truth)
    else:
        return 0.0

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
        for i in range(k):
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
    result.append(average_precision_score(one_hot_labels, pred_scores, average='samples'))

    # "NDCG", "NDCG@10", "NDCG@20", "NDCG@30", "NDCG@40", "NDCG@50"
    result.append(ndcg_score(one_hot_labels, pred_scores))
    for k in [10, 20, 30, 40, 50]:
        result.append(ndcg_score(one_hot_labels, pred_scores, k=k))

    # "Hit@10", "Hit@20", "Hit@30", "Hit@40", "Hit@50"
    for k in [10, 20, 30, 40, 50]:
        result.append(compute_hit_at_k(labels, ranked_indices, k=k))

    return result


def main():
    # Evaluate Tf-idf and BM-25

    hypo_to_prem = load_target_dict("test")
    premise_pool = load_premise_pool()

    # preprocessing
    for h in hypo_to_prem:
        for p in hypo_to_prem[h]:
            if p not in premise_pool:
                premise_pool.append(p)

    n = len(premise_pool)

    # build indexing
    vectorizer, index = tfidf_index(premise_pool)

    bm25 = bm25_index(premise_pool)

    label_indices = []
    one_hot_labels = []
    tfidf_scores = []
    bm25_scores = []

    for h in hypo_to_prem:
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

    tfidf_res = evaluate_metrics(label_indices, one_hot_labels, tfidf_scores)

    bm25_res = evaluate_metrics(label_indices, one_hot_labels, bm25_scores)


    tab1 = PrettyTable()
    tab1.field_names = ["Prec@Full Recall", "MAP", "NDCG", "NDCG@10", "NDCG@20", "NDCG@30", "NDCG@40", "NDCG@50", "Hit@10", "Hit@20", "Hit@30", "Hit@40", "Hit@50"]
    tab1.add_row(["{0:0.3f}".format(i) for i in tfidf_res])

    tab2 = PrettyTable()
    tab2.field_names = ["Prec@Full Recall", "MAP", "NDCG", "NDCG@10", "NDCG@20", "NDCG@30", "NDCG@40", "NDCG@50", "Hit@10", "Hit@20", "Hit@30", "Hit@40", "Hit@50"]
    tab2.add_row(["{0:0.3f}".format(i) for i in bm25_res])

    print()
    print("Tf-idf")
    print(tab1)
    print()
    print("BM25")
    print(tab2)



if __name__ == "__main__":
    main()

