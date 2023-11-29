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
    similarity_scores = cosine_similarity(query_tfidf, index)

    # Rank the documents based on their similarity scores
    ranked_documents = [(score, idx) for score, idx in zip(similarity_scores[0], range(index.shape[0]))]
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

    return ranked_idx


def bm25_query(bm25, query):
    token_query = query.split()
    scores = bm25.get_score(token_query)
    # .. same as tfidf..


# Evaluation

def evaluate_metrics(label, pred):


def main():
    # Evaluate Tf-idf and BM-25

    hypo_to_prem = load_target_dict("test")
    premise_pool = load_premise_pool()

    # preprocessing
    for h in hypo_to_prem:
        for p in hypo_to_prem[h]:
            if p not in premise_pool:
                premise_pool.append(p)

    # build indexing
    vectorizer, index = tfidf_index(premise_pool)

    for h in hypo_to_prem:
        label_index = []
        for p in hypo_to_prem[h]:
            label_index.append(premise_pool.index(p))
        ranked_index = tfidf_query(vectorizer, index, h, premise_pool)


    # run evaluations

    tfidf_res = ()

    bm25_res = ()



    tab1 = PrettyTable()
    tab1.field_names = ["Prec@Full Recall", "MAP", "NDCG", "NDCG@10", "NDCG@20", "NDCG@30", "NDCG@40", "NDCG@50", "Hit@10", "Hit@20", "Hit@30", "Hit@40", "Hit@50"]
    tab1.add_row(["{0:0.3f}".format(i) for i in tfidf_res])

    tab2 = PrettyTable()
    tab2.field_names = ["Prec@Full Recall", "MAP", "NDCG", "NDCG@10", "NDCG@20", "NDCG@30", "NDCG@40", "NDCG@50", "Hit@10", "Hit@20", "Hit@30", "Hit@40", "Hit@50"]
    tab2.add_row(["{0:0.3f}".format(i) for i in bm25_res])

    print(tab1)
    print()
    print(tab2)



if __name__ == "__main__":
    main()

