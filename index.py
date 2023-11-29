import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import torch

import faiss
# or choose the GPU implementation, for example:
# conda install -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl
# conda install -c pytorch/label/nightly -c nvidia faiss-gpu=1.7.4


class FaissIndex:

    def __init__(self, hypo_to_premises, premise_pool, model):
        # construct N x dim premise table
        # construct hypo to which indices in table are matching
        self._index = []            # index table from premise_pool
        self._hypo_to_indices = {}  # map hypothesis to positive indices
        self._premise_pool = premise_pool
        self._model = model

        hypotheses = set()
        model.eval()

        # construct hypo_to_indices mapping
        for h in hypo_to_premises:
            if h in hypotheses:
                continue
            hypotheses.add(h)
            
            self._hypo_to_indices[h] = set()
            for p in hypo_to_premises[h]:
                # find index of p in premise_pool
                if p not in self._premise_pool:
                    self._premise_pool.append(p)

                idx = self._premise_pool.index(p)

                self._hypo_to_indices[h].add(idx)

        # construct index pool
        self._premise_embeddings = model.encode(self._premise_pool)
        self._embed_dim = self._premise_embeddings.shape[1]
        self._index = faiss.IndexFlatIP(self._embed_dim)
        faiss.normalize_L2(self._premise_embeddings)
        self._index.add(self._premise_embeddings)

    def train_cosine_sim(self, embeddings):
        faiss.normalize_L2(embeddings)
        self._index.add(embeddings)

    def retrieve_index(self, query, k=5):
        # find index of query in premise_pool
        p_emb = np.array([self._model.encode(query)])
        faiss.normalize_L2(p_emb)
        distances, ann = self._index.search(p_emb, k=k)
        idx = ann[0]
        #sent = self._premise_pool[idx]
        return idx#, sent


class Index:

    def __init__(self, hypo_to_premises, premise_pool, model):

        # construct N x dim premise table
        # construct hypo to which indices in table are matching
        self._index = []
        self._hypo_to_indices = {}
        self._premise_to_embedding = {}
        hypotheses = set()
        premises = []
        idx = 0
        model.eval()
        for h in hypo_to_premises:
            if h in hypotheses:
                continue
            hypotheses.add(h)
            self._hypo_to_indices[h] = set()
            for p in hypo_to_premises[h]:
                if p in premises:
                    self._hypo_to_indices[h].add(premises.index(p))
                    continue
                premises.append(p)
                with torch.no_grad():
                    embedding = model.encode(p)
                self._index.append(embedding)
                self._premise_to_embedding[p] = embedding
                self._hypo_to_indices[h].add(idx)
                idx += 1
        premises = set(premises)
        for p in premise_pool:
            if p in premises:
                continue
            premises.add(p)
            with torch.no_grad():
                embedding = model.encode(p)
            self._index.append(embedding)
            self._premise_to_embedding[p] = embedding
        self._index = np.stack(self._index)

    def get_gt_premises(self, query, one_hot=False):
        gt_premise_idx = np.array(list(self._hypo_to_indices[query]))
        if not one_hot:
            return gt_premise_idx
        gt_premise_idx_one_hot = np.zeros((self._index.shape[0]))
        gt_premise_idx_one_hot[gt_premise_idx] = 1
        return gt_premise_idx_one_hot

    def get_similarity_scores(self, query_embedding):
        # Returns matrix 1 x embedding_dim
        return cosine_similarity(np.expand_dims(query_embedding, axis=0), self._index)

    def get_embedding(self, premises):
        # Expects a batch of premises
        embeddings = []
        for p in premises:
            embeddings.append(self._premise_to_embedding[p])
        embeddings = np.stack(embeddings)
        return embeddings