from itertools import product

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

    def retrieve_index(self, query, k=5, model=None):
        if model is None:
            model = self._model
        # find index of query in premise_pool
        p_emb = np.array([model.encode(query)])
        faiss.normalize_L2(p_emb)
        distances, ann = self._index.search(p_emb, k=k)
        idx = ann[0]
        #sent = self._premise_pool[idx]
        return idx#, sent


class Index:

    def __init__(self, hypo_to_premises, premise_pool, model):

        # construct N x D (embedding_dim) premise table
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
        # Returns matrix 1 x N
        return cosine_similarity(np.expand_dims(query_embedding, axis=0), self._index)

    def get_embedding(self, premises):
        # Expects a batch of premises
        embeddings = []
        for p in premises:
            embeddings.append(self._premise_to_embedding[p])
        embeddings = np.stack(embeddings)
        return embeddings

    def _gen_all_binary_combinations_of_premises(self):
        """
        Returns a binary table representing all possible ocmbinations of premises where each row is a possible combination.

        Ex: For a premise pool of size 3, returns
            [[0 0 0]
             [0 0 1]
             [0 1 0]
             [0 1 1]
             [1 0 0]
             [1 0 1]
             [1 1 0]
             [1 1 1]]
        """
        combinations = [i for i in product(range(2), repeat=self._index.shape[0])]
        return np.array(combinations)

    def get_relative_scores(self, query_embedding, solver="brute_force", debug=False):
        # Returns matrix 1 x N
        # Generate a relative score for every premise by solving the least squares problem between the query_embedding and the sum of arbitrary premise embeddings
        if solver != "brute_force":
            raise NotImplementedError("Only brute force solver is implemented")

        # Generate all possible combinations of premises
        combinations = self._gen_all_binary_combinations_of_premises()  # 2^N x N

        # Calculate all possible sums
        sums = combinations @ self._index  # 2^N x D

        # Calcualte all possible differences
        squared_diffs = ((np.expand_dims(query_embedding, axis=0) - sums)**2).sum(axis=1)  # 2^N
        if debug:
            print("Squared diffs:")
            print(squared_diffs)

        # Rank the differences from least to greatest
        sorted_idx = np.argsort(squared_diffs)

        # Iterate over the solutions from least difference to greatest difference until all premises have been iterated over
        # Rank premises from 1 to infinity, incrementing by 1
        premise_ranks = np.zeros(self._index.shape[0], dtype=int)  # keeps track of which premises have been returned
        rank = 1
        for i in sorted_idx:
            binary_chain = combinations[i]

            # In each solution, organize premises from largest norm to smallest
            premise_chain = self._index[binary_chain.astype(bool)]  # C (chain length) x D
            squared_norm = (premise_chain**2).sum(axis=1)
            sorted_chain_idx = np.argsort(squared_norm)[::-1]
            for j in sorted_chain_idx:
                if premise_ranks[j] < 1:
                    premise_ranks[j] = rank
                    rank += 1

            # Check if all premises have been assigned ranks
            if np.all(premise_ranks):
                break
        if debug:
            print("Premise ranks:")
            print(premise_ranks)

        # Convert ranks to similarity score
        similarity_scores = 1 - premise_ranks/rank
        similarity_scores = np.expand_dims(similarity_scores, axis=0)
        return similarity_scores