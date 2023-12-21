from itertools import product

# import miosqp
#from miosqp.solver import MIOSQP
import numpy as np
import scipy.sparse as spa
from sklearn.metrics.pairwise import cosine_similarity
from sympy.utilities.iterables import multiset_permutations
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

    def __init__(self, hypo_to_premises, premise_pool, model, max_chain_length=None, debug=False):
        """
        max_chain_length is the maximum length of premise chains this index will return. If not specified, the maximum chain
        length is equal to the total number of premises indexed.
        """

        self._construct_index(hypo_to_premises, premise_pool, model)
        self._construct_combinations(max_chain_length=max_chain_length, debug=debug)
        #self._construct_solver()

    def _construct_index(self, hypo_to_premises, premise_pool, model):
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

    def _construct_combinations(self, max_chain_length=None, debug=False):
        """
        Returns a binary table representing all possible combinations of premises where each row is a possible combination.
        Specify max_chain_length to limit the total number of premises in each combination (i.e. the sum of each row must
        be less than or equal to max_chain_length).

        Ex: For a premise pool of size 3 with max_chain_length of 2, returns
            [[0 0 1]
             [0 1 0]
             [0 1 1]
             [1 0 0]
             [1 0 1]
             [1 1 0]].
            It excludes the combination [1 1 1] (and [0 0 0]).
        """
        num_premises = self._index.shape[0]
        if not max_chain_length:
            max_chain_length = num_premises
        combinations = []
        for n in range(1, max_chain_length+1):
            binary_elements = [1] * n + [0] * (num_premises - n)
            for perm in multiset_permutations(binary_elements):
                combinations.append(perm)
        self._combinations = np.array(combinations)

    def _construct_solver(self):
        """
        minimize    0.5 x' P x + q' x

        subject to  l <= A x <= u
                    x[i] in Z for i in i_idx
                    i_l[i] <= x[i] <= i_u[i] for i in i_idx

        where       i_idx is a vector of indices of which variables are integer
                    i_l is the lower bound on integer variables
                    i_u is the upper bound on integer variables

        Formulation in code below comes from solving least squares between a query embedding, q, and all of the premise embeddings.
        """
        self._miqp_solver = MIOSQP()
        self.P = spa.csc_matrix(self._index @ self._index.T)
        dummy_q = np.zeros(self._index.shape[0])  # this will be updated at solving time

        # No constraints
        A = np.zeros((1, dummy_q.shape[0]))
        l = -1
        u = 1

        # Constrain every variable to be binary (0 or 1)
        i_idx = np.arange(0, self._index.shape[0], dtype=int)
        i_l = np.zeros(self._index.shape[0])
        i_u = np.ones(self._index.shape[0])

        miosqp_settings = {
                            # integer feasibility tolerance
                            'eps_int_feas': 1e-03,
                            # maximum number of iterations
                            'max_iter_bb': 1000,
                            # tree exploration rule
                            #   [0] depth first
                            #   [1] two-phase: depth first until first incumbent and then  best bound
                            'tree_explor_rule': 1,
                            # branching rule
                            #   [0] max fractional part
                            'branching_rule': 0,
                            'verbose': False,
                            'print_interval': 1}
        osqp_settings = {'eps_abs': 1e-03,
                         'eps_rel': 1e-03,
                         'eps_prim_inf': 1e-04,
                         'verbose': False}

        self._miqp_solver.setup(self.P, dummy_q, A, l, u, i_idx, i_l, i_u, miosqp_settings, osqp_settings)

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

    def _brute_force_solve(self, query_embedding, debug=False):

        # Generate all possible combinations of premises
        #print("Generate combination")
        #combinations = self._gen_all_binary_combinations_of_premises()  # 2^N x N
        # combinations = self._combinations

        # Calculate all possible sums
        #print("Calculate sum")
        sums = self._combinations @ self._index  # 2^N x D

        # Calcualte all possible differences
        #print("Calculate difference")
        squared_diffs = ((np.expand_dims(query_embedding, axis=0) - sums)**2).sum(axis=1)  # 2^N
        if debug:
            print("Squared diffs:")
            print(squared_diffs)

        return squared_diffs

    def _gen_similarity_scores_from_chains(self, combinations, scores, debug=False):

        # Rank the differences from least to greatest
        #print("start sorting")
        sorted_idx = np.argsort(scores)
        #print("finish sorting")

        # Pre-sort all premises by their squared norm in descending order
        squared_norm = (self._index**2).sum(axis=1)
        sorted_indices = np.argsort(squared_norm)[::-1]

        # Iterate over the solutions from least difference to greatest difference until all premises have been iterated over
        # Rank premises from 1 to infinity, incrementing by 1
        premise_ranks = np.zeros(self._index.shape[0], dtype=int)  # keeps track of which premises have been returned
        rank = 1
        for i in sorted_idx:
            binary_chain = combinations[i]

            # In each solution, organize premises from largest norm to smallest
            sorted_premise_chain = binary_chain[sorted_indices]
            premise_indices = sorted_indices[sorted_premise_chain.astype(bool)]  # C (chain length)
            for j in premise_indices:
                if premise_ranks[j] < 1:
                    premise_ranks[j] = rank
                    rank += 1

            # Check if all premises have been assigned ranks
            if np.all(premise_ranks):
                break
        if debug:
            print("Premise ranks:")
            print(premise_ranks)

        assert np.all(premise_ranks), "Not all premises were ranked. Results would be incorrect"

        # Convert ranks to similarity score
        similarity_scores = 1 - premise_ranks/rank
        similarity_scores = np.expand_dims(similarity_scores, axis=0)

        return similarity_scores

    def get_relative_scores(self, query_embedding, solver="bf", debug=False):
        # Returns matrix 1 x N
        # Generate a relative score for every premise by solving the least squares problem between the query_embedding and the sum of arbitrary premise embeddings

        if solver == "bf":
            combinations = self._combinations
            scores = self._brute_force_solve(query_embedding, debug=debug)
        elif solver == "bb":
            # XXX: Give a guess
            self._miqp_solver.update_vectors(q=-self._index@query_embedding)
            if debug:
                print("Solving...")
            results = self._miqp_solver.solve()  # returns a list of solutiosn along with their coreresponding scores
            combinations = results.all_x
            scores = results.all_scores
            rounded_combinations = []
            unique_scores = []
            for c in range(len(combinations)):
                rounded_c = (np.rint(combinations[c])).astype(int)
                already_have = False
                for r in rounded_combinations:
                    if np.array_equal(rounded_c, r):
                        already_have = True
                if not already_have:
                    rounded_combinations.append(rounded_c)
                    unique_scores.append(scores[c])

            # Compute distance for any premise not in a chain
            all_chains = np.stack(rounded_combinations)
            premsises_excluded = np.where(~all_chains.any(axis=0))[0]
            for p in premsises_excluded:
                x = np.zeros(self._index.shape[0], dtype=int)
                x[p] = 1
                rounded_combinations.append(x)
                unique_scores.append(0.5 * np.dot(x, self.P.dot(x)) + np.dot(-self._index@query_embedding, x))

            combinations = rounded_combinations
            scores = unique_scores

        else:
            raise NotImplementedError("Only brute force (bf) and branch-and-bound (bb) solvers are implemented")

        similarity_scores = self._gen_similarity_scores_from_chains(combinations, scores, debug=debug)

        return similarity_scores
