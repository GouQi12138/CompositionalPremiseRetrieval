import torch
from torch.utils.data import Dataset, DataLoader


""" Dataloader for constrastive learning on pairwise examples.

1) Hypothesis-Premise pairs, + distractors
2) All adjacent pairs in the same proof tree, + distractors
3) Random positive pairs from the same proof tree, + distractors

* Distractors are sampled within task2's distractor data
"""

class TripletDataset(Dataset):
    """ Stores triplets of (hypo, pos, neg) string tuples """
    def __init__(self, triplet_file):
        # read json

        self.triplets = [list of tuples]

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]


"""

class PairwiseDataset(Dataset):
    def __init__(self, premise_trees, hypotheses):
        self.premise_trees = premise_trees
        self.hypotheses = hypotheses

    def __len__(self):
        return len(self.premise_trees)

    def __getitem__(self, idx):
        premise_tree = self.premise_trees[idx]
        hypothesis = self.hypotheses[idx]
        return premise_tree, hypothesis

premise_trees = [...]  # List of premise trees
hypotheses = [...]  # List of hypotheses

dataset = PairwiseDataset(premise_trees, hypotheses)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
"""


## ---------- Pairwise data preprocess ----------

def generate_pairwise_data(stage='train', format='triplet' positive='adjacent', negative='cross_join')

# TODO:
# Taks2 data
# Options: train, dev, test
# Format: triplet(node, pos, neg)
# Postive: root-leaf, root-any, adjacent, random pair
# Negative:
#   cross_join(pair every positive pair with every negative distractor)
#   hypothesis-select from distractor
#   hypothesis-all distractors
#   pair with upper node only
#   pair with any node
# Reference paper: adjacent positve + cross_join negative; because also test on adjacent pairs, not hypo-prem
# Negative premise from all sentences - active entailment




