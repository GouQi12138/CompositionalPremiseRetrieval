import torch
from torch.utils.data import Dataset, DataLoader


""" Dataloader for constrastive learning on pairwise examples.

1) Hypothesis - Premise pairs, + distractors
2) All adjacent pairs in the same proof tree, + distractors
3) Random positive pairs from the same proof tree, + distractors

* Distractors are sampled within task2's distractor data
"""


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
