import json
import csv
from pathlib import Path

import pandas as pd

from torch.utils.data import Dataset
from sentence_transformers import InputExample


""" Dataloader for constrastive learning on pairwise examples.

1) Hypothesis-Premise pairs, + distractors
2) All adjacent pairs in the same proof tree, + distractors
3) Random positive pairs from the same proof tree, + distractors

* Distractors are sampled within task2's distractor data
"""

task2_all_files = {
        'dev':"../data/task_2/dev.jsonl",
        'train':"../data/task_2/train.jsonl",
        'test':"../data/task_2/test.jsonl"
    }


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

class PairwiseDataset(Dataset):
    """ Stores (hypothesis, +/-premise, 1/0) pairs """
    def __init__(self, triplet_file):
        # read file
        df = read_tsv(triplet_file)

        # check shape (N * 3)
        if len(df.shape)!=2 or df.shape[1]!=3:
            raise Exception("Format Error: tsv file has incorrect dimensions")

        # convert to list of pairs
        tuples = list(df.itertuples(index=False, name=None))
        pairs = [(t[0], t[1], 1) for t in tuples] + [(t[0], t[2], 0) for t in tuples]

        self.contrastive_pairs = pairs

    def __len__(self):
        return len(self.contrastive_pairs)

    def __getitem__(self, idx):
        return self.contrastive_pairs[idx]


class TripletDataset(Dataset):
    """ Stores triplets of (hypo, pos, neg) string tuples """
    def __init__(self, triplet_file):
        # read file
        df = read_tsv(triplet_file)

        # check shape (N * 3)
        if len(df.shape)!=2 or df.shape[1]!=3:
            raise Exception("Format Error: tsv file has incorrect dimensions")

        # convert to list of tuples
        tuples = list(df.itertuples(index=False, name=None))
        input_examples = []
        for (hyp, pos, neg) in tuples:
            input_examples.append(InputExample(texts=[ hyp, pos, neg ]))
        self.triplets = input_examples

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]


class TripletEvalDataset(Dataset):
    """ Stores dictionaries of (hypo, [pos], [neg]) string tuples """
    def __init__(self, triplet_file):
        # read file
        df = read_tsv(triplet_file)

        # check shape (N * 3)
        if len(df.shape)!=2 or df.shape[1]!=3:
            raise Exception("Format Error: tsv file has incorrect dimensions")

        # convert to list of tuples
        tuples = list(df.itertuples(index=False, name=None))
        dicts = []
        hypothesis = None
        positives = set()
        negatives = set()
        for (hyp, pos, neg) in tuples:
            if hypothesis is None:
                hypothesis = hyp
            if hyp!=hypothesis:
                # new hypo
                dicts.append({'query': hypothesis, 'positive': list(positives), 'negative': list(negatives)})
                hypothesis = hyp
                positives = set()
                negatives = set()

            positives.add(pos)
            negatives.add(neg)

        dicts.append({'query': hypothesis, 'positive': list(positives), 'negative': list(negatives)})

        self.triplets = dicts

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]



## ---------- Pairwise data preprocess ----------

def generate_pairwise_data(stage='train', format='triplet', positive='adjacent', negative='cross_join'):
    """ Create pairwise training set; save to tsv file """
    # stage: train, dev, test
    # positive: adjacent, root_leaf
    # negative: cross_join (from distractors)

    file = task2_all_files[stage]
    data = load_jsonl(file)

    result = []
    result_file_name = "../data/processed_pairwise/" + "pairwise_data_" + "_".join([stage, format, positive, negative]) + ".tsv"

    # create pairs for individual proof trees
    for tree in data:
        # ---- for each tree ----
        hypothesis = tree["hypothesis"]
        sentences = {"hypothesis":hypothesis}
        sentences.update(tree["meta"]["triples"])
        sentences.update(tree["meta"]["intermediate_conclusions"])
        proof = tree["proof"]
        distractor_idx = tree["meta"]["distractors"]

        if negative == 'cross_join':
            # --- negative : cross_join ---
            pos_list = []
            neg_list = []

            for dis in distractor_idx:
                neg_list.append(process_string(sentences[dis]))


            if positive == 'adjacent':
                # -- positive : adjacent nodes --
                steps = proof.split(";")
                for step in steps:
                    if not step or step.isspace():
                        continue
                    # stripe int text
                    step = step.split(":")[0]
                    advance = step.split("->")
                    node = advance[1].strip()
                    children = advance[0].split("&")
                    for child in children:
                        child = child.strip()
                        if not child:
                            continue
                        # construct node, child pair
                        pos_list.append([sentences[node], process_string(sentences[child])])


            elif positive == 'root_leaf':
                # -- positive : root-leaf pairs --
                tokens = proof.split()
                premise_idx = [token for token in tokens if token.startswith("sent")]

                premises = [sentences[idx] for idx in premise_idx]

                for premise in premises:
                    pos_list.append([hypothesis, process_string(premise)])


            # --- append cross-joined samples to result ---
            for pair in pos_list:
                for neg in neg_list:
                    result.append( pair + [neg] )


    # write list of lists(tuples) into tsv file
    write_tsv(result_file_name, result)


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


def process_string(sent):
    # to lower case
    # replace \\s+ with single space
    # add space in front of 's
    # replace ; with /
    sent = sent.lower()
    sent = sent.replace("'s", " 's")
    sent = sent.replace(";", " / ")
    sent = ' '.join(sent.split())
    return sent

# jsonl file loader
def load_jsonl(file):
    data = []
    file_path = Path(__file__).parent / file
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# tsv file operations
def write_tsv(file, data):
    file_path = Path(__file__).parent / file
    with open(file_path, 'w') as f:
        tsv_writer = csv.writer(f, delimiter='\t')
        tsv_writer.writerows(data)

def read_tsv(file):
    file_path = Path(__file__).parent / file
    df = pd.read_csv(file_path, delimiter='\t', header=None)
    return df



if __name__ == "__main__":
    generate_pairwise_data(stage="train", positive="adjacent")
    generate_pairwise_data(stage="dev", positive="adjacent")
    generate_pairwise_data(stage="test", positive="adjacent")
    generate_pairwise_data(stage="train", positive="root_leaf")
    generate_pairwise_data(stage="dev", positive="root_leaf")
    generate_pairwise_data(stage="test", positive="root_leaf")
    #data = TripletDataset("../data/processed_pairwise/pairwise_data_train_triplet_root_leaf_cross_join.tsv")

