import ujson as json
import csv
import random
from pathlib import Path

import pandas as pd

from torch.utils.data import Dataset
from sentence_transformers import InputExample



""" Dataloader for constrastive learning on compositional data.

1) Node-[Path] pairs, + distractors

* Distractors are created by substituting correct examples with task2's distractor data
"""

task2_all_files = {
        'dev':"../data/task_2/dev.jsonl",
        'train':"../data/task_2/train.jsonl",
        'test':"../data/task_2/test.jsonl"
    }



# ---------- Compositional dataloader ----------

class CompositionalDataset(Dataset):
    """ Stores dictionaries of {hypo, [true path], [false path]} """
    def __init__(self, dict_file=None, contrastive=False, dict=None):
        self.contrastive = contrastive

        if dict_file is None:
            # directly load dict
            self.strings = dict.strings
            self.triplets = dict.triplets
        else:
            # read file
            print("loading tsv")
            strings = read_tsv(dict_file + ".tsv")
            print("loading jsonl")
            data = load_jsonl(dict_file + ".jsonl")
            print("Finish loading", dict_file)

            # format string pool
            strings = strings[0].tolist()

            # format indices to dictionaries
            dicts = []
            for item in data:
                assert len(item) == 3
                """
                hyp = item[0]
                pos = item[1]
                neg = item[2]
                if contrastive:
                    dicts.append({'query': hyp, 'positive': pos, 'negative': neg})
                else:
                    dicts.append({'query': hyp, 'positive': pos})
                """

            self.strings = strings
            self.triplets = data    # dicts

        print("Number of strings:", len(self.strings))
        print("Number of triplets:", len(self.triplets))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        result = {}
        result['query'] = self.strings[triplet[0]]
        result['positive'] = [self.strings[idx] for idx in triplet[1]]
        # sample an index for distractor
        if self.contrastive:
            #negs = result['positive'].copy()
            #negs[random.choice(range(len(negs)))] = self.strings[random.choice(triplet[2])]
            #result['negative'] = negs
            result['negative'] = [self.strings[idx] for idx in triplet[2]]

        return result


class CompositionalEvalDataset(Dataset):
    """ Stores dictionaries of (hypo, [true path], [false path]) string tuples """
    def __init__(self, triplet_file, contrastive=False):
        self.contrastive = contrastive
        
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
                if contrastive:
                    dicts.append({'query': hypothesis, 'positive': list(positives), 'negative': list(negatives)})
                else:
                    dicts.append({'query': hypothesis, 'positive': list(positives)})
                #dicts.append(InputExample(texts=[ hypothesis, list(positives), list(negatives) ]))
                hypothesis = hyp
                positives = set()
                negatives = set()

            positives.add(pos)
            negatives.add(neg)

        if contrastive:
            dicts.append({'query': hypothesis, 'positive': list(positives), 'negative': list(negatives)})
        else:
            dicts.append({'query': hypothesis, 'positive': list(positives)})
        #dicts.append(InputExample(texts=[ hypothesis, list(positives), list(negatives) ]))

        self.triplets = dicts

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]



## ---------- Compositional data preprocess ----------

# In a premise tree, each non-leaf node is a hypothesis; the paths going down are its premises
# TODO: tune how to sample negative examples / how many distractors to use
# TODO: add multiple ways of combining paths to form positive examples - DONE

def generate_compositional_data(stage='train', format='dict', positive='adjacent', negative='cross_join'):
    """ Create pairwise training set; save to jsonl file """
    # stage: train, dev, test
    # positive: adjacent, root_leaf
    # negative: cross_join (from distractors)

    file = task2_all_files[stage]
    data = load_jsonl(file)

    #result = []
    strings = []
    appears = {}
    indices = []    # list of [hyp, [pos], [neg]]
    # result = strings + indices
    #indices.append(strings)

    result_file_name = "../data/compositional/" + "compositional_data_" + "_".join([stage, format, positive, negative]) # + ".jsonl"


    # create premises for individual proof trees
    print("Processing:", result_file_name)
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
            # Tree structure
            # Guaranteed bottom-up
            proof_trees = {}

            pos_list = []
            neg_list = []

            # Process distractors
            for dis in distractor_idx:
                neg_list.append(process_string(sentences[dis]))


            # Process positive tree
            if positive == 'adjacent':
                # -- positive : adjacent nodes --
                steps = proof.split(";")    # (a) ; (b) ; (c)
                for step in steps:
                    # x & y -> z
                    if not step or step.isspace():
                        continue
                    # stripe int text
                    step = step.split(":")[0]
                    advance = step.split("->")
                    node = advance[1].strip()           # z
                    children = advance[0].split("&")    # x, y
                    # iterate each child premise
                    for child in children:
                        child = child.strip()
                        if not child:
                            continue

                        # Add node, children to tree
                        # This creates all possible sub-trees
                        if node not in proof_trees:
                            proof_trees[node] = []
                        proof_trees[node].append(child)
        
            """
            elif positive == 'root_leaf':
                # -- positive : root-leaf pairs --
                tokens = proof.split()
                premise_idx = [token for token in tokens if token.startswith("sent")]
                premises = [sentences[idx] for idx in premise_idx]
                for premise in premises:
                    pos_list.append([hypothesis, process_string(premise)])
            """


            # Process positive paths, add to pos_list
            for node in proof_trees:
                # add individual proof trees; node is hypothesis
                search_positive_paths(proof_trees, node, proof_trees[node], sentences, pos_list)


            # --- append cross-joined samples to result ---
            for pair in pos_list:
                hyp_sample = pair[0]
                pos_sample = pair[1]
                for neg in neg_list:
                    # neg cross positive
                    # replace each pos element by each neg element once
                    for i in range(len(pos_sample)):
                        neg_sample = pos_sample.copy()
                        neg_sample[i] = neg
                        sample = [hyp_sample, pos_sample, neg_sample]
                        add_to_list(sample,    strings, appears, indices)
                """
                positives = pair[1]
                # every pos item replaced once
                for i in range(len(positives)):
                    negatives = positives.copy()
                    negatives[i] = neg_list[ i % len(neg_list) ]
                    #result.append( pair + [negatives] )
                    add_to_list(pair+[negatives],   strings, appears, indices)
                # randomly replace with distractors
                for neg in neg_list:
                    negatives = positives.copy()
                    # sample an index
                    i = random.choice(range(len(positives)))
                    negatives[i] = neg
                    #result.append( pair + [negatives] )
                    add_to_list(pair+[negatives],   strings, appears, indices)
                """


    # write list of lists(tuples) into jsonl file
    # process to save space by indexing
    #result += indices
    write_tsv(result_file_name + ".tsv", strings)
    write_jsonl(result_file_name + ".jsonl", indices)



def search_positive_paths(proof_trees, node, path, sentences, pos_list):
    # Add current path
    pos_list.append( [ sentences[node], [process_string(sentences[child]) if child.startswith('sent') else sentences[child] for child in path] ] )
    return
    # DFS to get all paths
    for i in range(len(path)):
        curr = path[i]
        if curr in proof_trees:
            # expand path by curr
            new_path = path[:i] + proof_trees[curr] + path[i+1:]
            search_positive_paths(proof_trees, node, new_path, sentences, pos_list)
            

def add_to_list(triplet, strings, appears, indices):
    assert len(triplet) == 3
    hyp = triplet[0]
    pos = triplet[1]
    neg = triplet[2]

    index = []

    # append hypo index
    index.append( get_index(hyp, strings, appears) )

    # positives
    positives = []
    for item in pos:
        positives.append( get_index(item, strings, appears) )
    index.append(positives)

    # negatives
    negatives = []
    for item in neg:
        negatives.append( get_index(item, strings, appears) )
    index.append(negatives)

    # append triplet
    indices.append(index)

def get_index(item, strings, appears):
    if item in appears:
        # item in set
        return appears[item]    # strings.index(item)
    else:
        # item not in set
        appears[item] = len(strings)
        strings.append(item)
        return len(strings)-1



# ----------------- Utility functions -----------------

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
    #prev_neg = []
    file_path = Path(__file__).parent / file
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            #if item[2] != prev_neg:
            #    prev_neg = item[2]
            #item[2] = prev_neg
            data.append(item)
    return data

def write_jsonl(file, data):
    file_path = Path(__file__).parent / file
    with open(file_path, 'w') as f:
        for line in data:
            json.dump(line, f)
            f.write('\n')

# tsv file operations
def write_tsv(file, data):
    file_path = Path(__file__).parent / file
    with open(file_path, 'w') as f:
        tsv_writer = csv.writer(f, delimiter='\n')
        tsv_writer.writerow(data)

def read_tsv(file):
    file_path = Path(__file__).parent / file
    df = pd.read_csv(file_path, delimiter='\t', header=None)
    return df



if __name__ == "__main__":
    generate_compositional_data(stage="train", positive="adjacent")
    generate_compositional_data(stage="dev", positive="adjacent")
    generate_compositional_data(stage="test", positive="adjacent")
    #generate_compositional_data(stage="train", positive="root_leaf")
    #generate_compositional_data(stage="dev", positive="root_leaf")
    #generate_compositional_data(stage="test", positive="root_leaf")
    print("Generation finished")
    data = CompositionalDataset("../data/compositional/compositional_data_train_dict_adjacent_cross_join")

