import json
import os
import sys
from pathlib import Path
import numpy as np


""" Load a premise pool as the truth knowledge base

Used for indexing (tf-idf, FAISS) IR
"""

# Options:
# 1) Premises used in *all* proof trees
# 2) All sentences from the worldtree corpus
# 3) * Premises + distractors used in all data


worldtree_file = "../data/worldtree_corpus_sentences_extended.json"
# task1 data provides the ground-truth used premises
task1_all_files = ["../data/task_1/dev.jsonl", "../data/task_1/train.jsonl", "../data/task_1/test.jsonl"]


# --------- Premise pool loader ----------
def load_premise_pool(source='worldtree'):
    """ Load a premise pool as the truth knowledge base
    Create the corpus using the optioned source

    Args:
        source : 1) usedPremises, 2) worldtree

    Returns:
        list: list of strings
    """

    premise_pool = []

    if source == 'usedPremises':
        used_premises = load_used_premises()
        premise_pool = list(used_premises.values())

    #elif source == 'something':
    #   premise_pool = [premise list]

    else:
        # default to worldtree corpus
        premise_dict = load_worldtree_corpus()
        premise_pool = list(premise_dict.values())

    return premise_pool
# ----------------------------------------



# jsonl file loader
def load_jsonl(file):
    data = []
    file_path = Path(__file__).parent / file
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_worldtree_corpus():
    """ Load the worldtree corpus

    Returns:
        dict: dict of strings indexed by unique id
    """
    worldtree_data = load_jsonl(worldtree_file)[0]
    return worldtree_data


def load_used_premises():
    """ Load the used premises from all proof trees

    Returns:
        dict: dict of strings indexed by unique id
    """
    premise_dict = {}

    for file in task1_all_files:

        all_data = load_jsonl(file)
        
        for tree in all_data:
            for sentType, premise in tree['meta']['worldtree_provenance'].items():
                if sentType.startswith("sent") and premise['uuid'] not in premise_dict:
                    premise_dict[premise['uuid']] = premise['original_text']

    return premise_dict



if __name__ == "__main__":
    premises = load_premise_pool()#("usedPremises")
    print(type(premises))
    print(len(premises))
    print(premises[0])
    print(premises[1])

