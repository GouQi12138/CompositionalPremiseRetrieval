import json
import os
import sys
from pathlib import Path
import numpy as np


""" Create the ground truth evaluation dataset between hypothesis and premises """

# Options:
# 1) all
# 2) train
# 3) dev
# 4) test

task1_all_files = {'dev':"../data/task_1/dev.jsonl", 'train':"../data/task_1/train.jsonl", 'test':"../data/task_1/test.jsonl"}
task2_all_files = {
        'dev':"../data/task_2/dev.jsonl",
        'train':"../data/task_2/train.jsonl",
        'test':"../data/task_2/test.jsonl"
    }


# --------- Evaluation data loader ----------
def load_target_dict(target='test'):
    """ Evaluate on the model such that the input is the hypothesis and the label is the premise set

    Returns:
        dict: key - hypothesis, value - {set of premises}
    """
    options = ['train', 'dev', 'test']

    eval_dict = {}

    if target not in options:
        # all
        for file in task1_all_files.values():
            data = load_jsonl(file)
            parse_labels(data, eval_dict)
    else:
        # specific
        data = load_jsonl(task1_all_files[target])
        parse_labels(data, eval_dict)

    return eval_dict


def parse_labels(problem_list, result_dict, core_concept=False):
    """ Add {hypo:{prems}} to result_dict """
    for tree in problem_list:

        hypothesis = tree['hypothesis']
        premises = []

        if core_concept:
            for premise in tree['meta']['core_concepts']:
                premises.append(process_string(premise))

        else:
            for sentType, premise in tree['meta']['triples'].items():
                if sentType.startswith("sent"):
                    premises.append(process_string(premise))


        if hypothesis not in result_dict:
            result_dict[hypothesis] = set()

        # Uniquely index hypothesis in evaluation setting, append all premises
        result_dict[hypothesis].update(premises)


def load_pool_dict(target='test'):
    """ Small premise pool for retrieval problem """
    pool_dict = {}
    pos_data = load_jsonl(task1_all_files[target])
    neg_data = load_jsonl(task2_all_files[target])

    for pos_tree in pos_data:
        hypothesis = pos_tree['hypothesis']
        sentences = []

        for sentType, sent in pos_tree['meta']['triples'].items():
            if sentType.startswith("sent"):
                sentences.append(process_string(sent))

        if hypothesis not in pool_dict:
            pool_dict[hypothesis] = set()

        pool_dict[hypothesis].update(sentences)

    for neg_tree in neg_data:
        hypothesis = neg_tree['hypothesis']
        sentences = []

        for sentType, sent in neg_tree['meta']['triples'].items():
            if sentType.startswith("sent"):
                sentences.append(process_string(sent))

        if hypothesis not in pool_dict:
            raise Exception("Neg tree not in dict")

        for sentence in sentences:
            if len(pool_dict[hypothesis]) < 25:
                pool_dict[hypothesis].add(sentence)

    return pool_dict

# ----------------------------------------


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



if __name__ == "__main__":
    sentences = load_pool_dict()
    premises = load_target_dict('test')
    print(type(premises))
    print(len(premises))
    count = {}
    for key, value in premises.items():
        #print(key)
        #print(value)
        size = len(value)
        if size in count:
            count[size] += 1
        else:
            count[size] = 1
        #break
        if len(sentences[key]) > 25:
            print(len(sentences[key]))
            raise Exception("Prem pool larger than 25")
        for sent in value:
            if sent not in sentences[key]:
                raise Exception("Prem not in pool")
    print("Check complete")
    for i in range(20):
        if i in count:
            print(i, ":", count[i])
    print(count)


