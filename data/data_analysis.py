import json
import os
import sys
from pathlib import Path
import numpy as np


# Analyze the files in entailment_trees_emnlp2021_data_v3

worldtree_file = "./worldtree_corpus_sentences_extended.json"

task3_all_file = "./task_3/all.jsonl"
task3_train_file = "./task_3/train.jsonl"
task3_dev_file = "./task_3/dev.jsonl"
task3_test_file = "./task_3/test.jsonl"

task1_dev_file = "./task_1/dev.jsonl"
task2_dev_file = "./task_2/dev.jsonl"


# load the jsonl file
def load_jsonl(file_path):
    data = []
    path = Path(__file__).parent / file_path
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


# print the data

worldtree_data = load_jsonl(worldtree_file)
print("Total sentences: ", len(worldtree_data[0]))

all_data = load_jsonl(task3_all_file)
train_data = load_jsonl(task3_train_file)
dev_data_task3 = load_jsonl(task3_dev_file)
test_data = load_jsonl(task3_test_file)

dev_data_task1 = load_jsonl(task1_dev_file)
dev_data_task2 = load_jsonl(task2_dev_file)


print("Total trees: :", len(all_data))
print("Total train trees: :", len(train_data))
print("Total dev trees: :", len(dev_data_task3))
print("Total test trees: :", len(test_data))



print("\n\nFormat for IR:")

print("\nTask 3 data\n")

print(dev_data_task3[0].keys())
# dict_keys(['id', 'context', 'question', 'answer', 'hypothesis', 'proof', 'meta'])
#print("id: ", dev_data_task3[0]['id'])
#print("context: ", dev_data_task3[0]['context'])
#print("question: ", dev_data_task3[0]['question'])
#print("answer: ", dev_data_task3[0]['answer'])
print("hypothesis: ", dev_data_task3[0]['hypothesis'])
#print("proof: ", dev_data_task3[0]['proof'])
print(dev_data_task3[0]['meta'].keys())
# dict_keys(['question_text', 'answer_text', 'triples', 'core_concepts', 'worldtree_provenance'])
print("question_text: ", dev_data_task3[0]['meta']['question_text'])
print("answer_text: ", dev_data_task3[0]['meta']['answer_text'])
print("triples: ", dev_data_task3[0]['meta']['triples'])
print("core_concepts: ", dev_data_task3[0]['meta']['core_concepts'])
print("worldtree_provenance: ", dev_data_task3[0]['meta']['worldtree_provenance'])



print("\n\nFormat for testing:")

print("\nTask 1 data\n")

print(dev_data_task1[0].keys())
# dict_keys(['id', 'context', 'question', 'answer', 'hypothesis', 'proof', 'full_text_proof', 'depth_of_proof', 'length_of_proof', 'meta'])
#print("id: ", dev_data_task1[0]['id'])
#print("context: ", dev_data_task1[0]['context'])
#print("question: ", dev_data_task1[0]['question'])
#print("answer: ", dev_data_task1[0]['answer'])
print("hypothesis: ", dev_data_task1[0]['hypothesis'])
print("proof: ", dev_data_task1[0]['proof'])
#print("full_text_proof: ", dev_data_task1[0]['full_text_proof'])
print("depth_of_proof: ", dev_data_task1[0]['depth_of_proof'])
print("length_of_proof: ", dev_data_task1[0]['length_of_proof'])
print(dev_data_task1[0]['meta'].keys())
# dict_keys(['question_text', 'answer_text', 'hypothesis_id', 'triples', 'distractors', 'distractors_relevance', 'intermediate_conclusions', 'core_concepts', 'step_proof', 'lisp_proof', 'polish_proof', 'worldtree_provenance', 'add_list', 'delete_list'])
#print("question_text: ", dev_data_task1[0]['meta']['question_text'])
#print("answer_text: ", dev_data_task1[0]['meta']['answer_text'])
print("hypothesis_id: ", dev_data_task1[0]['meta']['hypothesis_id'])
print("triples: ", dev_data_task1[0]['meta']['triples'])
print("distractors: ", dev_data_task1[0]['meta']['distractors'])
print("distractors_relevance: ", dev_data_task1[0]['meta']['distractors_relevance'])
print("intermediate_conclusions: ", dev_data_task1[0]['meta']['intermediate_conclusions'])
print("core_concepts: ", dev_data_task1[0]['meta']['core_concepts'])
print("step_proof: ", dev_data_task1[0]['meta']['step_proof'])
print("lisp_proof: ", dev_data_task1[0]['meta']['lisp_proof'])
print("polish_proof: ", dev_data_task1[0]['meta']['polish_proof'])
print("worldtree_provenance: ", dev_data_task1[0]['meta']['worldtree_provenance'])
print("add_list: ", dev_data_task1[0]['meta']['add_list'])
print("delete_list: ", dev_data_task1[0]['meta']['delete_list'])



print("\n\nFormat for training:")

print("\nTask 2 data\n")

print(dev_data_task2[0].keys())
# dict_keys(['id', 'context', 'question', 'answer', 'hypothesis', 'proof', 'full_text_proof', 'depth_of_proof', 'length_of_proof', 'meta'])
#print("id: ", dev_data_task2[0]['id'])
#print("context: ", dev_data_task2[0]['context'])
#print("question: ", dev_data_task2[0]['question'])
#print("answer: ", dev_data_task2[0]['answer'])
print("hypothesis: ", dev_data_task2[0]['hypothesis'])
print("proof: ", dev_data_task2[0]['proof'])
#print("full_text_proof: ", dev_data_task2[0]['full_text_proof'])
print("depth_of_proof: ", dev_data_task2[0]['depth_of_proof'])
print("length_of_proof: ", dev_data_task2[0]['length_of_proof'])
print(dev_data_task2[0]['meta'].keys())
# dict_keys(['question_text', 'answer_text', 'hypothesis_id', 'triples', 'distractors', 'distractors_relevance', 'intermediate_conclusions', 'core_concepts', 'step_proof', 'lisp_proof', 'polish_proof', 'worldtree_provenance', 'add_list', 'delete_list'])
#print("question_text: ", dev_data_task2[0]['meta']['question_text'])
#print("answer_text: ", dev_data_task2[0]['meta']['answer_text'])
print("hypothesis_id: ", dev_data_task2[0]['meta']['hypothesis_id'])
print("triples: ", dev_data_task2[0]['meta']['triples'])
print("distractors: ", dev_data_task2[0]['meta']['distractors'])
print("distractors_relevance: ", dev_data_task2[0]['meta']['distractors_relevance'])
print("intermediate_conclusions: ", dev_data_task2[0]['meta']['intermediate_conclusions'])
print("core_concepts: ", dev_data_task2[0]['meta']['core_concepts'])
print("step_proof: ", dev_data_task2[0]['meta']['step_proof'])
print("lisp_proof: ", dev_data_task2[0]['meta']['lisp_proof'])
print("polish_proof: ", dev_data_task2[0]['meta']['polish_proof'])
print("worldtree_provenance: ", dev_data_task2[0]['meta']['worldtree_provenance'])
print("add_list: ", dev_data_task2[0]['meta']['add_list'])
print("delete_list: ", dev_data_task2[0]['meta']['delete_list'])




