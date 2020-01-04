import os

import sys
sys.path.append('../')
sys.path.append('../libs/')
sys.path.append('../libs/pylingtools/')

import isanlp_srl_framebank as isanlp

from isanlp_srl_framebank import convert_corpus_to_brat

import json
import argparse

parser = argparse.ArgumentParser(description='Clear annotated corpus from verbs with not enough examples')
parser.add_argument("--annotated-corpus", nargs="?", dest="source", default="../data/preprocessed_framebank/annotated_corpus.json", help="annotated framebank file in .json format")
parser.add_argument("--output", nargs="?", dest="output", default="../data/cleared_corpus.json", help="path to output cleared file in .json format")
parser.add_argument("-n", "--min-n", nargs="?", dest="min_n", default=10, help="minimal number of examples per verb")

args = parser.parse_args()

print("1. Loading annotated corpus....")
with open(args.source, 'r') as f:
    data = json.load(f)
    
print("..Done!")
    
print("Number of examples: ", len(data))

min_n_examples = args.min_n

print("2. Clearing verbs....")
verb_index = convert_corpus_to_brat.create_verb_example_index(data)
print('Original number of verbs: ', len(verb_index))

stat = sorted([(verb, len(examples)) for verb, examples in verb_index.items()], 
              key = lambda x: x[1], reverse=True)


verbs_to_keep = [verb for verb, count in stat if count >= min_n_examples]

print("..Done!")
print('Number of left verbs: ', len(verbs_to_keep))

examples = list()

print("3. Reindexing examples....")

for verb in verbs_to_keep:
    indexes = verb_index[verb]
    
    for ind in indexes:
        examples.append((ind, data[ind]))

print("..Done!")
print('Number of framebank examples left: ', len(examples))
print("4. Saving cleared corpus....")
with open(args.output, 'w') as f:
    json.dump(examples, f)
    
print("..Done!")
