import os
import sys
import argparse

#sys.path.append('/notebook/framebank_parser/roleinduction/ri_tools/src/liash')
sys.path.append('../libs/pylingtools/')
sys.path.append('../libs/framebank_preprocessing/')
sys.path.append('../libs/')
import os
import time
import pickle
import isanlp
import json

from tqdm import tqdm

from pprint import pprint as pretty_

from isanlp.processor_remote import ProcessorRemote
from isanlp.ru.converter_mystem_to_ud import ConverterMystemToUd
from isanlp import PipelineCommon
from isanlp.ru.processor_mystem import ProcessorMystem
from isanlp.ru.processor_tokenizer_ru import ProcessorTokenizerRu
from isanlp.processor_sentence_splitter import ProcessorSentenceSplitter
from isanlp.wrapper_multi_process_document import WrapperMultiProcessDocument
from isanlp.wrapper_multi_process_document import split_equally

from convert_corpus_to_brat import make_text, create_verb_example_index

parser = argparse.ArgumentParser(description='Process frambank .json file to obtain linguistic data')
parser.add_argument("--cleared-corpus", nargs="?", dest="source", default="../data/cleared_corpus.json", help="preprocessed framebank file in .json format")
parser.add_argument("--output", nargs="?", dest="output", default="../data/results_final_fixed.pckl", help="path to output .pckl file")

args = parser.parse_args()


ppl = WrapperMultiProcessDocument([
    PipelineCommon([
        (
            ProcessorRemote('vmh2.isa.ru', 4466, 'default'),
            ['text'],
            {
                'sentences' : 'sentences', 
                 'tokens' : 'tokens',
                 'postag' : 'mystem_postags',
                 'lemma' : 'lemma'
            }
        ),
        (
            ProcessorRemote('exn40.isa.ru', 3334, 'default'), 
            ['text'], 
            {
                'syntax_dep_tree' : 'syntax_dep_tree'
            }
        ),
        (
            ConverterMystemToUd(),
            ['mystem_postags'],
            {
                'morph' : 'postag',
            }
        )
    ])
])

print("1. Reading data....")
input_data_path = args.source

with open(input_data_path, 'r') as f:
    data = json.load(f)
    
print("..Done!")    
print('Number of examples: ', len(data))

print("2.Processing....")
data_pts = split_equally(data, 5)

result_pts = []

for data_pt in data_pts:
    texts = [make_text(example, 0)[0] for (ex_id, example) in data_pt]
    result_pt = ppl(texts)
    result_pts.append(result_pt)
    
print("..Done!")

print("3.Final processing...")
final_res = []
for res in result_pts:
    final_res += res

ling_annots_to_fix = [(data[i][0], e) for (i, e) in enumerate(final_res)]

for ex_id, ling_annot in ling_annots_to_fix:
    ling_annot['morph'] = []
    for i in range(len(ling_annot['lemma'])):
        sent = ling_annot['postag'][i]
        ling_annot['morph'].append([])
        for j in range(len(sent)):
            ling_annot['morph'][i].append(ling_annot['postag'][i][j])
            ling_annot['postag'][i][j] = ling_annot['morph'][i][j].get('fPOS', '')

print("..Done!")

print("4.Saving results...")
with open(args.output, 'wb') as f:
    pickle.dump(ling_annots_to_fix, f)

print("..Done!")