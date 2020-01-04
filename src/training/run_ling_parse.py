import os
import sys
sys.path.append('../')

import argparse
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

from isanlp_srl_framebank.convert_corpus_to_brat import make_text, create_verb_example_index


class UdpipePipeline(object):

    def __init__(self, basic_processor, udpipe_processor):
        self.ppl = WrapperMultiProcessDocument([
                    PipelineCommon([
                        (
                            ProcessorRemote(basic_processor[0], basic_processor[1], 'default'),
                            ['text'],
                            {
                                'sentences' : 'sentences', 
                                'tokens' : 'tokens',
                                'postag' : 'mystem_postags',
                                'lemma' : 'lemma'
                            }
                        ),
                        (
                            ProcessorRemote(udpipe_processor[0], udpipe_processor[1], '0'), 
                            ['tokens', 'sentences'], 
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

        

    def __call__(self, texts):
        return self.ppl(texts)


def split_and_process(data, piplene_fn, n_splits=5):
    data_pts = split_equally(data, n_splits)

    result_pts = []

    for data_pt in data_pts:
        texts = [make_text(example, 0)[0] for (ex_id, example) in data_pt]
        result_pt = ppl(texts)
        result_pts.append(result_pt)

    results = []
    for res in result_pts:
        results += res

    return results

def fix_results(results, data):
    ling_annots = [(data[i][0], e) for (i, e) in enumerate(results)]

    for ex_id, ling_annot in ling_annots:
        ling_annot['morph'] = []
        for i in range(len(ling_annot['lemma'])):
            sent = ling_annot['postag'][i]
            ling_annot['morph'].append([])
            for j in range(len(sent)):
                ling_annot['morph'][i].append(ling_annot['postag'][i][j])
                ling_annot['postag'][i][j] = ling_annot['morph'][i][j].get('fPOS', '')

    return ling_annots


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process frambank .json file to obtain linguistic data')
    parser.add_argument("--cleared-corpus", nargs="?", dest="source", default="../data/cleared_corpus.json", help="preprocessed framebank file in .json format")
    parser.add_argument("--output", nargs="?", dest="output", default="../data/results_final_fixed.pckl", help="path to output .pckl file")
    parser.add_argument('--isanlp_proc')
    parser.add_argument('--udpipe_proc')

    args = parser.parse_args()
    input_file = args.source
    output_file = args.output
    isanlp_proc = args.isanlp_proc.split(':')
    udpipe_proc = args.udpipe_proc.split(':')

    ppl = UdpipePipeline(basic_processor=isanlp_proc, udpipe_processor=udpipe_proc)

    print("1.Reading data....")
    with open(input_file, 'r') as f:
        data = json.load(f)
    print("..Done!")    
    print('Number of examples: ', len(data))

    print("2.Processing....")
    results = split_and_process(data, ppl, 5)
    print("..Done!")
    
    print("3.Final processing...")
    ling_annotations = fix_results(results, data)
    print("..Done!")

    print("4.Saving results...")
    with open(args.output, 'wb') as f:
        pickle.dump(ling_annotations, f)
    print("..Done!")
