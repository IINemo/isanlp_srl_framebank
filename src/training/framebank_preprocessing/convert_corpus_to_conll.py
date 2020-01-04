# -*- coding: utf-8 -*-

import json
import pandas
import os
import StringIO
import logging

from collections import defaultdict


DATA_COLUMNS = ['form', 
                'lemma', 
                'xpos', 
                'upos',
                'feat',
                'head',
                'deprel',
                'rank',
                'sem',
                'sem2',
                'fillpred',
                'pred',
                'apred1',
                'rolepred1']


def convert_sent_to_csv(sent):
    result = pandas.DataFrame(columns = DATA_COLUMNS, 
                              index = range(1, len(sent) + 1),
                              dtype = unicode)
    
    for i in xrange(len(sent)):
        new_row = sent[i].copy()
        for col in DATA_COLUMNS:
            if col not in new_row:
                new_row[col] = u'_'
        
        if 'fillpred' in new_row:
            if isinstance(new_row['fillpred'], int):
                new_row['fillpred'] += 1
                
        result.iloc[i, :] = new_row
    
    strstrm = StringIO.StringIO()
    result.to_csv(strstrm, 
                  sep = '\t', 
                  encoding = 'utf8', 
                  header = False, 
                  quotechar = '@')
    
    return strstrm.getvalue()


def convert(input_file_path, output_dir_path, logger):
    logger.info('Loading corpus data...')
    with open(input_file_path, 'r') as f:
        json_data = json.load(f)
    logger.info('Done.')
    
    logger.info('Creating verb-example index...')
    verb_example_index = defaultdict(list)
    for ex_id, example in json_data.iteritems():
        for sent in example:
            for word in sent:
                if(word.get(u'rank', u'') == u'Предикат'):
                    if u'lemma' in word:
                        verb_example_index[word[u'lemma']].append(ex_id)
    logger.info('Done.')
    
    logger.info('Converting and saving...')
    for pred, ex_ids in verb_example_index.iteritems():
        output_pred_dir_path = os.path.join(output_dir_path, pred)
        if not os.path.exists(output_pred_dir_path):
            os.mkdir(output_pred_dir_path)
        
        for ex_id in ex_ids:
            example = json_data[ex_id]
            
            output_file_path = os.path.join(output_pred_dir_path, 
                                            str(ex_id) + u'.conll')
            with open(output_file_path, 'w') as f:
                for sent in example:
                    csv_sent = convert_sent_to_csv(sent)
                    csv_sent += '\n'
                    
                    f.write(csv_sent)
                    
    logger.info('Done.')


if __name__ == '__main__':
    def create_info_logger():
        import sys
        
        info_logger = logging.getLogger('converter_conll')
        info_logger.setLevel(logging.INFO)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s', 
                                      datefmt = '%Y-%m-%d %H:%M:%S')
        ch.setFormatter(formatter)
        info_logger.addHandler(ch)
        info_logger.propagate = False
        return info_logger
    
    import argparse

    parser = argparse.ArgumentParser(description = 
                                     'Converts annotated corpus from the JSON '
                                     'format into the CONLL tab-separator '
                                     'format. The format is a type of CSV with '
                                     '@ quoting symbol and sentences separated '
                                     'by double newline. Examples are grouped '
                                     'by a predicate.')
    parser.add_argument('--inputFile', '-I', 
                        required = True, 
                        help = 'Input file with annotations path (JSON).')
    parser.add_argument('--outputDir', '-O', 
                        required = True, 
                        help = 'The directory to store CONLL files.')

    args = parser.parse_args()

    convert(args.inputFile, args.outputDir, create_info_logger())
