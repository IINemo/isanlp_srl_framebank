# -*- coding: utf-8 -*-

import json
import pandas
import logging
import codecs


def prepare_annot_word(wd):
    
    def remove_quotes(string):
        quotes = '"«»,. \t' + u"'"
        return string.strip(quotes)
    
    def get_main_word(string):
        string = string.strip()
        
        bad = ' / '
        n = string.find(bad)
        if n == -1:
            bad = ' '
            n = string.find(bad)
    
        result = string
        if n != -1:
            result = string[n + len(bad) :].strip()
            
        return result
    
    result = wd
    result = get_main_word(result)
    result = remove_quotes(result)
    result = result.strip()
    result = result.lower()
    
    return result


def prepare_text_word(wd):
    return wd.lower()


def compare_tokens(annot_token, text_token):
    hyphen_annot = (annot_token.count('-') == 1)
    hyphen_text = (text_token.count('-') == 1)
    
    if (hyphen_annot and hyphen_text) or (not hyphen_annot and not hyphen_text):
        return annot_token == text_token
    else:
        annot_repr = annot_token.split('-')
        text_repr = text_token.split('-')
        
        for an_part in annot_repr:
            if an_part == 'то':
                continue
            
            if an_part in text_repr:
                return True
        
        return False


def find_token_in_sent(arg, text, predicate = None):
    annot_token = prepare_annot_word(arg['WordDep'])
    
    def f_find_impl(start, end, adv):
        for i in range(start, end, adv):
            token = text[i]
            text_token = prepare_text_word(token['form'])
            if compare_tokens(annot_token, text_token):
                return i
            
        return None
    
    if predicate:
        try_1 = (predicate[0] - 1, -1, -1)
        try_2 = (predicate[0] + 1, len(text), 1)
        
        if predicate[1] < arg['Offset']:
            try_1, try_2 = try_2, try_1
        
        match = f_find_impl(*try_1)

        if match is not None:
            return match

        return f_find_impl(*try_2)
    else:
        return f_find_impl(0, len(text), 1)


def find_all_tokens_in_text(arg, text):
    result = list()
    for sent in text:
        token = find_token_in_sent(arg, sent)
        
        if token is not None:
            result.append((token, sent))
    
    return result


def find_arg_positions(args, pred, sent):
    result = dict()
    for arg_idx, arg in enumerate(args):
        if str(arg['WordDep']) == 'nan':
            continue
        
        token_index = find_token_in_sent(arg, sent, predicate = pred)
            
        if token_index is not None:
            result[arg_idx] = token_index
    
    return result


def find_predicate_arguments(annot_pred, annot_args, text_data):
    sent = None
    text_pred_index = None
    chosen_arg_positions = list()
    
    if annot_pred is not None:
        all_pred_tokens = find_all_tokens_in_text(annot_pred, text_data)
        
        max_args = 0
        for pred_token, token_sent in all_pred_tokens:
            arg_positions = find_arg_positions(annot_args,
                                               (pred_token, annot_pred['Offset']), 
                                               token_sent)
            if len(arg_positions) > max_args:
                chosen_arg_positions = arg_positions
                max_args = len(arg_positions)
                sent = token_sent
                
                text_pred_index = pred_token
    
    if annot_pred is None or len(all_pred_tokens) == 0:
        max_args = 0
        for token_sent in text_data:
            arg_positions = find_arg_positions(annot_args, 
                                               None, 
                                               token_sent)
            if len(arg_positions) > max_args:
                chosen_arg_positions = arg_positions
                max_args = len(arg_positions)
                sent = token_sent
    
    return sent, text_pred_index, chosen_arg_positions


def check_role(role):
    if not role:
        return False
    
    if isinstance(role, int):
        return False
    
    if isinstance(role, float):
        return False
    
    srole = role.strip()
    if srole in ['-', '?', '0']:
        return False
    
    return True


def get_offset(offset_data, annot):
    if offset_data is None:
        return 0
    
    ex_index = annot.loc['ExIndex']
    item_index = annot.loc['ItemExIndex']
    
    try:
        return offset_data[(ex_index, item_index)]
    except KeyError:
        return 0
                                                          

def extract_args_and_pred(annot_data, offset_data, err_logger):
    annot_args = list()
    annot_pred = None
    
    for annot in annot_data:
        if (annot.loc['Rank'] == 'Предикат' and 
            str(annot.loc['WordDep']) != 'nan'):
            annot_pred = annot
            annot_pred['Offset'] = get_offset(offset_data, annot_pred)
        else:
            if not check_role(annot.loc['Role']):
                err_logger.warning('Bad role name of an argument. '
                                   'ExIndex: {}'.format(annot['ExIndex']))
                continue
            
            annot.loc['Offset'] = get_offset(offset_data, annot)
            annot_args.append(annot)
    
    return annot_pred, annot_args


def process_example(index, 
                    example_index, 
                    annot_data, 
                    text_data, 
                    err_logger,
                    offset_data):
    annot_pred, annot_args = extract_args_and_pred(annot_data, 
                                                   offset_data, 
                                                   err_logger)
    
    sent, text_pred_index, text_args = find_predicate_arguments(annot_pred, 
                                                                annot_args, 
                                                                text_data)
    
    if text_pred_index is not None:
        pred_token = sent[text_pred_index]
        pred_token['rank'] = annot_pred['Rank']
        pred_token['pred'] = annot_pred['ConstrIndex']

    for annot_arg_index, annot_arg in enumerate(annot_args):
        if str(annot_arg['WordDep']) == 'nan':
            err_logger.warning('Empty WordDep of an argument '
                               '(id: {}, index: {})'.format(example_index, 
                                                            index))
            continue
        
        if annot_arg_index not in text_args:
            pred_str = ''
            if annot_pred is not None:
                pred_str = annot_pred['WordDep']
                
            err_logger.error('Failed to locate token {} in example text ' 
                             '(id: {}, pred: {}, '
                             'index: {}).'.format(str(annot_arg['WordDep']).encode('utf8'), 
                                                  example_index,
                                                  pred_str.encode('utf8'),
                                                  index))
        else:
            token_index = text_args[annot_arg_index]
            token = sent[token_index]
            token['fillpred'] = text_pred_index
            token['rolepred1'] = annot_arg['Role'].strip()
            token['rank'] = annot_arg['Rank']


def process_examples(annots, text_data, err_logger, offset_data):
    curr_index = None
    example_data = list()
    for index, row in annots.iterrows():
        example_index = row['ExIndex']
        if curr_index is None:
            curr_index = example_index

        if example_index == curr_index:
            example_data.append(row)
        else: 
            if str(curr_index) in text_data:
                process_example(index, 
                                curr_index, 
                                example_data, 
                                text_data[str(curr_index)],
                                err_logger,
                                offset_data)
            else:
                err_logger.error('Absent example id {}.'.format(curr_index))
                
            example_data = [row]
            curr_index = example_index

    if str(curr_index) in text_data:
        process_example(index, 
                        curr_index, 
                        example_data, 
                        text_data[str(curr_index)],
                        err_logger,
                        offset_data)


def create_error_logger(err_file_path):
    logging.basicConfig(level = logging.DEBUG)
    logger = logging.getLogger('process_errors')
    logger.setLevel(logging.WARNING)
    logger.propagate = False
    fh = logging.FileHandler(err_file_path)
    fh.setLevel(logging.WARNING)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                  datefmt = '%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger


def build_offset_cache(offset_data):
    cache = dict()
    
    for _, row in offset_data.iterrows():
        if row['Type'] != 'Dep':
            continue
        
        cache[(row['ExIndex'], row['ItemExIndex'])] = row['Begin']
    
    return cache  


def print_annot_statistics(logger, annots):
    logger.info('Examples: {}.'.format(len(annots.loc[:, 'ExIndex'].unique())))
    logger.info('Predicates: {}'.format(len(annots[annots['Rank'] == 'Предикат'])))
    logger.info('Args: {}'.format(len(annots[annots['Rank'] != 'Предикат'])))
    

def print_corpus_statistics(logger, corpus):
    n_examples = 0
    n_args = 0
    n_preds = 0
    n_ex_with_args = 0
    for example in corpus.values():
        n_examples += 1
        
        have_args = False
        
        for sent in example:
            for word in sent:
                if 'rank' in word and word['rank'] == 'Предикат':
                    n_preds += 1
                    have_args = True
                elif 'rolepred1' in word:
                    n_args += 1
                    have_args = True
        
        if have_args:
            n_ex_with_args += 1
    
    logger.info('Examples: {}'.format(n_examples))
    logger.info('Predicates: {}'.format(n_preds))
    logger.info('Arguments: {}'.format(n_args))
    logger.info('N examples with args: {}'.format(n_ex_with_args))


def process(json_file_path, 
            annot_file_path, 
            output_file_path, 
            err_logger_path,
            offset_file_path = None,
            logger = logging.getLogger()):
    err_logger = create_error_logger(err_logger_path)
    
    logger.info('Loading annotation data (CSV)...')
    annots = pandas.read_csv(annot_file_path, sep = '\t', encoding = 'utf8')
    offset_data = None
    if offset_file_path:
        offset_data = pandas.read_csv(offset_file_path, 
                                      sep = '\t', 
                                      encoding = 'utf8')
        offset_data = build_offset_cache(offset_data)
    logger.info('Done. Got {} items with predicates.'.format(len(annots)))
    
    print_annot_statistics(logger, annots)

    logger.info('Loading text data (JSON)...')
    with open(json_file_path, 'r') as f:
        text_data = json.load(f)
    logger.info('Done.')

    logger.info('Processing data...')
    process_examples(annots, text_data, err_logger, offset_data)
    logger.info('Done.')
    
    print_corpus_statistics(logger, text_data)
        
    logger.info('Saving results...')
    del annots
    with codecs.open(output_file_path, 'w', encoding = 'utf8') as f:
        json.dump(text_data, f, ensure_ascii = False)
        
    logger.info('Done.')


if __name__ == '__main__':
    def create_info_logger():
        import sys
        
        info_logger = logging.getLogger('matcher')
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
                                     'Matches corpus (JSON) with annotations '
                                     '(CSV) and produces new corpus file (JSON) '
                                     'with annotations embedded.')
    
    parser.add_argument('--corpusFile', '-T', 
                        required = True, 
                        help = 'Path to JSON representation of a corpus.')
    parser.add_argument('--annotFile', '-A', 
                        required = True, 
                        help = 'Path to CSV representation of annotations.')
    parser.add_argument('--outputFile', '-O', 
                        required = True, help = 'An output file path (JSON).')
    parser.add_argument('--logFile', '-L', 
                        required = True, 
                        help = 'Log file path for errors.')
    parser.add_argument('--offsetFile', '-F', 
                        required = False,
                        default = None,
                        help = 'CSV file with approximate offsets for arguments.')
    
    args = parser.parse_args()

    process(args.corpusFile, 
            args.annotFile, 
            args.outputFile, 
            args.logFile, 
            args.offsetFile,
            create_info_logger())
