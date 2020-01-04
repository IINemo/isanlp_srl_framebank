# -*- coding: utf-8 -*-

import logging
import json
import os
import itertools
import pylingtools.format_brat as brt

from collections import defaultdict


def need_space(form, num, sent):
    if form in u'«([':
        return False
    
    if num < len(sent) - 1 and sent[num + 1][u'form'] in (u',.:;?!»)]'):
        return False
    
    return True


def make_text(example, offset):
    text = str()
    offset_index = dict()
    curr_offset = offset
    for sent_num, sent in enumerate(example):
        for word_num, word in enumerate(sent):
            form = word[u'form']
            text += form
            add_offset = len(form)
            if need_space(form, word_num, sent):
                text += u' '
                add_offset += 1
                
            offset_index[(sent_num, word_num)] = curr_offset
            curr_offset += add_offset
    
    return text, offset_index


get_next_id = itertools.count().__next__


def make_annotations_conf(role_index):
    basic_conf = \
'''
[entities]

Arg
Pred

[attributes]

[relations]

{}


[events]

'''

    roles = str()
    for _, role_id in sorted(role_index.items(), key = lambda e: e[0]):
        roles += 'Role_{} Arg1:Pred, Arg2:Arg\n'.format(role_id)
    
    return basic_conf.format(roles)


def make_visual_conf(role_index):
    basic_conf = \
u'''
[labels]

{}

[drawing]

Arg fgColor:black, bgColor:#FF7070, borderColor:darken
Pred fgColor:black, bgColor:lightgreen, borderColor:darken

{}

'''
    bad_roles = list()
    roles = unicode()
    for role_str, role_id in role_index.items():
        if role_str.startswith(u'false_'):
            bad_roles.append(u'Role_{}'.format(role_id))
            roles += u'Role_{} | {}\n'.format(role_id, role_str) # bug
        else:
            roles += u'Role_{} | {}\n'.format(role_id, role_str)  
    
    bad_roles_str = u''
    for role_str in bad_roles:
        bad_roles_str += u'{}\tcolor:red, dashArray:-, arrowHead:triangle-5\n'.format(role_str)
    
    return basic_conf.format(roles, bad_roles_str).encode('utf8') 
    

def fill_brat_entity(brat_entity, word, tp, offset):
    brat_entity.tp = tp
    brat_entity.bid = get_next_id()
    brat_entity.begin = offset
    brat_entity.end = brat_entity.begin + len(word['form'])
    brat_entity.form = word['form'].encode('utf8')


def convert_example(example, offset_index, role_index):
    brat_annots = list()
    pred = None
    for sent_num, sent in enumerate(example):
        for word_num, word in enumerate(sent):
            if u'rolepred1' not in word:
                continue
            
            arg = brt.BratEntity()
            fill_brat_entity(arg, 
                             word, 
                             u'Arg',
                             offset_index[(sent_num, word_num)])
            brat_annots.append(arg)
            
            if pred is None:
                pred_num = word[u'fillpred']
                pred_word = sent[pred_num]
                
                pred = brt.BratEntity()
                fill_brat_entity(pred, 
                                 pred_word, 
                                 u'Pred', 
                                 offset_index[(sent_num, pred_num)])
                brat_annots.append(pred)
            
            if word[u'rolepred1'] not in role_index:
                role_index[word[u'rolepred1']] = len(role_index) + 1
                
            rel = brt.BratRealtion()
            rel.tp = u'Role_' + str(role_index[word[u'rolepred1']]) 
            rel.bid = get_next_id()
            rel.dep = arg.bid
            rel.head = pred.bid
            brat_annots.append(rel)
            
            if u'rolepred2' in word:
                if word[u'rolepred1'] != word[u'rolepred2']:
                    false_role = u'false_' + word[u'rolepred2']
                    if false_role not in role_index:
                        role_index[false_role] = len(role_index) + 1
                    
                    rel = brt.BratRealtion()
                    rel.tp = u'Role_' + str(role_index[false_role]) 
                    rel.bid = get_next_id()
                    rel.dep = arg.bid
                    rel.head = pred.bid
                    brat_annots.append(rel)
    
    return brat_annots


def create_verb_example_index(json_data):
    verb_example_index = defaultdict(list)
    for ex_id, example in json_data.items():
        for sent in example:
            for word in sent:
                if(word.get('rank', '') == 'Предикат'):
                    if 'lemma' in word:
                        verb_example_index[word['lemma']].append(ex_id)
    
    return verb_example_index
    
    
def make_annotations_conf_synt(role_index):
    basic_conf = \
'''
[entities]

Token

[attributes]

[relations]

{}


[events]

'''
    roles = str()
    for _, role_id in sorted(role_index.items(), key = lambda e: e[0]):
        roles += 'Synt_{} Arg1:Token, Arg2:Token\n'.format(role_id)
    
    return basic_conf.format(roles)


def make_visual_conf_synt(role_index):
    basic_conf = \
u'''
[labels]

{}

[drawing]

#Token fgColor:black, bgColor:#FF7070, borderColor:darken
#Pred fgColor:black, bgColor:lightgreen, borderColor:darken

'''
    bad_roles = list()
    roles = unicode()
    for role_str, role_id in role_index.items():
        roles += u'Synt_{} | {}\n'.format(role_id, role_str)  
    
    return basic_conf.format(roles).encode('utf8') 

    
def convert_example_syntax(example, offset_index, role_index):
    synt_annots = list()
    for sent_num, sent in enumerate(example):
        word_index = {}
        for word_num, word in enumerate(sent):
            token = brt.BratEntity()
            fill_brat_entity(token, 
                             word, 
                             u'Token',
                             offset_index[(sent_num, word_num)])
            
            synt_annots.append(token)
            
            word_index[word_num] = token.bid
            
        for word_num, word in enumerate(sent):
            parent = word['parent']
            if parent != -1:
                link_name = word[u'link_name']
                if link_name not in role_index:
                    role_index[link_name] = len(role_index) + 1
                
                rel = brt.BratRealtion()
                rel.tp = u'Synt_' + str(role_index[link_name]) 
                rel.bid = get_next_id()
                rel.dep = word_index[word_num]
                rel.head = word_index[parent]
                synt_annots.append(rel)
    
    return synt_annots


def convert(input_file_path, output_dir_path, converter, logger):
    logger.info('Loading corpus data...')
    with open(input_file_path, 'r') as f:
        json_data = json.load(f)
    logger.info('Done.')
    
    logger.info('Creating verb-example index...')
    verb_example_index = create_verb_example_index(json_data)
    logger.info('Done.')
    
    role_index = dict()
    
    logger.info('Converting and saving...')
    dir_num = 0
    curr_dir_path = os.path.join(output_dir_path, str(dir_num))
    os.mkdir(curr_dir_path)
    for num, (pred, ex_ids) in enumerate(sorted(verb_example_index.items(), 
                                                key = lambda e: e[0])):
        if num > 0 and (num % 50) == 0:
            dir_num += 1
            curr_dir_path = os.path.join(output_dir_path, str(dir_num))
            os.mkdir(curr_dir_path)
                
        output_file_path = os.path.join(curr_dir_path, pred)
        
        brat_objects = list()
        offset = 0
        #print output_file_path.encode('utf8')
        #print type(output_file_path)
        with open(output_file_path.encode('utf8') + '.txt', 'a') as f:
            for ex_id in ex_ids:
                example = json_data[ex_id]
                
                text, offset_index = make_text(example, offset)
                f.write(text.encode('utf8'))
                f.write('\n\n')
                offset += len(text) + 2
                
                brat_objects += converter.convert_example(example, 
                                                          offset_index, 
                                                          role_index)
                
        brt.save_list_of_brat_objects(brat_objects, output_file_path.encode('utf8') + '.ann')
    
    logger.info('Done.')
    
    logger.info('Generating brat configuration files...')
    
    annotations_conf = converter.make_annotations_conf(role_index)
    with open(os.path.join(output_dir_path, 'annotation.conf'), 'w') as f:
        f.write(annotations_conf)
        
    visual_conf = converter.make_visual_conf(role_index)
    with open(os.path.join(output_dir_path, 'visual.conf'), 'w') as f:
        f.write(visual_conf)
        
    with open(os.path.join(output_dir_path, 'rel_name_index.json'), 'w') as f:
        json.dump(role_index, f, indent = 2)
        
    logger.info('Done.')

    
class ConverterSRL(object):
    @staticmethod
    def convert_example(example, offset_index, role_index):
        return convert_example(example, offset_index, role_index)
    
    @staticmethod
    def make_annotations_conf(role_index):
        return make_annotations_conf(role_index)
    
    @staticmethod
    def make_visual_conf(role_index):
        return make_visual_conf(role_index)
    
    
class ConverterSyntax(object):
    @staticmethod
    def convert_example(example, offset_index, role_index):
        return convert_example_syntax(example, offset_index, role_index)
    
    @staticmethod
    def make_annotations_conf(role_index):
        return make_annotations_conf_synt(role_index)
    
    @staticmethod
    def make_visual_conf(role_index):
        return make_visual_conf_synt(role_index)
    

if __name__ == '__main__':
    def create_info_logger():
        import sys
        
        info_logger = logging.getLogger('converter_brat')
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
                                     'format into the Brat format.')
    parser.add_argument('--inputFile', '-I', 
                        required = True, 
                        help = 'Input file with annotations path (JSON).')
    parser.add_argument('--outputDir', '-O', 
                        required = True, 
                        help = 'The directory to store Brat files.')
    parser.add_argument('--converter', '-C',
                        default = 'srl',
                        help = 'Converter syn or srl.')

    args = parser.parse_args()
    
    converter = None
    if args.converter == 'srl':
        converter = ConverterSRL
    elif args.converter == 'syn':
        converter = ConverterSyntax
        
    convert(args.inputFile, args.outputDir, converter, create_info_logger())
