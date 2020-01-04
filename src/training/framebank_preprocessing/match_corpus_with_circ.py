# -*- coding: utf-8 -*-

import json, csv, pyexling, codecs, argparse, logging
from pyexling import parser_morph
from pyexling import morph




def create_process_logger():
	logging.basicConfig(level = logging.DEBUG)
	logger = logging.getLogger('console_log')
	logger.setLevel(logging.DEBUG)
	logger.propagate = False

	sh = logging.StreamHandler()
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
	                      datefmt = '%Y-%m-%d %H:%M:%S')
	sh.setFormatter(formatter)

	logger.addHandler(sh)

	return logger



def create_file_logger(err_file_path):
	logging.basicConfig(level = logging.DEBUG)
	logger = logging.getLogger('file_log')
	logger.setLevel(logging.DEBUG)
	logger.propagate = False
	fh = logging.FileHandler(err_file_path)
	fh.setLevel(logging.DEBUG)
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
	                      datefmt = '%Y-%m-%d %H:%M:%S')
	fh.setFormatter(formatter)
	logger.addHandler(fh)

	return logger



def init_corpus(json_path):
	data = None
	with open(json_path, 'r') as f:
		data = json.load(f)
	return data


def init_framebank(framebank_path):
	titles_arr = []
	
	framebank_list = []
	with open(framebank_path, 'rb') as csv_file:
		framebank = csv.reader(csv_file, delimiter = '\t')
	
		for row in framebank:
			if not titles_arr:
				titles_arr = row
			else:
				new_element = {}
				for el_index in xrange(0, len(row)):
					el_title = titles_arr[el_index]
					new_element[el_title] = row[el_index]
				framebank_list.append(new_element)

	return framebank_list



def get_top(res):
	base_first_syntax = res['syntax'][0]['base']

	annot_morph = parser_morph.create_annot_morph(res["lexics"])[0]
	
	for lexeme_num in xrange(0, len(base_first_syntax)):
		lexeme_dict = base_first_syntax[lexeme_num]
		parentIndex = lexeme_dict['parentIndex']
		if parentIndex == 18446744073709551615:
			word = annot_morph[lexeme_num]
			lexema =  word.get_lexeme() 
			lexeme_type = lexema.get_lexeme_type()
			if lexeme_type == morph.LexemeType(0):
				continue
			elif lexeme_type == morph.LexemeType(10):
				for lexeme_num_2 in xrange(0, len(base_first_syntax)):
					lexeme_dict_2 = base_first_syntax[lexeme_num_2]
					if lexeme_dict_2['parentIndex'] == lexeme_num:
						word = annot_morph[lexeme_num_2]
						form =  word.get_form()
						return form
				continue
			else:
				return word.get_form()
	return None


def process_circ(host, 
				 service, 
				 json_input_file_path, 
				 input_file_path, 
				 output_file_path,
				 log_file_path):
	
	
	process_logger = create_process_logger()
	file_logger = create_file_logger(log_file_path)

	process_logger.info('Script is initializing corpus in memory')
	data = init_corpus(json_input_file_path)
	process_logger.info('Script is initializing framebank in memory')
	framebank_list = init_framebank(input_file_path)
	

	proc = pyexling.Processor(host, service)
	
	

	total_elements = 0
	circum_elements = 0
	analyze_fails = 0
	not_found_syntax_top = 0
	not_found_predic = 0
	not_found_phrase_top = 0
	not_example_index_in_data = 0
	complete_example = 0

	error_log = ''
	

	framebank_len = len(framebank_list)

	for element in framebank_list:
		element_str_pos = str(element['ConstrIndex']) + ' ' + str(element['ExIndex'])
		total_elements += 1
		# if total_elements == 101:
		# 	break
		if (total_elements % 100) == 0:
			process_logger.info("%(elements_computes)i examples computed from %(total_examples)i examples" % {
				'elements_computes' : total_elements,
				'total_examples' : framebank_len
			})
			
		word_found = False
		if element['Type'] == 'Circum':
			circum_elements += 1
			res = None
			try:
				res = proc.process(element['Phrase'])
			except:
				file_logger.warning('Analyze fail - %s' % element_str_pos)
				analyze_fails += 1
			if res:
				phrase_top = get_top(res)
				if phrase_top == None:
					file_logger.warning("Syntax top wasn't found - %s" % element_str_pos)
					not_found_syntax_top += 1
				else:
					example_index = element['ExIndex']

					predic_num = None
					if example_index in data:
						for sentence in data[example_index]:
							word_index = 0
							for word in sentence:
								if word.get('rank') == u'Предикат':
									predic_num = word_index
									break
								word_index += 1

							if predic_num:
								for word in sentence:
									if word['form'] == phrase_top:
										word_found = True
										word['role'] = element['Role']
										word['fillpred'] = predic_num
										complete_example += 1
										break
								if not word_found:
									file_logger.warning("Script didn't find phrase from framebank in corpus - %s" % element_str_pos)
									not_found_phrase_top += 1
								break
							if word_found:
								break
					else:
						not_example_index_in_data += 1
						file_logger.warning("Example index from framebank wasn't found in corpus - %s" % element_str_pos)
					if predic_num == None:
						file_logger.warning("Predicate top in example wasn't found - %s" % element_str_pos)
						not_found_predic += 1

    
	file_logger.info('Total elements in framebank: %i' % total_elements)
	file_logger.info('Elements with circum type in framebank: %i' % circum_elements)
	file_logger.info('Intellection cgi failed analyze phrase in elements: %i' % analyze_fails)
	file_logger.info("Examples index from framebank wasn't found in corpus: %i" % not_example_index_in_data)
	file_logger.info("Syntax top wasn't found: %i" % not_found_syntax_top)
	file_logger.info("Predicate wasn't found: %i" % not_found_predic)
	file_logger.info("Script didn't find phrase from framebank in corpus: %i" % not_found_phrase_top)
	file_logger.info('Complete examples: %i' % complete_example)


	process_logger.info('Script is saving corpus on disk')
	with codecs.open(output_file_path, 'w', encoding = 'utf8') as f:
		json.dump(data, f, encoding = 'utf8', ensure_ascii = False)
		
	process_logger.info('Script has finished work!')



parser = argparse.ArgumentParser(description='Peripheral role extracting from framebank')

parser.add_argument('--corpusFile', '-T', required = True, help = 'Path to JSON representation of a corpus.')
parser.add_argument('--annotFile', '-A', required = True, help = 'Path to CSV representation of annotations.')
parser.add_argument('--outputFile', '-O', required = True, help = 'An output file path (JSON).')
parser.add_argument('--logFile', '-L', required = True, help = 'Log file path for errors.')



host = 'nlp.isa.ru:8007'
service = '/cgi-bin/intellection.fcgi'
# input_file_path = '/home/artem/workspace/test/framebank/framebank_jupyter/framebank_anno_ex_circ.txt'
# json_input_file_path = '/home/artem/workspace/test/framebank/framebank_jupyter/annotated_corpus.json'
# output_file_path = '/home/artem/workspace/test/framebank/framebank_jupyter/1/new_annotated_corpus.json'

args = parser.parse_args()

input_file_path = args.annotFile
json_input_file_path = args.corpusFile
output_file_path = args.outputFile
log_file_path  = args.logFile




process_circ(host, 
			 service, 
			 json_input_file_path, 
			 input_file_path, 
			 output_file_path,
			 log_file_path)
