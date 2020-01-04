import xml.etree.ElementTree as ET
import json
import logging
import os
import codecs
from bs4 import BeautifulSoup
import string


class ErrorPrasing(ValueError):
    pass


class ProgressBar:
    def __init__(self, 
                 full_size, 
                 notify_diff = 0.5, 
                 logger = logging.getLogger()):
        self._full_size = full_size
        self._prev_progress = 0.
        self._notify_diff = notify_diff
        self._logger = logger
    
    def display_progress(self, prog):
        self._logger.info(u'{0:.2f}%'.format(self._prev_progress))
    
    def advance_progress(self, curr_size, force = False):
        curr_progress = 100. * curr_size / self._full_size
        
        if (force 
            or curr_progress - self._prev_progress >= self._notify_diff 
            or curr_size == self._full_size):
            self._prev_progress = curr_progress
            self.display_progress(self._prev_progress)


def strip_string(string):
    if string:
        return string.strip()

    return string


PUNCT_SET = set(string.punctuation)


def check_hyphen(text, curr_pos):
    if text[curr_pos] == '-':
        if (curr_pos > 0 
            and curr_pos < len(text) - 1 
            and text[curr_pos + 1] not in PUNCT_SET
            and text[curr_pos - 1] not in PUNCT_SET):
            return False
    
    return True


def extract_tokens(text):
    if text is None:
        return []
    
    tokens = list()
    prev_pos = 0
    for pos, ch in enumerate(text):
        if ch in PUNCT_SET and check_hyphen(text, pos):
            tok = strip_string(text[prev_pos : pos])
            if tok:
                tokens.append((tok, False))
                
            prev_pos = pos + 1
            tokens.append((ch, True))
    
    tok = strip_string(text[prev_pos :])
    if tok:
        tokens.append((tok, False))
        
    result = list()
    hyph = False
    for token, is_punct in tokens:
        if token == '-':
            if hyph:
                result.append(('--', True))
                hyph = False
            else:
                hyph = True
        else:
            if hyph:
                result.append(('-', True))
                hyph = False
            result.append((token, is_punct))
    
    if hyph:
        result.append(('-', True))
    
    return result


def extract_index_xml(line):
    split_index = line.find('\t')
    if split_index == -1:
        raise ErrorPrasing('invalid example string')
    
    return int(line[:split_index]), line[split_index: ]


def process_text_line(line, converter_errors_logger):
    ex_index, xml_text = extract_index_xml(line)
    
    xml_text = xml_text.strip()
    
    if not xml_text.startswith('<p'):
        if not xml_text.startswith('<se'):
            xml_text = '<se>' + xml_text
        
        xml_text = '<p>' + xml_text
    
    if not xml_text.endswith('</p>'):
        if not xml_text.endswith('</se>') and not xml_text.endswith('<p>'):
            xml_text = xml_text + '</se>'
            
        xml_text = xml_text + '</p>'
        
    xml_text = '<top>' + xml_text.strip() + '</top>'
    
    try:
        parsed_text = ET.fromstring(xml_text)
        return ex_index, parsed_text
    except ET.ParseError as e:
        converter_errors_logger.warning('Conventional XML parser failed on '
                                        'example {} due to "{}". Proceeding '
                                        'with a beautiful ' 
                                        'soup.'.format(ex_index, str(e)))        
    
    soup = BeautifulSoup(xml_text, 'lxml')
    xml_text = str(soup)
    parsed_text = ET.fromstring(xml_text)
    return ex_index, parsed_text    


def post_process_sentence(sent):
    return sent


def parse_sentence(sent):
    
    def make_simple_word(wd):
        return {u'form' : wd}
    
    def process_tails(txt):
        stext = strip_string(txt)
        result = list()
        if stext:
            tokens = extract_tokens(stext)
            result += [make_simple_word(t[0]) for t in tokens]
        return result
    
    result = list()
    
    result += process_tails(sent.text)
    
    for word in sent:
        text = next(word.itertext(), None)
        tokens = extract_tokens(text)
        first_non_puct = True
        for token, token_type in tokens:
            if not token_type and first_non_puct:
                new_word = make_simple_word(token)
                annots = word.find('.//ana')
                if annots is not None:
                    new_word[u'lemma'] = annots.get('lex', '')
                    new_word[u'feat'] = annots.get('gr', '')
                    new_word[u'sem'] = annots.get('sem', '')
                    new_word[u'sem2'] = annots.get('sem2', '')
                
                result.append(new_word)
                first_non_puct = False
            else:
                result.append(make_simple_word(token))
        
        result += process_tails(word.tail)
    
    result += process_tails(sent.tail)
    
    return post_process_sentence(result)


def create_converter_errors_logger(err_file_path):
    logging.basicConfig(level = logging.WARNING)
    logger = logging.getLogger('converter_errors')
    logger.setLevel(logging.WARNING)
    
    logger.propagate = False
    
    fh = logging.FileHandler(err_file_path)
    fh.setLevel(logging.WARNING)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                  datefmt = '%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    
    logger.addHandler(fh)
    
    return logger


def convert(text_file_path, 
            output_file_path, 
            err_file_path, 
            logger = logging.getLogger()):
    
    converter_errors_logger = create_converter_errors_logger(err_file_path)

    logger.info('Parsing input data...')
    prog_bar = ProgressBar(os.path.getsize(text_file_path), logger = logger)
    text_file = open(text_file_path, 'r', encoding='utf8')
    text_file.readline() # Skipping useless line
    
    prog_bar.advance_progress(0, True)

    result = dict()
    for num, line in enumerate(iter(text_file.readline, '')):
    #for num, line in enumerate(text_file):
        try:
            ex_index, parsed_text = process_text_line(line, 
                                                      converter_errors_logger)
#             print(ex_index)

            sentences = list()
            for sent in parsed_text.findall('.//se'):
                sentences.append(parse_sentence(sent))

            result[ex_index] = sentences
            prog_bar.advance_progress(text_file.tell())

        except (ET.ParseError, ErrorPrasing) as err:
            converter_errors_logger.error('Critical parsing error. Failed to '
                                          'proceed further on the line of the ' 
                                          'file: {} due to "{}"'.format(num, err))
            converter_errors_logger.error(line)

    print(len(result))
    text_file.close()
    logger.info('Done.')

    logger.info('Saving result...')
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)
    logger.info('Done.')


if __name__ == '__main__':
    
    def create_info_logger():
        import sys
        
        info_logger = logging.getLogger("converter")
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
                                     'Converts corpus in original XML-type '
                                     'into the proper JSON format.')
    
    parser.add_argument('--inputFile', '-I', required = True, 
                        help='Input file path (XML type).')
    parser.add_argument('--outputFile', '-O', required = True, 
                        help = 'Output file path (JSON).')
    parser.add_argument('--errorFile', '-E', required = True, 
                        help = 'Log file path (errors are logged here).')

    args = parser.parse_args()

    convert(args.inputFile, 
            args.outputFile, 
            args.errorFile, 
            create_info_logger())
