from tqdm import tqdm
from isanlp_srl_framebank.convert_corpus_to_brat import make_text


class FeatureModelingTool:
    def __init__(self, ling_cache, feature_extractor):
        self.ling_cache = ling_cache
        self.feature_extractor = feature_extractor
        self.error_examples = {}
        self.num_of_errors

    def find_address_by_offset(self, offset, ling_ann):
        for tok_num, tok in enumerate(ling_ann['tokens']):
            if tok.begin <= offset and offset < tok.end:
                break

        for sent_num, sent in enumerate(ling_ann['sentences']):
            if sent.begin <= tok_num and tok_num < sent.end:
                break

        return sent_num, tok_num - sent.begin

    def process_arg_pred(self, ex_id, pred, args, example):
        feature_sets = list()

        text, offset_index = make_text(example, 0)
        ling_ann = self.ling_cache[ex_id]

        pred_offset = offset_index[(pred[0], pred[1])]
        pred_ling_sent, pred_ling_word = self.find_address_by_offset(pred_offset, ling_ann)

        for arg in args:
            arg_offset = offset_index[(arg[0], arg[1])]
            arg_ling_sent, arg_ling_word = self.find_address_by_offset(arg_offset, ling_ann)

            lens = {
                'len_postags': len(ling_ann['postag']),
                'len_morph': len(ling_ann['morph']),
                'len_lemma': len(ling_ann['lemma']),
                'len_syntax': len(ling_ann['syntax_dep_tree'])
            }

            if arg_ling_sent > min(lens.values()) or len(set(lens.values())) != 1:
                lens['len_arg_ling_sent'] = arg_ling_sent
                if ex_id not in self.error_examples:
                    self.error_examples[ex_id] = []
                self.error_examples[ex_id].append((ex_id, lens, "length mismatch"))
                continue
                
            if arg_ling_sent != pred_ling_sent:
                self.num_of_errors += 1
                # We miss some examples due to mistakes in framebank or discrepancy in
                # automatica annotation of sentences.
                #print('Error #{}'.format(num_of_errors))
                continue

            fb_pred_word = example[pred[0]][pred[1]]
            fb_arg_word = example[arg[0]][arg[1]]
            
            sentence = ling_ann['sentences'][pred_ling_sent]
            tokens = [tok.text for tok in ling_ann['tokens']]
            tokens = tokens[sentence.begin:sentence.end]

            role = fb_arg_word['rolepred1']

            try:
                features = self.feature_extractor.extract_features(pred_ling_word,
                                                            arg_ling_word,
                                                            ling_ann['postag'][arg_ling_sent],
                                                            ling_ann['morph'][arg_ling_sent],
                                                            ling_ann['lemma'][arg_ling_sent],
                                                            ling_ann['syntax_dep_tree'][arg_ling_sent])
            except Exception as e:
                lens['len_arg_ling_sent'] = arg_ling_sent
                if ex_id not in self.error_examples:
                    self.error_examples[ex_id] = []
                self.error_examples[ex_id].append((ex_id, lens, str(e)))
                continue

            #feature_sets.append((features, role, ex_id, arg))
            feature_sets.append((features, role, ex_id, tokens, arg_ling_word, pred_ling_word))

        return feature_sets

    def process_example(self, ex_id, sentences):
        pred = None
        args = list()
        for sent_num, sent in enumerate(sentences):
            for word_num, word in enumerate(sent):
                if 'rank' in word and word['rank'] == 'Предикат':
                    pred = (sent_num, word_num)
                elif 'rolepred1' in word:
                    args.append((sent_num, word_num))

        return self.process_arg_pred(ex_id, pred, args, sentences)


    num_of_errors = 0

    def prepare_train_data(self, examples):
        feature_sets = []
        for ex_num, (ex_id, ex) in tqdm(enumerate(examples)):
            feature_sets += self.process_example(ex_id, ex)

        print('Number of training examples:', len(feature_sets))
        return feature_sets
    