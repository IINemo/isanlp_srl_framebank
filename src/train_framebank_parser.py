
# coding: utf-8
import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('../isanlp/src/')
sys.path.append('../src/isanlp_srl_framebank/')
sys.path.append('../libs/')
sys.path.append('../libs/pylingtools/')

from sklearn.feature_extraction import DictVectorizer
import multiprocessing as mp
from sklearn.preprocessing import LabelBinarizer
from isanlp_srl_framebank.processor_srl_framebank import FeatureModelDefault
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import pickle
import json
import isanlp
import time
import os
from tensorflow.python.keras import backend as K
import tensorflow as tf
from convert_corpus_to_brat import make_text
from isanlp.annotation_repr import CSentence
from gensim.models import KeyedVectors



config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

K.set_session(sess)


np.random.seed(31)


parser = argparse.ArgumentParser(
    prog="Feature Extractor", description="Extracts features for model training")
parser.add_argument(
    "--cleared-corpus",
    nargs="?", dest="corpus_file",
    default="../data/cleared_corpus.json",
    help="cleared framebank corpus file (in .json format)"
)
parser.add_argument(
    "--ling-data",
    nargs='?', dest='ling_data_file',
    default='../data/results_final_fixed.pckl',
    help='linguistic data about examples'
)
parser.add_argument(
    "--embeddings-file",
    nargs='?', dest='embeddings_file',
    default='../data/embeddings/ruscorpora_upos_skipgram_300_5_2018.vec',
    help="W2V-format embeddings file"
)
parser.add_argument(
    '--out-plain-features',
    nargs='?', dest='plain_features_file',
    default='../data/plain_features.npy',
    help='where to save plain features table (in .npy format)'
)
parser.add_argument(
    '--out-verb-embed',
    nargs='?', dest='verb_embed_file',
    default='../data/verb_embedded.npy',
    help='where to save embeddings for verbs (in .npy format)'
)
parser.add_argument(
    "--out-arg-embed",
    nargs='?', dest='arg_embed_file',
    default="../data/arg_embedded.npy",
    help="where to save embeddings for arguments (in .npy format)"
)

args = parser.parse_args()

print("1. Reading files..")
print(f"\tCorpus file: ({args.corpus_file})", end="....", flush=True)
with open(args.corpus_file, 'r') as f:
    examples = json.load(f)
print("OK", flush=True)
print(f'\tLing data file: ({args.ling_data_file})', end="....", flush=True)
with open(args.ling_data_file, 'rb') as f:
    ling_data = pickle.load(f)
ling_data_cache = {k: v for k, v in ling_data}
print("OK", flush=True)
print(f"\tEmbeddings file: ({args.embeddings_file})", end="....", flush=True)
embeddings = KeyedVectors.load_word2vec_format(
    args.embeddings_file, binary=False)
print("OK", flush=True)
print("..Done!")

error_examples = {}


def find_address_by_offset(offset, ling_ann):
    for tok_num, tok in enumerate(ling_ann['tokens']):
        if tok.begin <= offset and offset < tok.end:
            break

    for sent_num, sent in enumerate(ling_ann['sentences']):
        if sent.begin <= tok_num and tok_num < sent.end:
            break

    return sent_num, tok_num - sent.begin


def process_arg_pred(feature_extractor, ling_cache, ex_id, pred, args, example):
    feature_sets = list()

    text, offset_index = make_text(example, 0)
    ling_ann = ling_cache[ex_id]

    pred_offset = offset_index[(pred[0], pred[1])]
    pred_ling_sent, pred_ling_word = find_address_by_offset(
        pred_offset, ling_ann)

    for arg in args:
        arg_offset = offset_index[(arg[0], arg[1])]
        arg_ling_sent, arg_ling_word = find_address_by_offset(
            arg_offset, ling_ann)

        # print("-"*20)
        #print('ex_id: ', ex_id)
        #print('ling_ann_sent: ', arg_ling_sent)
        #print('total number of postags: ', len(ling_ann['postag']))
        #print('total number of morph featues: ', len(ling_ann['morph']))
        #print('total number of lemmas: ', len(ling_ann['lemma']))
        #print('total number of syntax trees: ', len(ling_ann['syntax_dep_tree']))

        lens = {
            'len_postags': len(ling_ann['postag']),
            'len_morph': len(ling_ann['morph']),
            'len_lemma': len(ling_ann['lemma']),
            'len_syntax': len(ling_ann['syntax_dep_tree'])
        }

        # print("-"*20)
        # print(ex_id)
        # print(lens)
        #print("arg_ling_sent: ", arg_ling_sent)

        if arg_ling_sent > min(lens.values()) or len(set(lens.values())) != 1:
            lens['len_arg_ling_sent'] = arg_ling_sent
            if ex_id not in error_examples:
                error_examples[ex_id] = []
            error_examples[ex_id].append((ex_id, lens, "length mismatch"))
            continue

        fb_pred_word = example[pred[0]][pred[1]]
        fb_arg_word = example[arg[0]][arg[1]]

        role = fb_arg_word['rolepred1']

        if arg_ling_sent != pred_ling_sent:
            global num_of_errors
            num_of_errors += 1
            # We miss some examples due to mistakes in framebank or discrepancy in
            # automatica annotation of sentences.
            #print('Error #{}'.format(num_of_errors))
            continue

        try:
            features = feature_extractor.extract_features(pred_ling_word,
                                                          arg_ling_word,
                                                          ling_ann['postag'][arg_ling_sent],
                                                          ling_ann['morph'][arg_ling_sent],
                                                          ling_ann['lemma'][arg_ling_sent],
                                                          ling_ann['syntax_dep_tree'][arg_ling_sent])
        except Exception as e:
            lens['len_arg_ling_sent'] = arg_ling_sent
            if ex_id not in error_examples:
                error_examples[ex_id] = []
            error_examples[ex_id].append((ex_id, lens, str(e)))
            continue

        feature_sets.append((features, role, ex_id, arg))

    return feature_sets


def process_example(feature_extractor, ling_cache, ex_id, sentences):
    pred = None
    args = list()
    for sent_num, sent in enumerate(sentences):
        for word_num, word in enumerate(sent):
            if 'rank' in word and word['rank'] == 'Предикат':
                pred = (sent_num, word_num)
            elif 'rolepred1' in word:
                args.append((sent_num, word_num))

    return process_arg_pred(feature_extractor, ling_cache, ex_id, pred, args, sentences)


num_of_errors = 0


def prepare_train_data(examples, ling_data_cache, feature_extractor):
    feature_sets = []
    for ex_num, (ex_id, ex) in tqdm(enumerate(examples)):
        feature_sets += process_example(feature_extractor,
                                        ling_data_cache, ex_id, ex)

    print('Number of training examples:', len(feature_sets))
    return feature_sets


#!!!: Choose feature model here
feature_model = FeatureModelDefault()

# from isanlp_srl_framebank.processor_srl_framebank import FeatureModelUnknownPredicates
# feature_model = FeatureModelUnknownPredicates()
# main_model_path = os.path.join(main_model_path_root, 'unknown_preds')
print("2. Extracting features...")
feature_sets = prepare_train_data(examples, ling_data_cache, feature_model)

data_for_pandas = []
for example in feature_sets:
    data_for_pandas_ex = {}
    data_for_pandas_ex['role'] = example[1]
    data_for_pandas_ex['ex_id'] = example[2]
    data_for_pandas_ex['arg_address'] = example[3]
    for elem in example[0]:
        for subelem in elem:
            if subelem is not None:
                data_for_pandas_ex.update(subelem)

    data_for_pandas.append(data_for_pandas_ex)

pd_data = pd.DataFrame(data_for_pandas)
pd_data = pd_data.sample(frac=1)
pd_data[:10]
del data_for_pandas
print("..Done!")
print("3. Post-processing..")
y_stat = pd_data.loc[:, 'role'].value_counts()
drop_ys = y_stat[y_stat < 180].index
clear_data = pd_data.drop(pd_data[pd_data.loc[:, 'role'].isin(drop_ys)].index)

repl_roles = {
    'агенс - субъект восприятия': 'субъект восприятия',
    'агенс - субъект ментального состояния': 'субъект ментального состояния',
    'результат / цель': 'результат',
    'место - пациенс': 'место',
    'говорящий - субъект психологического состояния': 'субъект психологического состояния'
}


def normalize_single_region(data, rep, val):
    data.loc[:, 'role'] = data.loc[:, 'role'].str.replace(rep, val)


for rep, val in repl_roles.items():
    normalize_single_region(clear_data, rep, val)

number_of_roles = len(clear_data.loc[:, 'role'].value_counts().index)
print('Number of roles: ', number_of_roles)
clear_data.loc[:, 'role'].value_counts()

y_orig = clear_data.loc[:, 'role']
X_orig = clear_data.drop('role', axis=1)

label_encoder = LabelBinarizer()
y = label_encoder.fit_transform(y_orig)
print("..Done!")


def make_embeded_form(word):
    if word:
        # return word[1].encode('utf8')
        return u"{}_{}".format(word[1], word[0])
    else:
        return word


class Embedder_map:
    def __init__(self, embeddings, X):
        self.X_ = X
        self.embeddings_ = embeddings

    def __call__(self, i):
        result = np.zeros((len(self.X_[0]),
                           self.embeddings_.vector_size))

        for j in range(len(self.X_[0])):
            word = self.X_[i][j]
            tag = word[0] if word else str()

            if tag == ARG_SPECIAL_TAG or tag == ARG_SPECIAL_TAG:
                result[j, :] = np.ones(self.embeddings_.vector_size)
            elif word and word in embeddings:
                result[j, :] = self.embeddings_[word]

        return result


def embed(X):
    pool = mp.Pool(4)
    result = pool.map(Embedder_map(embeddings, X), X.index, 1000)
    pool.close()
    return np.asarray(result)


class Embedder_single_map:
    def __init__(self, embeddings, X):
        self.X_ = X
        self.embeddings_ = embeddings

    def __call__(self, i):
        #word = make_embeded_form(self.X_[i])
        word = self.X_[i]
        if word in self.embeddings_:
            return self.embeddings_[word]
        else:
            return np.zeros((self.embeddings_.vector_size,))


def embed_single(embeddings, X):
    pool = mp.Pool(4)
    result = pool.map(Embedder_single_map(embeddings, X), X.index, 1000)
    pool.close()

    return np.asarray(result)


print("4. Computing word embeddings..")

embedded_args = embed_single(embeddings, X_orig.arg_lemma)
embedded_verbs = embed_single(embeddings, X_orig.pred_lemma)

print("..Done!")
print("5. Saving embeddings..")
np.save(args.verb_embed_file, embedded_verbs, allow_pickle=False)
np.save(args.arg_embed_file, embedded_args, allow_pickle=False)
print("..Done!")

#morph_feats = ['pos', 'case', 'anim', 'vform', 'zform', 'shform', 'pform', 'vvform', 'nform', 'time']

# all_feats = (['pred_lemma', 'rel_pos'] +
#              ['arg_' + e for e in morph_feats] +
#              ['pred_' + e for e in morph_feats])

# all_feats = (['pred_lemma', 'rel_pos', 'arg_prep'] +
#              ['arg_' + e for e in morph_feats] +
#              ['pred_' + e for e in morph_feats])

# all_feats = (['pred_lemma', 'rel_pos', 'arg_prep', 'link_name'] +
#              ['arg_' + e for e in morph_feats] +
#              ['pred_' + e for e in morph_feats])

#all_feats = ['pred_lemma', 'rel_pos', 'pred_pos', 'arg_case', 'syn_link_name', 'arg_pos', 'prepos', 'dist']

#categ_feats = [e for e in all_feats if X_orig[e].dtype in [str, object]]
#not_categ = [e for e in all_feats if e not in categ_feats]

#pred_lemma_vectorizer.fit_transform(X_orig.loc[:, ['pred_lemma']].to_dict(orient = 'records'))

print("6. Processing categorical features")
not_categ_features = {'arg_address', 'ex_id', 'rel_pos', 'arg_lemma'}
categ_feats = [
    name for name in X_orig.columns if name not in not_categ_features]
not_categ = ['rel_pos']
print('Category features:\n', categ_feats)
print('Not category features:\n', not_categ)

vectorizer = DictVectorizer(sparse=False)
one_hot_feats = vectorizer.fit_transform(
    X_orig.loc[:, categ_feats].to_dict(orient='records'))

not_categ_columns = np.concatenate(
    tuple(X_orig.loc[:, e].as_matrix().reshape(-1, 1) for e in not_categ), axis=1)
plain_features = np.concatenate((one_hot_feats, not_categ_columns), axis=1)
print("..Done!")
print("7. Saving features table")
np.save(args.plain_features_file, plain_features)
print("..Done!")
