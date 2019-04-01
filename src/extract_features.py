# coding: utf-8
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelBinarizer

from gensim.models import KeyedVectors

from tqdm import tqdm
import pandas as pd
import numpy as np

from feature_modeling import FeatureModelingTool
from embedding import embed_single

import multiprocessing as mp
import argparse
import pickle
import json
import isanlp
import time
import os

from isanlp_srl_framebank import make_text
from isanlp_srl_framebank.processor_srl_framebank import FeatureModelDefault
from isanlp.annotation_repr import CSentence

DEFAULT_REPL_ROLES = {
    'агенс - субъект восприятия': 'субъект восприятия',
    'агенс - субъект ментального состояния': 'субъект ментального состояния',
    'результат / цель': 'результат',
    'место - пациенс': 'место',
    'говорящий - субъект психологического состояния': 'субъект психологического состояния'
}

def generate_feature_dataframe(examples, feature_modelling_tool: FeatureModelingTool):
    feature_sets = feature_modelling_tool.prepare_train_data(examples)

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
    del data_for_pandas
    return pd_data

def clear_unfrequent_roles(pd_data, n=180):
    y_stat = pd_data.loc[:, 'role'].value_counts()
    drop_ys = y_stat[y_stat < n].index
    clear_data = pd_data.drop(pd_data[pd_data.loc[:, 'role'].isin(drop_ys)].index)
    return clear_data

def normalize_roles(data_frame, repl_roles):
    def normalize_single_region(data, rep, val):
        data.loc[:, 'role'] = data.loc[:, 'role'].str.replace(rep, val)


    for rep, val in repl_roles.items():
        normalize_single_region(data_frame, rep, val)

    number_of_roles = len(data_frame.loc[:, 'role'].value_counts().index)

    return data_frame, number_of_roles

def split_to_x_y(dataframe):
    y_orig = dataframe.loc[:, 'role']
    X_orig = dataframe.drop('role', axis=1)

    return X_orig, y_orig

def train_label_encoder(labels):
    label_encoder = LabelBinarizer()
    y = label_encoder.fit_transform(labels)
    return y, label_encoder

def get_feature_names(dataframe, not_categ_features={'arg_address', 'ex_id', 'rel_pos', 'arg_lemma'}):
    categ_feats = [
        name for name in dataframe.columns if name not in not_categ_features]
    not_categ = ['rel_pos']

    return categ_feats, not_categ

def train_feature_vectorizer(dataframe, categ_features):
    vectorizer = DictVectorizer(sparse=False)
    one_hot_feats = vectorizer.fit_transform(
        dataframe.loc[:, categ_features].to_dict(orient='records')
    )
    return one_hot_feats, vectorizer

def generate_features_array(dataframe, one_hot_feats, not_categ):
    not_categ_columns = np.concatenate(
        tuple(dataframe.loc[:, e].values.reshape(-1, 1) for e in not_categ), axis=1)
    plain_features = np.concatenate((one_hot_feats, not_categ_columns), axis=1)
    return plain_features

if __name__ == "__main__":

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
        nargs='?', dest='plain_features',
        default='../data/plain_features.npy',
        help='where to save plain features table (in .npy format)'
    )
    parser.add_argument(
        '--out-verb-embed',
        nargs='?', dest='verb_embed',
        default='../data/verb_embedded.npy',
        help='where to save embeddings for verbs (in .npy format)'
    )
    parser.add_argument(
        "--out-arg-embed",
        nargs='?', dest='arg_embed',
        default="../data/arg_embedded.npy",
        help="where to save embeddings for arguments (in .npy format)"
    )
    parser.add_argument(
        "--out-labels",
        nargs='?', dest="labels",
        default="../data/labels.npy",
        help='where to save labels'
    )
    parser.add_argument(
        "--out-feature-model",
        nargs="?", dest="feature_model",
        default="./feature_model.pckl",
        help='where to save feature model'
    )
    parser.add_argument(
        "--out-label-encoder",
        nargs="?", dest='label_encoder',
        default="./label_encoder.pckl",
        help="where to save label encoder"
    )
    parser.add_argument(
        "--out-feature-encoder",
        nargs='?', dest='feature_encoder',
        default="./feature_encoder.pckl",
        help='where to save feature encoder'
    )
    args = parser.parse_args()

    print("1. Reading files..")
    print(f"\tCorpus file: ({args.corpus_file})", end="....", flush=True)

    with open(args.corpus_file, 'r') as f:
        examples = json.load(f)

    print("Ok", flush=True)
    print(f'\tLing data file: ({args.ling_data_file})', end="....", flush=True)

    with open(args.ling_data_file, 'rb') as f:
        ling_data = pickle.load(f)
    ling_data_cache = {k: v for k, v in ling_data}

    print("Ok", flush=True)
    print(f"\tEmbeddings file: ({args.embeddings_file})", end="....", flush=True)

    embeddings = KeyedVectors.load_word2vec_format(
        args.embeddings_file, binary=False)

    print("Ok", flush=True)
    print("..Done!")

    print("2. Initializing feature models")
    feature_model = FeatureModelDefault()
    tool = FeatureModelingTool(ling_cache=ling_data_cache, feature_extractor=feature_model)
    print("..Done!")

    print("3. Extracting features")
    dataframe = generate_feature_dataframe(examples, tool)
    print("..Done!")

    print("4. Clearing and role normalizing")
    dataframe = clear_unfrequent_roles(dataframe)
    dataframe, n_roles = normalize_roles(dataframe, DEFAULT_REPL_ROLES)
    print(f"Number of roles: {n_roles}")
    print("..Done!")

    X, y = split_to_x_y(dataframe)
    print("")
    print("-"*40)
    print(f"Shapes: X: {X.shape}, Y: {y.shape}")
    print("-"*40)
    print("")

    print("5. Feature and label processing")
    print("\tLabel encoding...", end="", flush=True)
    y, label_encoder = train_label_encoder(y)
    print("Ok", flush=True)

    print("\tArgument and Predicate embedding...", end="", flush=True)
    embedded_args = embed_single(embeddings, dataframe.arg_lemma)
    embedded_verbs = embed_single(embeddings, dataframe.pred_lemma)
    print("Ok", flush=True)

    print("\tFeature vectorizing...", end="", flush=True)
    categorical, non_categorical = get_feature_names(dataframe)
    one_hot, vectorizer = train_feature_vectorizer(dataframe, categorical)
    plain_features = generate_features_array(dataframe, one_hot, non_categorical)
    print("Ok", flush=True)
    print("..Done!")

    print("6. Saving models")
    print("\tFeature Model...", end="", flush=True)
    with open(args.feature_model, 'wb') as f:
        pickle.dump(feature_model, f)
    print("Ok", flush=True)
    
    print("\tLabel Encoder...", end="", flush=True)
    with open(args.label_encoder, "wb") as f:
        pickle.dump(label_encoder, f)
    print("Ok", flush=True)

    print("\tFeature Encoder...", end="", flush=True)
    with open(args.feature_encoder, "wb") as f:
        pickle.dump(vectorizer, f)
    print("Ok", flush=True)
    print("..Done!")

    print("7. Saving results")
    print("\tLabels...", end="", flush=True)
    np.save(args.labels, y)
    print("Ok", flush=True)
    
    print("\tEmbeddings...", end="", flush=True)
    np.save(args.arg_embed, embedded_args)
    np.save(args.verb_embed, embedded_verbs)
    print("Ok", flush=True)

    print("\tPlain Features...", end="", flush=True)
    np.save(args.plain_features, plain_features)
    print("Ok", flush=True)
    print("..Done!")
