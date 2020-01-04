from gensim.models import KeyedVectors

from tqdm import tqdm
import pandas as pd
import numpy as np
import multiprocessing as mp
import argparse
import pickle
import json
import time
import os
import sys

sys.path.append('../')

from feature_modeling import FeatureModelingTool
from isanlp_srl_framebank.convert_corpus_to_brat import make_text
from isanlp_srl_framebank.processor_srl_framebank import FeatureModelDefault, FeatureModelUnknownPredicates
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
        data_for_pandas_ex['tokens'] = example[3]
        data_for_pandas_ex['arg_address'] = example[4]
        data_for_pandas_ex['prd_address'] = example[5]
        for elem in example[0]:
            for subelem in elem[:3]:
                if subelem is not None:
                    data_for_pandas_ex.update(subelem)

        data_for_pandas.append(data_for_pandas_ex)

    pd_data = pd.DataFrame(data_for_pandas)
    pd_data = pd_data.sample(frac=1)
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


def get_feature_names(dataframe, not_categ_features={'arg_address', 'ex_id', 'rel_pos', 'arg_lemma'}):
    categ_feats = [
        name for name in dataframe.columns if name not in not_categ_features]
    not_categ = ['rel_pos']

    return categ_feats, not_categ


def generate_features_array(dataframe, one_hot_feats, not_categ):
    not_categ_columns = np.concatenate(
        tuple(dataframe.loc[:, e].values.reshape(-1, 1) for e in not_categ), axis=1)
    plain_features = np.concatenate((one_hot_feats, not_categ_columns), axis=1)
    return plain_features

# del pd_data['pred_lemma']
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
        '--known-preds',
        default=True
    )
    parser.add_argument('--output-dir')

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

    print("2. Initializing feature models")
    if args.known_preds:
        feature_model = FeatureModelDefault()
    else:
        feature_model = FeatureModelUnknownPredicates()
        
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

    print("6. Saving models")
    print("\tFeature Model...", end="", flush=True)
    with open(os.path.join(args.output_dir, 'feature_model.pckl'), 'wb') as f:
        pickle.dump(feature_model, f)
    print("Ok", flush=True)
    
    print('Saving features...')
    with open(os.path.join(args.output_dir, 'features.pckl'), 'wb') as f:
        pickle.dump(dataframe, f)
    print('Done.')
    