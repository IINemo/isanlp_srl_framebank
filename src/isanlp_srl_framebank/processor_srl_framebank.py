from tensorflow.python.keras.models import load_model
from gensim.models import KeyedVectors
from deeppavlov.models.embedders.elmo_embedder import ELMoEmbedder
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import pickle
import json
import os
from collections import namedtuple
import sys

from .argument_extractor import ArgumentExtractor
from .predicate_extractor import PredicateExtractor
from .preposition_extract import extract_preposition

from isanlp.annotation import Event, TaggedSpan

import logging

logger = logging.getLogger('isanlp_srl_framebank')
logging.basicConfig(filename="sample.log", level=logging.INFO)


# TODO: limit role confidence
# TODO: fix logging
# TODO: Evaluation script for parser (in addition to evaluation of models)
# TOOD: more features for predicate
# TODO: non-core roles
# TODO: refactor

# select type of embeddings for each model here ('w2v' | 'elmo')
_KNOWN_PREDS_EMBEDDINGS = 'elmo'
_UNKNOWN_PREDS_EMBEDDINGS = 'w2v'

class FeatureModelDefault:
    def extract_features(self, pred, arg, postag,
                         morph, lemma, syntax_dep_tree):
        
        def prepos_lemma(position):
            if not position:
                return ''
            if type(position) == int:
                return lemma[position]
            return '~'.join([lemma[e] for e in position])
            
        arg_pos = postag[arg]
        arg_case = morph[arg].get('Case', '')
        pred_pos = postag[pred]
        arg_lemma = '{}_{}'.format(lemma[arg], arg_pos)
        pred_lemma = '{}_{}'.format(lemma[pred], pred_pos)
        syn_link_name = syntax_dep_tree[arg].link_name
        dist = 1. * abs(arg - pred) if pred != arg else 0.

        morph_features_arg = {name + '_arg': morph[arg].get(name, '')
                              for name in ['Aspect', 'Number',
                                           'Tense', 'Valency',
                                           'VerbForm', 'Animacy',
                                           'Gender']}

        prepos = prepos_lemma(extract_preposition(arg, postag, morph, lemma, syntax_dep_tree))

        features_categorical = {'dist': dist,
                                'arg_case': arg_case,
                                'pred_pos': pred_pos,
                                'arg_pos': arg_pos,
                                'syn_link_name': syn_link_name,
                                'pred_lemma': pred_lemma,
                                'prepos': prepos}
        features_categorical.update(morph_features_arg)

        features_noncat = {'rel_pos': 1. if arg < pred else -1.}
        features_pred_lemma = {'pred_lemma': pred_lemma}
        features_arg_lemma = {'arg_lemma': arg_lemma}

        # Tuple (categorical, embeddings, continues, position)
        return [(None, features_arg_lemma, None, arg),
                (None, features_pred_lemma, None, pred),
                (features_categorical, None, features_noncat, None)]


class FeatureModelUnknownPredicates:
    def extract_features(self, pred, arg, postag,
                         morph, lemma, syntax_dep_tree):
        features = FeatureModelDefault().extract_features(pred, arg,
                                                          postag, morph,
                                                          lemma,
                                                          syntax_dep_tree)
        del features[2][0]['pred_lemma']
        return features


class ModelProcessorSrlFramebank:
    def __init__(self, model_dir_path, embeddings_type):
        self._model_dir_path = model_dir_path,
        self._model = load_model(os.path.join(model_dir_path, 'neural_model.h5'))
        self._model._make_predict_function()
        self._embeddings_type = embeddings_type
        self._embeddings = self._load_embeddings()

        with open(os.path.join(model_dir_path, 'feature_encoder.pckl'), 'rb') as f:
            self._categorical_encoder = pickle.load(f)

        with open(os.path.join(model_dir_path, 'label_encoder.pckl'), 'rb') as f:
            self._roles = pickle.load(f)

        with open(os.path.join(model_dir_path, 'feature_model.pckl'), 'rb') as f:
            self._feature_model = pickle.load(f)
        
    def _load_embeddings(self):
        if type(self._model_dir_path) == tuple:
            self._model_dir_path = self._model_dir_path[0]
            
        if self._embeddings_type == 'w2v':
            embedding_path = os.path.join(self._model_dir_path, 'embeddings.vec')
            logger.info(f'Model has no embeddings! Loading {embedding_path}')
            return KeyedVectors.load_word2vec_format(embedding_path, binary=False)
        
        elif self._embeddings_type == 'elmo':
            embedding_path = "http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-wiki_600k_steps.tar.gz"
            logger.info(f'Model has no embeddings! Loading {embedding_path}')
            return ELMoEmbedder(embedding_path, elmo_output_names=['elmo'])
        
    def _vectorize_embeddings(self, feature_embeddings, sent_embeddings, position):
        def _w2v():
            results = []
            for word in feature_embeddings.values():
                if word in self._embeddings:
                    results.append(self._embeddings[word])
                else:
                    results.append(np.zeros((self._embeddings.vector_size,)))

            return np.concatenate(results)
                                
        def _elmo():
            return np.array(sent_embeddings[min(position, len(sent_embeddings)-1)])
                                
        if self._embeddings_type == 'w2v':
            return _w2v()
        
        if self._embeddings_type == 'elmo':
            return _elmo()
        
    def _embed_sentence(self, tokens):
        def _w2v():
            return []
                                
        def _elmo():
            return self._embeddings([tokens])[0]
        
        if self._embeddings_type == 'w2v':
            return _w2v()
        
        if self._embeddings_type == 'elmo':
            return _elmo()

class ProcessorSrlFramebank:
    def __init__(self,
                 model_dir_path,
                 predicate_extractor=PredicateExtractor(),
                 argument_extractor=ArgumentExtractor(),
                 enable_model_for_unknown_predicates=True,
                 enable_global_scoring=True,
                 delay_init=False):

        self.model_dir_path = model_dir_path
        self._predicate_extractor = predicate_extractor
        self._argument_extractor = argument_extractor
        self.enable_model_for_unknown_predicates = enable_model_for_unknown_predicates
        self._enable_global_scoring = enable_global_scoring

        self._model_known_preds = None
        self._model_unknown_preds = None
        if not delay_init:
            self.init()

    def init(self):
        if not self._model_known_preds:
            with open(os.path.join(self.model_dir_path, 'known_preds.json'), 'r', encoding='utf8') as f:
                self._known_preds = set(json.load(f))
                            
            logger.info('Loading the model for known predicates...')
            self._model_known_preds = ModelProcessorSrlFramebank(os.path.join(self.model_dir_path, 'known_preds'), 
                                                                 embeddings_type=_KNOWN_PREDS_EMBEDDINGS)
            logger.info('Model for known predicates is loaded.')

        if not self._model_unknown_preds:
            path_model_unknown_preds = os.path.join(self.model_dir_path, 'unknown_preds')
            if self.enable_model_for_unknown_predicates and os.path.exists(path_model_unknown_preds):
                self._model_unknown_preds = ModelProcessorSrlFramebank(path_model_unknown_preds, 
                                                                       embeddings_type=_UNKNOWN_PREDS_EMBEDDINGS)
                logger.info('Model for unknown predicates is loaded.')
            else:
                self._model_unknown_preds = None

    

    def _vectorize_categorical(self, model, feature_categ):
        return model._categorical_encoder.transform(feature_categ).reshape(-1)

    def _vectorize_features(self, model, features, sent_embed):
        result = []
        for feat in features:
            vectorized_feats = []
            if feat[0]:
                vectorized_feats.append(self._vectorize_categorical(model, feat[0]))
            if feat[1]:
                vectorized_feats.append(model._vectorize_embeddings(feat[1], sent_embed, feat[3]))
            if feat[2]:
                vectorized_feats.append(np.array(list(feat[2].values())))

            result.append(np.concatenate(vectorized_feats).reshape(1, -1))

        return result
    
    def _apply_threshold(self, predictions, threshold):
        predictions[predictions < threshold] = 0.
        
        return np.array(predictions)

    def _process_argument(self, model, pred, arg, sent_embed, sent_postag,
                          sent_morph, sent_lemma, sent_syntax_dep_tree):

        features = model._feature_model.extract_features(pred,
                                                         arg,
                                                         sent_postag,
                                                         sent_morph,
                                                         sent_lemma,
                                                         sent_syntax_dep_tree)
        assert features

        vectors = self._vectorize_features(model, features, sent_embed)

        logger.info('predicting')
        logger.info(str(model._model))
        logger.info(vectors[0].shape)
        res = model._model.predict(vectors)
        res = self._apply_threshold(res, threshold=.1)
        logger.info('Done.')

        return res

    def __call__(self, tokens, postag, morph, lemma, syntax_dep_tree):
        assert self._model_known_preds

        result = []
        tokens_counter = 0
        for sent_num in range(len(postag)):
            sent_postag = postag[sent_num]
            sent_morph = morph[sent_num]
            sent_lemma = lemma[sent_num]
            sent_syntax_dep_tree = syntax_dep_tree[sent_num]
            
            if 'elmo' in (self._model_known_preds._embeddings_type, self._model_unknown_preds._embeddings_type):
                sent_tokens = [token.text for token in tokens[tokens_counter:tokens_counter+len(sent_postag)]]
                tokens_counter += tokens_counter + len(sent_postag)
                if self._model_known_preds._embeddings_type == 'elmo':
                    sent_embeddings = self._model_known_preds._embed_sentence(sent_tokens)
                else:
                    sent_embeddings = self._model_unknown_preds._embed_sentence(sent_tokens)
                del sent_tokens
            else:
                sent_embeddings = [], []

            preds = self._predicate_extractor(sent_postag, sent_morph,
                                              sent_lemma, sent_syntax_dep_tree)

            sent_arg_roles = []
            for pred in preds:
                pred_arg_roles = []

                pred_lemma = sent_lemma[pred]
                if pred_lemma in self._known_preds or self._model_unknown_preds is None:
                    model = self._model_known_preds
                    logger.info('Using model for known predicates.')
                else:
                    model = self._model_unknown_preds
                    logger.info('Using model for unknown predicates.')

                args = self._argument_extractor(pred, sent_postag, sent_morph,
                                                sent_lemma, sent_syntax_dep_tree)

                if args:
                    arg_roles = [self._process_argument(model,
                                                        pred,
                                                        arg,
                                                        sent_embeddings,
                                                        sent_postag,
                                                        sent_morph,
                                                        sent_lemma,
                                                        sent_syntax_dep_tree)
                                 for arg in args]
                    
                    clean_args = []
                    clean_roles = []
                    for i, arg in enumerate(args):
                        if model._roles.inverse_transform(arg_roles[i]) != 'non_rel':
                            clean_args.append(arg)
                            clean_roles.append(arg_roles[i])
                    
                    args = clean_args
                    arg_roles = clean_roles
                    
                    if arg_roles:

                        if self._enable_global_scoring:
                            logger.info('Solving linear sum task.')
                            roles = [None for i in range(len(args))]
                            row_ind, col_ind = linear_sum_assignment(-1. * np.concatenate(arg_roles, axis=0))
                            logger.info('Finished.')

                            for row_ind, col_ind in zip(row_ind, col_ind):
                                roles[row_ind] = model._roles.classes_[col_ind]

                            for i in range(len(args)):
                                arg_annot = TaggedSpan(tag=roles[i], begin=args[i], end=args[i])
                                pred_arg_roles.append(arg_annot)
                        else:
                            logger.info('Using inverse transform.')
                            roles = model._roles.inverse_transform(np.concatenate(arg_roles))
                            roles = [str(e) for e in roles]
                            pred_arg_roles = []
                            for i in range(len(args)):
                                arg_annot = TaggedSpan(tag=roles[i], begin=args[i], end=args[i])
                                pred_arg_roles.append(arg_annot)

                        sent_arg_roles.append(Event(pred=(pred, pred), args=pred_arg_roles))

            result.append(sent_arg_roles)

        return result
