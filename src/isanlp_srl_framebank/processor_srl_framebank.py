from tensorflow.python.keras.models import load_model
from gensim.models import KeyedVectors
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import pickle
import json
import os
from collections import namedtuple

from .argument_extractor import ArgumentExtractor
from .predicate_extractor import PredicateExtractor
from .preposition_extract import extract_preposition

from isanlp.annotation import Event, TaggedSpan

import logging
logger = logging.getLogger('isanlp_srl_framebank')

#TODO: limit role confidence
#TODO: fix logging
#TODO: Evaluation script for parser (in addition to evaluation of models)
#TOOD: more features for predicate
#TODO: non-core roles
#TODO: refactor 

class FeatureModelDefault:
    def extract_features(self, pred, arg, postag, 
                         morph, lemma, syntax_dep_tree):
        arg_pos = postag[arg]
        arg_case = morph[arg].get('Case', '')
        pred_pos = postag[pred]
        arg_lemma = '{}_{}'.format(lemma[arg], arg_pos)
        pred_lemma = '{}_{}'.format(lemma[pred], pred_pos)
        syn_link_name = syntax_dep_tree[arg].link_name
        dist = 1. * abs(arg - pred) if pred != arg else 0.
        
        morph_features_arg = {name + '_arg' : morph[arg].get(name, '') 
                              for name in ['Aspect', 'Number', 
                                           'Tense', 'Valency', 
                                           'VerbForm', 'Animacy', 
                                           'Gender']}

        preposition_nums = extract_preposition(arg, postag, morph, lemma, syntax_dep_tree)
        prepos = '~'.join([lemma[e] for e in preposition_nums])
        
        features_categorical = {'dist' : dist, 
                                'arg_case' : arg_case,
                                'pred_pos' : pred_pos,
                                'arg_pos' : arg_pos,
                                'syn_link_name' : syn_link_name,
                                'pred_lemma' : pred_lemma,
                                'prepos' : prepos}
        features_categorical.update(morph_features_arg)
    
        features_noncat = {'rel_pos' : 1. if arg < pred else -1.}
        features_pred_lemma = {'pred_lemma' : pred_lemma}
        features_arg_lemma = {'arg_lemma' : arg_lemma}
        
        # Tuple (categorical, embeddings, continues)
        return [(None, features_arg_lemma, None),
                (None, features_pred_lemma, None),
                (features_categorical, None, features_noncat)]
    
    
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
    def __init__(self, model_dir_path):        
        self._model = load_model(os.path.join(model_dir_path, 'neural_model.h5'))
        self._model._make_predict_function()
            
        with open(os.path.join(model_dir_path, 'feature_encoder.pckl'), 'rb') as f:
            self._categorical_encoder = pickle.load(f)
            
        with open(os.path.join(model_dir_path, 'label_encoder.pckl'), 'rb') as f:
            self._roles = pickle.load(f)
        
        with open(os.path.join(model_dir_path, 'feature_model.pckl'), 'rb') as f:
            self._feature_model = pickle.load(f)
        

class ProcessorSrlFramebank:
    def __init__(self, 
                 model_dir_path,  
                 embeddings = None, 
                 predicate_extractor = PredicateExtractor(),
                 argument_extractor = ArgumentExtractor(), 
                 enable_model_for_unknown_predicates = True,
                 enable_global_scoring = True):
        if embeddings:
            self._embeddings = embeddings
        else:
            self._embeddings = KeyedVectors.load_word2vec_format(os.path.join(model_dir_path, 
                                                                              'embeddings.vec'), 
                                                                 binary = False)
            
        self._predicate_extractor = predicate_extractor
        self._argument_extractor = argument_extractor
        
        with open(os.path.join(model_dir_path, 'known_preds.json'), 'r', encoding = 'utf8') as f:
            self._known_preds = set(json.load(f))
        
        self._model_known_preds = ModelProcessorSrlFramebank(os.path.join(model_dir_path, 'known_preds'))
        logger.info('Model for known predicates is loaded.')
        
        path_model_unknown_preds = os.path.join(model_dir_path, 'unknown_preds')
        if enable_model_for_unknown_predicates and os.path.exists(path_model_unknown_preds):
            self._model_unknown_preds = ModelProcessorSrlFramebank(path_model_unknown_preds)
            logger.info('Model for unknown predicates is loaded.')
        else:
            self._model_unknown_preds = None
            
        self._enable_global_scoring = enable_global_scoring
    
    def _vectorize_embeddings(self, feature_embeddings):
        results = []
        for word in feature_embeddings.values():
            if word in self._embeddings:
                results.append(self._embeddings[word])
            else:
                results.append(np.zeros((self._embeddings.vector_size,)))
        
        return np.concatenate(results)
    
    def _vectorize_categorical(self, model, feature_categ):
        return model._categorical_encoder.transform(feature_categ).reshape(-1)

    def _vectorize_features(self, model, features):
        result = []
        for feat in features:
            vectorized_feats = []
            if feat[0]:
                vectorized_feats.append(self._vectorize_categorical(model, feat[0]))
            if feat[1]:
                vectorized_feats.append(self._vectorize_embeddings(feat[1]))
            if feat[2]:
                vectorized_feats.append(np.array(list(feat[2].values())))
            
            result.append(np.concatenate(vectorized_feats).reshape(1, -1))
        
        return result
    
    def _process_argument(self, model, pred, arg, sent_postag, 
                          sent_morph, sent_lemma, sent_syntax_dep_tree):
            
        features = model._feature_model.extract_features(pred, 
                                                         arg, 
                                                         sent_postag, 
                                                         sent_morph, 
                                                         sent_lemma, 
                                                         sent_syntax_dep_tree)
        
        vectors = self._vectorize_features(model, features)
        
        logger.info('predicting')
        res = model._model.predict(vectors)
        logger.info('Done.')
        
        return res
    
    def __call__(self, postag, morph, lemma, syntax_dep_tree):
        result = []
        for sent_num in range(len(postag)):
            sent_postag = postag[sent_num]
            sent_morph = morph[sent_num]
            sent_lemma = lemma[sent_num]
            sent_syntax_dep_tree = syntax_dep_tree[sent_num]
            
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
                if not args:
                    continue
                
                arg_roles = [self._process_argument(model,
                                                    pred, 
                                                    arg, 
                                                    sent_postag, 
                                                    sent_morph, 
                                                    sent_lemma, 
                                                    sent_syntax_dep_tree) 
                             for arg in args]
                
                if self._enable_global_scoring:
                    logger.info('Solving linear sum task.')
                    roles = [None for i in range(len(args))]
                    row_ind, col_ind = linear_sum_assignment(-1. * np.concatenate(arg_roles, axis = 0))
                    logger.info('Finished.')

                    for row_ind, col_ind in zip(row_ind, col_ind):
                        roles[row_ind] = model._roles.classes_[col_ind]

                    for i in range(len(args)):
                        arg_annot = TaggedSpan(tag = roles[i], begin = args[i], end = args[i])
                        pred_arg_roles.append(arg_annot)
                else:
                    logger.info('Using inverse transform.')
                    roles = model._roles.inverse_transform(np.concatenate(arg_roles))
                    roles = [str(e) for e in roles]
                    pred_arg_roles = []
                    for i in range(len(args)):
                        arg_annot = TaggedSpan(tag = roles[i], begin = args[i], end = args[i])
                        pred_arg_roles.append(arg_annot)
            
                sent_arg_roles.append(Event(pred = (pred,pred), args = pred_arg_roles))
            
            result.append(sent_arg_roles)
            
        return result
    