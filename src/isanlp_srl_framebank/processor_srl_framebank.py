import json
import logging
import os
import pickle

import numpy as np
import tensorflow.lite as tfl
from bert_serving.client import BertClient
from deeppavlov.models.embedders.elmo_embedder import ELMoEmbedder
from gensim.models import KeyedVectors
from isanlp.annotation import Event, TaggedSpan
from scipy.optimize import linear_sum_assignment

from .argument_extractor import ArgumentExtractor
from .predicate_extractor import PredicateExtractor
from .preposition_extract import extract_preposition

logger = logging.getLogger('isanlp_srl_framebank')
logging.basicConfig(filename="sample.log", level=logging.INFO)


# TODO: limit role confidence
# TODO: fix logging
# TODO: Evaluation script for parser (in addition to evaluation of models)
# TODO: more features for predicate
# TODO: non-core roles
# TODO: refactor


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
        self._model_dir_path = model_dir_path
        # -----------------------Using TFLite------------------------
        logger.info('Loading the model...')
        self._model = tfl.Interpreter(os.path.join(model_dir_path, 'neural_model.tflite'))
        logger.info('Done.')
        self._model.allocate_tensors()
        self._input_idxs = {x['name']: x['index'] for x in self._model.get_input_details()}
        self._output_idx = self._model.get_output_details()[0]['index']
        # -----------------------------------------------------------

        logger.info('Loading embeddings...')
        self._embeddings_type = embeddings_type
        self._embeddings = self._load_embeddings()
        logger.info('Done.')

        logger.info('Loading feature models...')
        with open(os.path.join(model_dir_path, 'feature_encoder.pckl'), 'rb') as f:
            self._categorical_encoder = pickle.load(f)

        with open(os.path.join(model_dir_path, 'label_encoder.pckl'), 'rb') as f:
            self._roles = pickle.load(f)

        with open(os.path.join(model_dir_path, 'feature_model.pckl'), 'rb') as f:
            self._feature_model = pickle.load(f)
        logger.info('Done.')

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

        elif self._embeddings_type == 'bert':
            bert_host = os.environ['BERT_HOST']
            return BertClient(ip=bert_host)

    def _vectorize_embeddings(self, feature_embeddings, sent_embeddings, position):
        def _w2v():
            results = []
            for word in feature_embeddings.values():
                if word in self._embeddings:
                    results.append(self._embeddings[word])
                else:
                    results.append(np.zeros((self._embeddings.vector_size,)))

            return np.concatenate(results)

        def _contextual():
            return np.array(sent_embeddings[min(position, len(sent_embeddings) - 1)])

        if self._embeddings_type == 'w2v':
            return _w2v()

        if self._embeddings_type in ['elmo', 'bert']:
            return _contextual()

    def _embed_sentence(self, tokens):
        def _w2v():
            return []

        def _elmo():
            return self._embeddings([tokens])[0]

        def _bert():
            res = self._embeddings.encode([tokens], is_tokenized=True, show_tokens=False)
            return res[0, 1:len(tokens) + 1, :]

        if self._embeddings_type == 'w2v':
            return _w2v()

        if self._embeddings_type == 'elmo':
            return _elmo()

        if self._embeddings_type == 'bert':
            return _bert()


class ProcessorSrlFramebank:
    def __init__(self,
                 model_dir_path,
                 known_preds_embeddings_type,
                 predicate_extractor=PredicateExtractor(),
                 argument_extractor=ArgumentExtractor(),
                 enable_model_for_unknown_predicates=False,
                 unknown_preds_embeddings_type=None,
                 enable_global_scoring=True,
                 delay_init=False,
                 threshold=0.1):

        self.model_dir_path = model_dir_path
        self._predicate_extractor = predicate_extractor
        self._argument_extractor = argument_extractor
        self.enable_model_for_unknown_predicates = enable_model_for_unknown_predicates
        self._enable_global_scoring = enable_global_scoring
        self._known_preds_embeddings_type = known_preds_embeddings_type
        self._unknown_preds_embeddings_type = unknown_preds_embeddings_type
        
        self._threshold = threshold

        self._model_known_preds = None
        self._model_unknown_preds = None
        if not delay_init:
            self.init()

    def init(self):
        if not self._model_known_preds:
            with open(os.path.join(self.model_dir_path, 'known_preds.json'), 'r', encoding='utf8') as f:
                self._known_preds = json.load(f)  # set(json.load(f))

            logger.info('Loading the model for known predicates...')
            self._model_known_preds = ModelProcessorSrlFramebank(os.path.join(self.model_dir_path, 'known_preds'),
                                                                 embeddings_type=self._known_preds_embeddings_type)
            logger.info('Model for known predicates is loaded.')

        if not self._model_unknown_preds:
            path_model_unknown_preds = os.path.join(self.model_dir_path, 'unknown_preds')
            if self.enable_model_for_unknown_predicates and os.path.exists(path_model_unknown_preds):
                self._model_unknown_preds = ModelProcessorSrlFramebank(path_model_unknown_preds,
                                                                       embeddings_type=self._unknown_preds_embeddings_type)
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
                tmp = self._vectorize_categorical(model, feat[0])
                vectorized_feats.append(tmp)
            if feat[1]:
                tmp = model._vectorize_embeddings(feat[1], sent_embed, feat[3])
                vectorized_feats.append(tmp)
            if feat[2]:
                tmp = np.array(list(feat[2].values()))
                vectorized_feats.append(tmp)

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
        args, verbs, plain_features = vectors
        logger.info('predicting')
        #logger.info(str(model._model))
        #logger.info(vectors[0].shape)

        model._model.set_tensor(model._input_idxs['arg_embed'], np.array(args, dtype=np.float32))
        model._model.set_tensor(model._input_idxs['pred_embed'], np.array(verbs, dtype=np.float32))
        model._model.set_tensor(model._input_idxs['input_categorical'], np.array(plain_features, dtype=np.float32))
        model._model.invoke()
        res = model._model.get_tensor(model._output_idx)

        res = self._apply_threshold(res, threshold=self._threshold)
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

            if self._model_known_preds._embeddings_type in ('elmo', 'bert'):
                sent_tokens = [token.text for token in tokens[tokens_counter:tokens_counter + len(sent_postag)]]
                tokens_counter += tokens_counter + len(sent_postag)
                sent_embeddings = self._model_known_preds._embed_sentence(sent_tokens)
            else:
                sent_embeddings = [], []

            preds = self._predicate_extractor(sent_postag, sent_morph,
                                              sent_lemma, sent_syntax_dep_tree)

            sent_arg_roles = []
            for pred in preds:
                pred_arg_roles = []

                pred_lemma = sent_lemma[pred]
                if pred_lemma in self._known_preds.keys() or self._model_unknown_preds is None:
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
                        if np.isclose(np.sum(arg_roles[i]), 0.):
                            continue
                            
                        arg_role_name = model._roles.inverse_transform(arg_roles[i])
                        
                        if pred_lemma in self._known_preds.keys():
                            if arg_role_name in self._known_preds.get(pred_lemma):
                                clean_args.append(arg)
                                clean_roles.append(arg_roles[i])
                        elif arg_role_name != 'non_rel':
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
