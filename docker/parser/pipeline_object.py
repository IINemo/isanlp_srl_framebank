from isanlp_srl_framebank.processor_srl_framebank import ProcessorSrlFramebank
from isanlp import PipelineCommon

PPL_SRL_FRAMEBANK = PipelineCommon([(ProcessorSrlFramebank('/models', 
                                                           enable_model_for_unknown_predicates=True, 
                                                           known_preds_embeddings_type='elmo',
                                                           unknown_preds_embeddings_type='elmo'),
                                     ['tokens', 'postag', 'morph', 'lemma', 'syntax_dep_tree'],
                                     {0 : 'srl'})
                                   ],
                                   name='default')