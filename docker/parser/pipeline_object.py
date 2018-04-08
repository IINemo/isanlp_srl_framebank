from isanlp_srl_framebank.processor_srl_framebank import ProcessorSrlFramebank
from isanlp import PipelineCommon

PPL_SRL_FRAMEBANK = PipelineCommon([(ProcessorSrlFramebank('/models'),
                                     ['postag', 'morph', 'lemma', 'syntax_dep_tree'],
                                     {0 : 'srl'})
                                   ],
                                   name='default')
