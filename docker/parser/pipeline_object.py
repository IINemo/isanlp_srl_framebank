from isanlp import PipelineCommon

from isanlp_srl_framebank.processor_srl_framebank import ProcessorSrlFramebank


def create_pipeline(delay_init=False):
    return PipelineCommon([(ProcessorSrlFramebank('/models'),
                            ['postag', 'morph', 'lemma', 'syntax_dep_tree'],
                            {0: 'srl'})],
                          name='default')
