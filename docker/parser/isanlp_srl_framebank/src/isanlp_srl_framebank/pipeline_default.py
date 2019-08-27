from isanlp.processor_remote import ProcessorRemote
from isanlp.processor_syntaxnet_remote import ProcessorSyntaxNetRemote
from isanlp import PipelineCommon
from isanlp.ru.converter_mystem_to_ud import ConverterMystemToUd


class PipelineDefault:
    def __init__(self, address_morph, address_syntax, address_srl):
        self._ppl = PipelineCommon([(ProcessorRemote(address_morph[0], address_morph[1], 'default'), 
                                     ['text'], 
                                     {'tokens' : 'tokens', 
                                      'sentences' : 'sentences',
                                      'postag' : 'mystem_postag',
                                      'lemma' : 'lemma'}),
                                     (ProcessorSyntaxNetRemote(address_syntax[0], address_syntax[1]),
                                      ['tokens', 'sentences'],
                                      {'syntax_dep_tree' : 'syntax_dep_tree'}),
                                     (ConverterMystemToUd(), 
                                      ['mystem_postag'],
                                      {'morph' : 'morph',
                                       'postag' : 'postag'}),
                                     (ProcessorRemote(address_srl[0], address_srl[1], 'default'),
                                      ['tokens', 'postag', 'morph', 'lemma', 'syntax_dep_tree'],
                                      {'srl' : 'srl'})])
        self._name = 'default'
    
    def __call__(self, *args, **kwargs):
        return self._ppl(*args, **kwargs)
    
    def get_processors(self):
        return self._ppl.get_processors()
    
