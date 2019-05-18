from .preposition_extract import in_complex_preposition, complex_preposition_child, get_children


class ArgumentExtractor:
    ARGUMENT_POSTAGS = {
        'NOUN',
        'PRON',
        'ADJ',
        'PROPN'
    }
    
    LN_HOMOGENEOUS = 'conj'
    LN_AGENT = 'nsubj'
    LN_DIRECTION = 'nmod'

    def __call__(self, pred_number, postags, morphs, lemmas, syntax_dep_tree):
        """ Return list of arguments for predicate in the sentence """

        arguments = self._get_own_args(pred_number, postags, morphs, lemmas, syntax_dep_tree)
        expanded_args = self._get_conj_args(pred_number, postags, morphs, lemmas, syntax_dep_tree)

        return arguments + expanded_args
        
        
    def _get_own_args(self, pred_number, postags, morphs, lemmas, syntax_dep_tree):
        """ Return list of arguments for predicate in the sentence """

        arguments = []
        children = get_children(pred_number, syntax_dep_tree)
            
        for child_number in children:
            lemma_child  = lemmas[child_number]
            postag_child = postags[child_number]
                
            prep = in_complex_preposition(child_number, postags, morphs, lemmas, syntax_dep_tree)
            if prep:
                arg_num = complex_preposition_child(prep, syntax_dep_tree)
                if arg_num is None:
                    continue
                        
                if postags[arg_num] not in self.ARGUMENT_POSTAGS:
                    continue
                    
                arguments.append(arg_num)
            else:
                if postags[child_number] not in self.ARGUMENT_POSTAGS:
                    continue
                    
                arguments.append(child_number)

        return arguments

    
    def _get_conj_args(self, pred_number, postags, morphs, lemmas, syntax_dep_tree):

        def is_homogen_pair(a, b):
            synt_b = syntax_dep_tree[b]
            return synt_b.link_name == self.LN_HOMOGENEOUS and synt_b.parent == a

        def find_linkname(predicate, linkname):
            return [agent for agent in syntax_dep_tree if agent.link_name == linkname and agent.parent == predicate]

        def expand_linkname(arguments, linkname):
            if not find_linkname(pred_number, linkname):
                first_argument = [arg for arg in arguments if
                                  syntax_dep_tree[arg].link_name == linkname][0]
                return [first_argument] + [argument for argument, arg_synt in enumerate(syntax_dep_tree) if
                                           is_homogen_pair(first_argument, argument)]
            return []

        conj_predicates = [number_c for number_c, synt_c in enumerate(syntax_dep_tree)
                           if is_homogen_pair(number_c, pred_number)]

        result = []
        for predicate in conj_predicates:
            arguments = self._get_own_args(predicate, postags, morphs, lemmas, syntax_dep_tree)
            result += expand_linkname(arguments, self.LN_AGENT)
            result += expand_linkname(arguments, self.LN_DIRECTION)

        return result
    