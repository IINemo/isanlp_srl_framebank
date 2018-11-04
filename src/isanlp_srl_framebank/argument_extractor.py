from .preposition_extract import COMPLEX_PREP_CASES


class ArgumentExtractor:
    ARGUMENT_POSTAGS = (
        'NOUN',
        'PRON',
        'ADJ',
    )

    LN_HOMOGENEOUS = 'conj'
    LN_AGENT = 'nsubj'
    LN_DIRECTION = 'nmod'

    def __call__(self, pred_number, postags, morphs, lemmas, syntax_dep_tree):
        """ Return list of arguments for predicate in the sentence """

        arguments = self._get_own_args(pred_number, postags, morphs, lemmas, syntax_dep_tree)
        expanded_args = self._get_conj_args(pred_number, postags, morphs, lemmas, syntax_dep_tree)

        return arguments + expanded_args

    def _get_own_args(self, pred_number, postags, morphs, lemmas, syntax_dep_tree):
        arguments = list()
        children = [child_number for child_number, child_syntax in enumerate(syntax_dep_tree) if
                    child_syntax.parent == pred_number]

        try:
            for child_number in children:
                lemma_child, postag_child = lemmas[child_number], postags[child_number]

                if postag_child in self.ARGUMENT_POSTAGS:

                    # check if there is a complex preposition
                    if lemma_child in COMPLEX_PREP_CASES and 'Case' in morphs[child_number]:
                        if morphs[child_number]['Case'] == COMPLEX_PREP_CASES[lemma_child]:
                            child_number = \
                                [nextword_number for nextword_number, nextword in enumerate(syntax_dep_tree) if
                                 nextword.parent == child_number][-1]

                if postags[child_number] in self.ARGUMENT_POSTAGS:
                    arguments.append(child_number)

        except IndexError:
            pass

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
