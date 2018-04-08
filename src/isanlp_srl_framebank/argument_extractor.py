from .preposition_extract import COMPLEX_PREP_CASES


class ArgumentExtractor:
    ARGUMENT_POSTAGS = (
        'NOUN',
        'PRON',
        'ADJ',
    )

    def __call__(self, pred_number, postags, morphs, lemmas, syntax_dep_tree):
        """ Return list of arguments for predicate in the sentence """

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
