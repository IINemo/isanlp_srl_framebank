PREP_POSTAGS = ('ADP',)

COMPLEX_PREP_PARTS = ('в', 'во',)

COMPLEX_PREP_CASES = {
    'течение': 'Acc',
    'продолжение': 'Acc',
    'заключение': 'Acc',
    'отсутствие': 'Acc',
    'отличие': 'Acc',
    'преддверие': 'Loc',
    'избежание': 'Acc',
}


def extract_preposition(arg_number, postags, morph, lemmas, syntax_dep_tree):
    """ Return list of prepositions for an argument in the sentence """

    children = [child_number for child_number, child_syntax in enumerate(syntax_dep_tree) if
                child_syntax.parent == arg_number]
    siblings = [child_number for child_number, child_syntax in enumerate(syntax_dep_tree) if
                child_syntax.parent == syntax_dep_tree[arg_number].parent]
    prepositions = list()

    for child_number in children:
        lemma_child, postag_child = lemmas[child_number], postags[child_number]

        if postag_child in PREP_POSTAGS:
            prepositions.append(child_number)

    for child_number in siblings:
        lemma_child, postag_child = lemmas[child_number], postags[child_number]

        if postag_child in PREP_POSTAGS:
            prepositions.append(child_number)

            if lemma_child in COMPLEX_PREP_PARTS:
                prep_candidates = [cand_number for cand_number in range(child_number, arg_number)]
                for candidate_number in prep_candidates:
                    lemma_candidate, postag_candidate = lemmas[candidate_number], postags[candidate_number]
                    if lemma_candidate in COMPLEX_PREP_CASES:
                        prepositions.append(candidate_number)

    return prepositions
