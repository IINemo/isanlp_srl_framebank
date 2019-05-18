class PredicateExtractor:
    POSTAGS_PRED = ('VERB', 'PRED')

    def __call__(self, postags, morph, lemmas, syntax_dep_tree):
        """ Return list of predicates in every sentence """

        predicates = []
        for word_number, postag in enumerate(postags):
            if postag in self.POSTAGS_PRED:
                predicates.append(word_number)

        return predicates


def log(tokens, lemmas, predicates, arguments):
    lettercounter = 0
    text = [token.text for token in tokens]

    length_sent = len(lemmas)
    try:
        print(' '.join(text[lettercounter:lettercounter + length_sent]),
              [(lemmas[predicate], lemmas[argument]) \
               for (predicate, argument) in (predicates, arguments)])

    except IndexError:
        pass
