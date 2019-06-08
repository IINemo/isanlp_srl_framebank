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
        result = list(set(arguments + expanded_args))

        possible_subject = self._get_subject(arguments, postags, syntax_dep_tree)
        if not possible_subject:
            possible_subject = self._get_adv_cascade_subject(pred_number, postags, morphs, lemmas, syntax_dep_tree)

        if possible_subject and possible_subject[1]:
            for i, n in enumerate(result):
                if n == possible_subject[0]:
                    result[i] = possible_subject[1]
        elif possible_subject and possible_subject[0] not in result:
            result.append(possible_subject[0])

        return result

    def _get_own_args(self, pred_number, postags, morphs, lemmas, syntax_dep_tree):

        arguments = []
        children = get_children(pred_number, syntax_dep_tree)

        for child_number in children:

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
            return [token for token in syntax_dep_tree if token.link_name == linkname and token.parent == predicate]

        def expand_linkname(arguments, linkname):
            if not find_linkname(pred_number, linkname):
                first_argument = [arg for arg in arguments if
                                  syntax_dep_tree[arg].link_name == linkname]
                if first_argument:
                    first_argument = first_argument[0]
                    return [first_argument] + [argument for argument, arg_synt in enumerate(syntax_dep_tree) if
                                               is_homogen_pair(first_argument, argument)]
                else:
                    return [argument for argument, arg_synt in enumerate(syntax_dep_tree) if
                            is_homogen_pair(first_argument, argument)]
            return []

        conj_predicates = [number_c for number_c, synt_c in enumerate(syntax_dep_tree)
                           if is_homogen_pair(number_c, pred_number)]

        result = []
        for predicate in conj_predicates:
            arguments = self._get_own_args(predicate, postags, morphs, lemmas, syntax_dep_tree)
            result += expand_linkname(arguments, self.LN_AGENT)

        return result

    def _get_subject(self, arguments, postags, syntax_dep_tree):

        def _find_subject_name(subject):
            if postags[subject] != 'NOUN':
                return None

            namelink = ['name', 'appos']
            result = [i for i, token in enumerate(syntax_dep_tree) if
                      token.link_name in namelink and token.parent == subject]
            if not result:
                return None
            return result[0]

        for argument in arguments:
            if syntax_dep_tree[argument].link_name in self.LN_AGENT:
                subject = argument
                name = _find_subject_name(subject)
                if name:
                    second_name = _find_subject_name(name)
                    if second_name:
                        name = second_name
                return subject, name
        return []

    def _get_first_part(self, pred_number, syntax_dep_tree):
        """ Return the first verb of quasi-complex predicates """

        xcomplink = 'xcomp'
        if syntax_dep_tree[pred_number].link_name == xcomplink:
            return syntax_dep_tree[pred_number].parent
        return None

    def _get_adv_cascade_subject(self, pred_number, postags, morphs, lemmas, syntax_dep_tree):
        """ Return a subject for participle phrases """

        possible_subject = None
        first_part = self._get_first_part(pred_number, syntax_dep_tree)
        if first_part:
            possible_subject = self._get_subject(
                self._get_own_args(first_part, postags, morphs, lemmas, syntax_dep_tree), postags, syntax_dep_tree)
            if not possible_subject:
                conjunct = self._get_direct_link(first_part, syntax_dep_tree, self.LN_HOMOGENEOUS)
                if conjunct:
                    possible_subject = self._get_subject(
                        self._get_own_args(conjunct, postags, morphs, lemmas, syntax_dep_tree), postags,
                        syntax_dep_tree)
                    if not possible_subject:
                        advcl = self._get_direct_link(conjunct, syntax_dep_tree, 'advcl')
                        if advcl:
                            possible_subject = self._get_subject(
                                self._get_own_args(advcl, postags, morphs, lemmas, syntax_dep_tree), postags,
                                syntax_dep_tree)
                            if not possible_subject:
                                first_part = self._get_first_part(advcl, syntax_dep_tree)
                                if first_part:
                                    possible_subject = self._get_subject(
                                        self._get_own_args(first_part, postags, morphs, lemmas, syntax_dep_tree),
                                        postags, syntax_dep_tree)

        return possible_subject

    def _get_direct_link(self, pred_number, syntax_dep_tree, linkname):

        if syntax_dep_tree[pred_number].link_name != linkname:
            return None
        return syntax_dep_tree[pred_number].parent
