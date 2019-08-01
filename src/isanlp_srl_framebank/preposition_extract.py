# Complex prepositions of three words etc. в зависимости от

PREP_POSTAGS = ('ADP',)


def get_children(word_num, syntax_dep_tree):
    return [child_number for child_number, child_syntax in enumerate(syntax_dep_tree) 
            if child_syntax.parent == word_num]


class IsComplexPreposition:
    COMPLEX_PREPS = [
        ('в', 'течение', {'Acc'}),
        ('в', 'продолжение', {'Acc'}),
        ('в', 'заключение', {'Acc'}),
        ('в', 'отсутствие', {'Acc'}),
        ('в', 'отличие', {'Acc'}),
        ('в', 'преддверие', {'Loc'}),
        ('в', 'избежание', {'Acc'}),
        ('в', 'цель', {'Loc'}),
        ('в', 'ход', {'Loc'}),
        ('в', 'качество', {'Loc'}),
        ('в', 'период', {}),
        ('в', 'случай', {'Loc'}),
        ('в', 'отношение', {'Loc'}),
        ('в', 'направление', {'Loc'}),
        ('в', 'процесс', {'Loc'}),
        ('в', 'результат', {'Loc', 'Abl'}),
        ('в', 'интерес', {'Loc'}),
        ('в', 'сила', {'Acc'}),
        ('в', 'сторона', {}),
        ('в', 'условие', {}),
        ('во', 'имя', {}),
        ('во', 'время', {}),
        ('по', 'повод', {'Loc'}),
        ('вместе', 'c', {}),
        ('неподалеку', 'от', {}),
        ('совместно', 'с', {}),
        ('за', 'счет', {}),
        ('под', 'предлог', {'Ins'}),
        ('под', 'действие', {'Ins'}),
        ('под', 'влияние', {'Ins'}),
        ('по', 'отношение', {'Dat'}),
        ('по', 'мера', {'Loc'}),
        ('по', 'причина', {'Dat'}),
        ('при', 'условие', {'Loc'}),
        ('при', 'помощь', {'Abl', 'Loc'}),
        ('независимо', 'от', {}),
        ('несмотря', 'на', {}),
        ('смотря', 'по', {}),
        ('исходя', 'из', {}),
        ('судя', 'по', {}),
        ('на', 'основа', {'Loc'}),
        ('на', 'протяжение', {}),
        ('с', 'помощь', {'Ins'}),
        ('с', 'цель', {}),
        ('со', 'сторона', {'Gen'}),
        ('недалеко', 'от', {}),
        ('справа', 'от', {}),
        ('слева', 'от', {}),
        ('на', 'основание', {}),
        ('рядом', 'с', {}),
        ('в', 'честь', {})
    ]
    
    COMPLEX_PREPS_START = set(e[0] for e in COMPLEX_PREPS)
    COMPLEX_PREPS_MIDDLE = {e[1] : e[2] for e in COMPLEX_PREPS}
    
    @classmethod
    def __call__(cls, head_number, morph, lemma, syntax_dep_tree):
        head_lemma = lemma[head_number]

        if head_lemma not in cls.COMPLEX_PREPS_START:
            return False
        
        next_number = head_number + 1
        if next_number >= len(lemma):
            return False
        
        next_lemma = lemma[next_number]
        
        prep_case = cls.COMPLEX_PREPS_MIDDLE.get(next_lemma, None)
        if prep_case is None:
            return False
        
        next_morph = morph[next_number]
        
        if prep_case:
            if next_morph.get('Case', None) not in prep_case:
                return False
        
        return (head_number, next_number)
    
is_complex_preposition = IsComplexPreposition.__call__


def in_complex_preposition(word_num, postag, morph, lemma, syntax_dep_tree):
    word_postag = postag[word_num]
    
    if word_postag in PREP_POSTAGS:
        return is_complex_preposition(word_num, morph, lemma, syntax_dep_tree)
    elif word_num > 0:
        return is_complex_preposition(word_num - 1, morph, lemma, syntax_dep_tree)
    else:
        return False

    
def extract_preposition(arg_number, postags, morph, lemmas, syntax_dep_tree):
    """ Returns preposition for a word in the sentence """

    #TODO: fix duplication
    #TODO: there was a list of words for complex preposition, as we use the whole preposition as a feature
    
    children = get_children(arg_number, syntax_dep_tree)

    for child_number in children:
        lemma_child, postag_child = lemmas[child_number], postags[child_number]

        if postag_child in PREP_POSTAGS:
            complex_prep = in_complex_preposition(child_number, postags, morph, 
                                                  lemmas, syntax_dep_tree)
            if complex_prep:
                return complex_prep
            else:
                return child_number

    siblings = get_children(syntax_dep_tree[arg_number].parent, syntax_dep_tree)
            
    for child_number in siblings:
        lemma_child, postag_child = lemmas[child_number], postags[child_number]

        if postag_child in PREP_POSTAGS:
            complex_prep = is_complex_preposition(child_number, morph, lemmas, syntax_dep_tree)
            if complex_prep:
                return complex_prep
            else:
                return child_number

    return None


def complex_preposition_child(complex_prep, syntax_dep_tree):
    """ Returns sibling of complex preposition """
    
    children = (e for e in get_children(complex_prep[1], syntax_dep_tree) 
                if (e not in complex_prep))
    try:
        return next(children)
    except:
        return None
