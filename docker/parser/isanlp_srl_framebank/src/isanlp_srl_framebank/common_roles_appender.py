from isanlp.annotation import TaggedSpan


def complement_conj(res):
    def get_links_for_predicate(sent_synt, word_position):
        return set([child.link_name for n, child in enumerate(sent_synt) if child.parent == word_position])
    
    def get_srl_tag_for_word(sent_srl, predicate_position, word_position):
        return [[arg.tag for arg in event.args if arg.begin == word_position] 
                for event in sent_srl if event.pred[0] == predicate_position][0][0]
    
    def get_srl_tags_for_predicate(predicate_srl):
        return [arg.tag for arg in predicate_srl.args]
        
    CONJ_LINKS = ('nsubj', 'dobj', 'nmod')
    CONJ_TAGS = ('агенс', 'пациенс', 'место')
    predicates = [[event.pred[0] for event in sentence] for sentence in res['srl']]

    for sent in range(len(predicates)):
        sent_synt = res['syntax_dep_tree'][sent]
        sent_srl = res['srl'][sent]

        for pos_in_srl, pos_in_sent in enumerate(predicates[sent]):
            pred_a_synt = sent_synt[pos_in_sent]
            pred_a_srl = sent_srl[pos_in_srl]

            if pred_a_synt.link_name == 'conj':
                # pred_a - current predicate; pred_b - its 'conj' parent
                pred_a_links = get_links_for_predicate(sent_synt, pos_in_sent)
                pred_b_links = get_links_for_predicate(sent_synt, pred_a_synt.parent)
                
                pred_b_srl = [srl for srl in sent_srl if srl.pred[0] == pred_a_synt.parent][0]

                siblings = [(n_s, sibling) for n_s, sibling in enumerate(sent_synt) if
                            sibling.parent == pred_a_synt.parent and 
                            sibling.link_name in CONJ_LINKS]

                expanded_tags = get_srl_tags_for_predicate(pred_a_srl)

                for sibling in siblings:
                    link = sibling[1].link_name
                    tag = get_srl_tag_for_word(sent_srl, pred_a_synt.parent, sibling[0])
                    
                    if tag not in expanded_tags and tag in CONJ_TAGS:
                        new_arg = TaggedSpan(begin=sibling[0], end=sibling[0], tag=tag)
                        res['srl'][sent][pos_in_srl].args.append(new_arg)
    