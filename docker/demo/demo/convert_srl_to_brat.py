from isanlp.processor_remote import ProcessorRemote
from itertools import count


class Relation:
    def __init__(self, parsed_json=None):
        if parsed_json:
            self.name, self.type, self.pair = parsed_json

    def convert(self):
        return [self.name, self.type, self.pair]


class Entity:
    def __init__(self, parsed_json=None):
        if parsed_json:
            self.name, self.type, [self.position] = parsed_json

    def convert(self):
        return [self.name, self.type, [self.position]]


class Attribute:
    _ids = count(0)

    def __init__(self, token_name=None, parsed_json=None):
        if parsed_json:
            self.name, self.type, self.token, self.body = parsed_json
        else:
            self.name = 'F%s' % next(self._ids)
            self.token = token_name

    def convert(self):
        return [self.name, self.type, self.token, self.body]


class Normalization(Attribute):
    def convert(self):
        return [self.name, "", self.token, "", "", self.body]


def createType(category, tp):
    return "{}~{}".format(category, tp)


class SrlConverter:
    def __call__(self, proc_res):
        Attribute._ids = count(0)
        coll_data_sem, doc_data_sem = self.get_sem_markup(proc_res)
        coll_data, doc_data = self.get_synt_markup(proc_res)

        return {"coll_data": coll_data,
                "doc_data": doc_data,
                "coll_data_sem": coll_data_sem,
                "doc_data_sem": doc_data_sem
                }

    def get_sem_markup(self, proc_res):
        doc_data_sem = self.convert_sem_attributes(srl=proc_res['srl'],
                                                   sentences=proc_res['sentences'],
                                                   tokens=proc_res['tokens'],
                                                   text=proc_res['text'])

        coll_data_sem = MarkupSem(doc_data_sem).convert()

        return coll_data_sem, doc_data_sem

    def get_synt_markup(self, proc_res):
        doc_data = self.convert_synt_attributes(syntax_dep_tree=proc_res['syntax_dep_tree'],
                                                sentences=proc_res['sentences'],
                                                tokens=proc_res['tokens'],
                                                postags=proc_res['postag'],
                                                morphs=proc_res['morph'],
                                                lemmas=proc_res['lemma'],
                                                text=proc_res['text'])

        coll_data = MarkupSynt(doc_data).convert()

        return coll_data, doc_data

    def convert_sem_attributes(self, srl, sentences, tokens, text):

        entities = list()
        relations = list()

        for sentence in range(len(sentences)):
            sentence_begin = sentences[sentence].begin

            for event in srl[sentence]:
                predicate = Entity()
                predicate.number = sentence_begin + event.pred[0]
                predicate.token = tokens[predicate.number]
                predicate.position = [predicate.token.begin, predicate.token.end]
                predicate.type = 'Predicate~'
                predicate.name = 'P%d' % predicate.position[0]
                entities.append(predicate.convert())

                number_args = len(event.args)

                for arg in range(number_args):
                    argument = Entity()
                    argument.number = sentence_begin + event.args[arg].begin
                    argument.token = tokens[argument.number]
                    argument.position = [argument.token.begin, argument.token.end]
                    argument.type = 'Argument~'
                    argument.name = 'A%d' % argument.position[0]
                    entities.append(argument.convert())

                    relation = Relation()
                    relation.name = createType(predicate.position[0], argument.position[0])
                    relation.pair = [["Predicate", predicate.name],
                                     ["Argument", argument.name]]
                    relation.type = createType('Role', '%s' % event.args[arg].tag)
                    relations.append(relation.convert())

        return {"text": text,
                "relations": relations,
                "entities": entities
                }

    def convert_synt_attributes(self, syntax_dep_tree, sentences, tokens, postags, morphs, lemmas, text):

        attributes = list()
        normalizations = list()
        entities = list()
        relations = list()

        for sentence in range(len(sentences)):
            sentence_begin = sentences[sentence].begin
            sentence_length = len(syntax_dep_tree[sentence])

            for token in range(sentence_length):
                entity = Entity()
                entity.number = sentence_begin + token
                entity.token = tokens[entity.number]
                entity.lemma = lemmas[sentence][token]
                entity.position = [entity.token.begin, entity.token.end]
                entity.address = {"sentence": sentence, "token": token}
                entity.type = createType('Token', '%s' % postags[sentence][token])
                entity.name = 'T%d' % entity.position[0]
                entities.append(entity)

        for entity in entities:
            sentence, token = entity.address["sentence"], entity.address["token"]
            parent_number = syntax_dep_tree[sentence][token].parent
            if parent_number > -1:
                parent = [ent for ent in entities if ent.address["sentence"] == entity.address["sentence"] and ent.address["token"] == parent_number][0]
                relation = Relation()

                relation.name = createType('S%d' % parent.position[0], entity.position[0])
                relation.type = 'Syntax~'
                relation.pair = [["Parent", parent.name],
                                 ["Child", entity.name]]

                relations.append(relation.convert())

            for tp, body in morphs[sentence][token].items():
                new_attr = Attribute(token_name=entity.name)
                if 'POS' in tp:
                    new_attr.type = createType('postag', 'postag')
                else:
                    new_attr.type = createType('Morph', tp)
                new_attr.body = body
                attributes.append(new_attr.convert())

            normalization = Normalization(token_name=entity.name)
            normalization.body = entity.lemma
            normalizations.append(normalization.convert())

        return {"text": text,
                "attributes": attributes,
                "normalizations": normalizations,
                "relations": relations,
                "entities": [entity.convert() for entity in entities]}


class MarkupSem:
    def __init__(self, parsed_json):
        self.entities = [Entity(ent) for ent in parsed_json["entities"]]
        self.relations = [Relation(rel) for rel in parsed_json["relations"]]

    def decoratorSemanticEntityType(self, obj, dec):
        if obj.type == "Predicate~":
            dec.update({"bgColor": "#abf395", "borderColor": "darken", "labels": ["Predicate", "Pred"]})
        elif obj.type == "Argument~":
            dec.update({"bgColor": "#f39191", "borderColor": "darken", "labels": ["Argument", "Arg"]})

        return dec

    def decoratorSemanticRelationType(self, obj, dec):
        if obj.type[:4] == "Role":
            dec.update({"color": "red", "labels": [obj.type[5:]], "type": obj.type})
        elif obj.type[:8] == "Semantic":
            dec.update({"dashArray": "3,3", "color": "green", "labels": [obj.tp]})

        return dec

    def create_args(self):
        return {
            "args": [{"role": "Predicate", "targets": ["Argument~"]}, {"role": "Argument", "targets": ["Argument~"]}]}

    def convert(self):
        entity_types = list()
        relation_types = list()

        for entity in self.entities:
            entity_types.append(self.decoratorSemanticEntityType(entity, {'type': entity.type}))

        entity_types = list({v['type']: v for v in entity_types}.values())

        for relation in self.relations:
            relation_types.append(self.decoratorSemanticRelationType(relation, self.create_args()))

        return {"entity_types": entity_types,
                "relation_types": relation_types
                }


class MarkupSynt:
    def __init__(self, parsed_json):
        self.entities = [Entity(ent) for ent in parsed_json["entities"]]
        self.relations = [Relation(rel) for rel in parsed_json["relations"]]
        self.attributes = [Attribute(parsed_json=attr) for attr in parsed_json["attributes"]]

    def createAttrType(self, type, name):
        return {"type": type,
                "name": name,
                "unused": False,
                "values": []}

    def createAttributeTypes(self, attributes):
        attrTypes = list()

        memo = dict()
        for attr in attributes:
            if attr.type == "lemma":
                continue

            tpName = attr.type.split('~')[1]

            if tpName not in memo:
                memo[tpName] = [attr.body]
                attrTypes.append(self.createAttrType(attr.type, tpName))
            else:
                memo[tpName].append(attr.body)

        for attrType in attrTypes:
            vals = dict()

            types = memo[attrType["name"]]

            if type(types) is str:
                vals.update({types: {"glyth": ""}})
            else:
                for t in types:
                    vals.update({t: {"glyth": ""}})

            attrType["values"] = vals

        return attrTypes

    def decoratorSyntaxEntityType(self, obj, dec):
        if obj.type == 'Token~':
            label = ['P']
        else:
            dirty_label = obj.type.split('~')[1]
            if dirty_label == 'PRON':
                label = ['Pr']
            else:
                label = dirty_label[:1]

        dec.update({"bgColor": "#7fa2ff", "borderColor": "darken", "labels": label})
        return dec

    def decoratorSyntaxRelationType(self, obj, dec):

        if obj.type == "fixed":
            dec.update({"color": "red", "labels": ["Syntax", "Synt", "S"]})
        else:
            dec.update({"color": "black", "labels": [obj.type] if obj.type else ["Syntax", "Synt", "S"]})

        return dec

    def convert(self):
        entity_attribute_types = self.createAttributeTypes(self.attributes)
        entity_types = list()
        relation_types = list()

        for entity in self.entities:
            entity_types.append(self.decoratorSyntaxEntityType(entity, {'type': entity.type}))

        for relation in self.relations:
            relation_types.append(self.decoratorSyntaxRelationType(relation, {}))

        return {'entity_attribute_types': entity_attribute_types,
                'entity_types': entity_types,
                'relation_types': relation_types}

