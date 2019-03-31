from . import annotation
import re
import io
import os


class BratBase(object):
    def __init__(self, cat):
        self.bid = ""
        self.tp = ""
        self.cat = cat
    
    def __str__(self):
        return "{}{}\t{}".format(self.cat, self.bid, self.tp)
        

class BratEntity(BratBase, annotation.Span):
    def __init__(self):
        super(BratEntity, self).__init__('T')
        self.form = ""
        
    def __str__(self):
        return "{} {} {}\t{}".format(super(BratEntity, self).__str__(), 
                                    self.begin, 
                                    self.end, 
                                    self.form)


class BratRealtion(BratBase):
    def __init__(self):
        super(BratRealtion, self).__init__('R')
        self.head = ""
        self.dep = ""
        
    def __str__(self):
        return "{} Arg1:T{} Arg2:T{}".format(super(BratRealtion, self).__str__(),
                                           self.head,
                                           self.dep)
    

"""Loads brat annotations from file. Returns entities, relations."""
def load_brat_annotation_file(file_path):
    fl = io.open(file_path, 'r', encoding = 'utf8')
    
    annots_entities = list()
    annots_relations = list()
    
    for line in fl:
        cols = line.strip().split(u'\t')
        
        annot_id = cols[0]
        annot_type = annot_id[0]
        
        if annot_type == u'R':
            annot = BratRealtion()
            annot.bid = annot_id[1:]
            
            annot_props = re.split(u" ", cols[1])
            annot.tp = annot_props[0]
            
            arg_str = annot_props[1:]
            args = {e.split(u':')[0] : e.split(u':')[1] for e in arg_str}
            
            annot.head = args[u'Arg1'][1:]
            annot.dep = args[u'Arg2'][1:]
            
            annots_relations.append(annot)
            
        elif annot_type == u'T':
            annot = BratEntity() 
            annot.bid = annot_id[1:]
            
            annotProps = re.split(u";| ", cols[1])
            annot.tp = annotProps[0]
            annot.begin = int(annotProps[1])
            annot.end = int(annotProps[-1])
            annot.text = cols[2]
            
            annots_entities.append(annot)
            
        else:
            continue
         
    return annots_entities, annots_relations


def load_brat_annotation_dir(dir_path):
    annotations = dict()
    for file_name in os.listdir(dir_path):
        if not file_name.endswith(".ann"):
            continue
        
        more_ent, more_rel = load_brat_annotation_file(os.path.join(dir_path, 
                                                                    file_name))
        annotations[os.path.splitext(file_name)[0]] = {'entities' : more_ent, 
                                                       'relations' : more_rel}
    
    return annotations


def save_list_of_brat_objects(lst, output_file_path):
    with open(output_file_path, 'w') as f:
        for obj in lst:
            print >>f, obj
    