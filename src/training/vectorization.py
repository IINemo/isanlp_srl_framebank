import numpy
import json
import os

from sklearn.preprocessing import OneHotEncoder


def save_vectors(save_path, vectors):
    for name, vec in vectors.items():
        numpy.save(os.path.join(save_path, name), vec['value'])
        vec['value'] = ''
    
    with open(os.path.join(save_path, 'struct.json'), 'w') as f:
        json.dump(vectors, f, indent = 4)
        

class VectorizedDataset:
    def __init__(self, data):
        self._data = data
        for k, v in self._data.items():
            if len(v['value'].shape) > 1:
                continue
            else:
                v['value'] = v['value'].reshape(-1, 1)
                
        self.size = -1
        
        for k, v in self._data.items():
            val = v['value']
            
            if self.size == -1:
                self.size = val.shape[0]
            else:
                assert(self.size == val.shape[0])
                
            if v['encoder']:
                v['one_hot'] = OneHotEncoder(sparse = False)
                v['one_hot'].fit(val) 
    
    def __getitem__(self, name):
        return self._data[name]['value']
    
    def get_onehot(self, name):
        return self._data[name].get('one_hot')
    
        
def load_vectors(vectors_path):
    with open(os.path.join(vectors_path, 'struct.json'), 'r') as f:
        struct = json.load(f)
    
    for name, vec in struct.items():
        vec['value'] = numpy.load(os.path.join(vectors_path, name + '.npy'))
    
    return VectorizedDataset(struct)


def vectorize_everything(encoder, feat, groups, f_convert, zero, padding):
    sep = feat.get('separator')
    depth = feat.get('depth')
    
    full_vec = []

    for _, group in groups:
        vec = []

        for row_id in group.index[:padding]:
            subvec = []
            
            if sep:
                vals = group.loc[row_id, feat['name']]
                for j in range(min(depth, len(vals))):
                    subvec.append(f_convert(encoder, vals[j]))

                subvec += [zero for _ in range(depth - len(subvec))]
            else:       
                subvec = f_convert(encoder, group.loc[row_id, feat['name']])
            
            vec.append(numpy.array(subvec))
                
        vec += [numpy.zeros(vec[0].shape) for _ in range(padding - len(vec))]
        full_vec.append(vec)
            
    return numpy.array(full_vec), encoder


def get_vec_id(encoder, val):
    sval = str(val)
    if sval not in encoder:
        encoder[sval] = len(encoder) + 1
    
    return encoder[sval]


def make_embeddings(word_vectors, feat, data, padding):
    emb_size = word_vectors.vector_size
    
    def transform_func(encoder, val):
        try:
            new_emb = word_vectors.word_vec(val)
        except KeyError:
            new_emb = numpy.zeros((emb_size,))
        
        return new_emb
    
    return vectorize_everything('', feat, data, transform_func, 
                                numpy.zeros((emb_size,)), padding)


def vectorize_bool(feat, data, padding):
    return vectorize_everything('', feat, data, lambda _, val: int((val - 0.5) * 2.), 0, padding)


def vectorize_categorical(feat, data, padding):
    encoder = {}
    return vectorize_everything(encoder, feat, data, get_vec_id, 0, padding)

def vectorize_continues(feat, data, padding):
    return vectorize_everything('', feat, data, lambda _, val: val, 0, padding)


def vectorize_features(data, word_vectors, feature_config):
    vectors = {}
    
    groupby = feature_config.get('groupby', None)
    padding = int(feature_config['padding']) if groupby else data.shape[0]
    
    groups = (data.groupby(feature_config['groupby']) if groupby 
              else [('', data)])

    for feat in feature_config['features']:
        print(feat['name'])
        
        tp = feat['type']
        if tp == 'embedding':
            vec, encoder = make_embeddings(word_vectors, feat, groups, padding)

        elif tp == 'bool':
            vec, encoder = vectorize_bool(feat, groups, padding)
        
        elif tp == 'cont':
            vec, encoder = vectorize_continues(feat, groups, padding)
            
        else:
            vec, encoder = vectorize_categorical(feat, groups, padding)

        if not groupby:
              vec = vec[0]
              
        vectors[feat['name']] = {'value' : vec, 'encoder' : encoder}
    
    return vectors


def select_vectors(vect_dict, indexes):
    result = {}
    for k,v in vect_dict.items():
        result[k] = v[indexes]
    
    return result
