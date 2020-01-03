from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
import fire
import pickle
import os
import numpy as np


def main(feature_path, output_dir, known_preds=True):
    with open(feature_path, 'rb') as f:
        X_train = pickle.load(f)
        
    y_train = X_train.loc[:, 'role']
    label_encoder = LabelBinarizer()
    y_train = label_encoder.fit_transform(y_train)

    with open(os.path.join(output_dir, 'label_encoder.pckl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    
    np.save(os.path.join(output_dir, 'labels.npy'), y_train)
    
    columns_to_ommit = ['tokens']
    X_train = X_train.drop('role', axis=1)
    
    if not known_preds and 'pred_lemma' in X_train.keys():
        X_train = X_train.drop(columns=['pred_lemma'])

    not_categ_features = {'arg_address', 'ex_id', 'rel_pos'}

    categ_feats = [name for name in X_train.drop(columns=columns_to_ommit).columns 
                   if name not in not_categ_features] 
    not_categ = ['rel_pos']
    print('Category features:\n', categ_feats)
    print('Not category features:\n', not_categ)

    vectorizer = DictVectorizer(sparse=False)
    vectorizer.fit(X_train[categ_feats].to_dict(orient='records'))
    one_hot_feats = vectorizer.transform(X_train[categ_feats].to_dict(orient='records'))
    print(one_hot_feats.shape)

    with open(os.path.join(output_dir, 'feature_encoder.pckl'), 'wb') as f:
        pickle.dump(vectorizer, f)
        
#     with open(os.path.join(output_dir, 'feature_vectors.pckl'), 'wb') as f:
#         pickle.dump(one_hot_feats, f)

    not_categ_columns = np.concatenate(tuple(X_train.loc[:, e].values.reshape(-1, 1) for e in not_categ), axis =1)
    plain_features = np.concatenate((one_hot_feats, not_categ_columns), axis = 1)
        
    np.save(os.path.join(output_dir, 'feature_vectors.npy'), plain_features)
        

if __name__ == "__main__":
    fire.Fire(main)
    