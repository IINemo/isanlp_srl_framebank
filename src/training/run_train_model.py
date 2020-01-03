import numpy as np
from models import SimpleModelFactory, SparseModelFactory
import pickle
from sklearn.model_selection import train_test_split
import fire
import os
import tensorflow as tf


def select_from_nparray_list(nparray_list, selector):
    return [e[selector] for e in nparray_list]


def main(input_dir, output_dir, batch_size=32, epochs=10):
    print('Loading data...')
    plain_features = np.load(os.path.join(input_dir, 'feature_vectors.npy'))
    arg_embedded = np.load(os.path.join(input_dir, 'elmo_args_whole.npy'))
    pred_embedded = np.load(os.path.join(input_dir, 'elmo_verbs_whole.npy'))
    y = np.load(os.path.join(input_dir, 'labels.npy'))
    n_examples = plain_features.shape[0]
    print('Done.')
    
    train_selector, test_selector = train_test_split(list(range(n_examples)), test_size=0.2, random_state=42)
    print(y[train_selector].shape)
    number_of_roles = y.shape[1]

    print('Training model...')
    factory = SparseModelFactory([plain_features.shape[1], pred_embedded.shape[1], arg_embedded.shape[1]], number_of_roles)
    model = factory.create_and_compile()
    model.fit(select_from_nparray_list([plain_features, pred_embedded, arg_embedded], train_selector),
              y[train_selector],
              epochs=epochs, batch_size=batch_size, validation_split=0.1, shuffle=True)
    print('Done.')

    print('Evaluating...')
    print(model.metrics_names)
    print(model.evaluate(select_from_nparray_list([plain_features, pred_embedded, arg_embedded], test_selector), 
                         y[test_selector]))
    print('Done.')
    
    print('Saving model...')
    save_model_path = os.path.join(output_dir, 'neural_model.h5')
    model.save(save_model_path)
    
    converter = tf.lite.TFLiteConverter.from_keras_model_file(save_model_path)
    tflite_model = converter.convert()
    with open(os.path.join(output_dir, 'neural_model.tflite'), 'wb') as f:
        f.write(tflite_model)
    
    print('Done.')

    
if __name__ == '__main__':
    fire.Fire(main)
    