import os
import sys
sys.path.append('../')

import fire
import pickle
import json


def run_command(command):
    if os.system(command) != 0:
        raise RuntimeError()

        
def work_with_one_model(cleared_corpus_path, ling_data, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    print('Extracting features.============================================')
    run_command(f'python ./run_extract_features.py --cleared-corpus={cleared_corpus_path}' +
              f' --ling-data={ling_data} --known-preds=true --output-dir={output_dir}')
    print('Done.===========================================================')
    
    print('Vectorize features.=============================================')
    feature_path = os.path.join(output_dir, 'features.pckl')
    run_command(f'python ./run_vectorize_features.py --feature_path={feature_path} --output_dir={output_dir}')
    print('Done.===========================================================')
    
    print('Generating embeddings.==========================================')
    run_command(f'python ./run_generate_embeddings.py --feature_path={feature_path} --output_dir={output_dir}')
    print('Done.===========================================================')
    
    print('Training model.=================================================')
    run_command(f'python ./run_train_model.py --input_dir={output_dir} --output_dir={output_dir}')
    print('Done.===========================================================')


def extract_known_predicates(features_path, workdir):
    with open(features_path, 'rb') as f:
        dataframe = pickle.load(f)
        
    known_preds = [e.split('_')[0] for e in dataframe.pred_lemma.tolist()]
    with open(os.path.join(workdir, 'known_preds.json'), 'w') as f:
        json.dump(known_preds, f)
        
def extract_known_predicated(cleared_corpus_path, workdir):
    def make_pred_dict(data_chunk, pred_dict):
        for sentence in data_chunk[1]:
            for word in sentence:
                pred_number = word.get('fillpred')
                if pred_number:
                    if not pred_dict.get(sentence[pred_number]['lemma']):
                        pred_dict[sentence[pred_number]['lemma']] = {word.get('rolepred1'): 1}
                    else:
                        if not pred_dict.get(sentence[pred_number]['lemma']).get(word.get('rolepred1')):
                            pred_dict[sentence[pred_number]['lemma']][word.get('rolepred1')] = 1
                        else:
                            pred_dict[sentence[pred_number]['lemma']][word.get('rolepred1')] += 1
                            
    def filter_roles(pred_dictionary, threshold=5):
        filtered_dict = {}
        for predicate in pred_dictionary.keys():
            new_pred = {}
            for (key, value) in pred_dictionary[predicate].items():
                if value > threshold:
                    new_pred[key] = value
            filtered_dict[predicate] = new_pred

        for predicate in filtered_dict.keys():
            if not filtered_dict[predicate]:
                top = sorted(pred_dict[predicate], key=pred_dict[predicate].get, reverse=True)[0]
                filtered_dict[predicate][top] = pred_dict[predicate][top]

        return filtered_dict
    
    with open(cleared_corpus_path, 'r') as f:
        data = json.load(f)
    
    pred_dict = {}                            
    for instance in data:
        make_pred_dict(instance, pred_dict)
        
    pred_dict = filter_roles(pred_dict)
    known_preds = {key: list(value.keys()) for key, value in pred_dictionary.items()}
    with open(os.path.join(workdir, 'known_preds.json'), 'w') as f:
        json.dump(known_preds, f)
    
def main(data_dir, workdir):
    cleared_corpus_path = os.path.join(data_dir, 'cleared_corpus.json')
    ling_data = os.path.join(data_dir, 'ling_data.pckl')
    
    print('Generating the model for known predicates**********************************')
    output_dir = os.path.join(workdir, 'known_preds')
    work_with_one_model(cleared_corpus_path, ling_data, output_dir)
    
    extract_known_predicates(os.path.join(output_dir, 'features.pckl'), workdir)

    print('Generating the model for unknown predicates********************************')
    output_dir = os.path.join(workdir, 'unknown_preds')
    work_with_one_model(cleared_corpus_path, ling_data, output_dir)


if __name__ == "__main__":
    fire.Fire(main)
    