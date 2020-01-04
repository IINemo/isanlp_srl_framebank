import os
import sys
sys.path.append('../')

import fire


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


def main(data_dir, workdir):
    cleared_corpus_path = os.path.join(data_dir, 'cleared_corpus.json')
    ling_data = os.path.join(data_dir, 'ling_data.pckl')
    
    print('Generating the model for known predicates**********************************')
    output_dir = os.path.join(workdir, 'known_preds')
    work_with_one_model(cleared_corpus_path, ling_data, output_dir)

    print('Generating the model for unknown predicates********************************')
    output_dir = os.path.join(workdir, 'unknown_preds')
    work_with_one_model(cleared_corpus_path, ling_data, output_dir)


if __name__ == "__main__":
    fire.Fire(main)
    