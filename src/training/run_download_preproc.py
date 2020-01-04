import fire
import sys
import os


def run_command(command):
    print('Running: ', command)
    if os.system(command) != 0:
        raise RuntimeError()


def main(workdir):
    if not os.path.exists(os.path.join(workdir, 'master.zip')):
        run_command(f'cd {workdir} && wget https://github.com/olesar/framebank/archive/master.zip')
        run_command(f'cd {workdir} && unzip master.zip')

    if not os.path.exists(os.path.join(workdir, 'exampleindex.csv')):
        run_command(f'cd {workdir} && wget http://nlp.isa.ru/framebank_parser/data/original_framebank/exampleindex.csv')
        
    run_command(f'python framebank_preprocessing/extract_corpus.py -I {workdir}/exampleindex.csv -O {workdir}/corpus.json -E err.log')
    
    run_command(f'python framebank_preprocessing/match_corpus_with_annots.py --corpusFile={workdir}/corpus.json --annotFile={workdir}/framebank-master/framebank_anno_ex_items.txt --logFile=log.txt --outputFile={workdir}/annotated_corpus.json') # offsetFile
    
    run_command(f'python ./run_prepare_dataset.py --annotated-corpus={workdir}/annotated_corpus.json --output={workdir}/cleared_corpus.json')
    

if __name__ == "__main__":
    fire.Fire(main)
