import fire
import sys
import os


def run_command(command):
    print('Running: ', command)
    if os.system(command) != 0:
        raise RuntimeError()


def main(workdir):
    run_command(f'cd {workdir} && wget https://github.com/olesar/framebank/archive/master.zip')
    run_command(f'cd {workdir} && unzip master.zip')
    run_command(f'python framebank_preprocessing/extract_corpus.py -I {workdir}/framebank-master/framebank_anno_ex_items.txt -O {workdir}/corpus.json -E err.log')
    run_command(f'python prepare_dataset.py --annotated-corpus={workdir}/corpus.json --output-dir={workdir}')


if __name__ == "__main__":
    fire.Fire(main)
