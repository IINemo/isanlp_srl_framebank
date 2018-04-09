#!/bin/bash

script_dir=$(dirname $0)
mkdir --parents $script_dir/isanlp_srl_framebank/src
rsync -r $script_dir/../../src/demo $script_dir/
rsync -r $script_dir/../../setup.py $script_dir/isanlp_srl_framebank/
rsync -r $script_dir/../../src/isanlp_srl_framebank $script_dir/isanlp_srl_framebank/src
docker build -t inemo/isanlp_srl_framebank_demo $script_dir
