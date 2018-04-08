#!/bin/bash

script_dir=$(dirname $0)
rsync -r $script_dir/../../src/demo $script_dir/
rsync -r $script_dir/../../src/isanlp_srl_framebank $script_dir/
docker build -t inemo/isanlp_srl_framebank_demo $script_dir
