#!/bin/bash

script_dir=$(dirname $0)
if [ ! -d $script_dir/isanlp_srl_framebank ]; then
    mkdir $script_dir/isanlp_srl_framebank;
fi
rsync -r $script_dir/../../src/ $script_dir/isanlp_srl_framebank/src
rsync -r $script_dir/../../models/ $script_dir/models
docker build -t inemo/isanlp_srl_framebank $script_dir

