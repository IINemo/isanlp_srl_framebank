# ./backdoor_srl_update.sh container_name

docker cp preposition_extract.py $1:/src/isanlp_srl_framebank/src/isanlp_srl_framebank/preposition_extract.py
docker cp argument_extractor.py $1:/src/isanlp_srl_framebank/src/isanlp_srl_framebank/argument_extractor.py
docker cp processor_srl_framebank.py $1:/src/isanlp_srl_framebank/src/isanlp_srl_framebank/processor_srl_framebank.py

docker cp ../../data/models_new/known_preds/feature_encoder.pckl $1:/models/known_preds/feature_encoder.pckl
docker cp ../../data/models_new/known_preds/feature_model.pckl $1:/models/known_preds/feature_model.pckl
docker cp ../../data/models_new/known_preds/label_encoder.pckl $1:/models/known_preds/label_encoder.pckl
docker exec $1 pip install tensorflow==1.12.0
docker cp ../../data/models_new/known_preds/neural_model.h5 $1:/models/known_preds/neural_model.h5

docker restart chistova_framebank
