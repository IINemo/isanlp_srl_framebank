### The script was added in case you don't want to rebuild the whole container only to change a script or a model
### Usage: ./backdoor_srl_update.sh container_name

docker exec $1 pip install -U pip
docker exec $1 pip install tensorflow==1.12.0 deeppavlov tensorflow-hub

### update pipeline object
docker cp ../../docker/parser/pipeline_object.py $1:/src/isanlp_srl_framebank/src/pipeline_object.py

docker cp preposition_extract.py $1:/src/isanlp_srl_framebank/src/isanlp_srl_framebank/preposition_extract.py
docker cp argument_extractor.py $1:/src/isanlp_srl_framebank/src/isanlp_srl_framebank/argument_extractor.py
docker cp processor_srl_framebank.py $1:/src/isanlp_srl_framebank/src/isanlp_srl_framebank/processor_srl_framebank.py
docker cp pipeline_default.py $1:/src/isanlp_srl_framebank/src/isanlp_srl_framebank/pipeline_default.py

### update known preds model
docker cp ../../data/models_new/known_preds/feature_encoder.pckl $1:/models/known_preds/feature_encoder.pckl
docker cp ../../data/models_new/known_preds/feature_model.pckl $1:/models/known_preds/feature_model.pckl
docker cp ../../data/models_new/known_preds/label_encoder.pckl $1:/models/known_preds/label_encoder.pckl
docker cp ../../data/models_new/known_preds/test_model_elmo.h5 $1:/models/known_preds/neural_model.h5
docker cp ../../data/ruscorpora_upos_skipgram_300_5_2018.vec $1:/models/known_preds/embeddings.vec

### update unknown preds model
docker cp ../../data/models_new/unknown_preds/feature_encoder.pckl $1:/models/unknown_preds/feature_encoder.pckl
docker cp ../../data/models_new/unknown_preds/feature_model.pckl $1:/models/unknown_preds/feature_model.pckl
docker cp ../../data/models_new/unknown_preds/label_encoder.pckl $1:/models/unknown_preds/label_encoder.pckl
docker cp ../../data/models_new/unknown_preds/test_model_w2v.h5 $1:/models/unknown_preds/neural_model.h5
docker cp ../../data/ruscorpora_upos_skipgram_300_5_2018.vec $1:/models/unknown_preds/embeddings.vec

### rollback to as-is models
# docker cp ../../models/known_preds/feature_encoder.pckl $1:/models/known_preds/feature_encoder.pckl
# docker cp ../../models/known_preds/feature_model.pckl $1:/models/known_preds/feature_model.pckl
# docker cp ../../models/known_preds/label_encoder.pckl $1:/models/known_preds/label_encoder.pckl
# docker cp ../../models/known_preds/neural_model.h5 $1:/models/known_preds/neural_model.h5
# docker cp ../../models/embeddings.vec $1:/models/known_preds/embeddings.vec

### docker cp ../../models/unknown_preds/feature_encoder.pckl $1:/models/unknown_preds/feature_encoder.pckl
# docker cp ../../models/unknown_preds/feature_model.pckl $1:/models/unknown_preds/feature_model.pckl
# docker cp ../../models/unknown_preds/label_encoder.pckl $1:/models/unknown_preds/label_encoder.pckl
# docker cp ../../models/unknown_preds/neural_model.h5 $1:/models/unknown_preds/neural_model.h5
# docker cp ../../models/embeddings.vec $1:/models/unknown_preds/embeddings.vec

docker restart $1