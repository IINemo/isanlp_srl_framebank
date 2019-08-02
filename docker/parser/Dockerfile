#FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

#ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

FROM inemo/isanlp_base:0.0.5

# RUN apt-get update && \ 
#     apt-get upgrade -y && \
#     apt-get install -y --no-install-recommends \
#         gnupg \
#         apt-transport-https ca-certificates \
#     && \
#     apt-get clean

# # nvidia-container-runtime
# ENV NVIDIA_VISIBLE_DEVICES all
# ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
# ENV NVIDIA_REQUIRE_CUDA "cuda>=9.0"
    
#############################################

RUN pip install -U pip
RUN python -m pip install -U cython

RUN pip install setuptools==41.0.1 scipy scikit-learn==0.19.1 gensim==3.6.0 smart-open==1.7.0 tensorflow==1.12.0 keras h5py deeppavlov tensorflow-hub

RUN python -c "import tensorflow as tf; print(tf.__version__)"

COPY isanlp_srl_framebank /src/isanlp_srl_framebank
COPY pipeline_object.py /src/isanlp_srl_framebank/src/pipeline_object.py
COPY models /models

ENV PYTHONPATH=/src/isanlp_srl_framebank/src/

# CMD [ "python", "/start.py", "-m", "pipeline_object", "-a", "create_pipeline" ]  ## Requires for isanlp > 0.0.5 (doesn't work with tensorflow)
CMD [ "python", "/start.py", "-m", "pipeline_object", "-a", "PPL_SRL_FRAMEBANK" ]
