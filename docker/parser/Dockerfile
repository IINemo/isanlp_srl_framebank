FROM inemo/isanlp_base

RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

ENV CUDNN_VERSION 6.0.21
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --allow-unauthenticated --no-install-recommends \
            libcudnn6=$CUDNN_VERSION-1+cuda8.0 \
            libcudnn6-dev=$CUDNN_VERSION-1+cuda8.0 && \
    rm -rf /var/lib/apt/lists/*



RUN pip  install -U pip
RUN python -m pip install -U cython
RUN pip install scipy sklearn gensim==3.2.0 tensorflow==1.4.0 keras h5py

COPY isanlp_srl_framebank /src/isanlp_srl_framebank
COPY pipeline_object.py /src/isanlp_srl_framebank/src/pipeline_object.py
COPY models /models

ENV PYTHONPATH=/src/isanlp_srl_framebank/src/

CMD [ "python", "/start.py", "-m", "pipeline_object", "-a", "PPL_SRL_FRAMEBANK" ]
