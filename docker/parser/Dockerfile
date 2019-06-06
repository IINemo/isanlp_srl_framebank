FROM inemo/isanlp_base


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gnupg \
    && \
    apt-get clean

RUN NVIDIA_GPGKEY_SUM=d1be581509378368edeec8c1eb2958702feedf3bc3d17011adbf24efacce4ab5 && \
    NVIDIA_GPGKEY_FPR=ae09fe4bbd223a84b2ccfce3f60f4b3d7fa2af80 && \
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub && \
    apt-key adv --export --no-emit-version -a $NVIDIA_GPGKEY_FPR | sed -n '/^--/,/^--/ p' > cudasign.pub && \
    echo "$NVIDIA_GPGKEY_SUM cudasign.pub" | sha256sum -c --strict - && \
    rm cudasign.pub && \
    echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/cuda.list

ENV CUDA_VERSION=8.0.61
LABEL com.nvidia.cuda.version="${CUDA_VERSION}"
ENV CUDA_PKG_VERSION=8-0=8.0.61-1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        cuda-nvrtc-$CUDA_PKG_VERSION \
        cuda-nvgraph-$CUDA_PKG_VERSION \
        cuda-cusolver-$CUDA_PKG_VERSION \
        cuda-cublas-8-0=8.0.61.2-1 \
        cuda-cufft-$CUDA_PKG_VERSION \
        cuda-curand-$CUDA_PKG_VERSION \
        cuda-cusparse-$CUDA_PKG_VERSION \
        cuda-npp-$CUDA_PKG_VERSION \
        cuda-cudart-$CUDA_PKG_VERSION \
    && \
    ln -s cuda-8.0 /usr/local/cuda && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/cuda.conf && \
    echo "/usr/lib/x86_64-linux-gnu" >> /etc/ld.so.conf.d/cuda.conf && \
    ldconfig

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/lib/x86_64-linux-gnu/nvidia/current/" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/x86_64-linux-gnu/nvidia/:$PATH

ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu/nvidia/current

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        nvidia-cuda-toolkit \
        cuda-core-$CUDA_PKG_VERSION \
        cuda-misc-headers-$CUDA_PKG_VERSION \
        cuda-command-line-tools-$CUDA_PKG_VERSION \
        cuda-nvrtc-dev-$CUDA_PKG_VERSION \
        cuda-nvml-dev-$CUDA_PKG_VERSION \
        cuda-nvgraph-dev-$CUDA_PKG_VERSION \
        cuda-cusolver-dev-$CUDA_PKG_VERSION \
        cuda-cublas-dev-8-0=8.0.61.2-1 \
        cuda-cufft-dev-$CUDA_PKG_VERSION \
        cuda-curand-dev-$CUDA_PKG_VERSION \
        cuda-cusparse-dev-$CUDA_PKG_VERSION \
        cuda-npp-dev-$CUDA_PKG_VERSION \
        cuda-cudart-dev-$CUDA_PKG_VERSION \
        cuda-driver-dev-$CUDA_PKG_VERSION && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LIBRARY_PATH
RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

ENV CUDNN_VERSION=6.0.21
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libcudnn6=$CUDNN_VERSION-1+cuda8.0 \
        libcudnn6-dev=$CUDNN_VERSION-1+cuda8.0 \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        software-properties-common \
        sudo \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

LABEL com.nvidia.volumes.needed="nvidia_driver"
LABEL com.nvidia.cuda.version=$CUDA_VERSION


#############################################

RUN pip  install -U pip
RUN python -m pip install -U cython

RUN pip install setuptools==41.0.1 scipy scikit-learn==0.19.1 gensim==3.6.0 smart-open==1.7.0 tensorflow-gpu==1.4.0 keras h5py
RUN python -c "import tensorflow as tf; print(tf.__version__)"

COPY isanlp_srl_framebank /src/isanlp_srl_framebank
COPY pipeline_object.py /src/isanlp_srl_framebank/src/pipeline_object.py
COPY models /models

ENV PYTHONPATH=/src/isanlp_srl_framebank/src/

CMD [ "python", "/start.py", "-m", "pipeline_object", "-a", "create_pipeline" ]
