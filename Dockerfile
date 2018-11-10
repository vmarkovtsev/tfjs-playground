FROM ubuntu:18.04

# essentials - deps of the other layers
RUN apt-get update \
    && apt-get install -y --no-install-suggests --no-install-recommends \
        ca-certificates curl gnupg1 locales \
    && echo "en_US.UTF-8 UTF-8" > /etc/locale.gen \
    && locale-gen \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# nvidia driver layer
################################################################################################
# NOTE: nvidia driver MUST be modprobe-d in the host, this layer only adds the userspace files #
################################################################################################
ENV NVIDIA_DRIVER_VERSION 390.25
RUN apt-get update \
    && apt-get install -y kmod \
    && mkdir -p /opt/nvidia && cd /opt/nvidia/ \
    && curl -L http://us.download.nvidia.com/XFree86/Linux-x86_64/${NVIDIA_DRIVER_VERSION}/NVIDIA-Linux-x86_64-${NVIDIA_DRIVER_VERSION}.run -o /opt/nvidia/driver.run \
    && chmod +x /opt/nvidia/driver.run \
    && /opt/nvidia/driver.run -a -s --no-nvidia-modprobe --no-kernel-module --no-unified-memory --no-x-check --no-opengl-files \
    && rm -rf /opt/nvidia \
    && apt-get remove -y kmod \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# cuda layer
ENV CUDA_VERSION 9.0.176
ENV CUDA_VERSION_DASH 9-0
ENV CUDA_VERSION_MAJOR 9.0
RUN apt-get update \
    && apt-get install -y gnupg \
    && curl -LO http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_${CUDA_VERSION}-1_amd64.deb \
    && dpkg -i cuda-repo-ubuntu1604_${CUDA_VERSION}-1_amd64.deb \
    && rm cuda-repo-ubuntu1604_${CUDA_VERSION}-1_amd64.deb \
    && apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub \
    && apt-get update \
    && apt-get -y install --no-install-suggests --no-install-recommends \
        cuda-cublas-${CUDA_VERSION_DASH} \
        cuda-cudart-${CUDA_VERSION_DASH} \
        cuda-cufft-${CUDA_VERSION_DASH} \
        cuda-curand-${CUDA_VERSION_DASH} \
        cuda-cusolver-${CUDA_VERSION_DASH} \
        cuda-cusparse-${CUDA_VERSION_DASH} \
    && apt upgrade -y \
    && sed -i 's#"$#:/usr/local/cuda-${CUDA_VERSION_MAJOR}/bin"#' /etc/environment \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# cudnn layer
RUN curl -L https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7_7.2.1.38-1+cuda${CUDA_VERSION_MAJOR}_amd64.deb -o libcudnn.deb \
    && dpkg -i libcudnn.deb \
    && rm libcudnn.deb

# node layer
RUN curl -sL https://deb.nodesource.com/setup_10.x | bash \
    && apt-get install -y --no-install-suggests --no-install-recommends nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# jupyter +tensorflowjs layer
RUN apt-get update \
    && apt-get install -y --no-install-suggests --no-install-recommends python3 python3-dev python3-distutils \
    && curl https://bootstrap.pypa.io/get-pip.py | python3 \
    && pip3 install jupyter tensorflowjs \
    && mkdir /root/.jupyter/ \
    && echo 'c.NotebookApp.token = ""' > /root/.jupyter/jupyter_notebook_config.py \
    && curl -o /usr/local/bin/hub2graph https://raw.githubusercontent.com/vmarkovtsev/hub/master/examples/hub2graph.py \
    && chmod +x /usr/local/bin/hub2graph \
    && apt-get remove -y python3-dev \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["jupyter", "notebook", "--allow-root", "--ip=0.0.0.0"]

# ijavascript layer
ENV NODE_PATH=/usr/lib/node_modules
RUN apt-get update \
    && apt-get install -y git gcc g++ libpython2.7-stdlib make \
    && npm --unsafe-perm install -g ijavascript-await \
    && ijsinstall \
    && echo "c.Spawner.env_keep.append('NODE_PATH')" >> /root/.jupyter/jupyter_notebook_config.py \
    && apt-get remove -y git gcc g++ libpython2.7-stdlib make \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# tfjs layer
RUN apt-get update \
    && apt-get install -y libgomp1 git gcc g++ libpython2.7-stdlib make \
    && npm --unsafe-perm install -g @tensorflow/tfjs-node-gpu @tensorflow/tfjs \
    && apt-get remove -y git gcc g++ libpython2.7-stdlib make \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# custom npm packages
RUN npm install -g csv-parse promisepipe browser-sync
