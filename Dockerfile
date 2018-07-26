FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

ENV PATH /opt/conda/bin:$PATH

RUN mkdir -p vectors/fasttext

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
         curl \
         bzip2 \
         git \
         unzip \
         make \
         g++ \
         gzip \
        && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/*

# download fasttext vectors for german language
RUN curl -o cc.de.300.bin.gz https://s3-us-west-1.amazonaws.com/fasttext-vectors/word-vectors-v2/cc.de.300.bin.gz \
    && gunzip -c cc.de.300.bin.gz > /vectors/fasttext/cc.de.300.bin \
    && rm cc.de.300.bin.gz

# download quantized fasttext vectors for german language
RUN curl -o cc.de.300_quantized.ftz.gz https://s3.eu-central-1.amazonaws.com/nlp-machine-learning-data/cc.de.300_quantized.ftz.gz \
    && gunzip -c cc.de.300_quantized.ftz.gz > /vectors/fasttext/cc.de.300_quantized.ftz \
    && rm cc.de.300_quantized.ftz.gz

# download elmo model
RUN curl -o elmo_weights.zip https://s3.eu-central-1.amazonaws.com/nlp-machine-learning-data/elmo_weights.zip \
    && mkdir -p vectors/elmo/hdf5 \
    && unzip elmo_weights.zip -d vectors/elmo/hdf5 \
    && rm elmo_weights.zip

# Install Anaconda.
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh \
    && curl -o ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && /bin/bash ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh

# Copy files.
COPY model/ model/
COPY evaluation.py evaluation.py
COPY requirements.txt requirements.txt

# Use python 3.6
RUN conda install python=3.6

# install python requirement
RUN pip install -r requirements.txt

# install spacy german package
RUN python -m spacy download de

# install bilm-tf for elmo vectors
RUN git clone https://github.com/allenai/bilm-tf.git && cd bilm-tf/ && python setup.py install

# Install fasttext
RUN curl -o v0.1.0.zip https://codeload.github.com/facebookresearch/fastText/zip/v0.1.0 \
    && unzip v0.1.0.zip && cd fastText-0.1.0 && make && cd .. && rm v0.1.0.zip
RUN git clone https://github.com/facebookresearch/fastText.git && pip install fastText/