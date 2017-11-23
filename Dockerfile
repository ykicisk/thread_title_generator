FROM nvidia/cuda:8.0-cudnn6-runtime

LABEL maintainer "ykic.p3@gmail.com"

# install apt packages
RUN apt-get update && apt-get -y install python3-pip curl mecab libmecab-dev mecab-ipadic-utf8 git make curl xz-utils file sudo language-pack-ja libhdf5-10

# install python packages
RUN pip3 install lxml mecab-python3 numpy tensorflow-gpu keras h5py gensim

# add docker user
RUN useradd -m -d /home/docker -s /bin/bash docker \
  && echo "docker:docker" | chpasswd \
  && mkdir -p /home/docker/.ssh \
  && chmod 700 /home/docker/.ssh

# create volume directory
RUN mkdir -p /home/docker/model

# install neologd
RUN mkdir -p /home/docker/install
WORKDIR /home/docker/install
RUN git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git
WORKDIR /home/docker/install/mecab-ipadic-neologd
RUN ./bin/install-mecab-ipadic-neologd -n -y -p "/usr/share/mecab/dic/mecab-ipadic-neologd"

# locale setting
RUN update-locale LANG=ja_JP.UTF-8 LANGUAGE=ja_JP:ja
ENV LANG ja_JP.UTF-8
ENV LC_ALL ja_JP.UTF-8
ENV LC_CTYPE ja_JP.UTF-8

# teardown setting
RUN chown -R docker:docker /home/docker
WORKDIR /home/docker
ENV HOME /home/docker
