#/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import glob
import logging
import argparse
from gensim.models import word2vec


def train(src_path, dst_path, dim, loglevel):
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(loglevel)

    sentences = []
    for fpath in glob.glob(src_path):
        sentences += word2vec.LineSentence(fpath)
    model = word2vec.Word2Vec(sentences, sg=1, size=dim,
                              min_count=1, window=7, hs=1,
                              negative=5)
    model.save(dst_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="src data path (glob expression)")
    parser.add_argument("dst", help="output path")
    parser.add_argument("--dim", "-d", type=int, default=128,
                        help="word vector dimension")
    parser.add_argument("--loglevel", type=int, default=20,
                        help="loglevel")
    args = parser.parse_args()
    train(args.src, args.dst, args.dim, args.loglevel)


if __name__ == "__main__":
    main()
