#/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import argparse
from gensim.models import word2vec


def test(model_path, word):
    model   = word2vec.Word2Vec.load(model_path)
    results = model.most_similar(positive=word, topn=10)
    for result in results:
        print(result[0], '\t', result[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="model path")
    parser.add_argument("word", help="test word")
    args = parser.parse_args()
    test(args.model, args.word)


if __name__ == "__main__":
    main()
