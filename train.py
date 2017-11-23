#/usr/bin/env python3
#-*- coding: utf-8 -*-
import os
import glob
import yaml
import argparse

import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence

from model import create_tokenizer, ThreadTitleGenerator



def texts_to_sequences_with_unk_seq(tokenizer, texts):
    nounk_seq = []
    unk_seq = []
    num_words = tokenizer.num_words
    for text in texts:
        seq = text_to_word_sequence(text, tokenizer.filters,
                                    tokenizer.lower, tokenizer.split)

        vect, unk_vect = [], []
        for w in seq:
            i = tokenizer.word_index.get(w, -1)
            if i == -1:
                raise Exception("Unknown word!")
            unk_idx = min(i, num_words+1)  # unk=num_words+1
            vect.append(i)
            unk_vect.append(unk_idx)
        nounk_seq.append(vect)
        unk_seq.append(unk_vect)
    return nounk_seq, unk_seq


def create_training_data(src_X, dst_X, max_len=20):
    expanded = []
    y = []
    for src_wv, dst_wv in zip(src_X, dst_X):
        if len(src_wv) > max_len:
            continue
        for i in range(1, len(src_wv)):
            expanded.append(src_wv[0:i])
            y.append(dst_wv[i])
    X = pad_sequences(expanded, maxlen=max_len, dtype="int32")
    return X, y


def train(conf_path):
    with open(conf_path) as f:
        conf = yaml.load(f)

    print("== initialize tokenizer ==")
    token_files = glob.glob(conf["input_token_files"])
    tokenizer = create_tokenizer(token_files,
                                 num_words=conf["num_vocab"])
    print("output vocab size:", tokenizer.num_words)
    print("| + <UNK> token")

    print("== load input tokens ==")
    input_sentences = []
    for fpath in token_files:
        with open(fpath) as f:
            for line in f:
                input_sentences.append(line.rstrip())

    src_X, dst_X = texts_to_sequences_with_unk_seq(tokenizer, input_sentences)
    X, y = create_training_data(src_X, dst_X, conf["max_title_tokens"])

    print("build model")
    print("> create instance")
    model = ThreadTitleGenerator(**conf["model_params"])
    print("> build model")
    model.build_model(conf["word2vec_model_path"],
                      tokenizer, conf["max_title_tokens"])
    print("> print summary")
    model.print_summary()

    print("train model")
    model.train(X, y, conf["model_path"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("conf", help="config file path (yaml)")
    args = parser.parse_args()
    train(args.conf)


if __name__ == "__main__":
    main()
