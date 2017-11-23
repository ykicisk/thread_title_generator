#/usr/bin/env python3
#-*- coding: utf-8 -*-
import os
import glob
import yaml
import argparse

from keras.utils import to_categorical
from keras.backend import tensorflow_backend as K

from model import create_tokenizer, ThreadTitleGenerator

START_TOKEN = "_start_"
END_TOKEN = "_end_"


def generate(conf_path, n, epoch, prefix_words, ignore_words):
    with open(conf_path) as f:
        conf = yaml.load(f)

    print("== initialize tokenizer ==")
    token_files = glob.glob(conf["input_token_files"])
    tokenizer = create_tokenizer(token_files,
                                 num_words=conf["num_vocab"])
    print("output vocab size:", tokenizer.num_words)
    print("| + <UNK> token")
    inverse_vocab = {idx:w for w, idx in tokenizer.word_index.items()}

    print("load model")
    print("> create instance")
    model = ThreadTitleGenerator(**conf["model_params"])
    print("> load model")
    model.load(conf["model_path"], epoch)
    print("> print summary")
    model.print_summary()
    print("generate words!")
    end_token_idx = tokenizer.word_index[END_TOKEN]
    prefix_tokens = [tokenizer.word_index[t]
                     for t in [START_TOKEN] + prefix_words]
    ignore_idx = [tokenizer.word_index[t] for t in ignore_words] \
                  + [conf["num_vocab"] + 1]  # unk_idx
    ret = model.gen_nbest(prefix_tokens, end_token_idx, ignore_idx, n=n)

    print(ret)

    print("convert to readable tokens")
    for tokens, prob in ret:
        title = [inverse_vocab.get(idx, "???") for idx in tokens]
        print(" ".join(title))
        print(prob)
    K.clear_session()


def main():
    description = """generate thread titles"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("conf", help="config file path (yaml)")
    parser.add_argument("-n", default=100, type=int,
                        help="num generation")
    parser.add_argument("--epoch", default=None, type=int,
                        help="target model epoch")
    parser.add_argument("--prefix", default=[], nargs="+",
                        help="prefix words")
    parser.add_argument("--ignore", default=[], nargs="+",
                        help="ignore words")
    args = parser.parse_args()
    generate(args.conf, args.n, args.epoch, args.prefix, args.ignore)


if __name__ == "__main__":
    main()
