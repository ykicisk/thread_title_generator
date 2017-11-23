#/usr/bin/env python3
#-*- coding: utf-8 -*-
import os
import glob
import argparse
import re
import unicodedata
import logging
import pickle
from collections import Counter

import MeCab


START_TOKEN = "_start_"
END_TOKEN = "_end_"

IGNORE_TOKENS = frozenset(["", "\\n", "\n", "、", "・", ".", "。"])
WARA_REGEX = re.compile("^w+$")
NUM_REGEX = re.compile("^[0-9]+$")


def process_text(text):
    normalized_text = unicodedata.normalize("NFKC", text)
    normalized_text = normalized_text.replace("。", " ")
    normalized_text = normalized_text.replace("彡(゚)(゚)", " NANJMINKAOMOJITOKEN ")
    normalized_text = normalized_text.replace("彡(^)(^)", " NIKONANJMINKAOMOJITOKEN ")
    normalized_text = normalized_text.replace("[無断転載禁止]©2ch.net", "")
    return normalized_text


def token_filter(token):
    if token in IGNORE_TOKENS:
        return False
    return True


def token_processor(token):
    if NUM_REGEX.match(token):
        return "_num_"
    if WARA_REGEX.match(token):
        return "WARATOKEN"

    return token


def tokenize(raw_path, dst_path,
             min_tokens, max_tokens, dict_path, loglevel):

    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(loglevel)

    mecab_args = "-d {0} -Owakati".format(dict_path)
    tokenizer = MeCab.Tagger(mecab_args)

    titles = []

    num_few_response_thread = 0
    for fpath in glob.glob(raw_path):
        with open(fpath) as f, open(dst_path, "w") as dst:
            for line in f:
                text = line.rstrip()
                if text.startswith("<"):
                    continue
                if len(text) < 1:
                    continue
                logger.debug("parse: %s", text)
                tokens = tokenizer.parse(process_text(text)).split(" ")
                processed = map(token_processor, filter(token_filter, tokens))
                processed = [START_TOKEN] + list(processed) + [END_TOKEN]
                logger.debug("processed tokens: %s", processed)
                if len(tokens) < min_tokens or len(tokens) > max_tokens:
                    logger.debug("> too long text. skiped.")
                    continue
                dst.write("{0}\n".format(" ".join(processed)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", help="raw data path (glob expression)")
    parser.add_argument("dst", help="output path")
    parser.add_argument("--min_tokens", type=int, default=0,
                        help="minumum token length")
    parser.add_argument("--max_tokens", type=int, default=999999,
                        help="maximum token length")
    parser.add_argument("--dict_path",
                        default="/usr/share/mecab/dic/mecab-ipadic-neologd",
                        help="vocaburary size")
    parser.add_argument("--loglevel", type=int, default=20,
                        help="loglevel")
    args = parser.parse_args()
    tokenize(args.raw, args.dst,
             args.min_tokens, args.max_tokens,
             args.dict_path, args.loglevel)


if __name__ == "__main__":
    main()
