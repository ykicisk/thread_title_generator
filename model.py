#/usr/bin/env python3
#-*- coding: utf-8 -*-
import copy
from collections import defaultdict, Counter, OrderedDict

import numpy as np
from gensim.models import word2vec
from keras import losses, optimizers, metrics
from keras.models import load_model, Model
from keras.layers import Input, Dense, Embedding, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def create_tokenizer(token_file_list, num_words=None):
    sentences = []
    for fpath in token_file_list:
        with open(fpath) as f:
            for line in f:
                sentences.append(line.rstrip())
    tokenizer = Tokenizer(num_words=num_words, lower=False, filters="")
    tokenizer.fit_on_texts(sentences)
    return tokenizer


class ThreadTitleGenerator(object):
    def __init__(self, batch_size, epochs, rnn_dim):
        self.batch_size = batch_size
        self.epochs = epochs
        self.rnn_dim = rnn_dim
        self.model = None

    def save(self, model_path):
        self.model.save(model_path)

    def load(self, model_path, epoch=None):
        if epoch is None:
            epoch = self.epochs
        self.model = load_model(model_path.format(epoch=epoch))

    def build_model(self, word2vec_path, tokenizer, max_seq):
        print(">> load word2vec model")
        w2v = word2vec.Word2Vec.load(word2vec_path)

        print(">> create embedding layer")
        original_weights = w2v.wv.syn0
        emb_dim = original_weights.shape[1]
        target_weigths_list = [np.zeros(emb_dim)]

        wcounts = list(tokenizer.word_counts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)

        for word, _ in wcounts:
            idx = w2v.wv.vocab[word].index
            target_weigths_list.append(original_weights[idx])

        embedding_matrix = np.vstack(target_weigths_list)
        print(">> embedding_matrix.shape:", embedding_matrix.shape)
        emb_layer = Embedding(
            input_dim=embedding_matrix.shape[0],
            output_dim=embedding_matrix.shape[1],
            weights=[embedding_matrix], trainable=True, mask_zero=True
        )

        print(">> build model")
        input = Input(shape=(max_seq,), dtype="int32", name="input")
        emb = emb_layer(input)
        # rnn = LSTM(self.rnn_dim, activation='relu')(emb)
        rnn = LSTM(self.rnn_dim, dropout=0.2, recurrent_dropout=0.2)(emb)
        h1 = Dropout(0.2)(rnn)
        output = Dense(tokenizer.num_words+2, activation='softmax')(h1)
        self.model = Model(inputs=input, outputs=output)

        loss = losses.sparse_categorical_crossentropy
        opti = optimizers.rmsprop()
        metr = [metrics.sparse_categorical_accuracy]

        self.model.compile(loss=loss, optimizer=opti, metrics=metr)

    def print_summary(self):
        self.model.summary()

    def train(self, X, y, fpath):
        cb_func = ModelCheckpoint(filepath=fpath, monitor='epoch', verbose=1,
                                  save_best_only=False, mode='auto')
        self.model.fit(X, y, callbacks=[cb_func], validation_split=0.1,
                       batch_size=self.batch_size, epochs=self.epochs)

    def gen_nbest(self, prefix, end_idx, ignore_idx,
                  n=100, pool_size=2048, search_width=36, min_len=10):
        if self.model is None:
            raise Exception("model is not trained")
        # [(prefix_tokens, prob), ...]

        input_layer = self.model.get_layer("input")
        max_len = input_layer.output_shape[1]

        initial_tokens = copy.deepcopy(prefix)
        gen_pool = [(initial_tokens, 1.0)]
        ret_pool = []

        for n_iter in range(max_len):
            print("generation iter:", n_iter)
            next_gen_pool = []

            input_tokens = [info[0] for info in gen_pool]
            X = pad_sequences(input_tokens, maxlen=max_len, dtype="int32")
            preds = self.model.predict(X)

            for gen_idx, info in enumerate(gen_pool):
                tokens, prob = info
                pred = preds[gen_idx]

                idx2prob = list(enumerate(pred))
                idx2prob.sort(key=lambda x:x[1], reverse=True)
                count = 0
                for token_idx, next_prob in idx2prob:
                    if token_idx == tokens[-1]:
                        continue
                    if token_idx in ignore_idx:
                        continue
                    new_tokens = tokens + [token_idx]
                    new_prob = prob * next_prob
                    if token_idx == end_idx:
                        if n_iter >= min_len:
                            ret_pool.append((new_tokens, new_prob))
                    else:
                        next_gen_pool.append((new_tokens, new_prob))
                    count += 1
                    if count >= search_width:
                        break
            next_gen_pool.sort(key=lambda x:x[1], reverse=True)
            if len(next_gen_pool) > pool_size:
                next_gen_pool = next_gen_pool[:pool_size]
            gen_pool = next_gen_pool

        ret_pool.sort(key=lambda x:x[1], reverse=True)
        return ret_pool[:n]
