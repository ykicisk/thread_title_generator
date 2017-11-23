# thead_title_generator

5chのなんJスレタイを生成する。

## 依存関係

* nvidia-docker
* https://github.com/eywalker/nvidia-docker-compose

## nvidia-docker上での作業をするときの操作

基本的にpythonスクリプトはDockerコンテナ内で動作させます。

```sh
# docker起動
$ sudo nvidia-docker-compose up -d

# コンテナ内で作業(dockerユーザ)
$ sudo nvidia-docker-compose exec --user docker dev /bin/bash
# コンテナ内で作業(root)
$ sudo nvidia-docker-compose exec dev /bin/bash

# docker終了
$ sudo nvidia-docker-compose down
```

## 手順

### wikipediaコーパスの取得・前処理

wikipediaコーパスはEmbedding層の事前学習に利用する。

#### 取得

基本はDocker内での作業

```sh
$ cd data/wikipedia
$ curl https://dumps.wikimedia.org/jawiki/latest/jawiki-latest-pages-articles.xml.bz2 -O
```

#### 前処理

```sh
$ cd data/wikipedia
$ curl https://raw.githubusercontent.com/attardi/wikiextractor/master/WikiExtractor.py -O
$ python3 WikiExtractor.py --filter_disambig_pages -b 50M -o extracted jawiki-latest-pages-articles.xml.bz2
$ find extracted -name 'wiki*' -exec cat {} \; > jawiki.xml
```
#### Tokenize

```sh
$ python3 tokenize_text.py data/wikipedia/jawiki.xml data/jawiki.txt
# サイズが大きすぎる場合は適当にサンプリングする
$ shuf -n 864558 data/jawiki.txt > data/jawiki_sampled.txt 
$ mv data/jawiki.txt data/wikipedia
```

### スレタイデータの取得・前処理

#### 取得

5chのなんJ板からスレタイとレス数を取得して、TSVで保存します。

```sh
$ python3 scrape_thread.py data/thread/raw
$ ls data/thread/raw
0000.tsv ...
```

#### 前処理

レスの数でフィルタリングします。
以下は3スレ以上のデータを使う場合。

```sh
$ cat data/thread/raw/*.tsv | awk -F'\t' '$2 > 2 {print}' | cut -f1 > data/thread/thread_dump.txt
```

#### Tokenize

```sh
$ python3 tokenize_text.py data/thread/thread_dump.txt data/thread.txt
```

### gensimによるword2vecの学習(Embedding層の事前学習)

```sh
# wikipedia, スレタイのデータからword2vecを学習
$ python3 word2vec_train.py "data/*.txt" data/w2v.dat

# 動作確認
$ python3 word2vec_test.py data/w2v.dat "東京"
大阪     0.9098623991012573
名古屋   0.8524906039237976
福岡     0.8452504873275757
札幌     0.7933300733566284
神戸     0.7872719764709473
関西     0.7716591358184814
神奈川   0.7698613405227661
京都     0.7634186744689941
埼玉     0.7461172342300415
千葉     0.7347638607025146

$ python3 word2vec_test.py data/w2v.dat "ラーメン"
うどん   0.8949257731437683
焼肉     0.8756559491157532
お好み焼き       0.8649905323982239
餃子     0.8645927906036377
寿司     0.8585028648376465
おでん   0.8419040441513062
丼       0.8334213495254517
味噌ラーメン     0.8308401107788086
麺       0.8289585113525391
焼きそば         0.8282044529914856
```

### スレタイ生成器の学習

GPUを使って生成器を学習します。

```sh
$ python3 train.py config.yaml
...
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input (InputLayer)           (None, 20)                0
_________________________________________________________________
embedding_1 (Embedding)      (None, 20, 128)           11004928
_________________________________________________________________
lstm_1 (LSTM)                (None, 128)               131584
_________________________________________________________________
dropout_1 (Dropout)          (None, 128)               0
_________________________________________________________________
dense_1 (Dense)              (None, 20002)             2580258
=================================================================
Total params: 13,716,770
Trainable params: 13,716,770
Non-trainable params: 0
_________________________________________________________________

...
```

### スレタイ生成

beam searchでn-bestを取得します。

```sh
$ python3 gen.py config.yaml -n 5 --prefix "ワイ"
...
_start_ ワイ 陰 キャ だ けど 駅 の トイレ で 泣い てる _end_
4.75191054636e-07
_start_ ワイ ( _num_ ) だ けど 駅 の トイレ で 泣い てる _end_
2.62613424411e-07
_start_ ワイ しか 見 て ない こと を し て しまう WARATOKEN _end_
3.5584431935e-08
_start_ ワイ しか 見 て ない こと に なっ て しまう WARATOKEN _end_
2.92392169464e-08
_start_ ワイ 「 お前 ら が 好き な ん です か ?」 _end_
2.72264028929e-08
```

