# Copyright (c) 2021 Jisang Yoon
# All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=no-name-in-module,logging-format-truncated
# pylint: disable=too-few-public-methods
import os
import time
import subprocess

import fire
import h5py
import numpy as np
import gensim
from gensim import downloader as api
from gensim.test.utils import datapath
from gensim import utils
from cusim import aux, IoUtils, CuLDA, CuW2V


LOGGER = aux.get_logger()
DOWNLOAD_PATH = "./res"
# DATASET = "wiki-english-20171001"
DATASET = "quora-duplicate-questions"
# DATA_PATH = f"./res/{DATASET}.stream.txt"
DATA_PATH = "./res/corpus.txt"
LDA_PATH = f"./res/{DATASET}-lda.h5"
PROCESSED_DATA_DIR = f"./res/{DATASET}-converted"
MIN_COUNT = 5
TOPK = 10

class MyCorpus:
  """An iterator that yields sentences (lists of str)."""

  def __iter__(self):
    corpus_path = datapath('lee_background.cor')
    for line in open(corpus_path):
      # assume there's one document per line, tokens separated by whitespace
      yield utils.simple_preprocess(line)


def download():
  if os.path.exists(DATA_PATH):
    LOGGER.info("%s already exists", DATA_PATH)
    return
  api.BASE_DIR = DOWNLOAD_PATH
  filepath = api.load(DATASET, return_path=True)
  LOGGER.info("filepath: %s", filepath)
  cmd = ["gunzip", "-c", filepath, ">", DATA_PATH]
  cmd = " ".join(cmd)
  LOGGER.info("cmd: %s", cmd)
  subprocess.call(cmd, shell=True)

def run_io():
  download()
  iou = IoUtils(opt={"chunk_lines": 10000, "num_threads": 1})
  iou.convert_stream_to_h5(DATA_PATH, 5, PROCESSED_DATA_DIR)


def run_lda():
  opt = {
    "data_path": DATA_PATH,
    "processed_data_dir": PROCESSED_DATA_DIR,
    # "skip_preprocess":True,
  }
  lda = CuLDA(opt)
  lda.train_model()
  lda.save_model(LDA_PATH)
  h5f = h5py.File(LDA_PATH, "r")
  beta = h5f["beta"][:]
  word_list = h5f["keys"][:]
  num_topics = h5f["alpha"].shape[0]
  for i in range(num_topics):
    print("=" * 50)
    print(f"topic {i + 1}")
    words = np.argsort(-beta.T[i])[:10]
    print("-" * 50)
    for j in range(TOPK):
      word = word_list[words[j]].decode("utf8")
      prob = beta[words[j], i]
      print(f"rank {j + 1}. word: {word}, prob: {prob}")
  h5f.close()

def run_cusim_w2v():
  opt = {
    # "c_log_level": 3,
    "data_path": DATA_PATH,
    "processed_data_dir": PROCESSED_DATA_DIR,
    "batch_size": 100000,
    "num_dims": 100,
    "word_min_count": MIN_COUNT,
    # "skip_preprocess":True,
  }
  w2v = CuW2V(opt)
  start = time.time()
  w2v.train_model()
  LOGGER.info("elapsed for cusim w2v training: %.4e sec", time.time() - start)

def run_gensim_w2v():
  start = time.time()
  model = gensim.models.Word2Vec(corpus_file=DATA_PATH, compute_loss=True,
                                 min_count=5, sg=True, hs=False, workers=8,
                                 alpha=0.001, negative=10, iter=1)
  loss = model.get_latest_training_loss()
  LOGGER.info("elapsed for gensim w2v training: %.4e sec, loss: %f",
              time.time() - start, loss)
  model.wv.save_word2vec_format("res/gensim.w2v.model", binary=False)

def dump_corpus():
  corpus = gensim.models.word2vec.LineSentence(DATA_PATH)
  fout = open("res/corpus.txt", "wb")
  for sentence in corpus:
    fout.write((" ".join(sentence) + "\n").encode("utf8"))
  fout.close()

def dump_corpus2():
  corpus = gensim.models.word2vec.LineSentence(DATA_PATH)
  word_count = {}
  for sentence in corpus:
    for word in sentence:
      word_count[word] = word_count.get(word, 0) + 1
  words = [word for word, count in word_count.items() if count >= 5]
  print(len(words))

def dump_corpus3():
  word_count = {}
  with open("res/corpus.txt", "rb") as fin:
    for line in fin:
      for word in line.decode("utf8").strip().split():
        word_count[word] = word_count.get(word, 0) + 1
  words = [word for word, count in word_count.items() if count >= 5]
  print(len(words))

def compare():
  cusim_words = []
  with open("res/quora-duplicate-questions-converted/keys.txt", "rb") as fin:
    for line in fin:
      word = line.decode("utf8").strip()
      cusim_words.append(word)
  gensim_words = []
  with open("res/gensim.w2v.model", "rb") as fin:
    for line in fin:
      word = line.decode("utf8").strip().split()[0]
      gensim_words.append(word)
  diff1 = list(set(gensim_words) - set(cusim_words))
  diff2 = list(set(cusim_words) - set(gensim_words))
  print(diff1[:5])


if __name__ == "__main__":
  fire.Fire()
