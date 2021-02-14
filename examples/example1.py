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

import tqdm
import fire
import h5py
import numpy as np

import gensim
from gensim import downloader as api
from gensim.test.utils import datapath

from cusim import aux, IoUtils, CuLDA, CuW2V


LOGGER = aux.get_logger()
DOWNLOAD_PATH = "./res"
# DATASET = "wiki-english-20171001"
DATASET = "quora-duplicate-questions"
# DATASET = "semeval-2016-2017-task3-subtaskBC"
# DATASET = "patent-2017"
DATA_PATH = f"./res/{DATASET}.stream.txt"
LDA_PATH = f"./res/{DATASET}-lda.h5"
PROCESSED_DATA_DIR = f"./res/{DATASET}-converted"
MIN_COUNT = 5
TOPK = 10
CUSIM_W2V_PATH = "./res/cusim.w2v.model"
GENSIM_W2V_PATH = "./res/gensim.w2v.model"

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
  sentences = gensim.models.word2vec.LineSentence(DATA_PATH)
  fout = open(DATA_PATH + ".tmp", "wb")
  LOGGER.info("convert %s by LineSentence in gensim", DATA_PATH)
  for sentence in tqdm.tqdm(sentences):
    fout.write((" ".join(sentence) + "\n").encode("utf8"))
  fout.close()
  os.rename(DATA_PATH + ".tmp", DATA_PATH)


def run_io():
  download()
  iou = IoUtils(opt={"chunk_lines": 10000, "num_threads": 8})
  iou.convert_stream_to_h5(DATA_PATH, 5, PROCESSED_DATA_DIR)


def run_lda():
  download()
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
  download()
  opt = {
    "data_path": DATA_PATH,
    "processed_data_dir": PROCESSED_DATA_DIR,
    "batch_size": 1000000,
    "num_dims": 100,
    "hyper_threads": 100,
    "epochs": 20,
    "lr": 0.001,
    "io": {
      "lower": False
    },
    "neg": 0,
    "skip_gram": False,
    # "pretrained_model": {
    #   "filename": "./res/gensim.w2v.model",
    # }
    # "skip_preprocess":True,
  }
  w2v = CuW2V(opt)
  start = time.time()
  w2v.train_model()
  LOGGER.info("elapsed for cusim w2v training: %.4e sec", time.time() - start)
  w2v.save_word2vec_format(CUSIM_W2V_PATH, binary=False)
  evaluate_w2v_model(CUSIM_W2V_PATH)

def run_gensim_w2v():
  start = time.time()
  model = gensim.models.Word2Vec(corpus_file=DATA_PATH, min_alpha=0.001,
                                 min_count=5, sg=True, hs=False, workers=8,
                                 alpha=0.001, negative=10, iter=10)
  LOGGER.info("elapsed for gensim w2v training: %.4e sec", time.time() - start)
  model.wv.save_word2vec_format(GENSIM_W2V_PATH, binary=False)
  LOGGER.info("gensim w2v model is saved to %s", GENSIM_W2V_PATH)
  evaluate_w2v_model(GENSIM_W2V_PATH)

def evaluate_w2v_model(model=GENSIM_W2V_PATH):
  LOGGER.info("load word2vec format model from %s", model)
  model = gensim.models.KeyedVectors.load_word2vec_format(model)
  results = model.wv.evaluate_word_pairs(datapath("wordsim353.tsv"),
                                         case_insensitive=False)
  LOGGER.info("evaluation results: %s", results)


if __name__ == "__main__":
  fire.Fire()
