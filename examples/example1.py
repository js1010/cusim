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

from cusim import aux, IoUtils, CuLDA, CuW2V


LOGGER = aux.get_logger()
DOWNLOAD_PATH = "./res"
# DATASET = "wiki-english-20171001"
DATASET = "quora-duplicate-questions"
DATA_PATH = f"./res/{DATASET}.stream.txt"
LDA_PATH = f"./res/{DATASET}-lda.h5"
PROCESSED_DATA_DIR = f"./res/{DATASET}-converted"
MIN_COUNT = 5
TOPK = 10


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
    "batch_size": 100000,
    "num_dims": 100,
    "io": {
      "lower": False
    }
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


if __name__ == "__main__":
  fire.Fire()
