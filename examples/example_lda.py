# Copyright (c) 2021 Jisang Yoon
# All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=no-name-in-module,logging-format-truncated
# pylint: disable=too-few-public-methods
import os
from os.path import join as pjoin
import time
import pickle
import subprocess

import tqdm
import fire
import wget
import h5py
import numpy as np

# import gensim
from gensim.models.ldamulticore import LdaMulticore

from cusim import aux, CuLDA

LOGGER = aux.get_logger()
# DATASET = "nips"
DATASET = "nytimes"
DIR_PATH = "./res"
BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/" \
           "bag-of-words/"

def download():
  if not os.path.exists(DIR_PATH):
    os.makedirs(DIR_PATH, exist_ok=True)

  if os.path.exists(pjoin(DIR_PATH, f"docword.{DATASET}.txt")):
    LOGGER.info("path %s already exists",
                pjoin(DIR_PATH, f"docword.{DATASET}.txt"))
    return

  # download docword
  filename = f"docword.{DATASET}.txt.gz"
  out_path = pjoin(DIR_PATH, filename)
  LOGGER.info("download %s to %s", BASE_URL + filename, out_path)
  wget.download(BASE_URL + filename, out=out_path)
  print()

  # decompress
  cmd = ["gunzip", "-c", out_path, ">",
         pjoin(DIR_PATH, f"docword.{DATASET}.txt")]
  cmd = " ".join(cmd)
  subprocess.call(cmd, shell=True)
  os.remove(pjoin(DIR_PATH, filename))

  # download vocab
  filename = f"vocab.{DATASET}.txt"
  out_path = pjoin(DIR_PATH, filename)
  LOGGER.info("download %s to %s", BASE_URL + filename, out_path)
  wget.download(BASE_URL + filename, out=out_path)
  print()

def run_cusim():
  download()
  data_path = pjoin(DIR_PATH, f"docword.{DATASET}.txt")
  keys_path = pjoin(DIR_PATH, f"vocab.{DATASET}.txt")
  processed_data_path = pjoin(DIR_PATH, f"docword.{DATASET}.h5")
  opt = {
    "data_path": data_path,
    "processed_data_path": processed_data_path,
    "keys_path": keys_path,
    "num_topics": 50,
    "num_iters_in_e_step": 10,
    "reuse_gamma": True,
    # "skip_preprocess": os.path.exists(processed_data_path),
  }
  start = time.time()
  lda = CuLDA(opt)
  lda.train_model()
  LOGGER.info("elapsed for training LDA using cusim: %.4e sec",
              time.time() - start)
  h5_model_path = pjoin(DIR_PATH, "cusim.lda.model.h5")
  lda.save_h5_model(h5_model_path)
  show_cusim_topics(h5_model_path)

def show_cusim_topics(h5_model_path, topk=10):
  h5f = h5py.File(h5_model_path, "r")
  beta = h5f["beta"][:, :].T
  keys = h5f["keys"][:]
  show_topics(beta, keys, topk, "cusim.topics.txt")

def build_gensim_corpus():
  corpus_path = pjoin(DIR_PATH, f"docword.{DATASET}.pk")
  if os.path.exists(corpus_path):
    LOGGER.info("load corpus from %s", corpus_path)
    with open(corpus_path, "rb") as fin:
      ret = pickle.loads(fin.read())
    return ret

  # get corpus for gensim lda
  data_path = pjoin(DIR_PATH, f"docword.{DATASET}.txt")
  LOGGER.info("build corpus from %s", data_path)
  docs, doc, curid = [], [], -1
  with open(data_path, "r") as fin:
    for idx, line in tqdm.tqdm(enumerate(fin)):
      if idx < 3:
        continue
      docid, wordid, count = line.strip().split()
      # zero-base id
      docid, wordid, count = int(docid) - 1, int(wordid) - 1, float(count)
      if 0 <= curid < docid:
        docs.append(doc)
        doc = []
      doc.append((wordid, count))
      curid = docid
    docs.append(doc)
  LOGGER.info("save corpus to %s", corpus_path)
  with open(corpus_path, "wb") as fout:
    fout.write(pickle.dumps(docs, 2))
  return docs

def run_gensim():
  docs = build_gensim_corpus()
  keys_path = pjoin(DIR_PATH, f"vocab.{DATASET}.txt")
  LOGGER.info("load vocab from %s", keys_path)
  id2word = {}
  with open(keys_path, "rb") as fin:
    for idx, line in enumerate(fin):
      id2word[idx] = line.strip()

  start = time.time()
  # 3 = real cores - 1
  lda = LdaMulticore(docs, num_topics=50, workers=None,
                     id2word=id2word, iterations=10)
  LOGGER.info("elapsed for training lda using gensim: %.4e sec",
              time.time() - start)
  model_path = pjoin(DIR_PATH, "gensim.lda.model")
  LOGGER.info("save gensim lda model to %s", model_path)
  lda.save(model_path)
  show_gensim_topics(model_path)

def show_gensim_topics(model_path=None, topk=10):
  # load beta
  model_path = model_path or pjoin(DIR_PATH, "gensim.lda.model")
  LOGGER.info("load gensim lda model from %s", model_path)
  lda = LdaMulticore.load(model_path)
  beta = lda.state.get_lambda()
  beta /= np.sum(beta, axis=1)[:, None]

  # load keys
  keys_path = pjoin(DIR_PATH, f"vocab.{DATASET}.txt")
  LOGGER.info("load vocab from %s", keys_path)
  with open(keys_path, "rb") as fin:
    keys = [line.strip() for line in fin]
  show_topics(beta, keys, topk, "gensim.topics.txt")

def show_topics(beta, keys, topk, result_path):
  LOGGER.info("save results to %s (topk: %d)", result_path, topk)
  fout = open(result_path, "w")
  for idx in range(beta.shape[0]):
    print("=" * 50)
    fout.write("=" * 50 + "\n")
    print(f"topic {idx + 1}")
    fout.write(f"topic {idx + 1}" + "\n")
    print("-" * 50)
    fout.write("-" * 50 + "\n")
    _beta = beta[idx, :]
    indices = np.argsort(-_beta)[:topk]
    for rank, wordid in enumerate(indices):
      word = keys[wordid].decode("utf8")
      prob = _beta[wordid]
      print(f"rank {rank + 1}. {word}: {prob}")
      fout.write(f"rank {rank + 1}. {word}: {prob}" + "\n")
  fout.close()


if __name__ == "__main__":
  fire.Fire()
