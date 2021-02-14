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
    # "skip_preprocess": True,
    "processed_data_path": processed_data_path,
    "keys_path": keys_path,
    "num_topics": 50,
    "num_iters_in_e_step": 20,
    # "skip_preprocess":True,
  }
  start = time.time()
  lda = CuLDA(opt)
  lda.train_model()
  LOGGER.info("elapsed for training LDA using cusim: %.4e sec",
              time.time() - start)
  lda.save_h5_model(pjoin(DIR_PATH, "cusim.lda.model.h5"))


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
  lda = LdaMulticore(docs, num_topics=5, workers=8, id2word=id2word)
  LOGGER.info("elapsed for training lda using gensim: %.4e sec",
              time.time() - start)
  lda.save(pjoin(DIR_PATH, "gensim.lda.model"))


if __name__ == "__main__":
  fire.Fire()
