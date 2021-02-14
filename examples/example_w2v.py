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

import gensim
from gensim import downloader as api
from gensim.test.utils import datapath

import nltk
from nltk.tokenize import RegexpTokenizer

from cusim import aux, CuW2V


LOGGER = aux.get_logger()
DOWNLOAD_PATH = "./res"
DATASET = "quora-duplicate-questions"
DATA_PATH = f"./res/{DATASET}.stream.txt"
PROCESSED_DATA_DIR = f"./res/{DATASET}-converted"
MIN_COUNT = 5
CUSIM_MODEL = "./res/cusim.w2v.model"
GENSIM_MODEL = "./res/gensim.w2v.model"

def download():
  if os.path.exists(DATA_PATH):
    LOGGER.info("%s already exists", DATA_PATH)
    return
  if not os.path.exists(DOWNLOAD_PATH):
    os.makedirs(DOWNLOAD_PATH, exist_ok=True)
  api.BASE_DIR = DOWNLOAD_PATH
  filepath = api.load(DATASET, return_path=True)
  LOGGER.info("filepath: %s", filepath)
  cmd = ["gunzip", "-c", filepath, ">", DATA_PATH]
  cmd = " ".join(cmd)
  LOGGER.info("cmd: %s", cmd)
  subprocess.call(cmd, shell=True)
  preprocess_data()

def preprocess_data():
  tokenizer = RegexpTokenizer(r'\w+')
  nltk.download("wordnet")
  lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
  fout = open(DATA_PATH + ".tmp", "wb")
  with open(DATA_PATH, "rb") as fin:
    for line in tqdm.tqdm(fin):
      line = line.decode("utf8").strip()
      line = preprocess_line(line, tokenizer, lemmatizer)
      fout.write((line + "\n").encode("utf8"))
  fout.close()
  os.rename(DATA_PATH + ".tmp", DATA_PATH)

def preprocess_line(line, tokenizer, lemmatizer):
  line = line.lower()
  line = tokenizer.tokenize(line)
  line = [token for token in line
          if not token.isnumeric() and len(token) > 1]
  line = [lemmatizer.lemmatize(token) for token in line]
  return " ".join(line)

def run_cusim():
  download()
  opt = {
    "data_path": DATA_PATH,
    "processed_data_dir": PROCESSED_DATA_DIR,
    "batch_size": 1000000,
    "num_dims": 100,
    "hyper_threads": 100,
    "epochs": 10,
    "lr": 0.001,
    "io": {
      "lower": False
    },
    "neg": 0,
    "skip_gram": False,
    "cbow_mean": False,
  }
  start = time.time()
  w2v = CuW2V(opt)
  w2v.train_model()
  LOGGER.info("elapsed for cusim w2v training: %.4e sec", time.time() - start)
  w2v.save_word2vec_format(CUSIM_MODEL, binary=False)
  evaluate_w2v_model(CUSIM_MODEL)

def run_gensim():
  download()
  start = time.time()
  model = gensim.models.Word2Vec(corpus_file=DATA_PATH, min_alpha=0.001,
                                 min_count=5, sg=False, hs=True, workers=8,
                                 alpha=0.001, negative=10, iter=10,
                                 cbow_mean=False)
  LOGGER.info("elapsed for gensim w2v training: %.4e sec", time.time() - start)
  model.wv.save_word2vec_format(GENSIM_MODEL, binary=False)
  LOGGER.info("gensim w2v model is saved to %s", GENSIM_MODEL)
  evaluate_w2v_model(GENSIM_MODEL)

def evaluate_w2v_model(model=GENSIM_MODEL):
  LOGGER.info("load word2vec format model from %s", model)
  model = gensim.models.KeyedVectors.load_word2vec_format(model)
  results = model.wv.evaluate_word_pairs(datapath("wordsim353.tsv"),
                                         case_insensitive=False)
  LOGGER.info("evaluation results: %s", results)




if __name__ == "__main__":
  fire.Fire()
