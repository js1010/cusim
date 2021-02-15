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
import pandas as pd

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
PROCESSED_DATA_DIR = "./res/{DATASET}-processed"
CUSIM_MODEL = "./res/cusim.w2v.model"
GENSIM_MODEL = "./res/gensim.w2v.model"


# common hyperparameters
MIN_COUNT = 5
LEARNING_RATE = 0.001
NEG_SIZE = 10
NUM_DIMS = 100
CBOW_MEAN = False
EPOCHS = 10


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

def run_cusim(skip_gram=False, hierarchical_softmax=False):
  download()
  opt = {
    "data_path": DATA_PATH,
    "processed_data_dir": PROCESSED_DATA_DIR,
    # "skip_preprocess": os.path.exists(PROCESSED_DATA_DIR),
    "num_dims": NUM_DIMS,
    "epochs": EPOCHS,
    "word_min_count": MIN_COUNT,
    "lr": 0.001,
    "io": {
      "lower": False
    },
    "neg": 0 if hierarchical_softmax else NEG_SIZE,
    "skip_gram": skip_gram,
    "cbow_mean": CBOW_MEAN,
  }
  start = time.time()
  w2v = CuW2V(opt)
  w2v.train_model()
  elapsed = time.time() - start
  LOGGER.info("elapsed for cusim w2v training: %.4e sec", elapsed)
  w2v.save_word2vec_format(CUSIM_MODEL, binary=False)
  return elapsed, evaluate_w2v_model(CUSIM_MODEL)

def run_gensim(skip_gram=False, hierarchical_softmax=False, workers=8):
  download()
  start = time.time()
  model = gensim.models.Word2Vec(corpus_file=DATA_PATH, workers=workers,
                                 sg=skip_gram, hs=hierarchical_softmax,
                                 min_alpha=LEARNING_RATE, min_count=MIN_COUNT,
                                 alpha=LEARNING_RATE, negative=NEG_SIZE,
                                 iter=EPOCHS, cbow_mean=CBOW_MEAN,
                                 size=NUM_DIMS)
  elapsed = time.time() - start
  LOGGER.info("elapsed for gensim w2v training: %.4e sec", elapsed)
  model.wv.save_word2vec_format(GENSIM_MODEL, binary=False)
  LOGGER.info("gensim w2v model is saved to %s", GENSIM_MODEL)
  return elapsed, evaluate_w2v_model(GENSIM_MODEL)

def evaluate_w2v_model(model=GENSIM_MODEL):
  LOGGER.info("load word2vec format model from %s", model)
  model = gensim.models.KeyedVectors.load_word2vec_format(model)
  results = model.wv.evaluate_word_pairs(datapath("wordsim353.tsv"),
                                         case_insensitive=False)
  LOGGER.info("evaluation results: %s", results)
  return results

def run_experiments(sg0=False, hs0=False):
  training_time = {"attr": "training_time"}
  pearson = {"attr": "pearson"}
  spearman = {"attr": "spearman"}
  for i in [1, 2, 4, 8]:
    elapsed, evals = run_gensim(sg0, hs0, i)
    training_time[f"{i} workers"] = elapsed
    pearson[f"{i} workers"] = evals[0][0]
    spearman[f"{i} workers"] = evals[1][0]
  elapsed, evals = run_cusim(sg0, hs0)
  training_time["GPU"] = elapsed
  pearson["GPU"] = evals[0][0]
  spearman["GPU"] = evals[1][0]
  df0 = pd.DataFrame([training_time, pearson, spearman])
  df0.set_index("attr", inplace=True)
  print(df0.to_markdown())


if __name__ == "__main__":
  fire.Fire()
