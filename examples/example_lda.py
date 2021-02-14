# Copyright (c) 2021 Jisang Yoon
# All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=no-name-in-module,logging-format-truncated
# pylint: disable=too-few-public-methods
import os
from os.path import join as pjoin
# import time
import subprocess

# import tqdm
import fire
import wget

# import gensim

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
    "keys_path": keys_path
    # "skip_preprocess":True,
  }
  lda = CuLDA(opt)
  lda.train_model()


if __name__ == "__main__":
  fire.Fire()
