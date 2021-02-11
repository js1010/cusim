# Copyright (c) 2021 Jisang Yoon
# All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=no-name-in-module,logging-format-truncated
import os
import subprocess
import fire

import h5py
import numpy as np
from gensim import downloader as api
from cusim import aux, IoUtils, CuLDA

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

def run_io():
  download()
  iou = IoUtils(opt={"chunk_lines": 10000, "num_threads": 8})
  iou.convert_stream_to_h5(DATA_PATH, 5, PROCESSED_DATA_DIR)


def run_lda():
  opt = {
    "data_path": DATA_PATH,
    "processed_data_dir": PROCESSED_DATA_DIR,
    "skip_preprocess":True,
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

if __name__ == "__main__":
  fire.Fire()
