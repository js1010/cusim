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

from cusim import aux, IoUtils

LOGGER = aux.get_logger()
DATASET = "nips"
DIR_PATH = "./res"
BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/" \
           "bag-of-words/"

def download():
  if not os.path.exists(DIR_PATH):
    os.makedirs(DIR_PATH, exist_ok=True)

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

def run_io():
  iou = IoUtils()
  data_path = pjoin(DIR_PATH, f"docword.{DATASET}.txt")
  h5_path = pjoin(DIR_PATH, f"docword.{DATASET}.h5")
  iou.convert_bow_to_h5(data_path, h5_path)

# def run_lda():
#   download()
#   opt = {
#     "data_path": DATA_PATH,
#     "processed_data_dir": PROCESSED_DATA_DIR,
#     # "skip_preprocess":True,
#   }
#   lda = CuLDA(opt)
#   lda.train_model()
#   lda.save_model(LDA_PATH)
#   h5f = h5py.File(LDA_PATH, "r")
#   beta = h5f["beta"][:]
#   word_list = h5f["keys"][:]
#   num_topics = h5f["alpha"].shape[0]
#   for i in range(num_topics):
#     print("=" * 50)
#     print(f"topic {i + 1}")
#     words = np.argsort(-beta.T[i])[:10]
#     print("-" * 50)
#     for j in range(TOPK):
#       word = word_list[words[j]].decode("utf8")
#       prob = beta[words[j], i]
#       print(f"rank {j + 1}. word: {word}, prob: {prob}")
#   h5f.close()


if __name__ == "__main__":
  fire.Fire()
