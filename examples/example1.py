# Copyright (c) 2021 Jisang Yoon
# All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=no-name-in-module,logging-format-truncated
import fire
from cusim import aux, IoUtils

LOGGER = aux.get_logger()
CORPORA_PATH = "res/corpora.txt"
MIN_COUNT = 5

def run():
  iou = IoUtils()
  iou.load_gensim_vocab(CORPORA_PATH, MIN_COUNT)


if __name__ == "__main__":
  fire.Fire()
