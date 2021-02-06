# Copyright (c) 2021 Jisang Yoon
# All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=no-name-in-module,logging-format-truncated
import fire
from cusim import aux
from cusim.ioutils import IoUtils

LOGGER = aux.get_logger()

def run():
  iou = IoUtils()
  iou.load_gensim_vocab("corpora.txt", 5)


if __name__ == "__main__":
  fire.Fire()
