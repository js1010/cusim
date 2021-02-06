# Copyright (c) 2021 Jisang Yoon
# All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=no-name-in-module,too-few-public-methods,no-member
from cusim import aux
from cusim.ioutils.ioutils_bind import IoUtilsBind


class IoUtils:
  def __init__(self, log_level=2):
    self.logger = aux.get_logger("ioutils", log_level=log_level)
    self.obj = IoUtilsBind(log_level)

  def load_gensim_vocab(self, filepath, min_count):
    self.obj.load_gensim_vocab(filepath, min_count)
