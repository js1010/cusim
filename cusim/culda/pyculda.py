# Copyright (c) 2021 Jisang Yoon
# All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=no-name-in-module,too-few-public-methods,no-member
import os
from os.path import join as pjoin

import json
import tempfile

# import h5py
import numpy as np

from cusim import aux
from cusim.culda.culda_bind import CuLDABind
from cusim.config_pb2 import CuLDAConfigProto

class CuLDA:
  def __init__(self, opt=None):
    self.opt = aux.get_opt_as_proto(opt or {}, CuLDAConfigProto)
    self.logger = aux.get_logger("culda", level=self.opt.py_log_level)

    tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)
    opt_content = json.dumps(aux.proto_to_dict(self.opt), indent=2)
    tmp.write(opt_content)
    tmp.close()

    self.logger.info("opt: %s", opt_content)
    self.obj = CuLDABind()
    assert self.obj.init(bytes(tmp.name, "utf8")), f"failed to load {tmp.name}"
    os.remove(tmp.name)

    self.words, self.num_words = None, None
    self.alpha, self.beta, self.grad_alpha, self.new_beta = \
      None, None, None, None

  def init_model(self):
    with open(pjoin(self.opt.data_dir, "keys.txt"), "r") as fin:
      self.words = [line.strip() for line in fin]
    self.num_words = len(self.words)
    self.alpha = \
      np.abs(np.uniform(shape=(self.opt.num_topics,))).astype(np.float32)
    self.beta = np.abs(np.uniform( \
      shape=(self.num_words, self.opt.num_topics))).astype(np.float32)
    self.beta /= np.sum(self.beta, axis=1)[None, :]
    self.grad_alpha = np.zeros(shape=self.alpha.shape, dtype=np.float32)
    self.new_beta = np.zeros(shape=self.beta.shape, dtype=np.float32)
    self.obj.load_model(self.alpha, self.beta, self.grad_alpha, self.new_beta)
