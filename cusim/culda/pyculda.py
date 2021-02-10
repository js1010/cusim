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

  def preprocess_data(self):
    if self.opt.skip_preprocess:
      return
    iou = IoUtils()
    if not self.opt.data_dir:
      self.opt.data_dir = tempfile.TemporaryDirectory().name
    iou.convert_stream_to_h5(self.opt.data_path, self.opt.data_dir)

  def init_model(self):
    # load voca
    self.logger.info("load key from %s", pjoin(self.opt.data_dir, "keys.txt"))
    with open(pjoin(self.opt.data_dir, "keys.txt"), "rb") as fin:
      self.words = [line.strip() for line in fin]
    self.num_words = len(self.words)
    self.logger.info("number of words: %d", self.num_words)

    # random initialize alpha and beta
    self.alpha = np.random.uniform( \
      size=(self.opt.num_topics,)).astype(np.float32)
    self.beta = np.random.uniform( \
      size=(self.num_words, self.opt.num_topics)).astype(np.float32)
    self.beta /= np.sum(self.beta, axis=0)[None, :]
    self.logger.info("alpha %s, beta %s initialized",
                     self.alpha.shape, self.beta.shape)

    # zero initialize grad alpha and new beta
    self.grad_alpha = np.zeros(shape=self.alpha.shape, dtype=np.float32)
    self.new_beta = np.zeros(shape=self.beta.shape, dtype=np.float32)

    # push it to gpu
    self.obj.load_model(self.alpha, self.beta, self.grad_alpha, self.new_beta)

  def train_model(self):
    self.preprocess_data()
    self.init_model()

  def _train_Estep(self, h5f):
    offset, size = 0, h5f["indptr"].shape[0] - 1
    steps = (size - 1) // batch_size + 1
    pbar = aux.Progbar(size)
    for step in range(steps):
      next_offset = min(size, offset + batch_size)
      indptr = h5f["indptr"][offset:next_offset + 1]
      beg, end = indptr[0], indptr[-1]
      indptr -= beg
      indices = h5f["indices"][beg:end]
      offset = next_offset

      self.obj.FeedData(indices, indptr, self.opt.num_iters_in_Estep)
      pbar.update(offset)

  def _train_Mstep(self):
    pass
