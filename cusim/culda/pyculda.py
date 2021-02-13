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

import h5py
import numpy as np
from scipy.special import polygamma as pg

from cusim import aux, IoUtils
from cusim.culda.culda_bind import CuLDABind
from cusim.config_pb2 import CuLDAConfigProto
from cusim.constants import EPS, WARP_SIZE


class CuLDA:
  def __init__(self, opt=None):
    self.opt = aux.get_opt_as_proto(opt or {}, CuLDAConfigProto)
    self.logger = aux.get_logger("culda", level=self.opt.py_log_level)

    assert self.opt.block_dim <= WARP_SIZE ** 2 and \
      self.opt.block_dim % WARP_SIZE == 0, \
      f"invalid block dim ({self.opt.block_dim}, warp size: {WARP_SIZE})"

    tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)
    opt_content = json.dumps(aux.proto_to_dict(self.opt), indent=2)
    tmp.write(opt_content)
    tmp.close()

    self.logger.info("opt: %s", opt_content)
    self.obj = CuLDABind()
    assert self.obj.init(bytes(tmp.name, "utf8")), f"failed to load {tmp.name}"
    os.remove(tmp.name)

    self.words, self.num_words, self.num_docs = None, None, None
    self.alpha, self.beta, self.grad_alpha, self.new_beta = \
      None, None, None, None

  def preprocess_data(self):
    if self.opt.skip_preprocess:
      return
    iou = IoUtils(self.opt.io)
    if not self.opt.processed_data_dir:
      self.opt.processed_data_dir = tempfile.TemporaryDirectory().name
    iou.convert_stream_to_h5(self.opt.data_path, self.opt.word_min_count,
                             self.opt.processed_data_dir)

  def init_model(self):
    # load voca
    data_dir = self.opt.processed_data_dir
    self.logger.info("load key from %s", pjoin(data_dir, "keys.txt"))
    with open(pjoin(data_dir, "keys.txt"), "rb") as fin:
      self.words = [line.strip() for line in fin]
    self.num_words = len(self.words)

    # count number of docs
    h5f = h5py.File(pjoin(data_dir, "token.h5"), "r")
    self.num_docs = h5f["indptr"].shape[0] - 1
    h5f.close()

    self.logger.info("number of words: %d, docs: %d",
                     self.num_words, self.num_docs)

    # random initialize alpha and beta
    np.random.seed(self.opt.seed)
    self.alpha = np.random.uniform( \
      size=(self.opt.num_topics,)).astype(np.float32)
    self.beta = np.random.uniform( \
      size=(self.num_words, self.opt.num_topics)).astype(np.float32)
    self.beta /= np.sum(self.beta, axis=0)[None, :]
    self.logger.info("alpha %s, beta %s initialized",
                     self.alpha.shape, self.beta.shape)

    # zero initialize grad alpha and new beta
    block_cnt = self.obj.get_block_cnt()
    self.grad_alpha = np.zeros(shape=(block_cnt, self.opt.num_topics),
                               dtype=np.float32)
    self.new_beta = np.zeros(shape=self.beta.shape, dtype=np.float32)
    self.logger.info("grad alpha %s, new beta %s initialized",
                     self.grad_alpha.shape, self.new_beta.shape)

    # push it to gpu
    self.obj.load_model(self.alpha, self.beta, self.grad_alpha, self.new_beta)

  def train_model(self):
    self.preprocess_data()
    self.init_model()
    h5f = h5py.File(pjoin(self.opt.processed_data_dir, "token.h5"), "r")
    for epoch in range(1, self.opt.epochs + 1):
      self.logger.info("Epoch %d / %d", epoch, self.opt.epochs)
      self._train_e_step(h5f)
      self._train_m_step()
    h5f.close()

  def _train_e_step(self, h5f):
    offset, size = 0, h5f["cols"].shape[0]
    pbar = aux.Progbar(size, stateful_metrics=["train_loss", "vali_loss"])
    train_loss_nume, train_loss_deno = 0, 0
    vali_loss_nume, vali_loss_deno = 0, 0
    while True:
      target = h5f["indptr"][offset] + self.opt.batch_size
      if target < size:
        next_offset = h5f["rows"][target]
      else:
        next_offset = h5f["indptr"].shape[0] - 1
      indptr = h5f["indptr"][offset:next_offset + 1]
      beg, end = indptr[0], indptr[-1]
      indptr -= beg
      cols = h5f["cols"][beg:end]
      vali = (h5f["vali"][beg:end] < self.opt.vali_p).astype(np.bool)
      offset = next_offset

      # call cuda kernel
      train_loss, vali_loss = \
        self.obj.feed_data(cols, indptr, vali, self.opt.num_iters_in_e_step)

      # accumulate loss
      train_loss_nume -= train_loss
      vali_loss_nume -= vali_loss
      vali_cnt = np.count_nonzero(vali)
      train_cnt = len(vali) - vali_cnt
      train_loss_deno += train_cnt
      vali_loss_deno += vali_cnt
      train_loss = train_loss_nume / (train_loss_deno + EPS)
      vali_loss = vali_loss_nume / (vali_loss_deno + EPS)

      # update progress bar
      pbar.update(end, values=[("train_loss", train_loss),
                               ("vali_loss", vali_loss)])
      if end == size:
        break

  def _train_m_step(self):
    self.obj.pull()

    # update beta
    self.new_beta[:, :] = np.maximum(self.new_beta, EPS)
    self.beta[:, :] = self.new_beta / np.sum(self.new_beta, axis=0)[None, :]
    self.new_beta[:, :] = 0

    # update alpha
    alpha_sum = np.sum(self.alpha)
    gvec = np.sum(self.grad_alpha, axis=0)
    gvec += self.num_docs * (pg(0, alpha_sum) - pg(0, self.alpha))
    hvec = self.num_docs * pg(1, self.alpha)
    z_0 = pg(1, alpha_sum)
    c_nume = np.sum(gvec / hvec)
    c_deno = 1 / z_0 + np.sum(1 / hvec)
    c_0 = c_nume / c_deno
    delta = (gvec - c_0) / hvec
    self.alpha -= delta
    self.alpha[:] = np.maximum(self.alpha, EPS)
    self.grad_alpha[:,:] = 0

    self.obj.push()

  def save_model(self, model_path):
    self.logger.info("save model path: %s", model_path)
    h5f = h5py.File(model_path, "w")
    h5f.create_dataset("alpha", data=self.alpha)
    h5f.create_dataset("beta", data=self.beta)
    h5f.create_dataset("keys", data=np.array(self.words))
    h5f.close()
