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

from cusim import aux, IoUtils
from cusim.cuw2v.cuw2v_bind import CuW2VBind
from cusim.config_pb2 import CuW2VConfigProto
from cusim.constants import EPS, WARP_SIZE

class CuW2V:
  def __init__(self, opt=None):
    self.opt = aux.get_opt_as_proto(opt or {}, CuW2VConfigProto)
    self.logger = aux.get_logger("culda", level=self.opt.py_log_level)

    assert self.opt.block_dim <= WARP_SIZE ** 2 and \
      self.opt.block_dim % WARP_SIZE == 0, \
      f"invalid block dim ({self.opt.block_dim}, warp size: {WARP_SIZE})"

    tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)
    opt_content = json.dumps(aux.proto_to_dict(self.opt), indent=2)
    tmp.write(opt_content)
    tmp.close()

    self.logger.info("opt: %s", opt_content)
    self.obj = CuW2VBind()
    assert self.obj.init(bytes(tmp.name, "utf8")), f"failed to load {tmp.name}"
    os.remove(tmp.name)

    self.words, self.word_count, self.num_words, self.num_docs = \
      None, None, None, None
    self.emb_in, self.emb_out = None, None

  def preprocess_data(self):
    if self.opt.skip_preprocess:
      return
    iou = IoUtils(aux.proto_to_dict(self.opt.io))
    if not self.opt.processed_data_dir:
      self.opt.processed_data_dir = tempfile.TemporaryDirectory().name
    iou.convert_stream_to_h5(self.opt.data_path, self.opt.word_min_count,
                             self.opt.processed_data_dir)

  def init_model(self):
    # load voca
    data_dir = self.opt.processed_data_dir
    keys_path = pjoin(data_dir, "keys.txt")
    count_path = pjoin(data_dir, "count.txt")
    self.logger.info("load key, count from %s, %s", keys_path, count_path)
    with open(keys_path, "rb") as fin:
      self.words = [line.strip() for line in fin]
    with open(count_path, "rb") as fin:
      self.word_count = np.array([float(line.strip()) for line in fin],
                                 dtype=np.float32)
    self.word_count = np.power(self.word_count, self.opt.count_power)
    self.num_words = len(self.words)
    assert len(self.words) == len(self.word_count)

    # count number of docs
    h5f = h5py.File(pjoin(data_dir, "token.h5"), "r")
    self.num_docs = h5f["indptr"].shape[0] - 1
    h5f.close()

    self.logger.info("number of words: %d, docs: %d",
                     self.num_words, self.num_docs)

    if self.opt.neg:
      self.obj.build_random_table( \
        self.word_count, self.opt.random_size, self.opt.num_threads)
    else:
      self.obj.build_huffman_tree(self.word_count)

    # random initialize alpha and beta
    np.random.seed(self.opt.seed)
    self.emb_in = np.random.normal( \
      size=(self.num_words, self.opt.num_dims)).astype(np.float32)
    out_words = self.num_words if self.opt.neg else self.num_words - 1
    self.emb_out = np.random.uniform( \
      size=(out_words, self.opt.num_dims)).astype(np.float32)
    self.logger.info("emb_in %s, emb_out %s initialized",
                     self.emb_in.shape, self.emb_out.shape)

    if self.opt.pretrained_model.filename:
      self.load_word2vec_model(**aux.proto_to_dict(self.opt.pretrained_model))

    # push it to gpu
    self.obj.load_model(self.emb_in, self.emb_out)

  def train_model(self):
    self.preprocess_data()
    self.init_model()
    h5f = h5py.File(pjoin(self.opt.processed_data_dir, "token.h5"), "r")
    for epoch in range(1, self.opt.epochs + 1):
      self.logger.info("Epoch %d / %d", epoch, self.opt.epochs)
      self._train_epoch(h5f)
    self.obj.pull()
    h5f.close()

  def _train_epoch(self, h5f):
    offset, size = 0, h5f["cols"].shape[0]
    pbar = aux.Progbar(size, stateful_metrics=["loss"])
    loss_nume, loss_deno = 0, 0
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
      offset = next_offset

      # call cuda kernel
      _loss_nume, _loss_deno = \
        self.obj.feed_data(cols, indptr.astype(np.int32))

      # accumulate loss
      loss_nume += _loss_nume
      loss_deno += _loss_deno
      loss = loss_nume / (loss_deno + EPS)

      # update progress bar
      pbar.update(end, values=[("loss", loss)])
      if end == size:
        break

  def save_h5_model(self, filename):
    self.logger.info("save h5 format model to %s", filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    h5f = h5py.File(filename, "w")
    h5f.create_dataset("emb_in", data=self.emb_in)
    h5f.create_dataset("emb_out", data=self.emb_out)
    h5f.create_dataset("keys", data=np.array(self.words))
    h5f.close()

  def save_word2vec_format(self, filename, binary=False, prefix=""):
    self.logger.info("save word2vec format model to %s, "
                     "binary: %s, prefix: '%s'", filename, binary, prefix)
    # save model compatible with gensim and original w2v code by Google
    with open(filename, "wb") as fout:
      fout.write(f"{self.num_words} {self.opt.num_dims}\n".encode("utf8"))
      for idx, word in enumerate(self.words):
        vec = self.emb_in[idx]
        if binary:
          fout.write(f"{prefix}{word} ".encode("utf8") + vec.tobytes())
        else:
          fout.write(f"{prefix}{word} "
                     f"{' '.join(repr(val) for val in vec)}\n".encode("utf8"))

  def load_word2vec_forrmat(self, filename, binary=False,
                            symmetry=False, no_header=False):
    self.logger.info("load pretrained model from %s", filename)
    # copy pretrained model to emb_out as well only if
    # we use negative sampling, NOT hierarchical softmax
    assert not symmetry or self.opt.neg, "no symmetry in hierarchical softmax"

    # read variable
    vector_dict = {}
    with open(filename, "rb") as fin:
      if not no_header:
        fin.readline()  # throw one line
      for line in fin:
        if binary:
          key, vec = line.split()
          vector_dict[key] = np.fromstring(vec, dtype=np.float32)
        else:
          line_vec = line.split()
          key = line_vec[0]
          vec = np.array([float(val) for val in line_vec[1:]],
                         dtype=np.float32)
          vector_dict[key] = vec

    # copy to variable
    word_idmap = {word: idx for idx, word in enumerate(self.words)}
    for key, vec in self.vector_dict:
      assert len(vec) == self.opt.num_dims
      if key not in word_idmap:
        continue
      idx = word_idmap[key]
      self.emb_in[idx, :] = vec
      if symmetry:
        self.emb_out[idx, :] = vec
