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

from cusim import aux
from cusim.ioutils.ioutils_bind import IoUtilsBind
from cusim.config_pb2 import IoUtilsConfigProto

class IoUtils:
  def __init__(self, opt=None):
    self.opt = aux.get_opt_as_proto(opt or {}, IoUtilsConfigProto)
    self.logger = aux.get_logger("ioutils", level=self.opt.py_log_level)

    tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)
    opt_content = json.dumps(aux.proto_to_dict(self.opt), indent=2)
    tmp.write(opt_content)
    tmp.close()

    self.logger.info("opt: %s", opt_content)
    self.obj = IoUtilsBind()
    assert self.obj.init(bytes(tmp.name, "utf8")), f"failed to load {tmp.name}"
    os.remove(tmp.name)

  def load_stream_vocab(self, filepath, min_count, keys_path):
    full_num_lines = self.obj.load_stream_file(filepath)
    pbar = aux.Progbar(full_num_lines, unit_name="line",
                        stateful_metrics=["word_count"])
    processed = 0
    while True:
      read_lines, word_count = \
        self.obj.read_stream_for_vocab(
          self.opt.chunk_lines, self.opt.num_threads)
      processed += read_lines
      pbar.update(processed, values=[("word_count", word_count)])
      if processed == full_num_lines:
        break
    self.obj.get_word_vocab(min_count, keys_path)

  def convert_stream_to_h5(self, filepath, min_count, out_dir,
                           chunk_indices=10000, seed=777):
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    keys_path = pjoin(out_dir, "keys.txt")
    token_path = pjoin(out_dir, "token.h5")
    self.logger.info("save key and token to %s, %s",
                     keys_path, token_path)
    self.load_stream_vocab(filepath, min_count, keys_path)
    full_num_lines = self.obj.load_stream_file(filepath)
    pbar = aux.Progbar(full_num_lines, unit_name="line")
    processed = 0
    h5f = h5py.File(token_path, "w")
    rows = h5f.create_dataset("rows", shape=(chunk_indices,),
                              maxshape=(None,), dtype=np.int32,
                              chunks=(chunk_indices,))
    cols = h5f.create_dataset("cols", shape=(chunk_indices,),
                              maxshape=(None,), dtype=np.int32,
                              chunks=(chunk_indices,))
    vali = h5f.create_dataset("vali", shape=(chunk_indices,),
                              maxshape=(None,), dtype=np.float32,
                              chunks=(chunk_indices,))
    indptr =  h5f.create_dataset("indptr", shape=(full_num_lines + 1,),
                                 dtype=np.int32, chunks=True)
    processed, offset = 1, 0
    indptr[0] = 0
    while True:
      read_lines, data_size = self.obj.tokenize_stream(
        self.opt.chunk_lines, self.opt.num_threads)
      _rows = np.empty(shape=(data_size,), dtype=np.int32)
      _cols = np.empty(shape=(data_size,), dtype=np.int32)
      _indptr = np.empty(shape=(read_lines,), dtype=np.int32)
      self.obj.get_token(_rows, _cols, _indptr)
      rows.resize((offset + data_size,))
      rows[offset:offset + data_size] = _rows + (processed - 1)
      cols.resize((offset + data_size,))
      cols[offset:offset + data_size] = _cols
      vali.resize((offset + data_size,))
      vali[offset:offset + data_size] = \
        np.random.uniform(size=(data_size,)).astype(np.float32)
      indptr[processed:processed + read_lines] = _indptr + offset
      offset += data_size
      processed += read_lines
      pbar.update(processed - 1)
      if processed == full_num_lines + 1:
        break
    h5f.close()
