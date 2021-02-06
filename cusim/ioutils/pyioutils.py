# Copyright (c) 2021 Jisang Yoon
# All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=no-name-in-module,too-few-public-methods,no-member
import os
import json
import tempfile
import tqdm

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

  def load_stream_vocab(self, filepath, min_count, chunk_lines=100000):
    full_num_lines = self.obj.load_stream_file(filepath)
    pbar = tqdm.trange(full_num_lines)
    while True:
      num_lines, finished = self.obj.read_stream_for_vocab(chunk_lines)
      pbar.update(num_lines)
      if finished:
        break
    pbar.close()
    self.obj.get_word_vocab(min_count)
