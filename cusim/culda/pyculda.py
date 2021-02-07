# Copyright (c) 2021 Jisang Yoon
# All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=no-name-in-module,too-few-public-methods,no-member
import os
# from os.path import join as pjoin

import json
import tempfile

# import h5py
# import numpy as np

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
