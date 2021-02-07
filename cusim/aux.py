# Copyright (c) 2021 Jisang Yoon
# All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
import os
import re
import sys
import json
import time
import logging
import logging.handlers
import numpy as np
import jsmin
from google.protobuf.json_format import Parse, MessageToDict

# get_logger and Option refer to
# https://github.com/kakao/buffalo/blob/
# 5f571c2c7d8227e6625c6e538da929e4db11b66d/buffalo/misc/aux.py
def get_logger(name=__file__, level=2):
  if level == 1:
    level = logging.WARNING
  elif level == 2:
    level = logging.INFO
  elif level == 3:
    level = logging.DEBUG
  logger = logging.getLogger(name)
  if logger.handlers:
    return logger
  logger.setLevel(level)
  sh0 = logging.StreamHandler()
  sh0.setLevel(level)
  formatter = logging.Formatter('[%(levelname)-8s] %(asctime)s '
                                '[%(filename)s] [%(funcName)s:%(lineno)d]'
                                '%(message)s', '%Y-%m-%d %H:%M:%S')
  sh0.setFormatter(formatter)
  logger.addHandler(sh0)
  return logger

# This function helps you to read non-standard json strings.
# - Handles json string with c++ style inline comments
# - Handles json string with trailing commas.
def load_json_string(cont):
  # (1) Removes comment.
  #     Refer to https://plus.google.com/+DouglasCrockfordEsq/posts/RK8qyGVaGSr
  cont = jsmin.jsmin(cont)

  # (2) Removes trailing comma.
  cont = re.sub(",[ \t\r\n]*}", "}", cont)
  cont = re.sub(",[ \t\r\n]*" + r"\]", "]", cont)

  return json.loads(cont)


# function read json file from filename
def load_json_file(fname):
  with open(fname, "r") as fin:
    ret = load_json_string(fin.read())
  return ret

# use protobuf to restrict field and types
def get_opt_as_proto(raw, proto_type=None):
  assert proto_type is not None
  proto = proto_type()
  # convert raw to proto
  Parse(json.dumps(Option(raw)), proto)
  err = []
  assert proto.IsInitialized(err), \
    f"some required fields are missing in proto {err}\n {proto}"
  return proto

def proto_to_dict(proto):
  return MessageToDict(proto, \
    including_default_value_fields=True, \
    preserving_proto_field_name=True)

def copy_proto(proto):
  newproto = type(proto)()
  Parse(json.dumps(proto_to_dict(proto)), newproto)
  return newproto

class Option(dict):
  def __init__(self, *args, **kwargs):
    args = [arg if isinstance(arg, dict)
            else load_json_file(arg) for arg in args]
    super().__init__(*args, **kwargs)
    for arg in args:
      if isinstance(arg, dict):
        for k, val in arg.items():
          if isinstance(val, dict):
            self[k] = Option(val)
          else:
            self[k] = val
    if kwargs:
      for k, val in kwargs.items():
        if isinstance(val, dict):
          self[k] = Option(val)
        else:
          self[k] = val

  def __getattr__(self, attr):
    return self.get(attr)

  def __setattr__(self, key, value):
    self.__setitem__(key, value)

  def __setitem__(self, key, value):
    super().__setitem__(key, value)
    self.__dict__.update({key: value})

  def __delattr__(self, item):
    self.__delitem__(item)

  def __delitem__(self, key):
    super().__delitem__(key)
    del self.__dict__[key]

  def __getstate__(self):
    return vars(self)

  def __setstate__(self, state):
    vars(self).update(state)

# reference: https://github.com/tensorflow/tensorflow/blob/
# 85c8b2a817f95a3e979ecd1ed95bff1dc1335cff/tensorflow/python/
# keras/utils/generic_utils.py#L483
class Progbar:
  # pylint: disable=too-many-branches,too-many-statements,invalid-name
  # pylint: disable=blacklisted-name,no-else-return
  """Displays a progress bar.
  Arguments:
      target: Total number of steps expected, None if unknown.
      width: Progress bar width on screen.
      verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
      stateful_metrics: Iterable of string names of metrics that should *not* be
        averaged over time. Metrics in this list will be displayed as-is. All
        others will be averaged by the progbar before display.
      interval: Minimum visual progress update interval (in seconds).
      unit_name: Display name for step counts (usually "step" or "sample").
  """

  def __init__(self,
               target,
               width=30,
               verbose=1,
               interval=0.05,
               stateful_metrics=None,
               unit_name='step'):
    self.target = target
    self.width = width
    self.verbose = verbose
    self.interval = interval
    self.unit_name = unit_name
    if stateful_metrics:
      self.stateful_metrics = set(stateful_metrics)
    else:
      self.stateful_metrics = set()

    self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                              sys.stdout.isatty()) or
                             'ipykernel' in sys.modules or
                             'posix' in sys.modules or
                             'PYCHARM_HOSTED' in os.environ)
    self._total_width = 0
    self._seen_so_far = 0
    # We use a dict + list to avoid garbage collection
    # issues found in OrderedDict
    self._values = {}
    self._values_order = []
    self._start = time.time()
    self._last_update = 0

    self._time_after_first_step = None

  def update(self, current, values=None, finalize=None):
    """Updates the progress bar.
    Arguments:
        current: Index of current step.
        values: List of tuples: `(name, value_for_last_step)`. If `name` is in
          `stateful_metrics`, `value_for_last_step` will be displayed as-is.
          Else, an average of the metric over time will be displayed.
        finalize: Whether this is the last update for the progress bar. If
          `None`, defaults to `current >= self.target`.
    """
    if finalize is None:
      if self.target is None:
        finalize = False
      else:
        finalize = current >= self.target

    values = values or []
    for k, v in values:
      if k not in self._values_order:
        self._values_order.append(k)
      if k not in self.stateful_metrics:
        # In the case that progress bar doesn't have a target value in the first
        # epoch, both on_batch_end and on_epoch_end will be called, which will
        # cause 'current' and 'self._seen_so_far' to have the same value. Force
        # the minimal value to 1 here, otherwise stateful_metric will be 0s.
        value_base = max(current - self._seen_so_far, 1)
        if k not in self._values:
          self._values[k] = [v * value_base, value_base]
        else:
          self._values[k][0] += v * value_base
          self._values[k][1] += value_base
      else:
        # Stateful metrics output a numeric value. This representation
        # means "take an average from a single value" but keeps the
        # numeric formatting.
        self._values[k] = [v, 1]
    self._seen_so_far = current

    now = time.time()
    info = ' - %.0fs' % (now - self._start)
    if self.verbose == 1:
      if now - self._last_update < self.interval and not finalize:
        return

      prev_total_width = self._total_width
      if self._dynamic_display:
        sys.stdout.write('\b' * prev_total_width)
        sys.stdout.write('\r')
      else:
        sys.stdout.write('\n')

      if self.target is not None:
        numdigits = int(np.log10(self.target)) + 1
        bar = ('%' + str(numdigits) + 'd/%d [') % (current, self.target)
        prog = float(current) / self.target
        prog_width = int(self.width * prog)
        if prog_width > 0:
          bar += ('=' * (prog_width - 1))
          if current < self.target:
            bar += '>'
          else:
            bar += '='
        bar += ('.' * (self.width - prog_width))
        bar += ']'
      else:
        bar = '%7d/Unknown' % current

      self._total_width = len(bar)
      sys.stdout.write(bar)

      time_per_unit = self._estimate_step_duration(current, now)

      if self.target is None or finalize:
        if time_per_unit >= 1 or time_per_unit == 0:
          info += ' %.0fs/%s' % (time_per_unit, self.unit_name)
        elif time_per_unit >= 1e-3:
          info += ' %.0fms/%s' % (time_per_unit * 1e3, self.unit_name)
        else:
          info += ' %.0fus/%s' % (time_per_unit * 1e6, self.unit_name)
      else:
        eta = time_per_unit * (self.target - current)
        if eta > 3600:
          eta_format = '%d:%02d:%02d' % (eta // 3600,
                                         (eta % 3600) // 60, eta % 60)
        elif eta > 60:
          eta_format = '%d:%02d' % (eta // 60, eta % 60)
        else:
          eta_format = '%ds' % eta

        info = ' - ETA: %s' % eta_format

      for k in self._values_order:
        info += ' - %s:' % k
        if isinstance(self._values[k], list):
          avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
          if abs(avg) > 1e-3:
            info += ' %.4f' % avg
          else:
            info += ' %.4e' % avg
        else:
          info += ' %s' % self._values[k]

      self._total_width += len(info)
      if prev_total_width > self._total_width:
        info += (' ' * (prev_total_width - self._total_width))

      if finalize:
        info += '\n'

      sys.stdout.write(info)
      sys.stdout.flush()

    elif self.verbose == 2:
      if finalize:
        numdigits = int(np.log10(self.target)) + 1
        count = ('%' + str(numdigits) + 'd/%d') % (current, self.target)
        info = count + info
        for k in self._values_order:
          info += ' - %s:' % k
          avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
          if avg > 1e-3:
            info += ' %.4f' % avg
          else:
            info += ' %.4e' % avg
        info += '\n'

        sys.stdout.write(info)
        sys.stdout.flush()

    self._last_update = now

  def add(self, n, values=None):
    self.update(self._seen_so_far + n, values)

  def _estimate_step_duration(self, current, now):
    """Estimate the duration of a single step.
    Given the step number `current` and the corresponding time `now`
    this function returns an estimate for how long a single step
    takes. If this is called before one step has been completed
    (i.e. `current == 0`) then zero is given as an estimate. The duration
    estimate ignores the duration of the (assumed to be non-representative)
    first step for estimates when more steps are available (i.e. `current>1`).
    Arguments:
      current: Index of current step.
      now: The current time.
    Returns: Estimate of the duration of a single step.
    """
    if current:
      # there are a few special scenarios here:
      # 1) somebody is calling the progress bar without ever supplying step 1
      # 2) somebody is calling the progress bar and supplies step one mulitple
      #    times, e.g. as part of a finalizing call
      # in these cases, we just fall back to the simple calculation
      if self._time_after_first_step is not None and current > 1:
        time_per_unit = (now - self._time_after_first_step) / (current - 1)
      else:
        time_per_unit = (now - self._start) / current

      if current == 1:
        self._time_after_first_step = now
      return time_per_unit
    else:
      return 0
