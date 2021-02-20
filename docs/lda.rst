LDA
===


Parameters
----------


- See `CuLDAConfigProto <https://github.com/js1010/cusim/blob/f12d18a65fc603b99350705b235d374654c87517/cusim/proto/config.proto#L27-L83>`_ 


Example Codes
-------------

- Full source code is in `examples/example_lda.py <https://github.com/js1010/cusim/blob/main/examples/example_lda.py>`_

- Before running example codes, run 

.. code-block:: shell

  pip install -r examples/requirements.txt


- Download and preprocess data

.. code-block:: python
  
  import os
  from os.path import join as pjoin
  import subprocess
  
  import wget

  DATASET = "nytimes"
  DIR_PATH = "./res"
  BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/" \
            "bag-of-words/"


  # download docword
  filename = f"docword.{DATASET}.txt.gz"
  out_path = pjoin(DIR_PATH, filename)
  wget.download(BASE_URL + filename, out=out_path)
  print()

  # decompress
  cmd = ["gunzip", "-c", out_path, ">",
         pjoin(DIR_PATH, f"docword.{DATASET}.txt")]
  cmd = " ".join(cmd)
  subprocess.call(cmd, shell=True)
  os.remove(pjoin(DIR_PATH, filename))

  # download vocab
  filename = f"vocab.{DATASET}.txt"
  out_path = pjoin(DIR_PATH, filename)
  wget.download(BASE_URL + filename, out=out_path)
  print()

- Train cusim word2vec

.. code-block:: python
  
  from cusim import CuLDA

  data_path = pjoin(DIR_PATH, f"docword.{DATASET}.txt")
  keys_path = pjoin(DIR_PATH, f"vocab.{DATASET}.txt")
  processed_data_path = pjoin(DIR_PATH, f"docword.{DATASET}.h5")
  opt = {
    "data_path": data_path,
    "processed_data_path": processed_data_path,
    "keys_path": keys_path,
    "num_topics": 50,
    "num_iters_in_e_step": 10,
    "reuse_gamma": True,
    # "skip_preprocess": os.path.exists(processed_data_path),
  }
  start = time.time()
  lda = CuLDA(opt)
  lda.train_model()


- Save and evaluate model

.. code-block:: python
  
  h5_model_path = pjoin(DIR_PATH, "cusim.lda.model.h5")
  lda.save_h5_model(h5_model_path)

  h5f = h5py.File(h5_model_path, "r")
  beta = h5f["beta"][:, :].T
  keys = h5f["keys"][:]
  topk = 10
  
  for idx in range(beta.shape[0]):
    print("=" * 50)
    print(f"topic {idx + 1}")
    print("-" * 50)
    _beta = beta[idx, :]
    indices = np.argsort(-_beta)[:topk]
    for rank, wordid in enumerate(indices):
      word = keys[wordid].decode("utf8")
      prob = _beta[wordid]
      print(f"rank {rank + 1}. {word}: {prob}")


Performance
-----------

- Data: `nytimes dataset <https://archive.ics.uci.edu/ml/datasets/bag+of+words>`_
- Topic Results
    - `cusim lda results <https://github.com/js1010/cusim/blob/main/examples/cusim.topics.txt>`_
    - `gensim lda results <https://github.com/js1010/cusim/blob/main/examples/gensim.topics.txt>`_
- Time Performance
    - Experimented in `AWS g4dn 2xlarge <https://aws.amazon.com/ec2/instance-types/g4/>`_ (One NVIDIA T4 and 8 vcpus of 8 Intel(R) Xeon(R) Platinum 8259CL CPU @ 2.50GHz)

+---------------------+-------------------+--------------------+
| attr                |   gensim (8 vpus) |   cusim (NVIDIA T4)|
+=====================+===================+====================+
| training time (sec) |           447.376 | **76.6972**        |
+---------------------+-------------------+--------------------+
