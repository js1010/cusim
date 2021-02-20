Word2Vec
========


Parameters
----------


- test

.. autoclass:: cusim.cuw2v.pycuw2v.CuW2V
   :members:
   :inherited-members:


Example Codes
-------------

- Full source code is in `github <https://github.com/js1010/cusim/blob/e29deb0a0a39a4b739aa1bc38ea9de897a8de8de/examples/example_w2v.py>`_

- before running example codes, run 

.. code-block:: shell

  pip install -r examples/requirements.txt


- Download and preprocess data

.. code-block:: python
  
  import os
  import subprocess

  import nltk
  from nltk.tokenize import RegexpTokenizer
  
  DOWNLOAD_PATH = "./res"
  DATASET = "quora-duplicate-questions"
  DATA_PATH = f"./res/{DATASET}.stream.txt"
  PROCESSED_DATA_DIR = f"./res/{DATASET}-processed"

  def preprocess_line(line, tokenizer, lemmatizer):
    line = line.lower()
    line = tokenizer.tokenize(line)
    line = [token for token in line
            if not token.isnumeric() and len(token) > 1]
    line = [lemmatizer.lemmatize(token) for token in line]
    return " ".join(line)
  
  # download
  api.BASE_DIR = DOWNLOAD_PATH
  filepath = api.load(DATASET, return_path=True)
  cmd = ["gunzip", "-c", filepath, ">", DATA_PATH]
  cmd = " ".join(cmd)
  subprocess.call(cmd, shell=True)
    
  # preprocess data
  tokenizer = RegexpTokenizer(r'\w+')
  nltk.download("wordnet")
  lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
  fout = open(DATA_PATH + ".tmp", "wb")
  with open(DATA_PATH, "rb") as fin:
    for line in tqdm.tqdm(fin):
      line = line.decode("utf8").strip()
      line = preprocess_line(line, tokenizer, lemmatizer)
      fout.write((line + "\n").encode("utf8"))
  fout.close()
  os.rename(DATA_PATH + ".tmp", DATA_PATH)

- Train cusim word2vec

.. code-block:: python
  
  from cusim import CuW2V

  MIN_COUNT = 5
  LEARNING_RATE = 0.001
  NEG_SIZE = 10
  NUM_DIMS = 100
  CBOW_MEAN = False
  EPOCHS = 10
  
  opt = {
    "data_path": DATA_PATH,
    "processed_data_dir": PROCESSED_DATA_DIR,
    # "skip_preprocess": os.path.exists(PROCESSED_DATA_DIR),
    "num_dims": NUM_DIMS,
    "epochs": EPOCHS,
    "word_min_count": MIN_COUNT,
    "lr": 0.001,
    "io": {
      "lower": False
    },
    "neg": 0 if hierarchical_softmax else NEG_SIZE,
    "skip_gram": skip_gram,
    "cbow_mean": CBOW_MEAN,
  }
  w2v = CuW2V(opt)
  w2v.train_model()


- Save and evaluate model

.. code-block:: python
  
  import gensim
  from gensim.test.utils import datapath

  CUSIM_MODEL = "./res/cusim.w2v.model" 
  
  w2v.save_word2vec_format(CUSIM_MODEL, binary=False)
  model = gensim.models.KeyedVectors.load_word2vec_format(model)
  results = model.wv.evaluate_word_pairs(datapath("wordsim353.tsv"),
                                         case_insensitive=False)
