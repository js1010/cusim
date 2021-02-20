Word2Vec
========


Parameters
----------


- see `CuW2VConfigProto <https://github.com/js1010/cusim/blob/f12d18a65fc603b99350705b235d374654c87517/cusim/proto/config.proto#L95-L159>`_ 


Example Codes
-------------

- Full source code is in `examples/example_w2v.py <https://github.com/js1010/cusim/blob/main/examples/example_w2v.py>`_

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

Performance
-----------

- Data: quora-duplicate-questions
- skip gram, hierarchical softmax

| attr                |   1 workers (gensim) |   2 workers (gensim) |   4 workers (gensim) |   8 workers (gensim) |   NVIDIA T4 (cusim) |
|:--------------------|---------------------:|---------------------:|---------------------:|---------------------:|--------------------:|
| training time (sec) |           892.596    |           544.212    |           310.727    |           226.472    |       **16.162**   |
| pearson             |             0.487832 |             0.487696 |             0.482821 |             0.487136 |       **0.492101** |
| spearman            |             0.500846 |             0.506214 |             0.501048 |         **0.506718** |            0.479468 |

- skip gram, negative sampling

| attr                |   1 workers (gensim) |   2 workers (gensim) |   4 workers (gensim) |   8 workers (gensim) |   NVIDIA T4 (cusim) |
|:--------------------|---------------------:|---------------------:|---------------------:|---------------------:|--------------------:|
| training time (sec) |           586.545    |           340.489    |           220.804    |           146.23     |       **33.9173**   |
| pearson             |             0.354448 |             0.353952 |             0.352398 |             0.352925 |        **0.360436** |
| spearman            |             0.369146 |             0.369365 |         **0.370565** |             0.365822 |        0.355204     |

- CBOW, hierarchical softmax

| attr                |   1 workers (gensim) |   2 workers (gensim) |   4 workers (gensim) |   8 workers (gensim) |   NVIDIA T4 (cusim) |
|:--------------------|---------------------:|---------------------:|---------------------:|---------------------:|--------------------:|
| training time (sec) |           250.135    |           155.121    |           103.57     |            73.8073   |        **6.20787**  |
| pearson             |             0.309651 |             0.321803 |             0.324854 |             0.314255 |        **0.480298** |
| spearman            |             0.294047 |             0.308723 |             0.318293 |             0.300591 |        **0.480971** |

- CBOW, negative sampling

| attr                |   1 workers (gensim) |   2 workers (gensim) |   4 workers (gensim) |   8 workers (gensim) |   NVIDIA T4 (cusim) |
|:--------------------|---------------------:|---------------------:|---------------------:|---------------------:|--------------------:|
| training time (sec) |           176.923    |           100.369    |            69.7829   |            49.9274   |        **9.90391**  |
| pearson             |             0.18772  |             0.193152 |             0.204509 |             0.187924 |        **0.368202** |
| spearman            |             0.243975 |             0.24587  |             0.260531 |             0.237441 |        **0.358042** |
