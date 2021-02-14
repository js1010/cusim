
def run_lda():
  download()
  opt = {
    "data_path": DATA_PATH,
    "processed_data_dir": PROCESSED_DATA_DIR,
    # "skip_preprocess":True,
  }
  lda = CuLDA(opt)
  lda.train_model()
  lda.save_model(LDA_PATH)
  h5f = h5py.File(LDA_PATH, "r")
  beta = h5f["beta"][:]
  word_list = h5f["keys"][:]
  num_topics = h5f["alpha"].shape[0]
  for i in range(num_topics):
    print("=" * 50)
    print(f"topic {i + 1}")
    words = np.argsort(-beta.T[i])[:10]
    print("-" * 50)
    for j in range(TOPK):
      word = word_list[words[j]].decode("utf8")
      prob = beta[words[j], i]
      print(f"rank {j + 1}. word: {word}, prob: {prob}")
  h5f.close()
