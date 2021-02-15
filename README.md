### Introduction

This project is to speed up various ML models (e.g. topic modeling, word embedding, etc) by CUDA. It would be nice to think of it as [gensim](https://github.com/RaRe-Technologies/gensim)'s GPU version project. As a starting step, I implemented the most widely used word embedding model, the [word2vec](https://arxiv.org/pdf/1301.3781.pdf) model, and the most representative topic model, the [LDA (Latent Dirichlet Allocation)](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) model.

### How to install

- install from source

```shell
# clone repo and submodules
git clone git@github.com:js1010/cusim.git && cd cusim && git submodule update --init

# install requirements
pip install -r requirements.txt

# generate proto
python -m grpc_tools.protoc --python_out cusim/ --proto_path cusim/proto/ config.proto

# install
python setup.py install
```

- pip installation will be available soon

### How to use

- `examples/example_w2v.py`, `examples/example_lda.py` and `examples/README.md` will be very helpful to understand the usage.
- paremeter description can be seen in `cusim/proto/config.proto`

### Performance

- [AWS g4dn 2xlarge instance](https://aws.amazon.com/ec2/instance-types/g4/) is used to the experiment. (One NVIDIA T4 GPU with 8 vcpus, Intel(R) Xeon(R) Platinum 8259CL CPU @ 2.50GHz)
- results can be reproduced by simply running `examples/example_w2v.py` and `examples/example_lda.py`
- To evaluate w2v model, I used `evaluate_word_pairs` function ([ref link](https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#evaluating)) in gensim, note that better performance on WS-353 test set does not necessarily mean that the model will workbetter in application as desribed on the link. However, it is good to be measured quantitively and fast training time will be at least very objective measure of the performaance.
  - I trained W2V model on `quora-duplicat-questions` dataset from gensim downloader api on GPU with cusim and compare the performance (both speed and model quality) with gensim.
- To evaluate LDA model, I found there is no good way to measure the quality of traing results quantitatively. But we can check the model by looking at the top words of each topic. Also, we can compare the training time quantitatively.
- W2V (skip gram, hierarchical softmax)

| attr                |   1 workers (gensim) |   2 workers (gensim) |   4 workers (gensim) |   8 workers (gensim) |   NVIDIA T4 (cusim) |
|:--------------------|---------------------:|---------------------:|---------------------:|---------------------:|--------------------:|
| training time (sec) |           892.596    |           544.212    |           310.727    |           226.472    |       **16.162**   |
| pearson             |             0.487832 |             0.487696 |             0.482821 |             0.487136 |       **0.492101** |
| spearman            |             0.500846 |             0.506214 |             0.501048 |         **0.506718** |            0.479468 |

- W2V (skip gram, negative sampling)

| attr                |   1 workers (gensim) |   2 workers (gensim) |   4 workers (gensim) |   8 workers (gensim) |   NVIDIA T4 (cusim) |
|:--------------------|---------------------:|---------------------:|---------------------:|---------------------:|--------------------:|
| training time (sec) |           586.545    |           340.489    |           220.804    |           146.23     |       **33.9173**   |
| pearson             |             0.354448 |             0.353952 |             0.352398 |             0.352925 |        **0.360436** |
| spearman            |             0.369146 |             0.369365 |         **0.370565** |             0.365822 |        0.355204     |

- W2V (CBOW, hierarchical softmax)

| attr                |   1 workers (gensim) |   2 workers (gensim) |   4 workers (gensim) |   8 workers (gensim) |   NVIDIA T4 (cusim) |
|:--------------------|---------------------:|---------------------:|---------------------:|---------------------:|--------------------:|
| training time (sec) |           250.135    |           155.121    |           103.57     |            73.8073   |        **6.20787**  |
| pearson             |             0.309651 |             0.321803 |             0.324854 |             0.314255 |        **0.480298** |
| spearman            |             0.294047 |             0.308723 |             0.318293 |             0.300591 |        **0.480971** |

- W2V (CBOW, negative sampling)

| attr                |   1 workers (gensim) |   2 workers (gensim) |   4 workers (gensim) |   8 workers (gensim) |   NVIDIA T4 (cusim) |
|:--------------------|---------------------:|---------------------:|---------------------:|---------------------:|--------------------:|
| training time (sec) |           176.923    |           100.369    |            69.7829   |            49.9274   |        **9.90391**  |
| pearson             |             0.18772  |             0.193152 |             0.204509 |             0.187924 |        **0.368202** |
| spearman            |             0.243975 |             0.24587  |             0.260531 |             0.237441 |        **0.358042** |

- LDA (`nytimes` dataset from https://archive.ics.uci.edu/ml/datasets/bag+of+words)
  - I found that setting `workers` variable in gensim LdaMulticore does not work properly (it uses all cores in instance anyway), so I just compared the speed between cusim with single GPU and gensim with 8 vcpus. 
  - One can compare the quality of modeling by looking at `examples/cusim.topics.txt` and `examples/gensim.topics.txt`.

| attr                |   gensim (8 vpus) |   cusim (NVIDIA T4)|
|:--------------------|------------------:|--------:|
| training time (sec) |           447.376 | **76.6972** |

### Future tasks

- support half precision
- support multi device (multi device implementation on LDA model will not be that hard, while multi device training on w2v may require some considerations)
- implement other models such as [FastText](https://arxiv.org/pdf/1607.04606.pdf), [BERT](https://arxiv.org/pdf/1810.04805.pdf), etc
