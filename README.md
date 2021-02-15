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

- [AWS P3 2xlarge instance](https://aws.amazon.com/ec2/instance-types/p3/) is used to the experiment. (One Tesla V100 GPU with 8 vcpus)
- results can be reproduced by running `examples/example_w2v.py` and `examples/example_lda.py`
- To evaluate w2v model, we used `evaluate_word_pairs` function ([ref link](https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#evaluating)) in gensim, note that better performance on WS-353 test set does not mean that the model will workbetter in application as desribed on the link. However, it is good to be measured quantitively and fast training time will be at least very objective measure of performaance.
  - I trained W2V model on quora-duplicat-questions dataset from gensim downloader api with cusim and the performance with gensim.
- To evaluate LDA model, I think there is no good way to measure the quality of traing results quantitatively. But we can check the model by looking at the top words of each topic. Also, we can compare the training time here.
- W2V (CBOW, negative sampling)

| attr          |   1 workers |   2 workers |   4 workers |   8 workers |      GPU |
|:--------------|------------:|------------:|------------:|------------:|---------:|
| training_time |  181.009    |  102.302    |   58.9811   |   47.7482   | **9.60324**  |
| pearson       |    0.203882 |    0.207705 |    0.221758 |    0.198408 | **0.331749** |
| spearman      |    0.25208  |    0.254706 |    0.275231 |    0.238611 | **0.295346** |

### Future tasks

- support half precision
- support multi device (multi device implementation on LDA model will not be that hard, while multi device training on w2v may require some considerations)
- implement other models such as FastText, BERT, etc
