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


### Future tasks

- support half precision
- support multi device (multi device implementation on LDA model will not be that hard, while multi device training on w2v may require some considerations)
- implement other models such as FastText, BERT, etc
