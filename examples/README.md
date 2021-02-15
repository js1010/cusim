### How to run example code

0. install requirements

```shell
pip install -r requirements.txt
```

1. first, it is good to know about python-fire in https://github.com/google/python-fire, if you haven't heard yet.

2. run w2v experiments on various setting (e.g. skip gram with hierarchical softmax)

```shell
python example_w2v.py run_experiments --sg0=True --hs0=True
```

7. run lda experiments 

```shell
python example_lda.py run_experiments
```
