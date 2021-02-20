.. cusim documentation master file, created by
   sphinx-quickstart on Sat Feb 20 13:36:31 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

CUSIM - Superfast implementation of Word2Vec and LDA
====================================================


CUSIM is a project to speed up various ML models (e.g. topic modeling, word embedding, etc) by CUDA. It would be nice to think of it as `gensim <https://github.com/RaRe-Technologies/gensim>`_'s GPU version project. As a starting step, I implemented the most widely used word embedding model, the `word2vec <https://arxiv.org/pdf/1301.3781.pdf>`_ model, and the most representative topic model, the `LDA (Latent Dirichlet Allocation) <https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf>`_ model.



.. toctree::
   :maxdepth: 2
   :caption: Contents

   Installation <install>
   Word2Vec <w2v>
   LDA <lda>
   Performance <performance>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
