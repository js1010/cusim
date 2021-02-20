.. cusim documentation master file, created by
   sphinx-quickstart on Sat Feb 20 13:36:31 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

CUSIM - Superfast implementation of Word2Vec and LDA
=================================


CUSIM is a project to speed up various ML models (e.g. topic modeling, word embedding, etc) by CUDA. It would be nice to think of it as gensim's GPU version project. As a starting step, I implemented the most widely used word embedding model, the word2vec model, and the most representative topic model, the LDA (Latent Dirichlet Allocation) model.


Resources 
---------

- `Source and issues on Github <https://github.com/js1010/cusim>`_

.. toctree::
   :maxdepth: 2
   :caption: Overview

   Installation
   GettingStarted
   examples
   Release Notes <https://github.com/js1010/cusim/releases>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
