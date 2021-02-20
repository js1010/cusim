Installation
============

install from pypi
-----------------

.. code-block:: shell

  pip install cusim


install from source
--------------------

.. code-block:: shell

  # clone repo and submodules
  git clone git@github.com:js1010/cusim.git && cd cusim && git submodule update --init

  # install requirements
  pip install -r requirements.txt

  # generate proto
  python -m grpc_tools.protoc --python_out cusim/ --proto_path cusim/proto/ config.proto

  # install
  python setup.py install
