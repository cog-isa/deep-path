FROM windj007/jupyter-keras-tools:8.0
MAINTAINER Roman Suvorov <rsuvorov@isa.ru>

RUN pip install -U git+https://github.com/hyperopt/hyperopt git+https://github.com/HIPS/Spearmint pymongo networkx
RUN apt-get install -yqq libav-tools python-tk
RUN pip install -U git+https://github.com/fchollet/keras
RUN pip install -U https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0-cp27-none-linux_x86_64.whl
