# Course projects API

This repo is called by [System Face Recognition](https://github.com/tks1998/system_face-recognition)

## Getting Started

We are still developing it so if you want to run it yourself, please follow the instructions

### Prerequisites

This repo is built on Ubuntu 18.04 with GPU support.

```
Install python3.7 follow [this tutorial](https://linuxize.com/post/how-to-install-python-3-7-on-ubuntu-18-04/) and [pip3](https://linuxize.com/post/how-to-install-pip-on-ubuntu-18.04/)
```

We recommend you using virtual environments 

### Installing

A step by step series of examples that tell you how to get a development env running

Installing some packages

```
apt update
apt-get install build-essential dkms (don't if you don't use CUDA)
apt-get install -y libsm6 libxext6 libxrender-dev libglib2.0-0 python3.7-dev git
```

Installing [**detectron2**](https://github.com/facebookresearch/detectron2)

```
pip install -U torch torchvision cython
pip install -U 'git+https://github.com/facebookresearch/fvcore.git' 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
git clone https://github.com/facebookresearch/detectron2 detectron2_repo
pip install -e detectron2_repo
```

Installing [**vggface**](https://github.com/rcmalli/keras-vggface)

```
pip install git+https://github.com/rcmalli/keras-vggface.git
pip install mtcnn
```

Installing **insightface**

```
pip install insightface
pip install mxnet-mkl
pip install mxnet-cu100 (for cuda 10.0)
```

## Library Versions

* [Django](https://www.djangoproject.com/) - version:
* [Django REST framework](https://www.django-rest-framework.org/#installation) - version:
* [OpenCV](https://opencv.org/) - version:
* [Tensorflow](https://www.tensorflow.org/install) - version: 


## API

Please see [usage](./cs-api.xlsx) for more information.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
