# Python support can be specified down to the minor or micro version
# (e.g. 3.6 or 3.6.3).
# OS Support also exists for jessie & stretch (slim and full).
# See https://hub.docker.com/r/library/python/ for all supported Python
# tags from Docker Hub.
FROM python:alpine

# If you prefer miniconda:
#FROM continuumio/miniconda3

LABEL Name=hatdieu-api Version=0.0.1
EXPOSE 3000

WORKDIR /hatdieu_api
ADD . /hatdieu_api

# Using pip:
RUN python3 -m pip install -r requirements.txt
RUN pip install -U torch torchvision cython
RUN pip install -U 'git+https://github.com/facebookresearch/fvcore.git' 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
RUN pip install -e detectron2_repo
CMD ["python3", "-m", "hatdieu-api"]

# Using pipenv:
#RUN python3 -m pip install pipenv
#RUN pipenv install --ignore-pipfile
#CMD ["pipenv", "run", "python3", "-m", "hatdieu-api"]

# Using miniconda (make sure to replace 'myenv' w/ your environment name):
#RUN conda env create -f environment.yml
#CMD /bin/bash -c "source activate myenv && python3 -m hatdieu-api"
