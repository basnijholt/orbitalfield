FROM andrewosh/binder-base
MAINTAINER Bas Nijholt <basnijholt@gmail.com>

USER root
RUN apt-get update
RUN apt-get install -y texlive-full

USER main
RUN conda install -y ipyparallel && ipcluster nbextension enable && conda install -y terminado

RUN conda install -y -n python3 -c basnijholt \ 
  'kwant==1.2.2' \
  'discretizer==0.2'

RUN conda install -y -n python3 \
  'holoviews==1.5.0' \
  'ipyparallel'

RUN /home/main/anaconda2/envs/python3/bin/ipython profile create --parallel --profile=python3

COPY ipcluster_config.py /home/main/.ipython/profile_python3/ipcluster_config.py
