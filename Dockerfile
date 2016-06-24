FROM andrewosh/binder-base
MAINTAINER Bas Nijholt <basnijholt@gmail.com>

USER main
RUN /bin/bash -c "source activate python3"
RUN conda update --all -y
RUN conda install -y -c basnijholt kwant==1.2.2
RUN conda install -y -c basnijholt discretizer==0.2
RUN conda install -y holoviews==1.5.0
RUN conda install -y ipyparallel

USER root
RUN apt-get update
RUN apt-get install -y texlive-full

USER main
