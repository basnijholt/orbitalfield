FROM andrewosh/binder-base
MAINTAINER Bas Nijholt <basnijholt@gmail.com>
RUN /bin/bash -c "source activate python3"
RUN conda install -y -c https://conda.anaconda.org/basnijholt kwant==1.2.2
RUN conda install -y -c https://conda.anaconda.org/basnijholt discretizer==0.2
RUN conda install -y holoviews==1.5.0
RUN conda install -y ipyparallel
USER root
RUN apt-get update
RUN apt-get install -y texlive-full
VOLUME ["/home/shared"]
