FROM andrewosh/binder-base
MAINTAINER Bas Nijholt <basnijholt@gmail.com>

USER main
RUN /bin/bash -c "source activate python3"
RUN conda install -y -n python3 -c basnijholt kwant==1.2.2
RUN conda install -y -n python3 -c basnijholt discretizer==0.2
RUN conda install -y -n python3 holoviews==1.5.0
RUN conda install -y -n python3 ipyparallel

USER root
RUN apt-get update
RUN apt-get install -y texlive-full

USER main
