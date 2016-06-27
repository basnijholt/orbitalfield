FROM andrewosh/binder-base
MAINTAINER Bas Nijholt <basnijholt@gmail.com>

USER root
RUN apt-get update
RUN apt-get install -y texlive-full

USER main
RUN conda install -y ipyparallel && ipcluster nbextension enable
RUN /bin/bash -c "source activate python3" && \
  conda install -y -c basnijholt kwant==1.2.2 && \
  conda install -y -c basnijholt discretizer==0.2 && \
  conda install -y holoviews==1.5.0 && \
  conda install -y terminado && \
  conda install -y ipyparallel && \
  ipcluster nbextension enable && \
  ipython profile create --parallel --profile=python3

RUN echo "env_path = '/home/main/anaconda2/envs/python3/bin/python'" >> /home/main/.ipython/profile_python3/ipcluster_config.py && \
  echo "c.LocalControllerLauncher.controller_cmd = [env_path, '-m', 'ipyparallel.controller']" >> /home/main/.ipython/profile_python3/ipcluster_config.py && \
  echo "c.LocalEngineLauncher.engine_cmd = [env_path, '-m', 'ipyparallel.engine']" >> /home/main/.ipython/profile_python3/ipcluster_config.py && \
  echo "c.LocalEngineSetLauncher.engine_cmd = [env_path, '-m', 'ipyparallel.engine']" >> /home/main/.ipython/profile_python3/ipcluster_config.py
