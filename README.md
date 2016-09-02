# Project on Orbital effect of magnetic field for Majorana phase diagram
Launch repository Binder (start an `ipcluster` in the IPython Clusters tab):
[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/basnijholt/orbitalfield)

See the "Orbital effect of magnetic field on the Majorana phase diagram" paper on [arXiv:1509.02675](http://arxiv.org/abs/1509.02675) [[pdf](http://arxiv.org/pdf/1509.02675)], [Phys. Rev. B 93, 235434 (2016)](http://dx.doi.org/10.1103/PhysRevB.93.235434).

By Bas Nijholt and Anton Akhmerov


This folder contains three IPython notebooks:
* [Phase-diagrams.ipynb](https://github.com/basnijholt/orbitalfield/blob/master/Phase-diagrams.ipynb)
* [Induced-gap-tuning.ipynb](https://github.com/basnijholt/orbitalfield/blob/master/Induced-gap-tuning.ipynb)
* [Paper-figures.ipynb](https://github.com/basnijholt/orbitalfield/blob/master/Paper-figures.ipynb)

All notebooks contain instructions of how it can be used.

## Phase-diagrams.ipynb
Find phase boundaries, the band gaps and Majorana decay lengths.

## Paper-figures.ipynb
Create all the figures used in the paper.

## Induced-gap-tuning.ipynb
Find the correct value for $\Delta$ to set the required $\Delta_\textrm{ind}$.


# Installation
If all package dependencies are met, the notebooks will work in Python 3 without
issues. In case it might not work we've created a Docker image that will create
an environment where everything will work.

First install Docker, see [instructions](https://docs.docker.com/installation/).

You can either build the image yourself or use a precompiled image.

To download and run just execute:
```
$ docker run -p 8888:8888 -v /path/to/downloaded/folder/:/home/jovyan/work/ basnijholt/kwant:orbitaleffect
```

or build the Docker image yourself, you should use `Dockerfile-notebook` (`Dockerfile` is for Binder) (will take ~20 min to build):
```
$ docker build --tag="basnijholt/kwant:orbitaleffect" /path/to/downloaded/folder/
```

```
$ docker run -p 8888:8888 -v /path/to/downloaded/folder/:/home/jovyan/work/ basnijholt/kwant:orbitaleffect
```

Now visit [localhost:8888/notebooks/orbitaleffect/](http://localhost:8888/notebooks/orbitaleffect/)

NOTE: If you are on OS X or Windows, Docker will show a IP address upon opening Docker
use this IP instead of localhost.
