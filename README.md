# Project on Orbital effect of magnetic field for Majorana phase diagram
Launch repository Binder (unfortunately Binder doesn't allow for starting `ipyparallel` engines, so only the plotting of the figures will work properly):
[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/basnijholt/orbitalfield)

By Bas Nijholt and Anton Akhmerov


This folder contains three IPython notebooks:
* Phase-diagrams.ipynb
* Induced-gap-tuning.ipynb
* Paper-figures.ipynb

All notebooks contain instructions of how it can be used.

## Phase-diagrams.ipynb
Find phase boundaries, the band gaps and Majorana decay lengths.

## Paper-figures.ipynb
Create all the figures used in the paper.

## Induced-gap-tuning.ipynb
Find the correct value for $\Delta$ to set the required $\Delta_\textrm{ind}$.


# Installation
If all package dependencies are met, the notebooks will work in Python 2 without
issues. In case it might not work we've created a Docker image that will create
an environment where everything will work.

First install Docker, see [instructions](https://docs.docker.com/installation/)

You can either build the image yourself or use a precompiled image.

To download and run just execute:
`$ docker run -p 8888:8888 -v /path/to/downloaded/folder/:/home/jovyan/work/ basnijholt/kwant:orbitaleffect`

OR build yourself (will take ~20 min to build):
`$ docker build --tag="basnijholt/kwant:orbitaleffect" /path/to/downloaded/folder/`

`$ docker run -p 8888:8888 -v /path/to/downloaded/folder/:/home/jovyan/work/ basnijholt/kwant:orbitaleffect`

Now visit [localhost:8888/notebooks/orbitaleffect/](http://localhost:8888/notebooks/orbitaleffect/)

NOTE: If you are on OS X or Windows, Docker will show a IP address upon opening Docker
use this IP instead of localhost.
