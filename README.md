# Signature Inference and Machine Learning
[![License](https://img.shields.io/:license-mit-blue.svg)](https://badges.mit-license.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

+ [This is the link to the arxiv article.](http://arxiv.org/abs/1911.02922)
+ [This is the link to slides for the talk given at the IWCIA.](https://karhunenloeve.github.io/SIML/docs/FAU-Beamer.pdf)

This repository **SIML** (**S**ignature **I**nference & **M**achine **L**earning) offers all functionalities and experiments for the paper *Persistent Homology as Stopping-Criterion for Voronoi Interpolation*. The functions are annotated. The repository is no longer maintained and is used for prototypical implementation of the project. It has been archived since publication. If you use the project, or share this project, please quote us as follows according to [DBLP (Natural Neighbor)](https://dblp.uni-trier.de/search?q=Persistent%20Homology%20as%20Stopping-Criterion%20for%20Natural%20Neighbor%20Interpolation) of [DBLP (Voronoi)](https://dblp.uni-trier.de/search?q=Persistent%20Homology%20as%20Stopping-Criterion%20for%Voronoi%20Interpolation).

## Citation
    @inproceedings{iwcia/MelodiaL20,
      author    = {Luciano Melodia and
                   Richard Lenz},
      editor    = {Tibor Lukic and
                   Reneta P. Barneva and
                   Valentin E. Brimkov and
                   Lidija Comic and
                   Natasa Sladoje},
      title     = {Persistent Homology as Stopping-Criterion for Voronoi Interpolation},
      booktitle = {Combinatorial Image Analysis - 20th International Workshop, {IWCIA}
                   2020, Novi Sad, Serbia, July 16-18, 2020, Proceedings},
      series    = {Lecture Notes in Computer Science},
      volume    = {12148},
      pages     = {29--44},
      publisher = {Springer},
      year      = {2020},
      url       = {https://doi.org/10.1007/978-3-030-51002-2\_3},
      doi       = {10.1007/978-3-030-51002-2\_3},
    }

## Persistent Homology as Stopping for Voronoi Interpolation
In this study the Voronoi interpolation is used to interpolate a set of points  drawn from a topological space with higher homology groups on its filtration. The technique is based on Voronoi tesselation, which induces a natural dual map to the Delaunay triangulation. Advantage is taken from this fact calculating the persistent homology on it after each iteration to capture the changing topology of the data. The boundary points are identified as critical. The Bottleneck and Wasserstein distance serve as a measure of quality between the original point set and the interpolation. If the norm of two distances exceeds a heuristically determined threshold, the algorithm terminates. We give the theoretical basis for this approach and justify its validity with numerical experiments.

## Requirements
For some of the packages written in `C++` with corresponding python bindings we use the `gcc` compiler. Please install `gcc` using one of the following commands for the linux distributions *Arch, Solus4* or *Ubuntu*:
```bash
 # Archlinux
 sudo pacman -S gcc

 # Solus4
 sudo eopkg install gcc
 # These are the requirements to run gcc for Solus4
 sudo eopkg install -c system.devel

 # Ubuntu
 sudo apt update
 sudo apt install build-essential
 sudo apt-get install python3-dev
 sudo apt-get install manpages-dev
 gcc --version
```

 Some packages are way easier to install using Anaconda. For the installation on several linux distributions please follow [this link](https://docs.anaconda.com/anaconda/install/linux/). Further the installation of our clustering prototype requires some python packages to be installed. We provide a requirements file, but here is a complete list for manual installation using `pip3` and `python 3`:
```bash
  pip3 install pandas
  pip3 install sklearn
  # Works only with gcc installed.
  pip3 install hdbscan
  pip3 install gudhi
  pip3 install matplotlib
  pip3 install tikzplotlib
  pip3 install scipy
  pip3 install pylab
  pip3 install ot
  pip3 install typing
  pip3 install sklearn
  pip3 install csv
  pip3 install numpy

  # Install Gudhi, easiest installation with Anaconda.
  # Gudhi is a library to compute persistent homology.
  conda install -c conda-forge gudhi
  conda install -c conda-forge/label/cf201901 gudhi
```

## References
- Coricos TDAToolbox, [https://github.com/Coricos/TdaToolbox](https://github.com/Coricos/TdaToolbox).
- Hirosm: Time Contrastive Learning, [https://github.com/hirosm/TCL](https://github.com/hirosm/TCL).
- Indoor WIFI Dataset: UJIIndoorLoc, [https://archive.ics.uci.edu/ml/datasets/UJIIndoorLoc](https://archive.ics.uci.edu/ml/datasets/UJIIndoorLoc).
- Nabihach Tane & CTane, [https://github.com/nabihach/FD_CFD_extraction](https://github.com/nabihach/FD_CFD_extraction).
- P. Fränti and S. Sieranoja: Clustering Basic Benchmarks, [http://cs.joensuu.fi/sipu/datasets/](http://cs.joensuu.fi/sipu/datasets/).
- PPoffice: Ant-Colony TSP-Solver, [https://github.com/ppoffice/ant-colony-tsp](https://github.com/ppoffice/ant-colony-tsp).
- Stwisdom: Optimizers, [https://github.com/stwisdom/urnn/blob/master/](https://github.com/stwisdom/urnn/blob/master/).
- Submanifolds Neural Persistence, [https://github.com/BorgwardtLab/Neural-Persistence](https://github.com/BorgwardtLab/Neural-Persistence).
