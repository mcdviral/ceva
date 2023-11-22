# ContEva - Continuous-time Evaluation

Light-weight package to create b-spline trajectory from 6DOF control points. The spline functions are based on the [basalt library](https://cvg.cit.tum.de/research/vslam/basalt).

Features:
* Quick inquiries on the spline (min/max time, order, knots, sampled pose)
* Create a random spline
* Deskew a pointcloud
* Fit a spline with sample data

# Prerequisites

Please ensure the following dependencies are installed

* Eigen3
```
sudo apt install libeigen3-dev
```
* fmt
```
sudo apt install libfmt-dev
```
* [Sophus](https://github.com/strasdat/Sophus)
```
cd;
git clone https://github.com/strasdat/Sophus;
cd Sophus; git checkout 1.22.10; mkdir build; cd build; cmake .. ; make -j$(nproc); sudo make install;
```
* Ceres Solver
```
## Third party programs----------------------------------------------------

# CMake
sudo apt-get install -y cmake
# google-glog + gflags
sudo apt-get install -y libgoogle-glog-dev libgflags-dev
# BLAS & LAPACK
sudo apt-get install -y libatlas-base-dev
# Eigen3
sudo apt-get install -y libeigen3-dev
# SuiteSparse and CXSparse (optional)
sudo apt-get install -y libsuitesparse-dev

## Download ceres---------------------------------------------------------

rm -rf ceres_2.1.0
git clone https://ceres-solver.googlesource.com/ceres-solver ceres_2.1.0; cd ceres_2.1.0; git checkout 2.1.0;

## Install ceres----------------------------------------------------------

mkdir -p ceres_bin; cd ceres_bin; cmake ..; make -j$(nproc); make test; sudo make install
```
# Download and Compile Ceva

```
git clone --recursive https://github.com/mcdviral/ceva; cd ceva; python3 setup.py install # Some system may need to use 'sudo python3 setup.py install'

```

# Demo
Navigate to the folder scripts for a demo scripts.

# License
This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License and is intended for non-commercial academic use.

# Reference
The package is created as an tool for the MCD dataset. If you use this package in other works please consider citing MCD as

```
@software{mcdviral2023,
  author  = {Anonymous},
  title   = {MCD: Diverse Large-Scale Multi-Campus Dataset for Robot Perception},
  license = {CC BY-NC-SA 4.0},
  url     = {https://mcdviral.github.io/},
  version = {1.0},
  year    = {2023},
  month   = {11}
}
```
