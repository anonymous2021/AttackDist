#!/bin/bash


# MNIST
python StaticAnalysis.py  --device=2 --module=0 --L=1 | tee mnist_Linf_stat.txt
python StaticAnalysis.py  --device=2 --module=0 --L=0 | tee mnist_L2_stat.txt

