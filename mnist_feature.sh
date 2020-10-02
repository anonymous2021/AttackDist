#!/bin/bash

# MNIST
python extract_feature.py  --device=6 --module=0 --L=1 | tee mnist_Linf_feature.txt
python extract_feature.py  --device=6 --module=0 --L=0 | tee mnist_L2_feature.txt

