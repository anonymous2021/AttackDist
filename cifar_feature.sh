#!/bin/bash

# CIFAR10
python extract_feature.py  --device=5 --module=1 --L=1 --batch=1000 | tee cifar_Linf_feature.txt
python extract_feature.py  --device=5 --module=1 --L=0 --batch=1000 | tee cifar_L2_feature.txt