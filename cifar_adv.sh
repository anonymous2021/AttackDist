#!/bin/bash




# CIFAR10
python generateAdv.py  --device=3 --module=1 --L=1
python generateAdv.py  --device=3 --module=1 --L=0

