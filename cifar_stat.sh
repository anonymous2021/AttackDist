#!/bin/bash


# CIFAR
python StaticAnalysis.py  --device=1 --module=1 --L=1 --batch=1000 | tee cifar_Linf_stat.txt
python StaticAnalysis.py  --device=1 --module=1 --L=0 --batch=1000 | tee cifar_L2_stat.txt

