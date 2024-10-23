#!/bin/bash 

python main.py --app ../A100_apps/llama3-8b-6144/ --config A100 --useMPI 0
python main.py --app ../A100_apps/llama3-8b-4096/ --config A100 --useMPI 0
python main.py --app ../A100_apps/llama3-8b-2048/ --config A100 --useMPI 0

python proprecess.py llama3-8b-2048
python proprecess.py llama3-8b-4096
python proprecess.py llama3-8b-6144