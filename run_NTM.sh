#!/bin/bash 

python main.py --app ../RTX3090_apps/Llama2-4096-32-514-fp16/ --config RTX3090 --useMPI 0
python main.py --app ../RTX3090_apps/Llama2-4096-32-258-fp16/ --config RTX3090 --useMPI 0
python main.py --app ../RTX3090_apps/Llama2-4096-32-130-fp16/ --config RTX3090 --useMPI 0

python proprecess.py Llama2-4096-32-130-fp16
python proprecess.py Llama2-4096-32-258-fp16
python proprecess.py Llama2-4096-32-514-fp16