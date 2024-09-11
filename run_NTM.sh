#!/bin/bash 
# /usr/bin/python3 main.py --app ../apps/Llama2-4096-32-34-fp16/ --sass --config RTX3090 --useMPI 0 --method_name NTM
# /usr/bin/python3 main.py --app ../apps/Llama2-4096-32-66-fp16/ --sass --config RTX3090 --useMPI 0 --method_name NTM
# /usr/bin/python3 main.py --app ../apps/Llama2-4096-32-130-fp16/ --sass --config RTX3090 --useMPI 0 --method_name NTM
source /staff/caoshiheng/venv/bin/activate


# python main.py --app ../apps/Llama2-4096-32-130-fp16/ --sass --config RTX3090 --useMPI 0 --method_name NTM
# python main.py --app ../apps/Llama2-4096-32-258-fp16/ --sass --config RTX3090 --useMPI 0 --method_name NTM
# mpiexec -n 4 python main.py --app ../apps/Llama2-4096-32-514-fp16/ --sass --config RTX3090 --useMPI 1 --method_name NTM
python main.py --app ../apps/Llama2-4096-32-514-fp16/ --sass --config RTX3090 --useMPI 0 --method_name NTM
# python proprecess.py Llama2-4096-32-130-fp16_NTM
# python proprecess.py Llama2-4096-32-258-fp16_NTM
python proprecess.py Llama2-4096-32-514-fp16_NTM
# python main.py --app ../apps/Llama2-4096-32-130-fp16_bs8/ --sass --config RTX3090 --useMPI 0
# python main.py --app ../apps/Llama2-4096-32-130-fp16_bs8/ --sass --config RTX3090 --useMPI 0 --method_name NTM
deactivate