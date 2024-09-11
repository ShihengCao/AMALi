#!/bin/bash 
# /usr/bin/python3 main.py --app ../apps/Llama2-4096-32-34-fp16/ --sass --config RTX3090 --useMPI 0 --method_name NTM
# /usr/bin/python3 main.py --app ../apps/Llama2-4096-32-66-fp16/ --sass --config RTX3090 --useMPI 0 --method_name NTM
# /usr/bin/python3 main.py --app ../apps/Llama2-4096-32-130-fp16/ --sass --config RTX3090 --useMPI 0 --method_name NTM
source /staff/caoshiheng/venv/bin/activate

python main.py --app ../apps/Llama2-4096-32-514-fp16/ --sass --config RTX3090 --useMPI 0 --method_name NTM --kernel 14 > log_14_ntm_03.log
# python main.py --app ../apps/Llama2-4096-32-514-fp16/ --sass --config RTX3090 --useMPI 0 --kernel 14 > log_14_.log
python main.py --app ../apps/Llama2-4096-32-514-fp16/ --sass --config RTX3090 --useMPI 0 --method_name NTM --kernel 22 > log_22_ntm_03.log
python main.py --app ../apps/Llama2-4096-32-514-fp16/ --sass --config RTX3090 --useMPI 0 --method_name NTM --kernel 24 > log_24_ntm_03.log
# python main.py --app ../apps/Llama2-4096-32-514-fp16/ --sass --config RTX3090 --useMPI 0 --kernel 24 > log_24_.log
python main.py --app ../apps/Llama2-4096-32-514-fp16/ --sass --config RTX3090 --useMPI 0 --method_name NTM --kernel 25 > log_25_ntm_03.log
# python main.py --app ../apps/Llama2-4096-32-514-fp16/ --sass --config RTX3090 --useMPI 0 --kernel 25 > log_25_.log
python main.py --app ../apps/Llama2-4096-32-514-fp16/ --sass --config RTX3090 --useMPI 0 --method_name NTM --kernel 34 > log_34_ntm_03.log
# python main.py --app ../apps/Llama2-4096-32-514-fp16/ --sass --config RTX3090 --useMPI 0 --kernel 34 > log_34_.log
python main.py --app ../apps/Llama2-4096-32-514-fp16/ --sass --config RTX3090 --useMPI 0 --method_name NTM --kernel 1395 > log_1395_ntm_03.log
# python main.py --app ../apps/Llama2-4096-32-514-fp16/ --sass --config RTX3090 --useMPI 0 --kernel 1395 > log_1395_.log
# python main.py --app ../apps/Llama2-4096-32-514-fp16/ --sass --config RTX3090 --useMPI 0 --method_name NTM --kernel 110 > log_110_ntm.log
# python main.py --app ../apps/Llama2-4096-32-514-fp16/ --sass --config RTX3090 --useMPI 0 --kernel 110 > log_110_.log
# python main.py --app ../apps/Llama2-4096-32-258-fp16/ --sass --config RTX3090 --useMPI 0
# python main.py --app ../apps/Llama2-4096-32-514-fp16/ --sass --config RTX3090 --useMPI 0

# python proprecess.py Llama2-4096-32-130-fp16_
# python proprecess.py Llama2-4096-32-258-fp16_
# python proprecess.py Llama2-4096-32-514-fp16_
# python main.py --app ../apps/Llama2-4096-32-130-fp16_bs8/ --sass --config RTX3090 --useMPI 0
# python main.py --app ../apps/Llama2-4096-32-130-fp16_bs8/ --sass --config RTX3090 --useMPI 0 --method_name NTM
deactivate