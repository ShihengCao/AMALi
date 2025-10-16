##############################################################################
# &copy 2017. Triad National Security, LLC. All rights reserved.

# This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration.

# All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

# Recall that this copyright notice must be accompanied by the appropriate open source license terms and conditions. Additionally, it is prudent to include a statement of which license is being used with the copyright notice. For example, the text below could also be included in the copyright notice file: This is open source software; you can redistribute it and/or modify it under the terms of the Performance Prediction Toolkit (PPT) License. If software is modified to produce derivative works, such modified software should be clearly marked, so as not to confuse it with the version available from LANL. Full text of the Performance Prediction Toolkit (PPT) License can be found in the License file in the main development branch of the repository.
##############################################################################

# Author: Yehia Arafa
# Last Update Date: April, 2021
# Copyright: Open source, must acknowledge original author

##############################################################################
from src.utils import functional_units_list, rptv_warp_select, sm_id_str_to_int
from typing import Generator
import os

def read_sass_trace_generator(sass_path: str) -> Generator[str, None, None]:
    """
    SASS trace Generator function to read SASS trace file line by line.
    Args:   sass_path: SASS trace file path
    Yields:     str: single line from the SASS trace file
    Raises:
        FileNotFoundError: file not found
        IOError: file cannot be read
    """
    if not os.path.exists(sass_path):
        raise FileNotFoundError(f"SASS trace file not found: {sass_path}")
    with open(sass_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                yield line

def parse(units_latency, sass_instructions, sass_path, logger):
    sass_trace = read_sass_trace_generator(sass_path)
    # sass_trace = open(sass_path,'r').read().strip().split('\n')

    task_list = {}
    dependency_map = {}
    warp_inst_count = {}
    count_gmem_reqs = 0
    opcnt_dict = {}
    opcnt_dict["ST"] = 0
    opcnt_dict["LD"] = 0
     
    for inst in sass_trace:
        inst_list = []
        splitted_inst = inst.split(" ")
        # new!! add sm_id to sass parser results
        sm_id = splitted_inst[0]
        # delete sm_id for not changing too many codes
        splitted_inst = splitted_inst[1:]
        # two kind of trace: sm_id warp_id \ sm_id cta_id_x cta_id_y cta_id_z warp_id
        if splitted_inst[1].isdigit():
            # new!! add cta_id_x, cta_id_y, cta_id_z to sass parser results
            cta_id_x = int(splitted_inst[0])
            cta_id_y = int(splitted_inst[1])  
            cta_id_z = int(splitted_inst[2])
            # delete cta_id_x, cta_id_y, cta_id_z for not changing too many codes
            splitted_inst = splitted_inst[3:]
            sm_id = sm_id +'#' + str(cta_id_x) + '#' + str(cta_id_y) + '#' + str(cta_id_z)

        warp_id = int(splitted_inst[0])
        current_inst = splitted_inst[1]
        opcodeAndOption = current_inst.split(".")
        opcode = opcodeAndOption[0]
        opcodeAndOption.pop(0)
        
        isOpcodeST = False
        # if opcode in uniform_insts_list: # add 'U' before register index in uniform instructions
        #     for i in range(len(opcodeAndOption)):
        #         opcodeAndOption[i] = 'U' + opcodeAndOption[i]
        
        if "ST" in opcode:
            opcnt_dict["ST"] += 1
        elif "LD" in opcode:
            opcnt_dict["LD"] += 1
        elif opcode in opcnt_dict:
            opcnt_dict[opcode] += 1
        else:
            opcnt_dict[opcode] = 1
        
        # create SM in warp_inst_count
        if sm_id in warp_inst_count:
            if warp_id in warp_inst_count[sm_id]:
                warp_inst_count[sm_id][warp_id] += 1
            else:
                warp_inst_count[sm_id][warp_id] = 1
        else:
            warp_inst_count[sm_id] = {}
            warp_inst_count[sm_id][warp_id] = 1
        #(1) type of inst
        if "LDG" in opcode:
            if "LDGSTS" in opcode:
                inst_list.append("GLOB_MEM_LD_STS")
            else:
                inst_list.append("GLOB_MEM_LD")
            inst_list.append("LD")
            count_gmem_reqs += 1
        elif "STG" in opcode:
            isOpcodeST = True
            inst_list.append("GLOB_MEM_ST")
            inst_list.append("ST")
            count_gmem_reqs += 1
        elif "LDL" in opcode:
            inst_list.append("LOCAL_MEM_LD")
            inst_list.append("LD")
        elif "STL" in opcode:
            isOpcodeST = True
            inst_list.append("LOCAL_MEM_ST")
            inst_list.append("ST")
        elif "LDS" in opcode:
            inst_list.append("SHARED_MEM_LD")
            inst_list.append("LD")
        elif "STS" in opcode:
            isOpcodeST = True
            inst_list.append("SHARED_MEM_ST")
            inst_list.append("ST")
        elif "LDC" in opcode:
            inst_list.append("CONST_MEM_ACCESS")
            inst_list.append("LD")
        elif "STC" in opcode:
            isOpcodeST = True
            inst_list.append("CONST_MEM_ACCESS")
            inst_list.append("ST")
        elif "LD" in opcode:
            inst_list.append("GLOB_MEM_LD")
            inst_list.append("LD")
            count_gmem_reqs += 1
        elif "ST" in opcode:
            isOpcodeST = True
            inst_list.append("GLOB_MEM_ST")
            inst_list.append("ST")
            count_gmem_reqs += 1
        elif "ATOM" in opcode or "RED" in opcode:
            inst_list.append(opcode)
            inst_list.append("") #for now just put an empty holder; need to be changed to the type of atomic operation later
        elif "BAR" in opcode:
            inst_list.append("BarrierSYNC")
        elif "MEMBAR" in opcode or "CCTL" in opcode:
            inst_list.append("MEMBAR")
        else:
            try:
                if "MUFU" in opcode:
                    unit = "SFU"
                    if "64" in opcodeAndOption:
                        unit_64 = "dSFU"
                        latency = units_latency[unit_64]
                    else:
                        latency = units_latency[unit]        
                # data type of accumulator influence the latency   
                elif "HMMA" in opcode:
                    flops = 0
                    if "16816" in opcodeAndOption:
                        flops = 16 * 8 * 16
                    elif "1688" in opcodeAndOption:
                        flops = 8 * 8 * 16
                    elif "848" in opcodeAndOption: 
                        flops = 8 * 8 * 8
                    else:
                        print("[Error] Unknown HMMA instruction")
                    
                    if "BF16" in opcodeAndOption:
                        unit = "bTCU"
                    elif "F32" in opcodeAndOption:
                        unit = "fTCU"
                    else:
                        unit = "hTCU"
                    latency = flops
                else:
                    unit = sass_instructions[opcode]
                    latency = units_latency[unit]
                
                # if "EXIT" in opcode:
                #     dependency_map[sm_id][warp_id] = {}
            except:
                print(opcode)
                print("\n[Error]\n"+"\""+current_inst+"\""+" is not available in SASS instructions table")
                exit(0)
            # add the instruction HW unit to the warp task_list
            inst_list.append(unit)
            # add the instruction latency to the warp task_list
            inst_list.append(latency)
        #(2) add current instruction dependencies
        destination = None      
        if sm_id not in dependency_map:
            dependency_map[sm_id] = {}
            dependency_map[sm_id][warp_id] = {}
        else:
            if warp_id not in dependency_map[sm_id]:
                dependency_map[sm_id][warp_id] = {}
        
        for i in range(2, len(splitted_inst)):
            if i == 2 and not isOpcodeST:
                destination = splitted_inst[i]
            else:
                source = splitted_inst[i]
                # check the source reigster used by the current instruction is already used by other instructions in the same warp or not
                if source in dependency_map[sm_id][warp_id]:
                    inst_list.append(dependency_map[sm_id][warp_id][source])

        if destination is not None:
            # store every register which is used by the current instruction to the dependency map
            dependency_map[sm_id][warp_id][destination] = warp_inst_count[sm_id][warp_id] - 1
            
        #(3) commit the instruction list to the task_list
        if sm_id in task_list:
            if warp_id not in task_list[sm_id]:
                task_list[sm_id][warp_id] = []
        else:
            task_list[sm_id] = {}
            task_list[sm_id][warp_id] = []
        task_list[sm_id][warp_id].append(inst_list)

    # logging
    sorted_opcnt = sorted(opcnt_dict.items(), key=lambda item: item[1], reverse=True)
    for key, value in sorted_opcnt:
        logger.write(key, value)
    logger.write("### SASS Trace Summary ###")
    logger.write("number of CTA:",len(task_list))
    task_len_cnt = {}
    warp_num = 0
    for sm in task_list:
        warp_num += len(task_list[sm])
        logger.write("CTA info:","sm id#CTA x#CTA y#CTA z:",sm, "warp number of this CTA:",len(task_list[sm]))
        warp_id_str = ""
        warp_instr_num_str = ""
        for warp in task_list[sm]:
            cur_warp_task_len = len(task_list[sm][warp])
            warp_id_str += "{:8d}".format(warp) + " "
            warp_instr_num_str += "{:8d}".format(cur_warp_task_len) + " "
            if cur_warp_task_len not in task_len_cnt:
                task_len_cnt[cur_warp_task_len] = 1
            else:
                task_len_cnt[cur_warp_task_len] += 1
        logger.write(warp_id_str)
        logger.write(warp_instr_num_str)
    logger.write("### SASS Trace Summary ###")
    logger.write()
    logger.write("number of warp:",warp_num)
    sorted_task_len_cnt = sorted(task_len_cnt.items(), key=lambda item: item[0], reverse=True)
    for key, value in sorted_task_len_cnt:
        logger.write("task len: {:d} number: {:d}".format(key,value))
    # end logging 
    # select representative warp using kmean clustering   
    import numpy as np
    from collections import Counter

    def transform_task_list(task_list, functional_units_list):
        flattened_warps = []
        warp_info = []  
        
        for sm_id in sorted(task_list.keys()):
            for warp_id in sorted(task_list[sm_id].keys()):
                warp_vector = task_list[sm_id][warp_id]
                unit_counter = Counter(item[0] for item in warp_vector if item)
                count_vector = [unit_counter.get(unit, 0) for unit in functional_units_list]
                flattened_warps.append(count_vector)
                warp_info.append((sm_id, warp_id))
        
        return np.array(flattened_warps), warp_info

    def get_original_sm_and_warp_ids(representative_indices, warp_info):
        if isinstance(representative_indices, list):
            return [warp_info[i] for i in representative_indices]  # 列表情况
        else:
            return warp_info[representative_indices]  # 单个值的情况

    kmeans_features, warp_info = transform_task_list(task_list, functional_units_list)
    all_center_warp_idx_list, representative_index = rptv_warp_select(kmeans_features)
    original_sm_and_warp_ids = get_original_sm_and_warp_ids(representative_index, warp_info)

    print(f"Representative warp - SM ID: {original_sm_and_warp_ids[0]}, Warp ID: {original_sm_and_warp_ids[1]}")
    logger.write(f"Representative warp - SM ID: {original_sm_and_warp_ids[0]}, Warp ID: {original_sm_and_warp_ids[1]}")
    print(f"Length of task_list of representative warp: {len(task_list[original_sm_and_warp_ids[0]][original_sm_and_warp_ids[1]])}")
    logger.write(f"Length of task_list of representative warp: {len(task_list[original_sm_and_warp_ids[0]][original_sm_and_warp_ids[1]])}")

    # total_warp_num = 0
    # for CTA_id in task_list:
    #     total_warp_num += len(task_list[CTA_id])
    rptv_warp_task_list = task_list[original_sm_and_warp_ids[0]][original_sm_and_warp_ids[1]]

    total_warp_num = 0
    # scan all CTA and Count warp number in all SM and sub-cores
    warp_num_count = {}
    warp_instr_num_count = {}
    active_SMs_set = set()
    for CTA_id in task_list:
        for warp_id in task_list[CTA_id]:
            sm_id = sm_id_str_to_int(CTA_id)
            active_SMs_set.add(sm_id)
            if sm_id not in warp_num_count:
                warp_num_count[sm_id] = [0] * 4
                warp_instr_num_count[sm_id] = [0] * 4
            warp_num_count[sm_id][warp_id % 4] += 1
            warp_instr_num_count[sm_id][warp_id % 4] += len(task_list[CTA_id][warp_id])
        total_warp_num += len(task_list[CTA_id])
    active_SMs_num = len(active_SMs_set)

    logger.write("### Warp number and intr number distribution ###")
    for sm_id in sorted(warp_num_count.keys()):
        logger.write("SM {:d} warp number:".format(sm_id), warp_num_count[sm_id])
        logger.write("SM {:d} warp instruction number:".format(sm_id), warp_instr_num_count[sm_id])
    logger.write("### Warp number and intr number distribution ###")

    return rptv_warp_task_list, count_gmem_reqs, original_sm_and_warp_ids, total_warp_num, active_SMs_num