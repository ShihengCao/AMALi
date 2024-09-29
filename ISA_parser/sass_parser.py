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
from src.utils import uniform_insts_list, functional_units_list, rptv_warp_select
from collections import Counter
import numpy as np

def parse(units_latency, sass_instructions, sass_path, logger):
    
    sass_trace = open(sass_path,'r').read().strip().split('\n')

    task_list = {}
    dependency_map = {}
    warp_inst_count = {}
    count_gmem_reqs = 0
    opcnt_dict = {}
    opcnt_dict["ST"] = 0
    opcnt_dict["LD"] = 0

    # insts_count = sass_instructions.copy()    
    # insts_count["ST"] = 0
    # insts_count["LD"] = 0
    # def flush_dict(input_dict:dict):
    #     for key in input_dict:
    #         input_dict[key] = 0
    # flush_dict(insts_count)
     
    for inst in sass_trace:
        inst_list = []

        splitted_inst = inst.split(" ")
        # new!! add sm_id to sass parser results
        sm_id = int(splitted_inst[0])
        # delete sm_id for not changing too many codes
        splitted_inst = splitted_inst[1:]
        
        warp_id = int(splitted_inst[0])
        current_inst = splitted_inst[1]
        opcodeAndOption = current_inst.split(".")
        opcode = opcodeAndOption[0]
        opcodeAndOption.pop(0)
        
        isOpcodeST = False

        if opcode in uniform_insts_list: # add 'U' before register index in uniform instructions
            for i in range(len(opcodeAndOption)):
                opcodeAndOption[i] = 'U' + opcodeAndOption[i]
        
        if "ST" in opcode:
            opcnt_dict["ST"] += 1
        elif "LD" in opcode:
            opcnt_dict["LD"] += 1
        elif opcode in opcnt_dict:
            opcnt_dict[opcode] += 1
        else:
            opcnt_dict[opcode] = 1

        # ??? warp_inst_count[warp_id] = 0 ???
        # if warp_id in warp_inst_count:
        #     warp_inst_count[warp_id] += 1
        # else:
        #     warp_inst_count[warp_id] = 0
        
        # create SM in warp_inst_count
        if sm_id in warp_inst_count:
            if warp_id in warp_inst_count[sm_id]:
                warp_inst_count[sm_id][warp_id] += 1
            else:
                warp_inst_count[sm_id][warp_id] = 0
        else:
            warp_inst_count[sm_id] = {}
            warp_inst_count[sm_id][warp_id] = 0
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
                    if "F32" in opcodeAndOption:
                        unit = "fTCU"
                    else:
                        unit = "hTCU"
                    latency = units_latency[unit]
                else:
                    unit = sass_instructions[opcode]
                    latency = units_latency[unit]

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
            dependency_map[sm_id][warp_id][destination] = warp_inst_count[sm_id][warp_id]
            
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

    logger.write("number of sm:",len(task_list))
    task_len_cnt = {}
    warp_num = 0
    for sm in task_list:
        warp_num += len(task_list[sm])
        logger.write("sm info:","sm id:",sm, "warp number of this sm:",len(task_list[sm]))
        for warp in task_list[sm]:
            cur_warp_task_len = len(task_list[sm][warp])

            logger.write("warp {:d} task_len: ".format(warp),cur_warp_task_len,end='; ')

            if cur_warp_task_len not in task_len_cnt:
                task_len_cnt[cur_warp_task_len] = 1
            else:
                task_len_cnt[cur_warp_task_len] += 1
            instruction_streaming_cnt = 0
            for task in task_list[sm][warp]:
                if task[0] == "EXIT":
                    logger.write(instruction_streaming_cnt + 1,end=' ')
                    instruction_streaming_cnt = 0
                else:
                    instruction_streaming_cnt += 1
            #logger.write(';',end=' ')
        logger.write()
    
    logger.write("number of warp:",warp_num)

    sorted_task_len_cnt = sorted(task_len_cnt.items(), key=lambda item: item[0], reverse=True)
    for key, value in sorted_task_len_cnt:
        logger.write("task len: {:d} number: {:d}".format(key,value))
    # end looging and return     
    def transform_task_list(task_list, functional_units_list):
        flattened_warps = []
        warp_info = []  # 用于存储每个warp的sm_id和warp_id
        
        for sm_id in sorted(task_list.keys()):
            for warp_id in sorted(task_list[sm_id].keys()):
                warp_vector = task_list[sm_id][warp_id]
                
                # 统计每个functional unit的出现次数
                unit_counter = Counter(item[0] for item in warp_vector if item)
                
                # 创建一个与functional_units_list对应的向量
                count_vector = [unit_counter.get(unit, 0) for unit in functional_units_list]
                
                # 计算总指令数
                total_instructions = sum(count_vector)
                
                # # 如果总指令数为0，我们将所有值设为0以避免除以0
                # if total_instructions == 0:
                #     normalized_vector = [0] * len(functional_units_list)
                # else:
                #     # 归一化处理，计算每种指令的占比
                #     normalized_vector = [count / total_instructions for count in count_vector]
                
                flattened_warps.append(count_vector)
                warp_info.append((sm_id, warp_id))
        
        return np.array(flattened_warps), warp_info

    # 假设task_list已经定义
    kmeans_features, warp_info = transform_task_list(task_list, functional_units_list)
    # print(kmeans_features)
    all_center_warp_idx_list, representative_index = rptv_warp_select(kmeans_features)
    print("all_center_warp_idx and represetative warp index")
    print(all_center_warp_idx_list, representative_index)
    # 根据representative_index找出对应的sm_id和warp_id
    def get_sm_and_warp_id(representative_index, warp_info):
        return warp_info[representative_index]
    # print(warp_info,r)
    sm_warp_pairs = get_sm_and_warp_id(representative_index, warp_info)
    print(len(task_list[sm_warp_pairs[0]][sm_warp_pairs[1]]))
    return task_list, count_gmem_reqs, sm_warp_pairs