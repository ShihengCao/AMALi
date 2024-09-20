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

from .utils import *
class Warp(object):
    '''
    class that represents a warp inside a block being executed on an SM
    '''
    def __init__(self, block, gpu, tasklist, kernel_id, SM_id, warp_id, avg_mem_lat, avg_atom_lat):
        
        self.gpu = gpu
        self.active = True # True if warp still has computations to run
        self.stalled_cycles = 0 # number of cycles before the warp can issue a new instruction
        self.block = block
        self.current_inst = 0 # current instruction in the tasklist
        self.tasklist = tasklist
        self.SM_id = SM_id
        self.id = warp_id

        self.completions = []
        self.syncing = False
        self.max_dep = 0
        self.average_memory_latency = avg_mem_lat
        self.average_atom_latency = avg_atom_lat
        self.kernel_id = kernel_id
        self.divergeStalls = 0

    def get_inst_latency(self, inst):
        if "GLOB" in inst[0]: 
            if "STS" in inst[0]:
                latency = int(self.average_memory_latency) + self.gpu.shared_mem_st_latency
            else:
                latency = int(self.average_memory_latency)
        elif inst[0] == "SHARED_MEM_ST":
            latency = self.gpu.shared_mem_st_latency
        elif inst[0] == "SHARED_MEM_LD":
            latency = self.gpu.shared_mem_ld_latency
        elif "LOCAL" in inst[0]:
            latency = self.gpu.local_mem_access_latency
        elif inst[0] == "CONST_MEM_ACCESS":
            latency = self.gpu.const_mem_access_latency
        elif inst[0] == "TEX_MEM_ACCESS":
            latency = self.gpu.tex_mem_access_latency
        elif inst[0] == "ATOMIC_OP":
            latency = int(self.average_atom_latency)
        elif inst[0] == "MEMBAR" or inst[0] == "BarrierSYNC":
            latency = 0
        elif inst[0] == "ATOMS":
            latency = self.gpu.shared_mem_st_latency + self.gpu.shared_mem_ld_latency
        elif "ATOM" in inst[0] or "RED" in inst[0]:
            latency = int(self.average_atom_latency)
        else:
            latency = inst[1]
        return latency
    
    def calculate_completion_and_issue(self):
        '''
        calculate the issue cycle and the completion cycle of every instruction

        Return:
            stall_reason_list: a list of stall reason, 0 means not stall, 1 means stall for memory, 2 means stall for computation
            access_info: 0 means L1 cache access, 1 means other memory access or compute
            hw_unit_list: a list of computation hw unit type, None means no computation hw unit used
            issues: a list of issue cycle of every instruction
        '''
        stall_reason_list = []
        mem_access_type_list = []
        hw_unit_list = []
        issue_cycle_list = [-1]
        # interval_list = []
        # stall_type_count = {}
        for inst in self.tasklist:
            self.current_inst += 1
            latency = self.get_inst_latency(inst)
            # caculate the latnecy for TCU
            if "TCU" in inst[0]:
                latency = latency / self.gpu.units_latency[inst[0]] 
            
            max_dep = -1
            stall_reason = 0
            # find the completion cycle of all dependence
            for i in inst[2:]:
                if self.completions[i] > max_dep:
                    max_dep = max(max_dep, self.completions[i])
                    stall_reason = i
            
            issue_cycle = 0
            # caculate the issue cycle of current instruction
            if max_dep > issue_cycle_list[-1]:
                # stall for some reason
                issue_cycle = max_dep + 1
                stall_inst = self.tasklist[stall_reason]
                if "MEM" in stall_inst[0]: # 1 means memData 2 means comData
                    stall_reason_list.append(1)
                else:
                    stall_reason_list.append(2)
                # if stall_inst[0] not in stall_type_count:
                #     stall_type_count[stall_inst[0]] = 1
                # else:
                #     stall_type_count[stall_inst[0]] += 1
                
            else:
                # not stall, serial issue next instruction
                stall_reason_list.append(0)
                issue_cycle = issue_cycle_list[-1] + 1

            issue_cycle_list.append(issue_cycle)
            self.completions.append(issue_cycle + latency)

            if inst[0] in functional_units_list:
                hw_unit_list.append(inst[0])
            elif "LD" in inst[0] or "ST" in inst[0] or "ATOM" in inst[0] or "RED" in inst[0]:
                hw_unit_list.append("LDST")
            elif "TCU" in inst[0]: # for TCU, we will use latency as the hw unit type
                hw_unit_list.append("TCU"+str(latency))
            else:
                hw_unit_list.append(None)

            if inst[0] in ["ATOMIC_OP", "GLOB_MEM_LD","LOCAL_MEM_LD", "GLOB_MEM_LD_STS"]:
                mem_access_type_list.append(1) # 1 means L1 cache access, 0 means other memory access or compute instrucion
            elif inst[0] in ["GLOB_MEM_ST","LOCAL_MEM_ST"]:
                mem_access_type_list.append(2)
            else:
                mem_access_type_list.append(0)
        # calculate the cycles which is the max element of self.completions
        total_cycle = max(issue_cycle_list)
        return stall_reason_list, mem_access_type_list, hw_unit_list, issue_cycle_list, total_cycle 
      
    def interval_analyze(self):
        '''
        process each instruction in the tasklist to get the interval profile
        '''
        self.current_inst = 0
        interval_list, total_intervals = [], 0
        stall_reason_list, cache_access_list, hw_unit_list, issue_cycle_list, total_cycle = self.calculate_completion_and_issue()
        def step_update(current_interval_length, mem_access_num, mem_read_num, functional_units, index):
            '''
            update the current_interval_length, mem_access_num, functional_units \\
            according to the current instruction
            '''
            current_interval_length += 1
            mem_access_num += 1 if cache_access_list[index] > 0 else 0
            mem_read_num += 1 if cache_access_list[index] == 1 else 0
            if hw_unit_list[index] is not None:
                if hw_unit_list[index] not in functional_units:
                    functional_units[hw_unit_list[index]] = 1
                else:
                    functional_units[hw_unit_list[index]] += 1   
            return current_interval_length, mem_access_num
        def initilaizing_variables():
            '''
            Returns:
                initialaizing values of functional_units, mem_access_num, mem_read_num, current_interval_length
            '''
            return {}, 0 , 0 , 0    
        # split the result of func <calculate_complete_and_issue> into intervals
        functional_units, mem_access_num, mem_read_num, current_interval_length = initilaizing_variables() 
        former_cycle = issue_cycle_list[0]
        for index in range(len(issue_cycle_list[1:])):
            issue_cycle = issue_cycle_list[index+1]
            if issue_cycle - former_cycle == 1:
                # active cycle of an interval
                current_interval_length, mem_access_num = step_update(current_interval_length, 
                                                                      mem_access_num,
                                                                      mem_read_num, 
                                                                      functional_units, 
                                                                      index)
            else:
                # found a stall event and create a new interval
                interval_list.append({"issue_stage":current_interval_length, 
                                           "functional_units":functional_units, 
                                           "mem_access_num":mem_access_num,
                                           "mem_read_num":mem_read_num,
                                           "end_cycle_idx":issue_cycle})
                total_intervals += 1
                # add the stall stage
                interval_list.append({"stall_stage":issue_cycle-former_cycle,
                                          "stall_type":stall_reason_list[index]})
                # reinitalize the variables
                functional_units, mem_access_num, mem_read_num, current_interval_length = initilaizing_variables()
                current_interval_length, mem_access_num = step_update(current_interval_length, 
                                                                      mem_access_num,
                                                                      mem_read_num, 
                                                                      functional_units, 
                                                                      index)
            former_cycle = issue_cycle
        # add the last interval
        interval_list.append({"issue_stage":current_interval_length, 
                            "functional_units":functional_units, 
                            "mem_access_num":mem_access_num,
                            "mem_read_num":mem_read_num,
                            "end_cycle_idx":issue_cycle})
        total_intervals += 1

        total_cycle = max(issue_cycle_list)
        # print(interval_list)
        return interval_list, total_cycle, total_intervals