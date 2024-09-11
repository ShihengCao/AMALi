##############################################################################

# Author: Shiheng Cao
# Last Update Date: January 2024
# Copyright: Open source, must acknowledge original author

##############################################################################

from .warps import Warp
# from .kernels import Kernel
# from .blocks import Block
from .MDM import process_MDM
from .helper_methods import ceil
# from .utils import rptv_warp_select

def process_GCoM(kernel, warp: Warp, warps_per_sub_core: int, active_SMs: int, umem_hit_rate: float):
		'''
			this function is used to calculate the GCoM of a Kernel
			
			Args:
				kernel: the simian kernel object
				pred_out: the output of the prediction stage
				warp: current warp to analyze
				interval_analysis: the result of interval analysis of the representative warp
				total_cycles: the total number of active cycles
				
			Returns:
				general_output_GCoM: the GCoM of the warp
		'''
		'''
			model sub-core
			1. the number of warps per sub-core
			2. the number of sub-cores per SM
			warps of sub-core = warps of SM / sub-cores per SM
		'''
		interval_list, total_cycles, total_intervals = warp.interval_analyze()
		# initial variable
		# warps_per_SM = pred_out["th_active_warps"]
		num_sub_cores_per_SM = kernel.acc.num_warp_schedulers_per_SM
		#warps_ij = int(ceil(warps_per_SM / num_sub_cores_per_SM,1)) # Warpsi,j is the number of warps the j-th sub-core of the i-th SM executes
		warps_ij = warps_per_sub_core
		warps_per_SM = warps_ij * num_sub_cores_per_SM
		issue_rate = 1 # 1 instruction is issued per cycle
		num_warp_inst = warp.current_inst
		total_num_warp_inst = num_warp_inst * warps_ij

		# calculate C_base_ij
		C_base_ij = total_num_warp_inst / issue_rate 
		actual_end = max(warp.completions)		
		C_base_ij += actual_end - total_cycles

		# calculate S_ComData_ij and S_MemData_ij
		S_ComData_ij = 0
		S_MemData_ij = 0
		# according to GPUMech, the probability of issuing an instruction
		P_inst = num_warp_inst / total_cycles
		
		if kernel.acc.warp_scheduling_policy == "GTO":		
			'''
				Use GTO warp schedule policy
			'''
			avg_int_insts = (num_warp_inst / total_intervals)
			for stage_info in interval_list:
				if "stall_stage" in stage_info:
					S_intv_k = stage_info["stall_stage"]
					P_warp = min(S_intv_k * P_inst, 1)
					cycle_other = int(P_warp* (warps_ij - 1) * avg_int_insts / issue_rate)
					if stage_info["stall_type"] == 2:
						S_ComData_ij += max(int(S_intv_k - cycle_other), 0) 
						# stall_cyc.append(S_intv_k)
					else:
						S_MemData_ij += max(int(S_intv_k - cycle_other), 0)
		else:  
			'''
				Use LRR warp schedule policy
			'''
			for stage_info in interval_list:
				if "stall_stage" in stage_info:
					S_intv_k = stage_info["stall_stage"]
					if stage_info["stall_type"] == 2:
						S_ComData_ij += max(int(S_intv_k - (warps_ij - 1) * P_inst), 0)
					else:
						S_MemData_ij += max(int(S_intv_k - (warps_ij - 1) * P_inst), 0)
		
		# calculate C_active_ij and C_idle_ij
		C_active_ij = C_base_ij + S_ComData_ij + S_MemData_ij
		C_idle_ij = 0 # we will calculate it later in kernel
		C_ij = C_active_ij + C_idle_ij
		'''
			Modeling the Cycles of a Core
		'''
		Si = 0
		S_ComStruct_i = 0
		S_MemStruct_i = 0
		'''		
			GCoM claim: the maximum number of warps that an SM can concurrently execute

			The maximum number of concurrent warps per SM is 32 on Turing (versus 64 on Volta)
			The maximum number of concurrent warps per SM remains the same as in Volta (i.e., 64) 
			for compute capability 8.0, while for compute capability 8.6 it is 48.

			see Nvidia doc https://docs.nvidia.com/cuda/turing-tuning-guide/index.html
			see Nvidia doc https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html
		'''
		num_cncr_warps = kernel.acc.max_active_warps_per_SM
		# num_cncr_warps = pred_out["th_active_warps"]
  
		# byte/cycle/SM test on ./l1_bw_32f microbenchmark in accel-sim
		B_L1_k = kernel.acc.l1_cache_bandwidth

		def issue_base(x, ik):
			'''
				return the base cycle to execute all instruction of the issue stage
				Inputs:
				x: number of warps execute in current issue stage
				ik: number of instruction of current issue stage

				Returns:
				issue_base: the issue_base of a SM
			'''
			if x == 0:
				return 0
			num_act_sub_core = min(x, num_sub_cores_per_SM)
			return ik * x / (num_act_sub_core * issue_rate)


		def issue_k_m(x,functional_units):
			'''
				return the stall cycle of SM caused by limited funcitonal units
				Inputs:
				x: number of warps execute in current issue stage
				functional_units: a dictionary of functional units which is used in current issue stage

				Returns:
				issue_km: the issue_km of a SM
			'''
			issue_km = -100
			num_act_sub_core = min(x, num_sub_cores_per_SM)
			for unit in functional_units:
				iim = kernel.acc.initial_interval[unit]
				im = functional_units[unit] # num of instuction which used the unit
				issue_km = max(issue_km, im * iim * x / (num_act_sub_core * issue_rate))		
			return issue_km			

		def issue_max(x, ik, fu, bk):
			'''
				return the max cycle of stall for multi warp concurrently execute
				Inputs:
				x: number of warps execute in current issue stage
				ik: number of instruction of current issue stage
				fu: list of functional units which is used in current issue stage
				bk: number of L1 cache access in current issue stage

				Returns:
				result: the max cycle of stall for multi warp concurrently execute
				is_mem: whether the stall is caused by L1 cache access
			'''
			if x == 0:
				return 0, False
			issue_base_var = issue_base(x, ik)
			issue_k_m_var = issue_k_m(x, fu)
			num_act_sub_core = min(x, num_sub_cores_per_SM)
			# one memory instruction like LDSM.16.M88.4 R72, [R51+UR8+0x800] will load data to the R72 register file 
			# to every thread which leads to 32bit * 32 / 8 = 128B L1 cache load
			issue_k_L1_var = (bk * 4 * kernel.acc.l1_cache_line_size / B_L1_k) * x / (num_act_sub_core * issue_rate)
			# issue_k_L1_var = bk * kernel.acc.l1_cache_access_latency * x / (num_act_sub_core * issue_rate)
			issue_k_L1_var = ceil(issue_k_L1_var, 1)
			result = max(issue_base_var, issue_k_m_var, issue_k_L1_var)
			is_mem = issue_k_L1_var > issue_k_m_var and issue_k_L1_var > issue_base_var
			return result, is_mem 
		
		def com_struct_and_mem_struct(x):
			com, mem = 0, 0
			for interval_info in interval_list:
				if "issue_stage" not in interval_info:
					continue
				ik = interval_info['issue_stage']
				fu = interval_info['functional_units'] if 'functional_units' in interval_info else {}
				mem_acesss_num = interval_info['mem_access_num']
				issue_max_var, is_mem = issue_max(x, ik, fu, mem_acesss_num)
				if is_mem:
					mem += issue_max_var - issue_base(x, ik)
				else:
					com += issue_max_var - issue_base(x, ik)
			return com, mem
		# print(warps_per_SM)
		com1, mem1 = com_struct_and_mem_struct(num_cncr_warps)
		com2, mem2 = com_struct_and_mem_struct(int(warps_per_SM % num_cncr_warps))
		S_ComStruct_i = com1 * (warps_per_SM // num_cncr_warps) + com2
		S_MemStruct_i = mem1 * (warps_per_SM // num_cncr_warps) + mem2
		
		MDM_output = process_MDM(interval_list, kernel, active_SMs, umem_hit_rate, warps_per_SM)
		
		S_MSHR_i = int(MDM_output["MSHR"])
		S_NoC_i = int(MDM_output["NoC"])
		S_Dram_i = int(MDM_output["Dram"])
		
		Si = S_ComStruct_i + S_MemStruct_i + S_MSHR_i + S_NoC_i + S_Dram_i
		C_active_i = C_ij + Si

		C_idle_i = 0 # we will calculate it later in kernel
		C = int(C_active_i + C_idle_i)
		
		general_GCoM_output = {
			"GCoM": C,
			"C_base_ij": C_base_ij,
			"S_ComData_ij": S_ComData_ij,
			"S_MemData_ij": S_MemData_ij,
			"C_idle_i": C_idle_i,
			"C_idle_ij": C_idle_ij,
			"S_ComStruct_i": S_ComStruct_i,
			"S_MemStruct_i": S_MemStruct_i,
			"S_MSHR_i": S_MSHR_i,
			"S_NoC_i": S_NoC_i,
			"S_Dram_i": S_Dram_i,
		}
		# print(general_GCoM_output)
		return general_GCoM_output, total_cycles

def get_idle_cycles(kernel, rptv_SM, rptv_warp, rptv_output, SM_block_list, active_SMs, umem_hit_rate):
	max_cycles_in_sub_core = -1
	for rptv_sub_core_block in rptv_SM:
		sub_core_warps_num = len(rptv_sub_core_block.warp_list)
		tmp_warp_GCoM_output = process_GCoM(kernel, rptv_warp, sub_core_warps_num, active_SMs, umem_hit_rate)
		if tmp_warp_GCoM_output["GCoM"] > max_cycles_in_sub_core:
			max_cycles_in_sub_core = tmp_warp_GCoM_output["GCoM"]
	if max_cycles_in_sub_core > rptv_output["GCoM"]:
		rptv_output["C_idle_ij"] += max_cycles_in_sub_core - rptv_output["GCoM"]
		rptv_output["GCoM"] += rptv_output["C_idle_ij"]

	max_cycles_in_SM = -1
	for SM in SM_block_list:
		for sub_core_block in SM:
			sub_core_warps_num = len(sub_core_block.warp_list)
			tmp_warp_GCoM_output = process_GCoM(kernel, rptv_warp, sub_core_warps_num, active_SMs, umem_hit_rate)
			if tmp_warp_GCoM_output["GCoM"] > max_cycles_in_SM:
				max_cycles_in_SM = tmp_warp_GCoM_output["GCoM"]
	if max_cycles_in_SM > rptv_output["GCoM"]:
		rptv_output["C_idle_i"] += max_cycles_in_SM - rptv_output["GCoM"]
		rptv_output["GCoM"] += rptv_output["C_idle_i"]

	def calculate_rptv_warp_GCoM(kernel, SM, 
							  represetative_index, all_center_warp_idx_list, 
							  pred_out,
							  total_warp_num,
							  max_warp_per_sub_core, max_warp_per_SM):
		# find the represetative warp based on the represetative index
		# def as an isolated func to break multi layer loops
		rptv_warp = None
		rptv_warp_GCoM_output = None
		tmp_idx = 0
		max_cycle = -1
		for SM in SM_block_list:
			# count the number of warps in the SM
			SM_warps_num = 0
			for tmp_block in SM:
				SM_warps_num += len(tmp_block.warp_list)
				
			for sub_core_block in SM:
				for warp in sub_core_block.warp_list:
					if tmp_idx == represetative_index:
						rptv_warp = warp
						# get the rptv_warp and process GCoM
						sub_core_warps_num = len(sub_core_block.warp_list)
						interval_analysis_result = rptv_warp.interval_analyze()
						pred_out["warps_instructions_executed"] = rptv_warp.current_inst * total_warp_num
						rptv_warp_GCoM_output = kernel.process_GCoM(rptv_warp, interval_analysis_result,
										sub_core_warps_num, SM_warps_num, pred_out["active_SMs"],
										pred_out["umem_hit_rate"])
						# return -1
						# candidate max cycles warp in the whole SM
						max_GCoM_output = kernel.process_GCoM(rptv_warp, interval_analysis_result,
										max_warp_per_sub_core, max_warp_per_SM, pred_out["active_SMs"],
										pred_out["umem_hit_rate"])
						# update the idle ij cycles
						kernel.update_c_idle_ij(SM, rptv_warp, interval_analysis_result, 
							rptv_warp_GCoM_output, pred_out["active_SMs"],
							pred_out["umem_hit_rate"])
					elif tmp_idx in all_center_warp_idx_list:
						sub_core_warps_num = len(sub_core_block.warp_list)
						interval_analysis_result = warp.interval_analyze()
						warp_GCoM_output = kernel.process_GCoM(warp, interval_analysis_result,
										sub_core_warps_num, SM_warps_num, pred_out["active_SMs"],
										pred_out["umem_hit_rate"])
						# print(warp_GCoM_output["GCoM"])
						if max_cycle < warp_GCoM_output["GCoM"]:
							max_cycle = warp_GCoM_output["GCoM"]
						# pass
					tmp_idx += 1
		# update C_idle_i
		max_cycle = max(max_cycle, max_GCoM_output["GCoM"])
		if max_cycle > rptv_warp_GCoM_output["GCoM"]:
			rptv_warp_GCoM_output["C_idle_i"] = max_cycle - rptv_warp_GCoM_output["GCoM"]
			rptv_warp_GCoM_output["GCoM"] += rptv_warp_GCoM_output["C_idle_i"]
		return rptv_warp_GCoM_output