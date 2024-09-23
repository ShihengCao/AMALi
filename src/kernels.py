##############################################################################
# &copy 2017. Triad National Security, LLC. All rights reserved.

# This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration.

# All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

# Recall that this copyright notice must be accompanied by the appropriate open source license terms and conditions. Additionally, it is prudent to include a statement of which license is being used with the copyright notice. For example, the text below could also be included in the copyright notice file: This is open source software; you can redistribute it and/or modify it under the terms of the Performance Prediction Toolkit (PPT) License. If software is modified to produce derivative works, such modified software should be clearly marked, so as not to confuse it with the version available from LANL. Full text of the Performance Prediction Toolkit (PPT) License can be found in the License file in the main development branch of the repository.
##############################################################################

# Author: Shiheng Cao
# Last Update Date: April, 2024
# Copyright: Open source, must acknowledge original author

##############################################################################


import time, importlib
from typing import Tuple

from .helper_methods import *
from .memory_model import *
from .blocks import Block
from .utils import print_output_info, rptv_warp_select, write_to_file, Logger
from .warps import Warp

class Kernel():

	def __init__(self, base_info, gpuNode, kernel_info):
		# super(Kernel, self).__init__(base_info)
		self.num = base_info
		## There is 1 Acc in Node replicated for each Kernel
		self.gpuNode = gpuNode
		self.acc = self.gpuNode.accelerators[0] 
		# get kernel_info
		self.kernel_id_real = self.num
		self.kernel_name = kernel_info["kernel_name"]
		self.kernel_id = int(kernel_info["kernel_id"])
		self.mem_traces_dir_path = kernel_info["mem_traces_dir_path"]
		self.kernel_grid_size = kernel_info["grid_size"]
		self.kernel_block_size = kernel_info["block_size"]
		self.kernel_num_regs = kernel_info["num_regs"]
		self.kernel_smem_size = kernel_info["smem_size"]
		self.ISA = "SASS"
		self.sass_file_path = kernel_info["sass_file_path"]
		# self.method_name = kernel_info["method_name"]

		## kernel local predictions outputs
		self.pred_out = {}
		pred_out = self.pred_out

		pred_out["app_path"] = kernel_info["app_path"]
		pred_out["kernel_id"] = self.kernel_id 
		pred_out["kernel_name"] = self.kernel_name	

		pred_out["ISA"] = self.ISA
		pred_out["total_num_workloads"] = 0
		pred_out["active_SMs"] = 0
		pred_out["max_active_blocks_per_SM"] = self.acc.max_active_blocks_per_SM
		# pred_out["blocks_per_SM_limit_warps"] = 0
		# pred_out["blocks_per_SM_limit_regs"] = 0
		# pred_out["blocks_per_SM_limit_smem"] = 0
		# pred_out["th_active_blocks"] = 0
		# pred_out["th_active_warps"] = 0
		# pred_out["th_active_threads"] = 0
		# pred_out["allocated_active_blocks_per_SM"] = 0
		pred_out["allocated_active_warps_per_block"] = 0
		# pred_out["num_workloads_per_SM"] = 0
		# pred_out["num_workloads_per_SM_new"] = 0
		pred_out["warps_instructions_executed"] = 0
		pred_out["ipc"] = 0.0
		# pred_out["l1_cache_bypassed"] = self.acc.l1_cache_bypassed
		# pred_out["tot_warps_instructions_executed"] = 0
		pred_out["AMAT"] = 0.0
		pred_out["ACPAO"] = 0.0
		pred_out["l1_parallelism"] = 0
		pred_out["l2_parallelism"] = 0
		pred_out["dram_parallelism"] = 0
		pred_out["simulation_time_parse"] = 0.0
		pred_out["simulation_time_memory"] = 0.0
		pred_out["simulation_time_compute"] = 0.0
		
		if self.kernel_block_size > self.acc.max_block_size:
			print_warning("block_size",str(self.acc.max_block_size))
			self.kernel_block_size = self.acc.max_block_size

		if self.kernel_num_regs > self.acc.max_registers_per_thread:
			print_warning("num_registers",str(self.acc.max_registers_per_thread))
			self.kernel_num_regs = self.acc.max_registers_per_thread
		# update shared memory size depending on the application configuration
		self.acc.update_shared_mem(self.kernel_smem_size)
		pred_out["total_num_workloads"]  = self.kernel_grid_size # the amount of blocks in the grid.
		pred_out["active_SMs"] = min(self.acc.num_SMs, pred_out["total_num_workloads"]) # if #blocks > #SMs, then all SM will be active. else active as many SMs as the #blocks
		pred_out["allocated_active_warps_per_block"] = int(ceil((float(self.kernel_block_size)/float(self.acc.warp_size)),1))

		self.logger = Logger(self.pred_out, kernel_info["log"])
		# return 0

	def kernel_call_GCoM(self, data, name, num):
		pred_out = self.pred_out
		tic = time.time()

		sass_parser = importlib.import_module("ISA_parser.sass_parser")
		self.kernel_tasklist, gmem_reqs = sass_parser.parse(units_latency = self.acc.units_latency, sass_instructions = self.acc.sass_isa,\
															sass_path = self.sass_file_path, logger = self.logger)
		toc = time.time()
		pred_out["simulation_time_parse"] = (toc - tic)
		# return -1			
		###### ---- memory performance predictions ---- ######
		tic = time.time()
		memory_stats_dict = get_memory_perf(pred_out["kernel_id"], self.mem_traces_dir_path, pred_out["total_num_workloads"], self.acc.num_SMs,\
													self.acc.l1_cache_size, self.acc.l1_cache_line_size, self.acc.l1_cache_associativity,\
													self.acc.l2_cache_size, self.acc.l2_cache_line_size, self.acc.l2_cache_associativity,\
													gmem_reqs, )
		toc = time.time()
		pred_out["simulation_time_memory"] = (toc - tic)
		'''
        new method
		get the value of AMAT in pred_out["AMAT"]
		and the value ACPAO in pred_out["ACPAO"]
		'''
		# AMAT: Average Memory Access Time (Cycles)
		if memory_stats_dict["gmem_tot_reqs"] != 0:
			l1_parallelism = self.acc.num_L1_cache_banks // self.acc.num_sub_cores # bank number per sub-core in L1 data cache
			l2_parallelism = int(memory_stats_dict["gmem_tot_diverg"])
			dram_parallelism = int(memory_stats_dict["gmem_tot_diverg"])
			# l2_parallelism = min(int(memory_stats_dict["gmem_tot_diverg"]),self.acc.num_l2_partitions)
			# dram_parallelism = min(int(memory_stats_dict["gmem_tot_diverg"]), self.acc.num_dram_channels)
			# l2_parallelism = int(memory_stats_dict["gmem_tot_diverg"]) if memory_stats_dict["gmem_tot_diverg"] < self.acc.num_l2_partitions else self.acc.num_l2_partitions
			# dram_parallelism = int(memory_stats_dict["gmem_tot_diverg"]) if memory_stats_dict["gmem_tot_diverg"] < self.acc.num_dram_channels else self.acc.num_dram_channels
			# if memory_stats_dict["gmem_ld_diverg"] >= self.acc.num_dram_channels\
			# or memory_stats_dict["gmem_st_diverg"] >= self.acc.num_dram_channels\
			# or memory_stats_dict["gmem_tot_diverg"] >= highly_divergent_degree:
			# 	l2_parallelism = memory_stats_dict["gmem_tot_diverg"] if memory_stats_dict["gmem_tot_diverg"] < self.acc.num_dram_channels else self.acc.num_dram_channels
			# 	dram_parallelism = memory_stats_dict["gmem_tot_diverg"] if memory_stats_dict["gmem_tot_diverg"] < self.acc.num_dram_channels else self.acc.num_dram_channels
				# l2_parallelism = self.num_dram_channels
				# dram_parallelism = self.num_dram_channels
				# l2_parallelism = self.num_l2_partitions
				# l2_parallelism = memory_stats_dict["gmem_tot_diverg"]
				# dram_parallelism = memory_stats_dict["gmem_tot_diverg"]
			pred_out["l1_parallelism"] = ceil(l1_parallelism, 1)
			pred_out["l2_parallelism"] = ceil(l2_parallelism, 1)
			pred_out["dram_parallelism"] = ceil(dram_parallelism, 1)
			l1_cycles_no_contention = memory_stats_dict["l1_sm_trans_gmem"] * self.acc.l1_cache_access_latency * (1/l1_parallelism)
			l2_cycles_no_contention = memory_stats_dict["l2_tot_trans_gmem"] * self.acc.l2_cache_from_l1_access_latency * (1/l2_parallelism)
			dram_cycles_no_contention = memory_stats_dict["dram_tot_trans_gmem"] * self.acc.dram_mem_from_l2_access_latency * (1/dram_parallelism)
			
			mem_cycles_no_contention = max(l1_cycles_no_contention, l2_cycles_no_contention) 
			mem_cycles_no_contention = max(mem_cycles_no_contention, dram_cycles_no_contention)
			mem_cycles_no_contention = ceil(mem_cycles_no_contention, 1)

			pred_out["l1_cycles_no_contention"] = l1_cycles_no_contention
			pred_out["l2_cycles_no_contention"] = l2_cycles_no_contention
			pred_out["dram_cycles_no_contention"] = dram_cycles_no_contention
			pred_out["mem_cycles_no_contention"] = mem_cycles_no_contention	
			tot_mem_cycles = ceil(mem_cycles_no_contention, 1)
			
			pred_out["AMAT"] = tot_mem_cycles/memory_stats_dict["gmem_tot_reqs"]
			pred_out["AMAT"] = ceil(pred_out["AMAT"], 1)

		# ACPAO: Average Cycles Per Atomic Operation
		# ACPAO = atomic operations latency / total atomic requests
		# atomic operations latency= (atomic & redcutions transactions * access latency of atomic & red requests)
		# if memory_stats_dict["atom_red_tot_trans"] != 0:
		# 	pred_out["ACPAO"] = (self.acc.atomic_op_access_latency * memory_stats_dict["atom_red_tot_trans"])\
		# 					/(memory_stats_dict["atom_tot_reqs"] + memory_stats_dict["red_tot_reqs"])
		pred_out["ACPAO"] = self.acc.atomic_op_access_latency
		pred_out.update(memory_stats_dict)
		###### ---- compute performance predictions ---- ######
		tic = time.time()
		# spawning all SMs which includes several warps
		SM_block_list = []
		total_warp_num = 0
		max_warp_per_sub_core = 0
		max_warp_per_SM = 0
		for key in self.kernel_tasklist:
			total_warp_num += len(self.kernel_tasklist[key])
			max_warp_per_SM = max(max_warp_per_SM, len(self.kernel_tasklist[key]))
			sub_core_list_in_SM = self.spawn_SM_tasklists(
				gpu=self.acc, 
				SM_id=key, 
				tasklist=self.kernel_tasklist[key], 
				kernel_id=self.kernel_id_real, 
				isa=self.ISA, 
				avg_mem_lat=pred_out["AMAT"], 
				avg_atom_lat=pred_out["ACPAO"],
			)
			for sub_core in sub_core_list_in_SM:
				max_warp_per_sub_core = max(max_warp_per_sub_core, len(sub_core.warp_list))
			SM_block_list.append(sub_core_list_in_SM)
		'''
			select the representive warps and count warp num
			use Kmeans algorithm
		'''
		kmeans_features = []
		for SM in SM_block_list:
			for sub_core_block in SM:
				for warp in sub_core_block.warp_list:
					_, total_cycles, _ = warp.interval_analyze()
					warp_total_inst = warp.current_inst
					kmeans_features.append([warp_total_inst / total_cycles, warp_total_inst])
		all_center_warp_idx_list, represetative_index = rptv_warp_select(kmeans_features)

		
		rptv_warp_GCoM_output = self.calculate_GCoM(SM_block_list, 
								total_warp_num, represetative_index,
								pred_out,
								max_warp_per_sub_core, max_warp_per_SM,)
		# calculate the simulation time
		toc = time.time()
		# fill up the pred_out values
		pred_out["simulation_time_compute"] = (toc - tic)
		pred_out.update(rptv_warp_GCoM_output)
		# calculate the ipc
		pred_out["ipc"] = pred_out["warps_instructions_executed"] / pred_out["GCoM"]
		# write output to file
		write_to_file(pred_out)
		# logging
		self.logger.write("all_center_warp_idx and represetative warp index")
		self.logger.write(all_center_warp_idx_list, represetative_index)
		self.logger.write("pred_out:")
		self.logger.write(pred_out)
		self.logger.write("rptv_warp_GCoM_output:")
		self.logger.write(rptv_warp_GCoM_output)
		# print output info		
		print_output_info(pred_out, rptv_warp_GCoM_output)

	def spawn_SM_tasklists(self, gpu, SM_id, tasklist, kernel_id, isa, avg_mem_lat, avg_atom_lat):
		'''
			return a list of Blocks to run on one SM each with allocated number of warps
		'''
		new_tasklists = []  		
		block_list = []	
		for i in range(gpu.num_warp_schedulers_per_SM):
			new_tasklists.append({})

		for key in tasklist:
			# allocate each warp to a sub-core(warp scheduler)
			new_tasklists[key % gpu.num_warp_schedulers_per_SM][key] = tasklist[key]

		for i in range(gpu.num_warp_schedulers_per_SM):
			block_list.append(Block(gpu, SM_id, i,
						   new_tasklists[i], kernel_id, isa, avg_mem_lat, avg_atom_lat))		
		return block_list
	def calculate_GCoM(self, SM_block_list:list, 
								total_warp_num:int, represetative_index:int,
								pred_out:dict,
								max_warp_per_sub_core:int, max_warp_per_SM:int,):
		# find the represetative warp based on the represetative index
		rptv_warp = None
		rptv_warp_GCoM_output = None
		tmp_idx = 0
		max_sub_core_instr = -1
		C_idle_i_orig = 0

		for cur_SM in SM_block_list:
			# count the number of warps in the SM
			cur_SM_warps_num = 0
			for tmp_block in cur_SM:
				cur_SM_warps_num += len(tmp_block.warp_list)
				
			for cur_sub_core in cur_SM:
				for cur_warp in cur_sub_core.warp_list:
					# update the max instruction number across all SMs
					inst_cnt = 0
					for tmp_warp in cur_sub_core.warp_list:
						inst_cnt += len(tmp_warp.tasklist)					
					max_sub_core_instr = max(max_sub_core_instr, inst_cnt)

					if tmp_idx == represetative_index:
						rptv_warp = cur_warp
						# get the rptv_warp and process GCoM
						sub_core_warps_num = len(cur_sub_core.warp_list)
						interval_analysis_result = rptv_warp.interval_analyze()
						pred_out["warps_instructions_executed"] = rptv_warp.current_inst * total_warp_num # used in calculating ipc

						self.logger.write("profiling rtpv warp")
						self.logger.write("args:",
							"sub_core_warps_num:",sub_core_warps_num, 
							"cur_SM_warps_num:",cur_SM_warps_num, 
							"pred_out['active_SMs']:",pred_out["active_SMs"],
							"pred_out['umem_hit_rate']:",pred_out["umem_hit_rate"])
							
						rptv_warp_GCoM_output = self.process_GCoM(rptv_warp, interval_analysis_result,
										sub_core_warps_num, cur_SM_warps_num, pred_out["active_SMs"],
										pred_out["umem_hit_rate"])
						# update the idle ij cycles
						if max_warp_per_sub_core > sub_core_warps_num:
							self.update_c_idle_ij(cur_SM, rptv_warp, sub_core_warps_num, interval_analysis_result, rptv_warp_GCoM_output,
								pred_out["active_SMs"], cur_SM_warps_num, pred_out["umem_hit_rate"])	
						# else:
						# 	rptv_warp_GCoM_output["C_idle_ij_ID"] = 0
						'''
							calculate the candidate max cycles warp in the whole SM
							this is to calculate the max cycles depending on max warp number in a SM
						'''
						if max_warp_per_SM > cur_SM_warps_num:
							
							max_GCoM_by_warp_num = self.process_GCoM(rptv_warp, interval_analysis_result,
											max_warp_per_SM // self.acc.num_warp_schedulers_per_SM, max_warp_per_SM, 
											pred_out["active_SMs"], pred_out["umem_hit_rate"])
							
							self.logger.write("profiling rtpv warp based on max warp number in the SM")
							self.logger.write(max_GCoM_by_warp_num)

							if max_GCoM_by_warp_num["GCoM"] > rptv_warp_GCoM_output["GCoM"]:
								C_idle_i_orig = max_GCoM_by_warp_num["GCoM"] - rptv_warp_GCoM_output["GCoM"]
						rptv_warp_GCoM_output["C_idle_i"] = C_idle_i_orig
						rptv_warp_GCoM_output["GCoM"] += rptv_warp_GCoM_output["C_idle_i"]

					tmp_idx += 1

		return rptv_warp_GCoM_output
		
	def process_GCoM(self, warp: Warp, 
				  interval_analysis_result: Tuple, 
				  warps_per_sub_core: int, warps_per_SM: int,
				  active_SMs: int, umem_hit_rate: float,):
		'''
			this function is used to calculate the GCoM of a Kernel
			
			Args:
				warp: the warp to calculate the GCoM
				interval_analysis_result: the result of interval analysis
				warps_per_sub_core: the number of warps per sub-core
				warps_per_SM: the number of warps per SM
				active_SMs: the number of active SMs
				umem_hit_rate: the hit rate of the memory
			Returns:
				general_output_GCoM: the GCoM of the warp
		'''
		'''
			model sub-core
			1. the number of warps per sub-core
			2. the number of sub-cores per SM
			warps of sub-core = warps of SM / sub-cores per SM
		'''
		interval_list, total_cycles, total_intervals = interval_analysis_result
		# initial variables
		num_sub_cores_per_SM = self.acc.num_warp_schedulers_per_SM
		warps_ij = warps_per_sub_core # Warpsi,j is the number of warps the j-th sub-core of the i-th SM executes
		issue_rate = 1 # 1 instruction is issued per cycle
		num_warp_inst = warp.current_inst
		total_num_warp_inst = num_warp_inst * warps_ij	
		# calculate C_base_ij
		C_base_ij = total_num_warp_inst / issue_rate 
		actual_end = max(warp.completions)		
		C_base_ij += actual_end - total_cycles
		# initialize S_ComData_ij and S_MemData_ij
		S_ComData_ij = 0
		S_MemData_ij = 0
		# according to GPUMech, the probability of issuing an instruction
		P_inst = num_warp_inst / total_cycles
		# calculate S_ComData_ij and S_MemData_ij
		if self.acc.warp_scheduling_policy == "GTO":		
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
					else:
						S_MemData_ij += max(int(S_intv_k - cycle_other), 0)
		elif self.acc.warp_scheduling_policy == "LRR":  
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
		else:
			print("Error: unsupported warp scheduling policy")
			exit(1)
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
			GCoM claim: num_cncr_warps is the maximum number of warps that an SM can concurrently execute

			The maximum number of concurrent warps per SM is 32 on Turing (versus 64 on Volta)
			The maximum number of concurrent warps per SM remains the same as in Volta (i.e., 64) 
			for compute capability 8.0, while for compute capability 8.6 it is 48.

			see Nvidia doc https://docs.nvidia.com/cuda/turing-tuning-guide/index.html
			see Nvidia doc https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html
		'''
		num_cncr_warps = self.acc.max_active_warps_per_SM
		# byte/cycle/SM test on ./l1_bw_32f microbenchmark in accel-sim
		B_L1_k = self.acc.l1_cache_bandwidth

		def issue_base(x, ik):
			'''
				return the base cycle to execute all instruction of the issue stage.

				Args:
					x: number of warps execute in current issue stage
					ik: number of instruction of current issue stage
				Returns:
					issue_base: the issue_base of a SM
			'''
			if x == 0:
				return 0
			num_act_sub_core = min(x, num_sub_cores_per_SM)
			return ik * x / (num_act_sub_core * issue_rate)

		def issue_k_m(x,functional_units, ik):
			'''
				return the stall cycle of SM caused by limited funcitonal units.

				Args:
					x: number of warps execute in current issue stage on the SM
					functional_units: a dictionary of functional units which is used in current issue stage
				Returns:
					issue_km: the issue_km of a SM
			'''
			issue_km = -100
			max_unit = None
			num_act_sub_core = min(x, num_sub_cores_per_SM)
			for unit in functional_units:	
				iim = self.acc.initial_interval[unit]
				im = functional_units[unit] # num of instuction which used the unit
				cur_issue_km = im * iim * x / (num_act_sub_core * issue_rate)
				if cur_issue_km > issue_km:
					max_unit = unit

				issue_km = max(issue_km, cur_issue_km)
			return issue_km, max_unit			

		def issue_max(x, ik, fu, bk):
			'''
				return the max cycle of stall for multi warp concurrently execute.

				Args:
					x: number of warps execute in current issue stage
					ik: number of instruction of current issue stage
					fu: list of functional units which is used in current issue stage
					bk: number of L1 cache access in current issue stage
				Returns:
					result: the max cycle of stall for multi warp concurrently execute
					is_mem: whether the stall is caused by L1 cache access
			'''
			if x == 0:
				return 0, 0, None
			issue_base_var = issue_base(x, ik)
			issue_k_m_var, unit = issue_k_m(x, fu, ik)
			num_act_sub_core = min(x, num_sub_cores_per_SM)
			# one memory instruction like LDSM.16.M88.4 R72, [R51+UR8+0x800] will load data to the R72 register file 
			# to every thread which leads to 32bit * 32 / 8 = 128B L1 cache load
			issue_k_L1_var = (bk * 4 * self.acc.l1_cache_line_size / B_L1_k) * x / (num_act_sub_core * issue_rate)
			# issue_k_L1_var = bk * kernel.acc.l1_cache_access_latency * x / (num_act_sub_core * issue_rate)
			issue_k_L1_var = ceil(issue_k_L1_var, 1)
			result = max(issue_base_var, issue_k_m_var, issue_k_L1_var)
			is_mem = issue_k_L1_var > issue_k_m_var and issue_k_L1_var > issue_base_var
			return result, is_mem, unit 
		
		def com_struct_and_mem_struct(x):
			com, mem = 0, 0
			all_struct_info = {}
			for interval_info in interval_list:
				if "issue_stage" not in interval_info:
					continue
				ik = interval_info["issue_stage"]
				fu = interval_info["functional_units"] if "functional_units" in interval_info else {}
				mem_acesss_num = interval_info["mem_access_num"]
				issue_max_var, is_mem, unit = issue_max(x, ik, fu, mem_acesss_num)
				issue_base_var = issue_base(x, ik)
				if is_mem:
					mem += issue_max_var - issue_base_var
					interval_info["struct_info"] = {"type:":"L1 cache", "value": issue_max_var, "base": issue_base_var}
					if "L1 cache" not in all_struct_info:
						all_struct_info["L1 cache"] = issue_max_var - issue_base_var
					else:
						all_struct_info["L1 cache"] += issue_max_var - issue_base_var
				else:
					com += issue_max_var - issue_base_var
					interval_info["struct_info"] = {"type:":unit, "value": issue_max_var, "base": issue_base_var}
					if unit not in all_struct_info:
						all_struct_info[unit] = issue_max_var - issue_base_var
					else:
						all_struct_info[unit] += issue_max_var - issue_base_var
			return com, mem, all_struct_info

		com1, mem1, all_struct_info1 = com_struct_and_mem_struct(num_cncr_warps)
		com2, mem2, all_struct_info2 = com_struct_and_mem_struct(warps_per_SM % num_cncr_warps)
		S_ComStruct_i = com1 * (warps_per_SM // num_cncr_warps) + com2
		S_MemStruct_i = mem1 * (warps_per_SM // num_cncr_warps) + mem2
		if warps_per_SM // num_cncr_warps > 0:
			self.logger.write(all_struct_info1)
		if warps_per_SM % num_cncr_warps > 0:
			self.logger.write(all_struct_info2)
			
		
		MDM_output = self.process_MDM(interval_list, active_SMs, umem_hit_rate, warps_per_SM)
		
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
			"C_idle_ij": C_idle_ij,
			"S_ComStruct_i": S_ComStruct_i,
			"S_MemStruct_i": S_MemStruct_i,
			"S_MSHR_i": S_MSHR_i,
			"S_NoC_i": S_NoC_i,
			"S_Dram_i": S_Dram_i,			
			"C_idle_i": C_idle_i,
		}

		self.logger.write("-----------")
		self.logger.write("total_cycles:",total_cycles)
		self.logger.write("-----------")
		self.logger.write("num_warp_inst:",num_warp_inst)
		self.logger.write("-----------")
		instr_idx = 0
		for interval in interval_list:
			if "issue_stage" in interval:
				self.logger.write(instr_idx, interval, end=' ')
				instr_idx += interval["issue_stage"]
			else:
				self.logger.write(interval)
		self.logger.write("------------")

		return general_GCoM_output

	def update_c_idle_ij(self, rptv_SM, rptv_warp: Warp, rptv_sub_core_warps_num: int, interval_analysis_result, rptv_output: dict, active_SMs: int, SM_warps_num: int, umem_hit_rate: float,):
		'''
			update the c idle ij of the representative warp.

			Args:
				rptv_SM: the SM block of the representative warp
				rptv_warp: Warp, representative warp
				interval_analysis_result: interval analysis results of the representative warp
				rptv_output: dict, GCoM output of the representative warp
				active_SMs: int, the number of active SMs
				umem_hit_rate: the hit rate of the representative warp

			Return:
				None, but update the c idle ij of the representative warp
		'''
		max_cycles_in_sub_core = -1
		max_sub_core_warp_num = rptv_sub_core_warps_num # initialize as the warp number of the warps on the sub-core which is executing the rptv warp
		max_inst_cnt = -1
		for cur_sub_core in rptv_SM:
			# calculate the number of instructions of the warps on the sub-core which is executing the rptv warp
			inst_cnt = 0
			for tmp_warp in cur_sub_core.warp_list:
				inst_cnt += len(tmp_warp.tasklist)	
			max_inst_cnt = max(max_inst_cnt, inst_cnt)
			# if sub_core_warps_num > max_sub_core_warp_num, calculate the candidate max sub-core cycles
			sub_core_warps_num = len(cur_sub_core.warp_list) 
			if sub_core_warps_num > max_sub_core_warp_num:
				max_sub_core_warp_num = sub_core_warps_num
				self.logger.write("profiling warp based on max sub_core warp number")
				tmp_warp_GCoM_output = self.process_GCoM(rptv_warp, interval_analysis_result, 
												sub_core_warps_num, SM_warps_num, active_SMs, umem_hit_rate)
				if tmp_warp_GCoM_output["GCoM"] > max_cycles_in_sub_core:
					max_cycles_in_sub_core = tmp_warp_GCoM_output["GCoM"]

		if max_cycles_in_sub_core > rptv_output["GCoM"]:
			rptv_output["C_idle_ij"] += max_cycles_in_sub_core - rptv_output["GCoM"]
			rptv_output["GCoM"] += rptv_output["C_idle_ij"]

	def process_MDM(self, interval_analysis, num_SM: int, umem_hit_rate: float, W: int):
		"""
			This function is used to process the MDM.

			Args:
				interval_analysis: result of interval analysis
				active_SMs: the number of active SMs
				umem_hit_rate: the hit rate of the representative warp
				th_active_warps: the number of warps concurrently executed on an SM
			
			Returns:
				result: a dictionary of MSHR, NoC, and Dram latency
		"""
		result = {"MSHR":0, "NoC":0, "Dram": 0}					
		num_MSHR = 4096 
		for interval_info in interval_analysis:
			if "issue_stage" not in interval_info:
				continue
			M_access = interval_info["mem_access_num"]
			M_read = interval_info["mem_read_num"]
			M_write = M_access - M_read
			'''
				MDM models the lockup-free characteristics of the streaming L1 D$s 
				by assuming a large number of MSHR entries (e.g., 4096 in [ 50 ]). 
				GCoM follows MDM's approach and models the lockup-free characteristics 
				by assuming a sufficient number of MSHR entries per L1 D$.
				However in accel-sim they claim that num_mshr is 256
			'''
			is_MD = M_read * W > num_MSHR
			M = min(M_read * W, num_MSHR) + M_write * W

			L_Min_LLC= self.acc.l1_cache_access_latency
			LLC_Miss_Rate = 1 - umem_hit_rate
			L_Min_Dram = self.acc.dram_mem_access_latency
			# L_no_contention = L_Min_LLC + LLC_Miss_Rate * L_Min_Dram
			#S_MSHR = (M_read * W // num_MSHR - 1) * S_Mem if is_MD else 0
			S_MSHR = 0

			L_noc_service = self.acc.dram_clockspeed * (self.acc.l1_cache_line_size / self.acc.noc_bandwidth)
			L_dram_service = self.acc.dram_clockspeed * (self.acc.l2_cache_line_size / self.acc.dram_bandwidth) * LLC_Miss_Rate
			is_NoC_saturated = L_noc_service * M * num_SM > L_Min_LLC + L_Min_Dram
			is_Dram_saturated = L_dram_service * M * num_SM > L_Min_LLC + L_Min_Dram

			S_NoC = num_SM * M * L_noc_service if is_NoC_saturated and is_MD else num_SM * M * L_noc_service * 0.5
			S_Dram = num_SM * M * L_dram_service if is_Dram_saturated and is_MD else num_SM * M * L_dram_service * 0.5
			# S_Mem = L_no_contention + S_NoC + S_Dram
			result["MSHR"] += S_MSHR
			result["NoC"] += S_NoC
			result["Dram"] += S_Dram

		return result