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

# from .helper_methods import *
from .memory_model import *
from .utils import print_output_info, sm_id_str_to_int, write_to_file, Logger, print_warning, ceil, floor
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
		self.sass_file_path = kernel_info["sass_file_path"]
		self.external_rptv_warp_selector = kernel_info["external_rptv_warp_selector"]
		## kernel local predictions outputs
		self.pred_out = {}	
		self.Idle_cycle_method = "AMALi" # GCoM or AMALi
		self.Tensor_core_ii_scale_factor = self.acc.num_TC_units_per_SM // self.acc.num_warp_schedulers_per_SM # this is to handle that there are multiple TCUs in one warp scheduler, for example in Volta and Turing, for newer architectures, this value may be 1
		
		pred_out = self.pred_out
		pred_out["app_path"] = kernel_info["app_path"]
		pred_out["app_name"] = kernel_info["app_name"]
		pred_out["kernel_id"] = self.kernel_id 
		pred_out["kernel_name"] = self.kernel_name	
		pred_out["max_active_blocks_per_SM"] = self.acc.max_active_blocks_per_SM
		pred_out["allocated_active_warps_per_block"] = 0
		pred_out["warps_instructions_executed"] = 0
		pred_out["cpi"] = 0.0
		pred_out["kernel_launch_intercept"] = self.acc.kernel_launch_overhead
		pred_out["AMAT"] = 0.0
		pred_out["ACPAO"] = 0.0
		pred_out["l1_parallelism"] = 0
		pred_out["l2_parallelism"] = 0
		pred_out["dram_parallelism"] = 0
		pred_out["simulation_time_parse"] = 0.0
		pred_out["simulation_time_memory"] = 0.0
		pred_out["simulation_time_compute"] = 0.0
		pred_out["grid_size"]  = self.kernel_grid_size # the amount of blocks in the grid.	
		pred_out["active_SMs"] = 0 # will be updated later in kernel_call_GCoM

		self.logger = Logger(self.pred_out, kernel_info["log"])
		if self.kernel_block_size > self.acc.max_block_size:
			print_warning("block_size",str(self.acc.max_block_size))
			self.kernel_block_size = self.acc.max_block_size

		if self.kernel_num_regs > self.acc.max_registers_per_thread:
			print_warning("num_registers",str(self.acc.max_registers_per_thread))
			self.kernel_num_regs = self.acc.max_registers_per_thread
		pred_out["allocated_active_warps_per_block"] = int(ceil((float(self.kernel_block_size)/float(self.acc.warp_size)),1))
		# calculate the maximum active blocks per SM based on the warps, registers and shared memory
		pred_out["blocks_per_SM_limit_warps"] = int(min(pred_out["max_active_blocks_per_SM"],\
				int(floor((self.acc.max_active_warps_per_SM/pred_out["allocated_active_warps_per_block"]),1))))

		if self.kernel_num_regs == 0: pred_out["blocks_per_SM_limit_regs"] = pred_out["max_active_blocks_per_SM"]
		else:
			allocated_regs_per_warp = ceil((self.kernel_num_regs*self.acc.warp_size),self.acc.register_allocation_size)
			allocated_regs_per_SM = int(floor((self.acc.max_registers_per_block/allocated_regs_per_warp),\
				self.acc.num_warp_schedulers_per_SM))
			pred_out["blocks_per_SM_limit_regs"] = int(floor((allocated_regs_per_SM/pred_out\
				["allocated_active_warps_per_block"]),1) * floor((self.acc.max_registers_per_SM/\
				self.acc.max_registers_per_block),1))

		if self.kernel_smem_size == 0:
			pred_out["blocks_per_SM_limit_smem"] = pred_out["max_active_blocks_per_SM"]
		else:
			smem_per_block = ceil(self.kernel_smem_size, self.acc.smem_allocation_size)
			pred_out["blocks_per_SM_limit_smem"] = int(floor((self.acc.shared_mem_config_list[-1]/smem_per_block),1))
		
		pred_out["allocated_active_blocks_per_SM"] = min(pred_out["blocks_per_SM_limit_warps"],\
													pred_out["blocks_per_SM_limit_regs"],\
													pred_out["blocks_per_SM_limit_smem"])		
		
		# update shared memory size depending on the application configuration
		self.acc.update_shared_mem(ceil(self.kernel_smem_size, self.acc.smem_allocation_size) * pred_out["allocated_active_blocks_per_SM"])

		# calculate kernel launch latency
		slope = self.acc.slope_alpha * pred_out["allocated_active_warps_per_block"] ** 2 \
			- self.acc.slope_beta * pred_out["allocated_active_warps_per_block"] + self.acc.slope_gamma
		self.kernel_launch_latency = ceil(slope * pred_out["grid_size"] + pred_out["kernel_launch_intercept"],1) if pred_out["grid_size"] < 2048 else 0 # if the grid size is too large, set as 0 for out of distribution

	def kernel_call_GCoM(self, data, name, num):
		pred_out = self.pred_out
		tic = time.time()

		sass_parser = importlib.import_module("ISA_parser.sass_parser")
		self.kernel_tasklist, gmem_reqs, represetative_sm_warp_pair, total_warp_num, pred_out["active_SMs"], unbanlance_sms, max_sub_core_instr_from_traces = sass_parser.parse(units_latency = self.acc.units_latency, sass_instructions = self.acc.sass_isa,\
															sass_path = self.sass_file_path, logger = self.logger, external_rptv_warp_selector = self.external_rptv_warp_selector)
		toc = time.time()
		pred_out["simulation_time_parse"] = (toc - tic)
		# return -1			
		# print("memory args:", gmem_reqs, self.acc.l1_cache_size, self.acc.l1_cache_line_size, self.acc.l1_cache_associativity,)
		###### ---- memory performance predictions ---- ######
		tic = time.time()
		memory_stats_dict = get_memory_perf(pred_out["kernel_id"], self.mem_traces_dir_path, pred_out["grid_size"], self.acc.num_SMs,\
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
			l2_parallelism = max(int(memory_stats_dict["gmem_tot_diverg"]),1)
			dram_parallelism = max(int(memory_stats_dict["gmem_tot_diverg"]),1)
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
		rptv_warp_GCoM_output = self.calculate_GCoM(represetative_sm_warp_pair, total_warp_num, pred_out, unbanlance_sms, max_sub_core_instr_from_traces)
		# calculate the simulation time
		toc = time.time()
		# fill up the pred_out values
		pred_out["simulation_time_compute"] = (toc - tic)
		pred_out.update(rptv_warp_GCoM_output)
		# calculate the cpi
		pred_out["cpi"] =   pred_out["AMALi"] / pred_out["warps_instructions_executed"]
		# write output to file
		write_to_file(pred_out)
		# logging
		self.logger.write("pred_out:")
		self.logger.write(pred_out)
		self.logger.write("rptv_warp_GCoM_output:")
		self.logger.write(rptv_warp_GCoM_output)
		# print output info		
		print_output_info(pred_out, rptv_warp_GCoM_output)
	
	def calculate_GCoM(self, represetative_sm_warp_pair:tuple, total_warp_num:int, pred_out:dict,unbanlance_sms:list, max_sub_core_instr_from_traces:int):
		# total_warp_num = 0
		# # scan all CTA and Count warp number in all SM and sub-cores
		# warp_num_count = []
		# warp_instr_num_count = []
		# active_SMs_set = set()
		# for _ in range(self.acc.num_SMs):
		# 	warp_num_count.append([0] * self.acc.num_warp_schedulers_per_SM)
		# 	warp_instr_num_count.append([0] * self.acc.num_warp_schedulers_per_SM)
		# for CTA_id in self.kernel_tasklist:
		# 	for warp_id in self.kernel_tasklist[CTA_id]:
		# 		sm_id = sm_id_str_to_int(CTA_id)
		# 		active_SMs_set.add(sm_id)
		# 		warp_num_count[sm_id][warp_id % self.acc.num_warp_schedulers_per_SM] += 1
		# 		warp_instr_num_count[sm_id][warp_id % self.acc.num_warp_schedulers_per_SM] += len(self.kernel_tasklist[CTA_id][warp_id])
		# 	total_warp_num += len(self.kernel_tasklist[CTA_id])
		# pred_out["active_SMs"] = len(active_SMs_set)
		# # print warp num and instr num distribution across SMs and sub-cores
		# self.logger.write("### Warp number and intr number distribution ###")
		# for sm_id in range(self.acc.num_SMs):
		# 	self.logger.write("SM {:d} warp number:".format(sm_id), warp_num_count[sm_id])
		# 	self.logger.write("SM {:d} warp instruction number:".format(sm_id), warp_instr_num_count[sm_id])
		# self.logger.write("### Warp number and intr number distribution ###")
		# find the represetative warp based on the represetative index
		rptv_sm_hashtag_CTA_id, rptv_warp_id = represetative_sm_warp_pair[0], represetative_sm_warp_pair[1]
		rptv_SM_id = sm_id_str_to_int(rptv_sm_hashtag_CTA_id)		
		# Calculate mean warp per SM
		mean_CTA_per_SM = pred_out["grid_size"] // pred_out["active_SMs"]
		max_CTA_per_SM = mean_CTA_per_SM if pred_out["grid_size"] % pred_out["active_SMs"] == 0 else mean_CTA_per_SM + 1

		mean_warp_per_SM = mean_CTA_per_SM * pred_out["allocated_active_warps_per_block"]
		max_warp_per_SM = max_CTA_per_SM * pred_out["allocated_active_warps_per_block"]
		mean_warp_per_sub_core = ceil(mean_warp_per_SM / self.acc.num_warp_schedulers_per_SM, 1)
		max_warp_per_sub_core = mean_warp_per_sub_core if mean_warp_per_SM % self.acc.num_warp_schedulers_per_SM == 0 else mean_warp_per_sub_core + 1
		self.logger.write("Mean warp per SM:", mean_warp_per_SM)
		self.logger.write("Max warp per SM:", max_warp_per_SM)
		self.logger.write("Mean warp per sub-core:", mean_warp_per_sub_core)
		self.logger.write("Max warp per sub-core:", max_warp_per_sub_core)
		# Find closest SM to mean
		# closest_sm_to_mean = 0
		# min_diff = float('inf')
		# for sm_id in range(self.acc.num_SMs):
		# 	sm_warp_count = sum(warp_num_count[sm_id])
		# 	if sm_warp_count == 0:
		# 		continue
		# 	diff = abs(sm_warp_count - mean_warp_per_SM)
		# 	if diff < min_diff:
		# 		min_diff = diff
		# 		closest_sm_to_mean = sm_id
		# self.logger.write("Closest SM to mean:", closest_sm_to_mean, "with warp count:", sum(warp_num_count[closest_sm_to_mean]))
		# mean_warp_per_SM = sum(warp_num_count[closest_sm_to_mean])
		# mean_warp_per_sub_core = ceil(mean_warp_per_SM / self.acc.num_warp_schedulers_per_SM, 1)

		# find the max warp number in each SM and sub-core
		# max_warp_per_SM = max([sum(warp_num_count[sm_id]) for sm_id in range(self.acc.num_SMs)])
		# max_warp_per_sub_core = max(warp_num_count[closest_sm_to_mean]) # max warp number in the sub-core of the closest SM to mean
		# max_instr_num_SM = max([sum(warp_instr_num_count[sm_id]) for sm_id in range(self.acc.num_SMs)])
		# max_instr_num_sub_core = max([max(warp_instr_num_count[sm_id]) for sm_id in range(self.acc.num_SMs)])
		# initialize the representative warp
		rptv_warp = Warp(0, self.acc, self.kernel_tasklist, 
				   self.kernel_id_real, rptv_SM_id , rptv_warp_id, pred_out["AMAT"], pred_out["ACPAO"])
		rptv_warp_GCoM_output = None
		# rptv_SM_warps_num = sum(warp_num_count[rptv_SM_id])
		# rptv_sub_core_warps_num = warp_num_count[rptv_SM_id][rptv_warp_id % self.acc.num_warp_schedulers_per_SM]
		# rptv_sub_core_instr_num = warp_instr_num_count[rptv_SM_id][rptv_warp_id % self.acc.num_warp_schedulers_per_SM]
		# run interval analysis on the represetative warp
		interval_analysis_result = rptv_warp.interval_analyze()
		pred_out["warps_instructions_executed"] = rptv_warp.current_inst * total_warp_num # used in calculating cpi
		max_instr_sub_core = rptv_warp.current_inst * max_warp_per_SM // self.acc.num_warp_schedulers_per_SM if unbanlance_sms else max_sub_core_instr_from_traces
		gcom_arg_CTA = min(mean_CTA_per_SM, pred_out["allocated_active_blocks_per_SM"])
		gcom_arg_SM_warp_num = gcom_arg_CTA * pred_out["allocated_active_warps_per_block"]
		gcom_arg_sub_core_warp_num = max(gcom_arg_SM_warp_num // self.acc.num_warp_schedulers_per_SM, 1)
		
		# logging
		self.logger.write("profiling rtpv warp, number of instructions:",len(rptv_warp.tasklist))
		self.logger.write("args:",
			"CTA_per_SM:", gcom_arg_CTA,
			"concurrent_warps_per_sub_core:", gcom_arg_sub_core_warp_num, 
			"concurrent_warps_per_SM:", gcom_arg_SM_warp_num, 
		)
		rptv_warp_GCoM_output = self.process_GCoM(rptv_warp, interval_analysis_result,
													gcom_arg_sub_core_warp_num, 
													gcom_arg_SM_warp_num, 
													pred_out["active_SMs"],
													pred_out["umem_hit_rate"])
		print(rptv_warp_GCoM_output)
		self.logger.write("rptv_warp_GCoM_output:", rptv_warp_GCoM_output)
		# idle_ij_output = None
		# idle_i_output = None
		# if self.Idle_cycle_method == "GCoM":
		# 	# update the idle ij cycles based on the max warp number in a sub-core of rptv SM
		# 	if max_warp_per_sub_core > mean_warp_per_sub_core:
		# 		self.logger.write("profiling rtpv warp based on max warp number in the sub-core:", max_warp_per_sub_core)
		# 		idle_ij_repeat_times = ceil(max_warp_per_sub_core // (pred_out["concurrent_warps_per_SM"] // self.acc.num_warp_schedulers_per_SM), 1)
		# 		idle_ij_output = self.output_scaler(rptv_warp_GCoM_output, idle_ij_repeat_times)
		# 	# update the idle i cycles based on the max warp number in a SM
		# 	if max_warp_per_SM > mean_warp_per_SM:
		# 		if pred_out["concurrent_warps_per_SM"] >= max_warp_per_SM:
		# 			self.logger.write("Warning: the max warp number in a SM is smaller than the concurrent warps per SM. Processing based on max warp number in a SM:", max_warp_per_SM)
		# 			idle_i_output = self.process_GCoM(rptv_warp, interval_analysis_result,
		# 											max_warp_per_sub_core, 
		# 											max_warp_per_SM, 
		# 											pred_out["active_SMs"],
		# 											pred_out["umem_hit_rate"])
		# 		else: # pred_out["concurrent_warps_per_SM"] < max_warp_per_SM:
		# 			self.logger.write("profiling rtpv warp based on max warp number in the SM:", max_warp_per_SM)
		# 			idle_i_repeat_times = ceil(max_warp_per_SM // pred_out["concurrent_warps_per_SM"], 1)
		# 			idle_i_output = self.output_scaler(rptv_warp_GCoM_output, idle_i_repeat_times)
		# 			if max_warp_per_SM % pred_out["concurrent_warps_per_SM"] != 0:
		# 				self.logger.write("profiling rtpv warp based on the remaining warp number in the SM:", max_warp_per_SM % pred_out["concurrent_warps_per_SM"])
		# 				less_CTA_output = self.process_GCoM(rptv_warp, interval_analysis_result,
		# 												ceil(max_warp_per_SM % pred_out["concurrent_warps_per_SM"] / self.acc.num_warp_schedulers_per_SM,1), 
		# 												max_warp_per_SM % pred_out["concurrent_warps_per_SM"], 
		# 												pred_out["active_SMs"],
		# 												pred_out["umem_hit_rate"])
		# 				idle_i_output = {key: idle_i_output[key] + less_CTA_output[key] for key in idle_i_output}
		
		repeat_times = ceil(mean_CTA_per_SM // gcom_arg_CTA, 1)
		if repeat_times > 1:
			rptv_warp_GCoM_output = self.output_scaler(rptv_warp_GCoM_output, repeat_times)
		# schedule less CTA for rptv warp
		if mean_CTA_per_SM % gcom_arg_CTA != 0:
			less_CTA_num = mean_CTA_per_SM % gcom_arg_CTA
			self.logger.write("profiling rtpv warp based on the remaining CTA on SM:", less_CTA_num)
			less_CTA_output = self.process_GCoM(rptv_warp, interval_analysis_result,
											ceil(less_CTA_num * pred_out["allocated_active_warps_per_block"] / self.acc.num_warp_schedulers_per_SM,1), 
											less_CTA_num * pred_out["allocated_active_warps_per_block"], 
											pred_out["active_SMs"],
											pred_out["umem_hit_rate"])
			rptv_warp_GCoM_output = {key: rptv_warp_GCoM_output[key] + less_CTA_output[key] for key in rptv_warp_GCoM_output}
			rptv_warp_GCoM_output["GCoM+TCM"] -= less_CTA_output["drain"] 
			rptv_warp_GCoM_output["drain"] -= less_CTA_output["drain"]
		# add idle cycles
		# if self.Idle_cycle_method == "GCoM":
		# 	if idle_i_output is not None:
		# 		rptv_warp_GCoM_output["C_idle_i_orig"] = idle_i_output["GCoM+TCM"] - rptv_warp_GCoM_output["GCoM+TCM"]
		# 	if idle_ij_output is not None:
		# 		rptv_warp_GCoM_output["C_idle_ij_orig"] = idle_ij_output["GCoM+TCM"] - rptv_warp_GCoM_output["GCoM+TCM"]
			# rptv_warp_GCoM_output["GCoM+TCM"] += rptv_warp_GCoM_output["C_idle_i_orig"]
			# rptv_warp_GCoM_output["GCoM+TCM"] += rptv_warp_GCoM_output["C_idle_ij_orig"]

		if self.Idle_cycle_method == "AMALi":		
			if max_instr_sub_core > rptv_warp_GCoM_output["selected"]:
				rptv_warp_GCoM_output["C_idle_i_ID"] = max_instr_sub_core - rptv_warp_GCoM_output["selected"]
			else:
				rptv_warp_GCoM_output["C_idle_i_ID"] = 0
			# rptv_warp_GCoM_output["GCoM+ID"] = rptv_warp_GCoM_output["GCoM+TCM"] + rptv_warp_GCoM_output["C_idle_i_ID"] # + rptv_warp_GCoM_output["C_idle_ij_ID"]
		# add the kernel launch overhead
		rptv_warp_GCoM_output["no_instructions_and_imc_miss"] = self.kernel_launch_latency
		rptv_warp_GCoM_output["GCoM+TCM+KLL"] = rptv_warp_GCoM_output["GCoM+TCM"] + rptv_warp_GCoM_output["no_instructions_and_imc_miss"]
		rptv_warp_GCoM_output["AMALi"] = rptv_warp_GCoM_output["GCoM+TCM+KLL"] + rptv_warp_GCoM_output["C_idle_i_ID"]
		# + rptv_warp_GCoM_output["C_idle_i_orig"] + rptv_warp_GCoM_output["C_idle_ij_orig"]
		return rptv_warp_GCoM_output
		
	def process_GCoM(self, warp: Warp, 
				  interval_analysis_result: Tuple, # interval_list, total_cycles, total_intervals
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
				
				selected: Warp was selected by the micro scheduler and issued an instruction.
				wait: Warp was stalled waiting on a fixed latency execution dependency.
				math_pipe_throttle: Warp was stalled waiting for the execution pipe to be available.
				long_scoreboard: Warp was stalled waiting for a scoreboard dependency on a L1TEX (local, global, surface, texture) operation.
				short_scoreboard: Warp was stalled waiting for a scoreboard dependency on a lg (memory input/output) operation (not to L1TEX). Share memory
				drain: Warp was stalled after EXIT waiting for all outstanding memory operations to complete so that warp’s resources can be freed.
				tex_throttle: Warp was stalled waiting for the L1 instruction queue for texture operations to be not full.(e.g. XU/MUFU, ADU) 
				lg_throttle: Warp was stalled waiting for the L1 instruction queue for local and global (LG) memory operations to be not full.			
		'''
		'''
			model sub-core
			1. the number of warps per sub-core
			2. the number of sub-cores per SM
			warps of sub-core = warps of SM / sub-cores per SM
		'''
		interval_list, total_cycles, _ = interval_analysis_result
		# initial variables
		num_sub_cores_per_SM = self.acc.num_warp_schedulers_per_SM
		warps_ij = warps_per_sub_core # Warpsi,j is the number of warps the j-th sub-core of the i-th SM executes
		issue_rate = 1 # 1 instruction is issued per cycle
		num_warp_inst = warp.current_inst
		total_num_warp_inst = num_warp_inst * warps_ij	
		# calculate selected
		selected = total_num_warp_inst / issue_rate 
		actual_end = max(warp.completions)		
		drain = actual_end - total_cycles
		# initialize wait and long_scoreboard
		wait = 0
		long_scoreboard = 0
		short_scoreboard = 0
		# calculate wait, long_scoreboard and short_scoreboard
		for stage_info in interval_list:
			if "stall_stage" in stage_info:
				if stage_info["stall_type"] == 2:
					wait += stage_info["stall_stage"]
				else:
					if stage_info["stall_type"] == 1:
						long_scoreboard += stage_info["stall_stage"]
					else:
						short_scoreboard += stage_info["stall_stage"]
		# calculate C_active_ij and C_idle_ij
		C_active_ij = selected + wait + long_scoreboard + short_scoreboard + drain
		C_idle_ij = 0 # we will calculate it later in kernel
		C_ij = C_active_ij + C_idle_ij

		'''
			Modeling the Cycles of a Core
		'''
		Si = 0
		math_pipe_throttle = 0
		tex_throttle = 0
		'''		
			GCoM claim: num_cncr_warps is the maximum number of warps that an SM can concurrently execute

			The maximum number of concurrent warps per SM is 32 on Turing (versus 64 on Volta)
			The maximum number of concurrent warps per SM remains the same as in Volta (i.e., 64) 
			for compute capability 8.0, while for compute capability 8.6 it is 48.

			see Nvidia doc https://docs.nvidia.com/cuda/turing-tuning-guide/index.html
			see Nvidia doc https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html
		'''
		# num_cncr_warps = self.acc.max_active_warps_per_SM
		num_cncr_warps = warps_per_SM
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
			issue_km, max_unit = -100, None
			num_act_sub_core = min(x, num_sub_cores_per_SM)
			# We handle TCU specially, because TCU may have different initial interval with other functional units, for example TCU32.0 TCU16.0 TCU8.0 and so on and we need accumulate them together. 
			# For units in functional_units dictionary, they are already accmulated, so every unit will occur once, except TCU 
			tensor_core_iim = 0
			for unit in functional_units:				
				if "TCU" in unit:
					iim = float(unit[3:]) / self.Tensor_core_ii_scale_factor # remove "TCU" prefix, e.g., unit = "TCU32.0", then iim = 32.0
				else:
					iim = self.acc.initial_interval[unit]
				im = functional_units[unit] # num of instuction which used the unit
				cur_issue_km = im * iim * x / (num_act_sub_core * issue_rate)
				if "TCU" in unit:
					tensor_core_iim += im * iim * x / (num_act_sub_core * issue_rate) 
				if cur_issue_km > issue_km:
					max_unit = unit

				issue_km = max(issue_km, cur_issue_km)
			if tensor_core_iim > issue_km:
				max_unit = "TCU"
			issue_km = max(issue_km, tensor_core_iim)
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
			# to every thread which leads to 32bit * 32 / 8 = 128B L1 cache load, so 4 bytes per instruction
			issue_k_L1_var = (bk * 4 * self.acc.l1_cache_line_size / B_L1_k) * x
			# issue_k_L1_var = bk * kernel.acc.l1_cache_access_latency * x / (num_act_sub_core * issue_rate)
			issue_k_L1_var = ceil(issue_k_L1_var, 1)
			result = max(issue_base_var, issue_k_m_var, issue_k_L1_var)
			is_mem = issue_k_L1_var > issue_k_m_var and issue_k_L1_var > issue_base_var
			return result, is_mem, unit 
		
		def com_struct_and_mem_struct(x):
			result_cm = [0,0] # [com_struct, mem_struct]
			all_struct_info = {}
			for interval_info in interval_list:
				if "issue_stage" not in interval_info:
					continue
				ik = interval_info["issue_stage"]
				fu = interval_info["functional_units"] if "functional_units" in interval_info else {}
				mem_acesss_num = interval_info["mem_access_num"]
				issue_max_var, is_mem, unit = issue_max(x, ik, fu, mem_acesss_num)
				issue_base_var = issue_base(x, ik)
				ids = 1 if is_mem else 0
				result_cm[ids] += issue_max_var - issue_base_var
				interval_info_type = "L1 cache" if is_mem else unit
				if interval_info_type not in all_struct_info:
					all_struct_info[interval_info_type] = issue_max_var - issue_base_var
				else:
					all_struct_info[interval_info_type] += issue_max_var - issue_base_var
			return result_cm, all_struct_info

		result_cm1, all_struct_info1 = com_struct_and_mem_struct(num_cncr_warps)
		com1, mem1 = result_cm1
		result_cm2, all_struct_info2 = com_struct_and_mem_struct(warps_per_SM % num_cncr_warps)
		com2, mem2 = result_cm2
		math_pipe_throttle = com1 * (warps_per_SM // num_cncr_warps) + com2
		tex_throttle = mem1 * (warps_per_SM // num_cncr_warps) + mem2
		if warps_per_SM // num_cncr_warps > 0:
			self.logger.write(all_struct_info1)
		if warps_per_SM % num_cncr_warps > 0:
			self.logger.write(all_struct_info2)
		
		MDM_output = self.process_MDM(interval_list, active_SMs, umem_hit_rate, warps_per_SM)
		
		S_MSHR_i = float(MDM_output["MSHR"])
		S_NoC_i = float(MDM_output["NoC"])
		S_Dram_i = float(MDM_output["Dram"])
		
		lg_throttle = S_MSHR_i + S_NoC_i + S_Dram_i
		Si = math_pipe_throttle + tex_throttle + lg_throttle
		C_active_i = C_ij + Si

		C_idle_i = 0 # we will calculate it later in kernel
		C = C_active_i + C_idle_i
		
		general_GCoM_output = {
			"AMALi": 0,
			"selected": selected,
			"wait": wait,
			"drain": drain,
			"long_scoreboard": long_scoreboard,
			"short_scoreboard": short_scoreboard,
			"C_idle_ij_orig": C_idle_ij,
			"C_idle_ij_ID": 0,
			"math_pipe_throttle": math_pipe_throttle,
			"tex_throttle": tex_throttle,
			"lg_throttle": lg_throttle,
			"S_MSHR_i": S_MSHR_i,
			"S_NoC_i": S_NoC_i,
			"S_Dram_i": S_Dram_i,			
			# "C_idle_i_orig": C_idle_i,
			"C_idle_i_ID": 0,	
			"no_instructions_and_imc_miss": 0,		
			"GCoM+TCM": C,
		}

		self.logger.write("-----------")
		self.logger.write("total_cycles:",total_cycles)
		self.logger.write("-----------")
		self.logger.write("num_warp_inst:",num_warp_inst)
		self.logger.write("-----------")
		instr_idx = 0
		for interval in interval_list:
			if "issue_stage" in interval:
				self.logger.write('instr_idx:', instr_idx, interval, end=' ')
				instr_idx += interval["issue_stage"]
			else:
				self.logger.write(interval)
		self.logger.write("------------")

		return general_GCoM_output

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

	
	def output_scaler(self, output_dict, scaler):
		'''
			scale the output dictionary by the scaler
			Args:
				output_dict (dict): a dictionary of outputs
				scaler (float): a scaler to scale the outputs
			Returns:
				scaled_output_dict (dict): a dictionary of scaled outputs
		'''
		keys_reduction = ["selected","wait","drain","long_scoreboard","short_scoreboard","math_pipe_throttle","tex_throttle", "lg_throttle", "S_MSHR_i", "S_NoC_i", "S_Dram_i"]
		keys_need_scale = ["selected","wait","long_scoreboard","short_scoreboard","math_pipe_throttle","tex_throttle", "lg_throttle", "S_MSHR_i", "S_NoC_i", "S_Dram_i"]
		scaled_output_dict = {}
		for key in output_dict:
			if isinstance(output_dict[key], (int, float)) and key in keys_need_scale:
				scaled_output_dict[key] = output_dict[key] * scaler
			else:
				scaled_output_dict[key] = output_dict[key]
		scaled_output_dict["GCoM+TCM"] = 0
		for key in keys_reduction:
			scaled_output_dict["GCoM+TCM"] += scaled_output_dict[key] 
		return scaled_output_dict