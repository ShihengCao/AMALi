		# print(interval_analysis)
		# interval analysis end
		# '''
		# model sub-core
		# 1. the number of warps per sub-core
		# 2. the number of sub-cores per SM
		# warps of sub-core = warps of SM / sub-cores per SM
		# '''
		# warps_per_SM = self.kernel_block_size * self.kernel_grid_size // pred_out["active_SMs"] // self.acc.warp_size
		# # TODO the value of num_sub_cores_per_SM 目前使用的是scheduler数量
		# num_sub_cores_per_SM = self.acc.num_warp_schedulers_per_SM
		# warps_ij = warps_per_SM / num_sub_cores_per_SM # Warpsi,j is the number of warps the j-th sub-core of the i-th SM executes
		# issue_rate = 1 # 1 issue per cycle
		# num_warp_inst = representive_warp.current_inst
		# total_num_warp_inst = num_warp_inst * warps_ij
		# C_base_ij = total_num_warp_inst / issue_rate
		# S_ComData_ij = 0
		# S_MemData_ij = 0
		# # TODO the value of P_inst 执行的指令，除以全部cycles
		# P_inst = num_warp_inst / total_cycles
		
		# # print(representive_warp.interval_list)			
		# # print(interval_analysis)
		# # print(mem_stall_num,mem_stall_total_cycles,com_stall_num,com_stall_total_cycles)
		
		# '''
		# Use GTO warp schedule policy
		# '''
		# # avg_int_insts = (num_warp_inst / total_intervals)
		# # for stage_info in interval_analysis:
		# # 	if "stall_stage" in stage_info:
		# # 		S_intv_k = stage_info["stall_stage"]
		# # 		P_warp = min(S_intv_k * P_inst, 1)
		# # 		cycle_other = int(P_warp* (warps_ij - 1) * avg_int_insts / issue_rate)
		# # 		if stage_info["stall_type"] == 2:
		# # 			S_ComData_ij += max(S_intv_k - cycle_other, 0)
		# # 		else:
		# # 			S_MemData_ij += max(S_intv_k - cycle_other, 0)
		# '''
		# Use LRR warp schedule policy
		# '''
		# for stage_info in interval_analysis:
		# 	if "stall_stage" in stage_info:
		# 		S_intv_k = stage_info["stall_stage"]
		# 		if stage_info["stall_type"] == 2:
		# 			S_ComData_ij += max(int(S_intv_k - (warps_ij - 1) * P_inst), 0)
		# 		else:
		# 			S_MemData_ij += max(int(S_intv_k - (warps_ij - 1) * P_inst), 0)

		# C_active_ij = C_base_ij + S_ComData_ij + S_MemData_ij
		# # print(C_active_ij, S_ComData_ij, S_MemData_ij)
		# '''
		# Modeling the Cycles of a Core
		# '''
		# Si = 0
		# S_ComStruct_i = 0
		# S_MemStruct_i = 0
		# # TODO 目前使用的是acc中设置的最大活跃warp数量
		# # num_cncr_warps = self.acc.max_active_warps_per_SM
		# num_cncr_warps = pred_out["th_active_warps"]
		# # TODO bk B_L1_k 的取值，目前使用L1_cache_access_latency乘一个系数近似模拟
		# # TODO 什么是the effective L1 D$ bandwidth of the k-th interval
		# # bk = int(pred_out["memory_stats"]["gmem_tot_reqs"] * pred_out["memory_stats"]["umem_hit_rate"])
		# # B_L1_k = 4
		# L1_latency = self.acc.l1_cache_access_latency#/num_cncr_warps #int(bk / B_L1_k)
		# # print(issue_kL1)

		# def issue_base(x, ik):
		# 	if x == 0:
		# 		return 0
		# 	num_act_sub_core = min(x, num_sub_cores_per_SM)
		# 	return ik * x / (num_act_sub_core * issue_rate)


		# def issue_k_m(x,functional_units):
		# 	issue_km = -100
		# 	num_act_sub_core = min(x, num_sub_cores_per_SM)
		# 	# TODO 这样获得the functional unit's initiation interval
		# 	# iim = 4
		# 	for unit in functional_units:
		# 		iim = self.acc.units_latency[unit]
		# 		im = functional_units[unit]
		# 		issue_km = max(issue_km, im * iim * x / (num_act_sub_core * issue_rate))		
		# 	return issue_km			
		# # def issue_k_L1(x, L1_access_num, bandwidth):
		# # 	return ceil(L1_access_num / bandwidth , 1) * x
		# def issue_max(x, ik, fu, bk):
		# 	if x == 0:
		# 		return 0
		# 	issue_base_var = issue_base(x, ik)
		# 	issue_k_m_var = issue_k_m(x, fu)
		# 	num_act_sub_core = min(x, num_sub_cores_per_SM)
		# 	issue_k_L1_var = L1_latency * bk * x / num_act_sub_core
		# 	# print(issue_base_var, issue_k_m_var, issue_k_L1_var)
		# 	return max(issue_base_var, issue_k_m_var, issue_k_L1_var), issue_k_L1_var > issue_k_m_var and issue_k_L1_var > issue_base_var
		# def com_struct_and_mem_struct(x):
		# 	com, mem = 0, 0
		# 	for interval_info in interval_analysis:
		# 		if "issue_stage" not in interval_info:
		# 			continue
		# 		ik = interval_info['issue_stage']
		# 		fu = interval_info['functional_units'] if 'functional_units' in interval_info else {}
		# 		mem_acesss_num = interval_info['mem_access_num']
		# 		issue_max_var, is_mem = issue_max(x, ik, fu, mem_acesss_num)
		# 		if is_mem:
		# 			mem += issue_max_var - issue_base(x, ik)
		# 		else:
		# 			com += issue_max_var - issue_base(x, ik)
		# 	return com, mem
		# # print(num_cncr_warps, warps_per_SM)
		# com1, mem1 = com_struct_and_mem_struct(num_cncr_warps)
		# com2, mem2 = com_struct_and_mem_struct(int(warps_per_SM % num_cncr_warps))
		# # print(com2,mem2)
		# S_ComStruct_i = com1 * (warps_per_SM // num_cncr_warps) + com2
		# S_MemStruct_i = mem1 * (warps_per_SM // num_cncr_warps) + mem2
		
		# Si = S_ComStruct_i + S_MemStruct_i
		# C_active_i = C_active_ij + Si

		# C_idle_i = 0
		# C = int(C_active_i + C_idle_i)
		# print_dict = {
		# 	"kernel_name": self.kernel_name,
		# 	"GCoM": C,
		# 	"C_base_ij": C_base_ij,
		# 	"S_ComData_ij": S_ComData_ij,
		# 	"S_MemData_ij": S_MemData_ij,
		# 	"S_ComStruct_i": S_ComStruct_i,
		# 	"S_MemStruct_i": S_MemStruct_i,
		# 	"total_instr_executed": num_warp_inst * warps_per_SM
		# }
		# print(print_dict)

'''
		interval analysis get the value of 

			mem_stall_num, mem_stall_total_cycles, 
			com_stall_num, com_stall_total_cycles
			total_intervals
'''
'''
		# total_cycles = representive_warp.interval_profile()
		# print(representive_warp.completions)
		# index = 1
		# current_len = 1
		# interval_analysis = []
		# functional_units = {}
		# mem_acesss_num = 0

		# mem_stall_num = 0
		# mem_stall_total_cycles = 0
		# com_stall_num = 0
		# com_stall_total_cycles = 0
		# total_intervals = 0
		# if representive_warp.hw_unit_list[0] not in functional_units:
		# 	functional_units[representive_warp.hw_unit_list[0]] = 1
		# else:
		# 	functional_units[representive_warp.hw_unit_list[0]] += 1
		# interval_list = representive_warp.interval_list
		# # print(len(representive_warp.interval_list), len(representive_warp.access_info))
		# while index < len(interval_list):
		# 	if interval_list[index] != interval_list[index-1]:
		# 		if interval_list[index-1] == 0:
		# 			interval_analysis.append({"issue_stage":current_len, "functional_units":functional_units, "mem_access_num":mem_acesss_num})
		# 			total_intervals += 1
		# 		else:
		# 			if interval_list[index-1] == 1:
		# 				mem_stall_num += 1
		# 				mem_stall_total_cycles += current_len
		# 			elif interval_list[index-1] == 2:
		# 				com_stall_num += 1
		# 				com_stall_total_cycles += current_len
		# 			interval_analysis.append({"stall_stage":current_len,"stall_type":interval_list[index-1]})
		# 		current_len = 1
		# 		functional_units = {}
		# 		mem_acesss_num = 0
		# 		if representive_warp.hw_unit_list[index] is not None:
		# 			# print(representive_warp.hw_unit_list[index], functional_units)
		# 			if representive_warp.hw_unit_list[index] not in functional_units:
		# 				functional_units[representive_warp.hw_unit_list[index]] = 1
		# 			else:
		# 				functional_units[representive_warp.hw_unit_list[index]] += 1
		# 		mem_acesss_num += representive_warp.access_info[index]
		# 	else:
		# 		if representive_warp.hw_unit_list[index] is not None:
		# 			# print(representive_warp.hw_unit_list[index], functional_units)
		# 			if representive_warp.hw_unit_list[index] not in functional_units:
		# 				functional_units[representive_warp.hw_unit_list[index]] = 1
		# 			else:
		# 				functional_units[representive_warp.hw_unit_list[index]] += 1
		# 		mem_acesss_num += representive_warp.access_info[index]
		# 		current_len += 1
		# 	index += 1

		# if interval_list[-1] == 0:
		# 	interval_analysis.append({"issue_stage":current_len, "functional_units":functional_units, "mem_access_num":mem_acesss_num})
		# else:
		# 	if interval_list[index-1] == 1:
		# 		mem_stall_num += 1
		# 		mem_stall_total_cycles += current_len
		# 	elif interval_list[index-1] == 2:
		# 		com_stall_num += 1
		# 		com_stall_total_cycles += current_len			
		# 	interval_analysis.append({"stall_stage":current_len,"stall_type":interval_list[index-1]})
		'''