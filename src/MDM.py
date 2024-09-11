##############################################################################

# Author: Shiheng Cao
# Last Update Date: January 2024
# Copyright: Open source, must acknowledge original author

##############################################################################
def process_MDM(interval_analysis, kernel, active_SMs, umem_hit_rate, th_active_warps):
    """
        This function is used to process the MDM.
        Args:
            interval_analysis: result of interval analysis
            kernel: Kernel object
            pred_out: prediction output
        
        Returns:
            result: a dictionary of MSHR, NoC, and Dram latency
    """
    result = {"MSHR":0, "NoC":0, "Dram": 0}
    for interval_info in interval_analysis:
        if "issue_stage" not in interval_info:
            continue
        M_read = interval_info["mem_access_num"] // 2
        M_write = interval_info["mem_access_num"] - M_read
        W = th_active_warps #the number of warps concurrently executed on an SM
        '''
            MDM models the lockup-free characteristics of the streaming L1 D$s 
            by assuming a large number of MSHR entries (e.g., 4096 in [ 50 ]). 
            GCoM follows MDM's approach and models the lockup-free characteristics 
            by assuming a sufficient number of MSHR entries per L1 D$.
            However in accel-sim they claim that num_mshr is 256
        '''
        num_MSHR = 4096 
        is_MD = M_read * W > num_MSHR
        # steaming cache, set is_MD always true
        # is_MD = True
        M = min(M_read * W, num_MSHR) + M_write * W

        L_Min_LLC= kernel.acc.l1_cache_access_latency
        LLC_Miss_Rate = 1 - umem_hit_rate
        L_Min_Dram = kernel.acc.dram_mem_access_latency
        # L_no_contention = L_Min_LLC + LLC_Miss_Rate * L_Min_Dram
        #S_MSHR = (M_read * W // num_MSHR - 1) * S_Mem if is_MD else 0
        S_MSHR = 0

        L_noc_service = kernel.acc.dram_clockspeed * (kernel.acc.l1_cache_line_size / kernel.acc.noc_bandwidth)
        L_dram_service = kernel.acc.dram_clockspeed * (kernel.acc.l2_cache_line_size / kernel.acc.dram_bandwidth) * LLC_Miss_Rate
        num_SM = active_SMs
        is_NoC_saturated = L_noc_service * M * num_SM > L_Min_LLC + L_Min_Dram
        is_Dram_saturated = L_dram_service * M * num_SM > L_Min_LLC + L_Min_Dram

        S_NoC = num_SM * M * L_noc_service if is_NoC_saturated and is_MD else num_SM * M * L_noc_service * 0.5
        S_Dram = num_SM * M * L_dram_service if is_Dram_saturated and is_MD else num_SM * M * L_dram_service * 0.5
        # S_Mem = L_no_contention + S_NoC + S_Dram
        result["MSHR"] += S_MSHR
        result["NoC"] += S_NoC
        result["Dram"] += S_Dram

    return result