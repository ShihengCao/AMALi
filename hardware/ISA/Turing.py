##############################################################################
# SASS instructions adpoted from:
# - https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html

# Instructions Latencies adopted from: 
# Y. Arafa et al., "Low Overhead Instruction Latency Characterization for NVIDIA GPGPUs," HPEC'19
# https://github.com/NMSU-PEARL/GPUs-ISA-Latencies

##############################################################################

units_latency = {

    #ALU Units Latencies
    "iALU"              :   4,
    "fALU"              :   4,
    "hALU"              :   4,
    "dALU"              :   45,
    "Uniform"           :   2,

    "SFU"               :   4,
    "dSFU"              :   12,

    "bTCU"              :   64,
    "iTCU"              :   64,
    "hTCU"              :   64,

    "BRA"               :   4,
    #Memory Units Latencies
    "dram_mem_access"   :   279,
    "l1_cache_access"   :   30,
    "l2_cache_access"   :   231,
    "local_mem_access"  :   279,
    "const_mem_access"  :   8,
    "shared_mem_ld"     :   30,
    "shared_mem_st"     :   30,
    "tex_mem_access"    :   410,
    "tex_cache_access"  :   92,
    "atomic_operation"  :   245,
    #Kernel_launch_ovhd get from Accel-sim benchmark
    "kernel_launch_ovhd"    :   2119,
    "slope_alpha"   :  1.0956e-03, # KLL model  0.0036
    "slope_beta"   :  -1.5389e-02, # KLL model  0.0366
    "slope_gamma"   :  9.3524e-01, # KLL model 1.1891

}

# we will calculate later in generate accelerator
initial_interval = {
    # Initiation interval (II) = threadsPerWarp / #FULanes
    "iALU"              :   0,
    "fALU"              :   0,
    "hALU"              :   0,
    "dALU"              :   0,

    "SFU"               :   0,

    "LDST"              :   0,    
    "BRA"               :   0,
    "Uniform"           :   0.5, # in hot chips 31 "RTX ON THE NVIDIA TURING GPU", note UDP 1 instr / 2 clk, so we set 0.5 here
    # we will not use initial_interval for TCU for now
    # "bTCU"              :   256, # compute BF16 accumulator FP32
    # "hTCU"              :   256, # accumulator FP16
    # "fTCU"              :   256, # accumulator FP32
    # "bTCU"              :   320, # compute BF16 accumulator FP32
    # "hTCU"              :   320, # accumulator FP16
    # "fTCU"              :   320, # accumulator FP32
}

sass_isa = {

    # Integer Instructions
    "BMSK"              : "iALU",
    "BREV"              : "iALU",
    "FLO"               : "iALU",
    "IABS"              : "iALU",
    "IADD"              : "iALU",
    "IADD3"             : "iALU",
    "IADD32I"           : "iALU",
    "IDP"               : "iALU",
    "IDP4A"             : "iALU",
    "IMAD"              : "iALU",
    "IMNMX"             : "iALU",
    "IMUL"              : "iALU",
    "IMUL32I"           : "iALU",
    "ISCADD"            : "iALU",
    "ISCADD32I"         : "iALU",
    "ISETP"             : "iALU",
    "LEA"               : "iALU",
    "LOP"               : "iALU",
    "LOP3"              : "iALU",
    "LOP32I"            : "iALU",
    "POPC"              : "iALU",
    "SHF"               : "iALU",
    "SHL"               : "iALU",
    "SHR"               : "iALU",
    "VABSDIFF"          : "iALU",
    "VABSDIFF4"         : "iALU",
    #Single Precision Floating Instructions
    "FADD"              : "fALU",
    "FADD32I"           : "fALU",
    "FCHK"              : "fALU",
    "FFMA32I"           : "fALU",
    "FFMA"              : "fALU",
    "FMNMX"             : "fALU",
    "FMUL"              : "fALU",
    "FMUL32I"           : "fALU",
    "FSEL"              : "fALU",
    "FSET"              : "fALU",
    "FSETP"             : "fALU",
    "FSWZADD"           : "fALU",
    #Half Precision Floating Instructions
    "HADD2"             : "hALU",
    "HADD2_32I"         : "hALU",
    "HFMA2"             : "hALU",
    "HFMA2_32I"         : "hALU",
    "HMNMX2"            : "hALU",
    "HMUL2"             : "hALU",
    "HMUL2_32I"         : "hALU",
    "HSET2"             : "hALU",
    "HSETP2"            : "hALU",
    #Double Precision Floating Instructions
    "DADD"              : "dALU",
    "DFMA"              : "dALU",
    "DMUL"              : "dALU",
    "DSETP"             : "dALU",
    #SFU Special Instructions
    "MUFU"              : "SFU",
    #Tensor Core
    "BMMA"              : "bTCU",
    "IMMA"              : "iTCU",
    "HMMA"              : "hTCU",
    #Conversion Instructions
    "F2F"               : "iALU",
    "F2I"               : "iALU",
    "I2F"               : "iALU",
    "I2I"               : "iALU",
    "I2IP"              : "iALU",
    "FRND"              : "iALU",
    #Movement Instructions
    "MOV"               : "iALU",
    "MOV32I"            : "iALU",
    "PRMT"              : "iALU",
    "SEL"               : "iALU",
    "SGXT"              : "iALU",
    "SHFL"              : "iALU",
    #Predicate Instructions
    "PLOP3"             : "iALU",
    "PSETP"             : "iALU",
    "P2R"               : "iALU",
    "R2P"               : "iALU",
    #Uniform Datapath Instructions
    "R2UR"              : "Uniform",
    "S2UR"              : "Uniform",
    "UBMSK"             : "Uniform",
    "UBREV"             : "Uniform",
    "UCLEA"             : "Uniform",
    "UFLO"              : "Uniform",
    "UIADD3"            : "Uniform",
    "UIADD3.64"         : "Uniform",
    "UIMAD"             : "Uniform",
    "UISETP"            : "Uniform",
    "ULDC"              : "Uniform",
    "ULEA"              : "Uniform",
    "ULOP"              : "Uniform",
    "ULOP3"             : "Uniform",
    "ULOP32I"           : "Uniform",
    "UP2UR"             : "Uniform",
    "UMOV"              : "Uniform",
    "UP2UR"             : "Uniform",
    "UPLOP3"            : "Uniform",
    "UPOPC"             : "Uniform",
    "UPRMT"             : "Uniform",
    "UPSETP"            : "Uniform",
    "UR2UP"             : "Uniform",
    "USEL"              : "Uniform",
    "USGXT"             : "Uniform",
    "USHF"              : "Uniform",
    "USHL"              : "Uniform",
    "USHR"              : "Uniform",
    "VOTEU"             : "Uniform",
    #Control Instructions
    "BMOV"              : "BRA",
    "BPT"               : "BRA",
    "BRA"               : "BRA",
    "BREAK"             : "BRA",
    "BRX"               : "BRA",
    "BRXU"              : "BRA",
    "BSSY"              : "BRA",
    "BSYNC"             : "BRA",
    "CALL"              : "BRA",
    "EXIT"              : "BRA",
    "JMP"               : "BRA",
    "JMX"               : "BRA",
    "KILL"              : "BRA",
    "NANOSLEEP"         : "BRA",
    "RET"               : "BRA",
    "RPCMOV"            : "BRA",
    "RTT"               : "BRA",
    "WARPSYNC"          : "BRA",
    "YIELD"             : "BRA",
    #Miscellaneous Instructions
    "B2R"               : "iALU",
    "BAR"               : "iALU",
    "CS2R"              : "iALU",
    "DEPBAR"            : "iALU",
    "GETLMEMBASE"       : "iALU",
    "LEPC"              : "iALU",
    "NOP"               : "iALU",
    "PMTRIG"            : "iALU",
    "R2B"               : "iALU",
    "S2R"               : "iALU",
    "SETCTAID"          : "iALU",
    "SETLMEMBASE"       : "iALU",
    "VOTE"              : "iALU"
}