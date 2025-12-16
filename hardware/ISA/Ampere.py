##############################################################################
# SASS instructions adpoted from:
# - https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html

# Instructions Latencies adopted from: 
# Y. Arafa et al., "Low Overhead Instruction Latency Characterization for NVIDIA GPGPUs," HPEC'19
# https://github.com/NMSU-PEARL/GPUs-ISA-Latencies
# M. Khairy et al, "Accel-Sim: An Extensible Simulation Framework for Validated GPU Modeling" ISCA'20
# https://github.com/accel-sim/accel-sim-framework

# Memory Latencies adopted from:
# H. Abdelkhalik et al., "Demystifying the Nvidia Ampere Architecture through Microbenchmarking and Instruction-level Analysis"
# https://arxiv.org/pdf/2208.11174.pdf
# M. Khairy et al, "Accel-Sim: An Extensible Simulation Framework for Validated GPU Modeling" ISCA'20
# https://github.com/accel-sim/accel-sim-framework

##############################################################################

units_latency = {

    #ALU Units Latencies
    "iALU"              :   4,
    "fALU"              :   4,
    "hALU"              :   4,
    "dALU"              :   4,
    "Uniform"           :   2,

    "SFU"               :  23,
    # for tensor core we use FMA/cycle instead and will calculate the latency later, the following values are useless
    "iTCU"              :  4,
    "hTCU"              :  128, # accumulator FP16
    "fTCU"              :  64, # accumulator FP32
    "dTCU"              :  64,

    "BRA"               :  1,
    #Memory Units Latencies
    "dram_mem_access"   :   290,
    "l1_cache_access"   :   33,
    "l2_cache_access"   :   200,
    "local_mem_access"  :   290,
    "const_mem_access"  :   290,
    "shared_mem_ld"     :   23,
    "shared_mem_st"     :   19,# ld 23 st 19
    "tex_mem_access"    :   290,
    "tex_cache_access"  :   86,
    "atomic_operation"  :   245,
    #Kernel_launch_ovhd get from Accel-sim benchmark
    "kernel_launch_ovhd"    :   1980,
    "slope_alpha"   :  0.0036, # KLL model 
    "slope_beta"   :  0.0366, # KLL model 
    "slope_gamma"   :  1.1891, # KLL model 
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
    # "iTCU"              :   32 / 1,
    # "hTCU"              :   128, # accumulator FP16
    # "fTCU"              :   64, # accumulator FP32
    # "dTCU"              :   0, # Ampere not support double precision, this only support in Tesla_Ampere

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
    # "CCTL"              : "iALU",
    # Single-Precision Floating Instructions
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
    # Half-Precision Floating Instructions
    "HADD2"             : "hALU",
    "HADD2_32I"         : "hALU",
    "HFMA2"             : "hALU",
    "HFMA2_32I"         : "hALU",
    "HMNMX2"            : "hALU",
    "HMUL2"             : "hALU",
    "HMUL2_32I"         : "hALU",
    "HSET2"             : "hALU",
    "HSETP2"            : "hALU",
    # Double-Precision Floating Instructions
    "DADD"              : "dALU",
    "DFMA"              : "dALU",
    "DMUL"              : "dALU",
    "DSETP"             : "dALU",
    # SFU Special Instructions
    "MUFU"              : "SFU",
    #Tensor Core
    "IMMA"              : "iTCU",
    "HMMA"              : "hTCU",
    "BMMA"              : "iTCU",
    "DMMA"              : "dTCU",
    # Conversion Instructions
    "F2F"               : "iALU",
    "F2I"               : "iALU",
    "I2F"               : "iALU",
    "I2I"               : "iALU",
    "I2IP"              : "iALU",
    "I2FP"              : "iALU",
    "F2IP"              : "iALU",
    "F2FP"              : "iALU",
    "FRND"              : "iALU",
    # Movement Instructions
    "MOV"               : "iALU",
    "MOV32I"            : "iALU",
    "MOVM"              : "iALU",
    "PRMT"              : "iALU",
    "SEL"               : "iALU",
    "SGXT"              : "iALU",
    "SHFL"              : "iALU",
    # Predicate Instructions
    "PLOP3"             : "iALU",
    "PSETP"             : "iALU",
    "P2R"               : "iALU",
    "R2P"               : "iALU",
   #Uniform Datapath Instructions
    "R2UR"              : "Uniform",
    "REDUX"             : "Uniform",
    "S2UR"              : "Uniform",
    "UBMSK"             : "Uniform",
    "UBREV"             : "Uniform",
    "UCLEA"             : "Uniform",
    "UF2FP"             : "Uniform",
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
    "JMXU"              : "BRA",
    "KILL"              : "BRA",
    "NANOSLEEP"         : "BRA",
    "RET"               : "BRA",
    "RPCMOV"            : "BRA",
    "RTT"               : "BRA",
    "WARPSYNC"          : "BRA",
    "YIELD"             : "BRA",
    # Miscellaneous Instructions
    "B2R"               : "iALU",
    "BAR"               : "iALU",
    "CS2R"              : "iALU",
    "DEPBAR"            : "iALU",
    "GETLMEMBASE"       : "iALU",
    "LEPC"              : "iALU",
    "NOP"               : "iALU",
    "PMTRIG"            : "iALU",
    "S2R"               : "iALU",
    "SETCTAID"          : "iALU",
    "SETLMEMBASE"       : "iALU",
    "VOTE"              : "iALU"
}
