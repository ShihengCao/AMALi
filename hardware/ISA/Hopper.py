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
    "dALU"              :   8,

    "SFU"               :  23,
    # for tensor core we use FMA/cycle instead and will calculate the latency later, the following values are useless
    "iTCU"              :  512,
    "hTCU"              :  512, # accumulator FP16
    "fTCU"              :  512, # accumulator FP32
    "bTCU"              :  1024, # accumulator INT8
    "dTCU"              :  512,

    "BRA"               :  1,
    #Memory Units Latencies
    "dram_mem_access"   :   192,
    "l1_cache_access"   :   40,
    "l2_cache_access"   :   225,
    "local_mem_access"  :   192,
    "const_mem_access"  :   192,
    "shared_mem_ld"     :   23,
    "shared_mem_st"     :   19,# ld 23 st 19
    "tex_mem_access"    :   194,
    "tex_cache_access"  :   86,
    "atomic_operation"  :   325,
    #Kernel_launch_ovhd get from Accel-sim benchmark
    "kernel_launch_ovhd"    :   4166.61,
    "slope_alpha"   :  4.9561e-04, # KLL model 
    "slope_beta"   :  -1.3946e-03, # KLL model 
    "slope_gamma"   :  1.0081e+00, # KLL model 
}

initial_interval = {

    # Initiation interval (II) = threadsPerWarp / #FULanes
    "iALU"              :   0,
    "fALU"              :   0,
    "hALU"              :   0,
    "dALU"              :   0,

    "SFU"               :   0,

    "LDST"              :   0,    
    "BRA"               :   0,
    # we will not use initial_interval for TCU for now
    #"iTCU"              :   320,
    #"hTCU"              :   320, # accumulator FP16 fma/clk/tensor core
    #"fTCU"              :   320, # accumulator FP32
    #"dTCU"              :   320, # Ampere not support double precision, this only support in Tesla_Ampere

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
    "VHMNMX"            : "iALU",
    "VIADD"             : "iALU",
    "VIADDMNMX"         : "iALU",
    "VIMNMX"            : "iALU",
    "VIMNMX3"           : "iALU",
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
    # Warpgroup Instructions
    "BGMMA"             : "iTCU",
    "HGMMA"             : "hTCU",
    "IGMMA"             : "iTCU",
    "QGMMA"             : "fTCU",
    "WARPGROUP"         : "BRA",
    "WARPGROUPSET"      : "BRA",
    # Tensor Memory Access Instructions
    "UBLKCP"            : "BRA",
    "UBLKPF"            : "BRA",
    "UBLKRED"           : "BRA",
    "UTMACCTL"          : "BRA",
    "UTMACMDFLUSH"      : "BRA",
    "UTMALDG"           : "BRA",
    "UTMAPF"            : "BRA",
    "UTMAREDG"          : "BRA",
    "UTMASTG"           : "BRA",
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
    "R2UR"              : "iALU",
    "REDUX"             : "iALU",
    "S2UR"              : "iALU",
    "UBMSK"             : "iALU",
    "UBREV"             : "iALU",
    "UCLEA"             : "iALU",
    "UF2FP"             : "iALU",
    "UFLO"              : "iALU",
    "UIADD3"            : "iALU",
    "UIADD3.64"         : "dALU",
    "UIMAD"             : "iALU",
    "UISETP"            : "iALU",
    "ULDC"              : "iALU",
    "ULEA"              : "iALU",
    "ULOP"              : "iALU",
    "ULOP3"             : "iALU",
    "ULOP32I"           : "iALU",
    "UP2UR"             : "iALU",
    "UMOV"              : "iALU",
    "UP2UR"             : "iALU",
    "UPLOP3"            : "iALU",
    "UPOPC"             : "iALU",
    "UPRMT"             : "iALU",
    "UPSETP"            : "iALU",
    "UR2UP"             : "iALU",
    "USEL"              : "iALU",
    "USGXT"             : "iALU",
    "USHF"              : "iALU",
    "USHL"              : "iALU",
    "USHR"              : "iALU",
    "VOTEU"             : "iALU",

    #Control Instructions
    "UCGABAR_ARV"       : "BRA", # new !!
    "UCGABAR_WAIT"      : "BRA", # new !!
    "USETMAXREG"        : "BRA", # new !!  
    "USETSHMSZ"         : "BRA", # new !!
    "ACQBULK"           : "BRA",
    "BMOV"              : "BRA",
    "BPT"               : "BRA",
    "BRA"               : "BRA",
    "BREAK"             : "BRA",
    "BRX"               : "BRA",
    "BRXU"              : "BRA",
    "BSSY"              : "BRA",
    "BSYNC"             : "BRA",
    "SYNCS"             : "BRA", # new ~=!
    "CALL"              : "BRA",
    "CCTL"              : "BRA",
    "CCTLL"             : "BRA",
    "CCTLT"             : "BRA",
    "CGAERRBAR"         : "BRA",
    "ELECT"             : "BRA",
    "ENDCOLLECTIVE"     : "BRA",
    "EXIT"              : "BRA",
    "JMP"               : "BRA",
    "JMX"               : "BRA",
    "JMXU"              : "BRA",
    "KILL"              : "BRA",
    "NANOSLEEP"         : "BRA",
    "PREEXIT"           : "BRA",# new ~~
    "RET"               : "BRA",
    "RPCMOV"            : "BRA",
    # "REDAS"             : "BRA",
    "WARPSYNC"          : "BRA",
    "YIELD"             : "BRA",
    "FENCE"             : "BRA", # new~=!
    "MATCH"             : "BRA", # new ~=!
    # Miscellaneous Instructions
    "B2R"               : "iALU",
    "BAR"               : "iALU",
    "CS2R"              : "iALU",
    "DEPBAR"            : "iALU",
    "GETLMEMBASE"       : "iALU",
    "LEPC"              : "iALU",
    "NOP"               : "iALU",
    "PMTRIG"            : "iALU",
    # "R2B"               : "iALU",
    "S2R"               : "iALU",
    # "S2UR"              : "iALU",
    "SETCTAID"          : "iALU",
    "SETLMEMBASE"       : "iALU",
    "VOTE"              : "iALU"
}
