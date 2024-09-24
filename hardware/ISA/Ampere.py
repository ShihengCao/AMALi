##############################################################################
# SASS instructions adpoted from:
# - https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html

# Instructions Latencies adopted from: 
# Y. Arafa et al., "Low Overhead Instruction Latency Characterization for NVIDIA GPGPUs," HPEC'19
# https://github.com/NMSU-PEARL/GPUs-ISA-Latencies

# Memory Latencies adopted from:
# H. Abdelkhalik et al., "Demystifying the Nvidia Ampere Architecture through Microbenchmarking and Instruction-level Analysis"
# https://arxiv.org/pdf/2208.11174.pdf

##############################################################################

units_latency = {

    #ALU Units Latencies
    "iALU"              :   4,
    "fALU"              :   4,
    "hALU"              :   4,
    "dALU"              :   4,

    "SFU"               :  23,
    "dSFU"              :  16,
    "iTCU"              :  4,
    "hTCU"              :  16, # accumulator FP16
    "fTCU"              :  32, # accumulator FP32
    "dTCU"              :  64,

    "BRA"               :  1,
    # "EXIT"              :  4,
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
}

initial_interval = {

    # Initiation interval (II) = threadsPerWarp / #FULanes
    "iALU"              :   32 / 16,
    "fALU"              :   32 / 32,
    "hALU"              :   32 / 32,
    "dALU"              :   32 / 0.5,

    "SFU"               :   32 / 4,
    "dSFU"              :   32 / 4,

    "iTCU"              :   32 / 1,
    "hTCU"              :   32 / 1, # accumulator FP16
    "fTCU"              :   32 / 1, # accumulator FP32
    "dTCU"              :   32 / 1,
    
    "BRA"               :   32 / 32,
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
    "HMNMX2"            : "hTCU", # new !!
    "BMMA"              : "iTCU", # new !!
    "DMMA"              : "dTCU",
    # Conversion Instructions
    "F2F"               : "iALU",
    "F2I"               : "iALU",
    "I2F"               : "iALU",
    "I2I"               : "iALU",
    "I2IP"              : "iALU",
    "I2FP"              : "iALU",# new !!
    "F2IP"              : "iALU",# new !!
    "F2FP"              : "iALU",# new !!
    "FRND"              : "iALU",
    # Movement Instructions
    "MOV"               : "iALU",
    "MOV32I"            : "iALU",
    "MOVM"              : "iALU",# new !!
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
    "REDUX"             : "iALU",# new !!
    "S2UR"              : "iALU",
    "UBMSK"             : "iALU",
    "UBREV"             : "iALU",
    "UCLEA"             : "iALU",
    "UF2FP"             : "iALU",# new !!
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
    "JMXU"              : "BRA",# new !!
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
    # "R2B"               : "iALU",
    "S2R"               : "iALU",
    # "S2UR"              : "iALU",
    "SETCTAID"          : "iALU",
    "SETLMEMBASE"       : "iALU",
    "VOTE"              : "iALU"
}


ptx_isa = { # ---> (ptx v.72)

    # Integer Instructions
    "add"               : "iALU",
    "sub"               : "iALU",
    "mul"               : "iALU",
    "mad"               : "iALU",
    "mul24lo"           : ["iALU", 8],
    "mul24hi"           : ["iALU", 15],
    "mad24lo"           : ["iALU", 8],
    "mad24hi"           : ["iALU", 15],
    "sad"               : "iALU",
    "mad"               : "iALU",
    "div"               : ["iALU", 71],
    "rem"               : ["iALU", 71],
    "abs"               : "iALU",
    "neg"               : "iALU",
    "min"               : "iALU",
    "max"               : "iALU",
    "popc"              : "iALU",
    "clz"               : ["iALU", 8],
    "bfind"             : "iALU",
    "fns"               : "iALU",
    "brev"              : ["iALU", 8],
    "bfe"               : ["iALU", 125],
    "bfi"               : ["iALU", 125],
    "dp4a"              : "iALU",
    "dp2a"              : "iALU",
    "ret"               : "iALU",
    "exit"              : "iALU",
    "bar"               : "iALU",
    # Logic and Shift Instructions
    "and"               : "iALU",
    "or"                : "iALU",
    "not"               : "iALU",
    "xor"               : "iALU",
    "cnot"              : ["iALU", 8],
    "lop3"              : "iALU",
    "shf"               : "iALU",
    "shl"               : "iALU",
    "shr"               : "iALU",
    # Extended-Precision Integer Instructions
    "addc"              : "iALU",
    "sub.cc"            : "iALU",
    "subc"              : "iALU",
    "mad.cc"            : "iALU",
    "madc"              : "iALU",
    # Single-Precision Floating Instructions
    "ftestp"            : "fALU",
    "fcopysign"         : "fALU",
    "fadd"              : "fALU",
    "fsub"              : "fALU",
    "fmul"              : "fALU",
    "ffma"              : "fALU",
    "fmad"              : "fALU",
    "fdiv"              : ["fALU", 335],
    "fabs"              : "fALU",
    "fneg"              : "fALU",
    "fmin"              : "fALU",
    "fmax"              : "fALU",
    "frcp"              : ["SFU", 60],
    "Fastfrcp"          : ["SFU", 23],
    "fsqrt"             : ["SFU", 60],
    "Fastfsqrt"         : ["SFU", 18],
    "frsqrt"            : ["SFU", 60],
    "Fastfrsqrt"        : ["SFU", 18],
    "fsin"              : ["SFU", 8],
    "fcos"              : ["SFU", 8],
    "fex2"              : ["SFU", 16],
    "flg2"              : ["SFU", 16],
    "ftanh"             : ["SFU", 8],
    # Half-Precision Floating Instructions
    "hfadd"             : "hALU",
    "hfsub"             : "hALU", 
    "hfmul"             : "hALU",
    "hfma"              : "hALU",  
    "hneg"              : "hALU",
    "habs"              : "hALU",  
    "hmin"              : "hALU", 
    "hmax"              : "hALU", 
    "htanh"             : "hALU", 
    "hex2"              : "hALU",
    # Double-Precision Floating Instructions
    "dadd"              : "dALU",
    "dsub"              : "dALU",
    "dmul"              : "dALU",
    "dmad"              : "dALU",
    "dfma"              : "dALU",
    "dabs"              : "dALU",
    "dneg"              : "dALU",
    "dmin"              : "dALU",
    "dmax"              : "dALU",
    "dmax"              : "dALU",
    "ddiv"              : ["dALU", 506],
    "Fastddiv"          : ["dALU", 159],
    "drcp"              : ["dALU", 114],
    "dsqrt"             : ["dALU", 106],
    "drsqrt"            : ["dALU", 106],
    # Tensor Core
    "wmma"              : "iTCU",
    "hwmma"             : "hTCU",
    # Conversion & Movement Instructions
    "mov"               : "iALU",
    "shfl"              : "iALU",
    "prmt"              : "iALU",
    "cvta"              : "iALU",
    "cvt"               : "iALU",
    # Comparision & Selection Instructions
    "set"               : "iALU",
    "setp"              : "iALU",
    "selp"              : "iALU",
    # Control Instructions
    "bra"               : "BRA",
    "call"              : "BRA",
    
}