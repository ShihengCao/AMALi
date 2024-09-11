# * GPU Microarchitecture adopted from:
# - [1] https://www.techpowerup.com/gpu-specs/geforce-rtx-2080-ti.c3305
# - [2] https://images.nvidia.cn/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf
# - [3] https://arxiv.org/pdf/1903.07486.pdf
# - [4] https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-7-x
cc_configs = {

    "warp_size"         		  :	32,
    "max_block_size"			  :	1024,
    "smem_allocation_size"		  :	256,
    "max_registers_per_SM"		  :	65536, # 64KB [2]
    "max_registers_per_block"	  :	65536,
    "max_registers_per_thread"    : 255,
    "register_allocation_size"	  : 256,
    "max_active_blocks_per_SM"    : 32,
    "max_active_threads_per_SM"   : 1024

}