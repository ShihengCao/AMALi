# * GPU Microarchitecture adopted from:
# - [1] https://www.techpowerup.com/gpu-specs/geforce-rtx-3090.c3622
# - [2] https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
# - [3] https://arxiv.org/pdf/2208.11174.pdf
# - [4] https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-8-x
cc_configs = {

    "warp_size"         		  :	32,
    "max_block_size"			  : 1024,
    "smem_allocation_size"		  :	256,
    "max_registers_per_SM"		  :	65536, # 64KB [2]
    "max_registers_per_block"	  :	65536,
    "max_registers_per_thread"    : 255,
    "register_allocation_size"	  : 256,
    "max_active_blocks_per_SM"    : 24,
    "max_active_threads_per_SM"   : 1536

}