##############################################################################
# &copy 2017. Triad National Security, LLC. All rights reserved.

# This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration.

# All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

# Recall that this copyright notice must be accompanied by the appropriate open source license terms and conditions. Additionally, it is prudent to include a statement of which license is being used with the copyright notice. For example, the text below could also be included in the copyright notice file: This is open source software; you can redistribute it and/or modify it under the terms of the Performance Prediction Toolkit (PPT) License. If software is modified to produce derivative works, such modified software should be clearly marked, so as not to confuse it with the version available from LANL. Full text of the Performance Prediction Toolkit (PPT) License can be found in the License file in the main development branch of the repository.
##############################################################################

# Author: Yehia Arafa
# Last Update Date: April, 2021
# Copyright: Open source, must acknowledge original author

##############################################################################


from .warps import Warp

class Block(object):
    '''
        class that represents a block (CTA) being executed on an SM
    '''
    def __init__(self, gpu, id, sub_core_id, tasklist, kernel_id, isa, avg_mem_lat, avg_atom_lat):
        # self.num_warps = num_warps 
        self.num_active_warps = 0
        self.sync_warps = 0
        # SM id cause we only spawn one thread block on each SM
        self.id = id
        # sub-core id
        self.sub_core_id = sub_core_id
        self.active = False
        self.waiting_to_execute = True
        self.actual_end = 0
        self.warp_list = []
        
        if isa == "SASS":
            # Here: len(self.kernel_tasklist) <= self.num_warps
            for warp in tasklist:
                self.warp_list.append(Warp(self, gpu, tasklist[warp], kernel_id, id, warp, avg_mem_lat, avg_atom_lat))
