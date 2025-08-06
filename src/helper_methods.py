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


import os, sys, math, time
from scipy import special as sp


def dump_output(pred_out):

    kernel_prefix = str(pred_out["kernel_id"])+"_"+pred_out["ISA"] +"_g"+pred_out["granularity"]
    output_path = os.path.join(pred_out["app_path"],"output")
    if not os.path.exists(output_path):  
        os.makedirs(output_path)
    outF = open(os.path.join(output_path, "kernel_"+kernel_prefix+".out"), "w+")

def print_config_error(config_name, flag=0):
	if flag == 1:
		print("\n[Error]\nGPU Compute Capabilty \"" +config_name+"\" is not supported")
		sys.exit(1)
	elif flag == 2:
		print("\n[Error]\n\""+config_name+"\" is not defined in the hardware compute capability file")
		sys.exit(1)
	else:
		print("\n[Error]\n\""+config_name+"\" config is not defined in the hardware configuration file")
		sys.exit(1)


def print_warning(arg1, arg2, flag=False):
	if flag:
		print("\n[Warning]\n\"" + arg1 + "\" is not defined in the config file "+\
		"assuming L1 cache is "+ arg2 + "\n")
	else:
		print("\n[Warning]\n\"" + arg1 + "\" can't be more than " + arg2\
		 	+" registers\n assuming \"" + arg1 + "\" = " + arg2 + "\n")


def ceil(x, s):
	return s * math.ceil(float(x)/s)

def qfunc(arg):
    return 0.5-0.5*sp.erf(arg/1.41421)

def floor(x, s):
    return s * math.floor(float(x)/s)
