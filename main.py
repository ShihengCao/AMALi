##############################################################################################################################################################################################################################################################
# &copy 2017. Triad National Security, LLC. All rights reserved.

# This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration.

# All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, 
# irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

# Recall that this copyright notice must be accompanied by the appropriate open source license terms and conditions. Additionally, it is prudent to include a statement of which license is being used with the copyright notice. For example, 
# the text below could also be included in the copyright notice file: This is open source software; you can redistribute it and/or modify it under the terms of the Performance Prediction Toolkit (PPT) License. If software is modified to produce derivative works,
# such modified software should be clearly marked, so as not to confuse it with the version available from LANL. Full text of the Performance Prediction Toolkit (PPT) License can be found in the License file in the main development branch of the repository.
##############################################################################################################################################################################################################################################################

# Author: Yehia Arafa
# Last Update Date: April, 2021
# Copyright: Open source, must acknowledge original author

##########################################################

# import csv
import sys, os, getopt, importlib
from simian import Simian, Entity
from src.kernels import Kernel
from src.post_process import parse_log, parse_out

def usage():
    print("\n[USAGE]\n\
    [option 1] To simulate all kernels of the application:\n\
    python main.py -a <your application path> -c <target GPU hardware configuration> -k <kernel id> -um <useMPI> -l <log> -f <force_delete>\n\n\
    [option 2] To choose a specific kernel, add the kernel id:\n\
    -k <target kernel id>\n\n\
    [MPI] For scalability, add mpirun call before program command:\n\
    mpirun -np <number of processes> python main.py -a <your application path> -c <target GPU hardware configuration> -k <kernel id> -um <useMPI> -l <log> -f <force_delete>\n\n\
    [force_delete] To delete existing outputs and logs directories before running the simulation:\n\
    -f <force_delete>\n\
    [log] To enable logging:\n\
    -l <log>\n\
    [useMPI] To enable MPI support:\n\
    -um <useMPI>\n\
    [kernel id] To specify a kernel id:\n\
    -k <kernel id>\n\
    [target GPU hardware configuration] To specify the target GPU hardware configuration:\n\
    -c <target GPU hardware configuration>\n\
    [application path] To specify the application path:\n\
    -a <your application path>\n\
    Example:\n\
    python main.py -a ./apps/your_app -c A100 -k 1 -um 1 -l 1 -f\n") 

def get_current_kernel_info(kernel_id, app_name, app_path, app_config, log):

    current_kernel_info = {}
    current_kernel_info["app_path"] = app_path
    current_kernel_info["kernel_id"] = kernel_id
    current_kernel_info["log"] = log

    ###########################
    ## kernel configurations ##
    ###########################
    kernel_id = "kernel_"+kernel_id

    try:
        kernel_config = getattr(app_config, kernel_id)
    except:
        print(str("\n[Error]\n<<")+str(kernel_id)+str(">> doesn't exists in app_config file"))
        sys.exit(1)

    try:
        kernel_name = kernel_config["kernel_name"]
    except:
        print(str("\n[Error]\n")+str("\"kernel_name\" configuration is missing"))
        sys.exit(1)
    current_kernel_info["kernel_name"] = kernel_name

    try:
        kernel_smem_size = kernel_config["shared_mem_bytes"]
    except:
        print(str("\n[Error]\n")+str("\"shared_mem_bytes\" configuration is missing"))
        sys.exit(1)
    current_kernel_info["smem_size"] = kernel_smem_size

    try:
        kernel_grid_size = kernel_config["grid_size"]
    except:
        print(str("\n[Error]\n")+str("\"grid_size\" configuration is missing"))
        sys.exit(1)
    current_kernel_info["grid_size"] = kernel_grid_size

    try:
        kernel_block_size = kernel_config["block_size"]
    except:
        print(str("\n[Error]\n")+str("\"block_size\" configuration is missing"))
        sys.exit(1)
    current_kernel_info["block_size"] = kernel_block_size
    
    try:
        kernel_num_regs = kernel_config["num_registers"]
    except:
        print(str("\n[Error]\n")+str("\"num_registers\" configuration is missing"))
        sys.exit(1)
    current_kernel_info["num_regs"] = kernel_num_regs

    ##################
    ## memory trace ##
    ##################
    # mem_trace_file = kernel_id+".mem"
    mem_trace_file = "memory_traces"
    mem_trace_file_path = app_path + mem_trace_file

    if not os.path.exists(mem_trace_file_path):
        print(str("\n[Error]\n")+str("<<memory_traces>> directory doesn't exists in ")+app_name+str(" application directory"))
        sys.exit(1)
    current_kernel_info["mem_traces_dir_path"] = mem_trace_file_path

    ################
    ## ISA Parser ##
    ################
    current_kernel_info["sass_file_path"] = ""

    sass_file = "sass_traces/"+kernel_id+".sass"
    if "/" in app_name:
        sass_file = app_name.split("/")[-1]+"sass_traces/"+kernel_id+".sass"
    sass_file_path = app_path + sass_file

    if not os.path.isfile(sass_file_path):
        print(str("\n[Error]\n")+str("sass instructions trace file: <<")+str(sass_file)+str(">> doesn't exists in ")+app_name +\
                str(" application directory"))
        sys.exit(1)

    current_kernel_info["ISA"] = "SASS"
    current_kernel_info["sass_file_path"] = sass_file_path

    return current_kernel_info

def main():

    all_kernels = False
    useMPI = False
    log = True
    app_path = ""
    app_name = ""
    gpu_config_file = ""
    kernel_id = -1
    kernels_info = []
    instructions_type = "SASS"

    full_cmd_arguments = sys.argv
    argument_list = full_cmd_arguments[1:]
    short_options = "h:a:c:k:um:l:f"
    long_options = ["help", "app=", "config=", "kernel=","useMPI=", "log=", "force_delete"]

    try:
        arguments, values = getopt.getopt(argument_list, short_options, long_options)
    except getopt.error as err:
        print("\n[Error]")
        print(str(err))
        usage()
        sys.exit(1)

    if len(argument_list) == 1:
        for current_argument, current_value in arguments:
            if current_argument in ("-h", "--help"):
                usage()
                sys.exit(2)
    elif len(argument_list) > 11 or len(argument_list) < 2:
        print("\n[Error]\nincorrect number arguments")
        usage()
        sys.exit(1)

    for current_argument, current_value in arguments:
        if current_argument in ("-a", "--app"):
            if not os.path.exists(current_value):
                print("\n[Error]\n<<"+current_value+">> doesn't exists in apps directory")
                sys.exit(1)
            if not os.path.isdir(current_value):
                print("\n[Error]\n<<"+current_value+">> is not a directory")
                sys.exit(1)
            if not os.path.exists(current_value + "/app_config.py"):
                print("\n[Error]\n<<app_config.py>> file doesn't exists in <<"+current_value+">> directory")
                sys.exit(1)
            if not os.path.exists(current_value + "/sass_traces"):
                print("\n[Error]\n<<sass_traces>> directory doesn't exists in <<"+current_value+">> directory")
                sys.exit(1)
            if not os.path.exists(current_value + "/memory_traces"):
                print("\n[Error]\n<<memory_traces>> directory doesn't exists in <<"+current_value+">> directory")
                sys.exit(1)
            if current_value[-1] != '/':
                current_value = current_value + '/'
            app_path = current_value
            app_name = app_path.split('/')[-2]
        elif current_argument in ("-c", "--config"):
            gpu_config_file = current_value
        elif current_argument in ("-k", "--kernel"):
            kernel_id = current_value
        elif current_argument in ("-um", "--useMPI"):
            useMPI = True if current_value == '1' else False
        elif current_argument in ("-l", "--log"):
            log = True if current_value == '1' else False
        elif current_argument in ("-f", "--force_delete"):
            if app_name == "":
                print("\n[Error]\n place -a <your application path> before -f <force_delete>")
                usage()
                sys.exit(1)
            if os.path.exists("./outputs/{}".format(app_name)):
                import shutil
                shutil.rmtree("./outputs/{}".format(app_name))
                print("Deleted existing outputs directory")
            if os.path.exists("./logs{}".format(app_name)):
                import shutil
                shutil.rmtree("./logs{}".format(app_name))
                print("Deleted existing logs directory")
    ######################
    ## specific kernel? ##
    ######################
    if kernel_id == -1:
        all_kernels = True

    ###############
    ## app name ##
    ###############
    sys.path.append(app_path)
    #####################################
    ## target hardware configiguration ##
    #####################################
    try:
        gpu_config_file
    except NameError:
        print("\n[Error]\nmissing target GPU hardware configuration")
        usage()
        sys.exit(1)

    try:
        gpu_configs = importlib.import_module("hardware."+gpu_config_file)
    except:
        print(str("\n[Error]\n")+str("GPU hardware config file provided doesn't exist\n"))
        sys.exit(1)
    
    ##############################
    ## Target ISA Latencies ##
    ##############################
    try:
        print(f"Attempting to import from: hardware.ISA.{gpu_configs.uarch['gpu_arch']}")
        ISA = importlib.import_module("hardware.ISA."+gpu_configs.uarch["gpu_arch"])
    except Exception as e:
        print(f"\n[Error]\nFailed to import ISA for <<{gpu_configs.uarch['gpu_arch']}>>")
        print(f"Error details: {str(e)}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Contents of hardware/ISA directory:")
        print(os.listdir("hardware/ISA"))
        sys.exit(1)
    units_latency = ISA.units_latency
    sass_isa = ISA.sass_isa
    initial_interval = ISA.initial_interval

    gpu_configs.uarch["sass_isa"] = sass_isa
    gpu_configs.uarch["units_latency"] = units_latency
    gpu_configs.uarch["initial_interval"] = initial_interval

    try:
        compute_capability = importlib.import_module("hardware.compute_capability."+str(gpu_configs.uarch["compute_capabilty"]))
    except:
        print("\n[Error]\ncompute capabilty for <<"+gpu_configs.uarch["compute_capabilty"]+">> doesn't exists in hardware/compute_capabilty directory")
        sys.exit(1)

    ##############################
    ## app configiguration file ##
    ##############################
    # 构建 app_config.py 的完整路径
    app_config_path = os.path.join(app_path, 'app_config.py')

    # 使用 importlib 动态加载模块
    spec = importlib.util.spec_from_file_location("app_config", app_config_path)
    app_config = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(app_config)
        # 打印 app_config.py 的绝对路径
        print(f"app_config module path: {os.path.abspath(app_config_path)}")
    except FileNotFoundError:
        print(f"\n[Error]\n<app_config.py> file doesn't exist in \"{app_path}\" directory")
        sys.exit(1)
    except Exception as e:
        print(f"\n[Error]\nFailed to import <app_config.py>: {e}")
        sys.exit(1)

    app_kernels_id = app_config.app_kernels_id
    app_output_dir = app_path.split('/')[-2]
    # if ./outputs not exist then make it
    if not os.path.exists("./outputs"):
        os.makedirs("./outputs")
    if all_kernels == True:    
        if app_output_dir in os.listdir("./outputs"):
            complete_files = os.listdir(os.path.join("./outputs",app_output_dir))
            for file in complete_files:
                cur_id = int(file.split('_')[0])
                if cur_id not in app_kernels_id:
                    continue
                app_kernels_id.remove(cur_id)
        for kernel_id in app_kernels_id:
            kernels_info.append(get_current_kernel_info(str(kernel_id), app_name, app_path, app_config, log))
    else:
        kernels_info.append(get_current_kernel_info(kernel_id, app_name, app_path, app_config, log))

    ############################
    # Simian Engine parameters #
    ############################
    if useMPI:
        
        simianEngine = Simian("Analytical models", useMPI=True, opt=False, appPath = app_path, ISA=instructions_type)
   
        gpuNode = GPUNode(gpu_configs.uarch, compute_capability.cc_configs, len(kernels_info))  
        print('+'+'-'*30)
        for i in range (len(kernels_info)):
            k_id = i 
            # Add Entity and sched Event only if Hash(Entity_name_i) % MPI.size == MPI.rank
            simianEngine.addEntity("Kernel", Kernel, k_id, len(kernels_info), gpuNode, kernels_info[i])
            simianEngine.schedService(1, "kernel_call_GCoM", None, "Kernel", k_id)

        simianEngine.run()
        simianEngine.exit()
    else:
        print(f'Number of Kernels in current app is: {len(kernels_info)}')
        gpuNode = GPUNode(gpu_configs.uarch, compute_capability.cc_configs, len(kernels_info))  
        print('+'+'-'*30)
        for i in range (len(kernels_info)):
            k_id = i 
            cur_kernel = Kernel(k_id, gpuNode, kernels_info[i])
            cur_kernel.kernel_call_GCoM(None, "Kernel", k_id)
    
    print("complete analysis and start parsing output")
    parse_log(app_name)
    parse_out(app_name)
    print("complete parsing")

class GPUNode(object):
	"""
	Class that represents a node that has a GPU
	"""
	def __init__(self, gpu_configs, gpu_configs_cc, num_kernels):
		self.num_accelerators = 1 # modeling a node that has 1 GPU for now
		self.accelerators = []
		self.gpu_configs = gpu_configs
		self.gpu_configs_cc = gpu_configs_cc
		#print("GPU node generated")
		self.generate_target_accelerators(num_kernels)

	#generate GPU accelerators inside the node
	def generate_target_accelerators(self, num_kernels):
		accelerators = importlib.import_module("src.accelerators")
		for i in range(self.num_accelerators):
			self.accelerators.append(accelerators.Accelerator(self, i, self.gpu_configs, self.gpu_configs_cc, num_kernels))

if __name__ == "__main__":
	main()

