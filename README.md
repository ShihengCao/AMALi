# AMALI: An Analytical Model for Accurately Modeling LLM Inference on Modern GPUs

This repository contains the source code for [AMALi \[ISCA '25\]](https://doi.org/10.1145/3695053.3731064). AMALi is a GPU analytical model. The tool is focused on NVIDIA GPUs.

In shorts, how to use AMALi

```bash
# trace apps
LD_PRELOAD=/path/to/AMALi/tracing_tool/tracer.so {exe}
# run analysis 
# simple use. The following will profiling with A100 config and app_name is 'nvidia'
python main.py --app ~/Benchmarks/DeepBench-master/nvidia/ --config A100
python run_post_process.py nvidia # this will parse log and generate .csv file in output/nvidia/
# --app is the path of tracing files, and the last dir will be used as app_name
# -f is force delete prevous outputs and logs of current app, 
# -l control log mode is logging or not (default 1).
python main.py --app {app_path} --config {config_name} --useMPI {0,1} --kernel {kernel_id} -f -l {0,1}
# if do not use MPI
python main.py --app {app_path} --config {config_name} --kernel {kernel_id} -f -l {0,1}
# without kernel_id args, AMALi will analyze all kernel by default
python main.py --app {app_path} --config {config_name} -f -l {0,1}
# or run with mpi, NOT test fully!!
PATH=/path/to/mpich/mpich-install/bin:$PATH ; export PATH
mpiexec -n {parallellism} python main.py --app {app_path} --config {config_name} --useMPI {0,1} --kernel {kernel_id} -f -l {0,1}
```

## Dependencies

### Simulation

- Linux OS
- python v>3.10
  - conda install greenlet joblib
  - pip install -r requirements.txt
- GCC > v5.x tested with 7.3.1 and 9 on centos 8
- make
- glibc
- MPICH v.3.2.3 (if you plan to use the PDES engine to run multiple kernels in parallel)

### Extracting the traces

- A GPU device with compute capability = 3.5 or later
- Software dependencies for extracting the memory traces and the SASS instructions traces are in the ***tracing_tool*** directory

#### see *dependecies* for the packages and versions tested on

## Steps for running  

Running simulation is straightforward. Here are the steps:

1. **Extract the traces of the application**
    - Go to ***tracing_tool*** folder and follow the instructions in the Readme file to build the tracing tool files
    - The ***tracing_tool*** extracts the application memory trace (automatically output a folder named ***memory_traces***) and the application SASS trace (automatically output a folder named ***sass_traces***). It also outputs a configuration file named **app_config.py** that has all information about the application kernels
    - For example, to get the traces for a certain application you have to call the tracer.so file that was built from the ***tracing_tool*** before running the application:

      ```bash
      LD_PRELOAD=/path/to/AMALi/tracing_tool/tracer.so ./2mm.out
      LD_PRELOAD=/path/to/AMALi/tracing_tool/tracer.so python test.py
      ```

2. **Build the Reuse Distance tool**
   - Go to ***reuse_distance_tool*** and follow the instructions in the Readme file to build the code

3. **Modeling the correct GPU configurations**  

    The ***hardware*** folder has an example of multiple hardware configurations. You can choose to model these or define your own in a new file. You can also define the ISA latencies numbers, and the compute capability configurations inside ***hardware/ISA*** and ***hardware/compute_capability***, respectively

4. **Running the simulations**
    - TO RUN:
    ```bash
    python main.py --app {app_path} --config {config_name} --useMPI {0,1} --kernel {kernel_id} -f -l {0,1}
    ```

    For example, running 2mm application on A100 with sass traces. Assuming that 2mm path is *"/home/test/Workloads/2mm"*

    ```bash
    python main.py --app /home/test/Workloads/2mm --config A100 --useMPI 0 -l 1
    ```
    **Kernels are ordered in the *app_config.py* file. Please refer to the file to know the information of kernels and the orders**

5. **Reading the output**

    The performance results are found inside each application file path. Outputs are per kernel. 

    ```bash
    python run_post_process.py 2mm
    ```

  this will generate a csv file

## Papers
- If you find this a helpful tool in your research, please consider citing as:

    ```bibtex
    @inproceedings{Cao2025AMALI,
      author = {Shiheng Cao and Junmin Wu and Junshi Chen and Hong An and Zhibin Yu},
      title = {AMALI: An Analytical Model for Accurately Modeling LLM Inference on Modern GPUs},
      year = {2025},
      booktitle = {Proceedings of the ACM/IEEE International Symposium on Computer Architecture},
      series = {ISCA '25}
    }
    ```

  AMALi is implemented based on the open-source project: [PPT-GPU](https://github.com/lanl/PPT?tab=readme-ov-file) and other projects including [Accel-sim](https://github.com/yonsei-hpcp/gcom), [GCoM](https://github.com/yonsei-hpcp/gcom), [MDM](https://github.com/wanglu1991/MDM-instrumentation)

According to license of PPT-GPU, keep its license

## License

&copy 2017. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration.

All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

Recall that this copyright notice must be accompanied by the appropriate open source license terms and conditions. Additionally, it is prudent to include a statement of which license is being used with the copyright notice. For example, the text below could also be included in the copyright notice file: This is open source software; you can redistribute it and/or modify it under the terms of the Performance Prediction Toolkit (PPT) License. If software is modified to produce derivative works, such modified software should be clearly marked, so as not to confuse it with the version available from LANL. Full text of the Performance Prediction Toolkit (PPT) License can be found in the License file in the main development branch of the repository.
