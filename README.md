# AMALi: An Analytical Model for Accurately Modeling LLM Inference on Modern GPUs

This repository contains the source code for [AMALi (ISCA '25)](https://doi.org/10.1145/3695053.3731064). AMALi is a GPU analytical model focused on NVIDIA GPUs.

## Citation

If you find this tool helpful in your research, please consider citing:

```bibtex
@inproceedings{Cao2025AMALI,
  author = {Shiheng Cao and Junmin Wu and Junshi Chen and Hong An and Zhibin Yu},
  title = {AMALI: An Analytical Model for Accurately Modeling LLM Inference on Modern GPUs},
  year = {2025},
  booktitle = {Proceedings of the ACM/IEEE International Symposium on Computer Architecture},
  series = {ISCA '25}
}
```

## Requirements

- OS: Linux
- Python >= 3.10
  - `pip install -r requirements.txt`
- GCC >= 5.x (tested with 7.3.1 and 9 on CentOS 8)
- `make`, `glibc`
- (Optional) MPICH 3.2.3 (if you plan to use the PDES engine / MPI mode; not fully tested)

For tracing tool dependencies (CUDA/NVBit/driver constraints), see `tracing_tool/README.md`.

## Quickstart

Run the following commands from the `AMALi/` directory (this README's directory).

### 0) Install Python dependencies

```bash
pip install -r requirements.txt
```

### 1) Trace an application

Run the target program with the tracer preloaded. The tracer generates `memory_traces/`, `sass_traces/`, and `app_config.py` in the current working directory.

```bash
LD_PRELOAD=/path/to/AMALi/tracing_tool/tracer.so <your_exe>
```

### 2) Build the reuse-distance tool

```bash
cd reuse_distance_tool && make
```

### 3) Run analysis

`--app` should point to the *trace directory* (it must contain `app_config.py`, `memory_traces/`, and `sass_traces/`).

```bash
# A concrete example path: analyze traces under ~/Benchmarks/DeepBench/nvidia/gemm/ with A100 config.
python main.py --app ~/Benchmarks/DeepBench/nvidia/gemm --config A100
```

By default, `app_name` is the last directory name of `--app` (here: `gemm`). Logs and outputs are written to `logs/<app_name>/` and `outputs/<app_name>/` under this repo.

### 4) Post-process results

```bash
python run_post_process.py gemm
```

This parses logs and generates `.csv` files under `outputs/gemm/`.

## CLI Reference

```text
--app <path>        Trace directory path (must contain app_config.py, memory_traces/, sass_traces/)
--config <name>     GPU config name (e.g., A100)
--kernel <id>       Analyze a specific kernel; omit to analyze all kernels; !! Without --kernel, AMALi will filter out the kernels that have already been analyzed based on the output folder.
--name <name>       Override app_name (defaults to last dir name of --app)
-l, --log {0,1}     Enable logging (default 1)
-f, --force_delete  Remove existing outputs/logs for this app_name
--useMPI {0,1}      MPI support (experimental / not fully tested)
```

## MPI (Experimental)

If you want to try running with MPI, prepend `mpiexec` and enable `--useMPI 1` (not fully tested):

```bash
mpiexec -n <parallelism> python main.py --app <app_path> --config <config_name> --useMPI 1 --kernel <kernel_id> -f -l 1
```

## Result Explanation

```text
AMALi = selected + wait + drain + long_scoreboard + short_scoreboard +
        math_pipe_throttle + lg_throttle + mio_throttle +
        C_idle_i_ID + no_instructions_and_imc_miss

mio_throttle = S_MSHR_i + S_NoC_i + S_Dram_i
```

## License

AMALi is implemented based on the open-source project: [PPT-GPU](https://github.com/lanl/PPT?tab=readme-ov-file) and other projects including [Accel-sim](https://github.com/accel-sim/accel-sim-framework), [GCoM](https://github.com/yonsei-hpcp/gcom), [MDM](https://github.com/wanglu1991/MDM-instrumentation).

According to license of PPT-GPU, keep its license:

&copy; 2017. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration.

All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

Recall that this copyright notice must be accompanied by the appropriate open source license terms and conditions. Additionally, it is prudent to include a statement of which license is being used with the copyright notice. For example, the text below could also be included in the copyright notice file: This is open source software; you can redistribute it and/or modify it under the terms of the Performance Prediction Toolkit (PPT) License. If software is modified to produce derivative works, such modified software should be clearly marked, so as not to confuse it with the version available from LANL. Full text of the Performance Prediction Toolkit (PPT) License can be found in the License file in the main development branch of the repository.
