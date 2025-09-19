#!/bin/bash

# Run ncu profiling with gpc cycles elapsed metrics
ncu --metrics "regex:.*gpc__cycles_elapsed.*" -o kll_repeat_10 -f ./kll

# Export profiling results to CSV
ncu -i kll_repeat_10.ncu-rep --csv --page raw --metrics gpc__cycles_elapsed.avg,gpc__cycles_elapsed.max,gpc__cycles_elapsed.min > kll_repeat_10.csv

# Calculate cycle averages
python cal_cycle_avg.py kll_repeat_10.csv kll_avg.csv

# get the analysis and plots
python kll_analysis.py