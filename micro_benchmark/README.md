generate parameter of kernel launch latency model

```bash
make clean && make
./run_kll_mb.sh
```
The output messages will be some messages like:
```bash
[2144.1615, 2125.6947, 2122.9079, 2082.0757] 
avg intercept 2118.70995486408
quad fit: slope = 1.0956e-03 * block_size^2 + -1.5389e-02 * block_size + 9.3524e-01
R² = 0.9996
```
then the configs of ISA will be 
"kernel_launch_ovhd"    :   2119,
"slope_alpha"   :  1.0956e-03,
"slope_beta"   :  -1.5389e-02,
"slope_gamma"   :  9.3524e-01,
