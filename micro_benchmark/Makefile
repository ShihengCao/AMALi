NVCC = nvcc
CFLAGS = -O2

kll: kernel_launch_latency.cu
	$(NVCC) $(CFLAGS) -o kll kernel_launch_latency.cu

clean:
	rm -f kll

.PHONY: clean