# PyTorch Benchmarks with oneDNN Graph
This is a fork of TorchBench to evaluate PyTorch performance with oneDNN Graph.


## Installation
Please refer to the main repo for instructions


## Running Model Benchmarks
Please run `compare_llga.sh`, which compares the performance of some models with respect to oneDNN Graph & PyTorch JIT OFI (Optimize For Inference).
The BF16 tests should be run on Xeon Cooper Lake platforms or beyond.


### Using pytest-benchmark driver
`pytest test_bench.py` invokes the benchmark driver.  See `--help` for a complete list of options.

To use oneDNN Graph, please use `--fuser fuser3`.
To use BFloat16, please use `--precision bf16`.
Please refer to `compare_llga.sh` for an example.


### Using `run.py` for simple debugging or profiling
Sometimes you may want to just run train or eval on a particular model, e.g. for debugging or profiling.  Rather than relying on __main__ implementations inside each model, `run.py` provides a lightweight CLI for this purpose, building on top of the standard BenchmarkModel API.

```
python run.py <model> [-d {cpu,cuda}] [-m {eager,jit}] [-t {eval,train}] [--profile] [--precision {bf16, fp32}] [--fuser {fuser3}]
```

### Other requirements
Please preload Intel OpenMP & tcmalloc/jemalloc for experimentation.
We run `compare_llga.sh` like this -

```
KMP_AFFINITY=granularity=fine,verbose,compact,1,0 KMP_BLOCKTIME=1 KMP_SETTINGS=1 MKL_NUM_THREADS=26 OMP_NUM_THREADS=26 numactl --membind=0 --cpunodebind=0 ./compare_llga.sh
```
