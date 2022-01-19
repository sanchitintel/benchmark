# PyTorch Benchmarks for oneDNN Graph evaluation
This is a collection of open source benchmarks used to evaluate PyTorch performance.

`torchbenchmark/models` contains copies of popular or exemplary workloads which have been modified to
(a) expose a standardized API for benchmark drivers, (b) enable JIT for inference,
 (c) contain a miniature version of train/test data and a dependency install script.

This fork is mostly based on changes in an [older fork](https://github.com/chunyuan-w/benchmark/tree/chunyuan/llga_preview2)  of TorchBench. 

## Installation
The benchmark suite should be self contained in terms of dependencies,
except for the torch products which are intended to be installed separately so
different torch versions can be benchmarked.


## Using a low-noise machine
Apart from the instructions mentioned in the TorchBench repo, the only addendum here is that benchmarking some models might entail run-to-run variation. We've observed that using libtcmalloc produces more reproducible results (with less run-to-run variation). We also recommend preloading Intel OpenMP.


## Running Model Benchmarks

It's recommeded that you simply run `compare_llga.sh` to compare (NNC + OFI) JIT performance with oneDNN Graph JIT performance.
Currently, for PyTorch, only FP32 inference is supported by oneDNN Graph, but IPEX (Intel PyTorch Extensions) supports more datatypes with oneDNN Graph. Just FYI, LLGA (Low-Level Graph API) is synonymous with oneDNN Graph.

There are currently two top-level scripts for running the models.

`test.py` offers the simplest wrapper around the infrastructure for iterating through each model and installing and executing it.

`test_bench.py` is a pytest-benchmark script that leverages the same infrastructure but collects benchmark statistics and supports pytest filtering.

In each model repo, the assumption is that the user would already have all of the torch family of packages installed (torch, torchtext, torchvision, ...) but it installs the rest of the dependencies for the model.

### Using `test.py`
`python test.py` will execute the APIs for each model, as a sanity check.  For benchmarking, use `test_bench.py`.  It is based on unittest, and supports filtering via CLI.

For instance, to run the resnet50 model on CPU for inference mode:
```
python test.py -k test_eval[test_resnet50-cpu-jit] -- fuser llga --ignore_machine_config
```



### Using pytest-benchmark driver
`pytest test_bench.py` invokes the benchmark driver.  See `--help` for a complete list of options.


### Running individual models to debug
```
python run.py <model> -d cpu,cuda -m jit -t eval --fuser llga
