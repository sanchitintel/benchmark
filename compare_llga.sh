export OMP_NUM_THREADS=1

models="test_bench.py::TestBenchNetwork::test_eval[resnext50_32x4d-cpu-jit] \
test_bench.py::TestBenchNetwork::test_eval[resnet50-cpu-jit] \
test_bench.py::TestBenchNetwork::test_eval[densenet121-cpu-jit] \
test_bench.py::TestBenchNetwork::test_eval[squeezenet1_1-cpu-jit] \
test_bench.py::TestBenchNetwork::test_eval[vgg16-cpu-jit] \
test_bench.py::TestBenchNetwork::test_eval[alexnet-cpu-jit] \
test_bench.py::TestBenchNetwork::test_eval[mobilenet_v2-cpu-jit] \
test_bench.py::TestBenchNetwork::test_eval[mnasnet1_0-cpu-jit] \
test_bench.py::TestBenchNetwork::test_eval[timm_resnest-cpu-jit] \
test_bench.py::TestBenchNetwork::test_eval[resnet18-cpu-jit] \
test_bench.py::TestBenchNetwork::test_eval[shufflenet_v2_x1_0-cpu-jit] \
test_bench.py::TestBenchNetwork::test_eval[mobilenet_v3_large-cpu-jit] \
test_bench.py::TestBenchNetwork::test_eval[timm_regnet-cpu-jit] \
test_bench.py::TestBenchNetwork::test_eval[timm_vision_transformer-cpu-jit] \
test_bench.py::TestBenchNetwork::test_eval[timm_efficientnet-cpu-jit] \
test_bench.py::TestBenchNetwork::test_eval[timm_vovnet-cpu-jit]"

python3.8 -m pytest --ignore_machine_config $models --fuser llga --dynamic_bs 128 --cpu_only --benchmark-json llga_dynamic.json
python3.8 -m pytest --ignore_machine_config $models --fuser llga --cpu_only --benchmark-json llga_static.json
python3.8 -m  compare.py llga_dynamic.json llga_static.json
