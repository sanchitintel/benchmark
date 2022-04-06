models="test_bench.py::TestBenchNetwork::test_eval[resnext50_32x4d-cpu-jit] \
test_bench.py::TestBenchNetwork::test_eval[resnet50-cpu-jit] \
test_bench.py::TestBenchNetwork::test_eval[squeezenet1_1-cpu-jit] \
test_bench.py::TestBenchNetwork::test_eval[vgg16-cpu-jit] \
test_bench.py::TestBenchNetwork::test_eval[alexnet-cpu-jit] \
test_bench.py::TestBenchNetwork::test_eval[mobilenet_v2-cpu-jit] \
test_bench.py::TestBenchNetwork::test_eval[mnasnet1_0-cpu-jit] \
test_bench.py::TestBenchNetwork::test_eval[resnet18-cpu-jit] \
test_bench.py::TestBenchNetwork::test_eval[shufflenet_v2_x1_0-cpu-jit] \
test_bench.py::TestBenchNetwork::test_eval[mobilenet_v3_large-cpu-jit]"

pytest --ignore_machine_config $models --fuser llga --dtype fp32 --cpu_only --benchmark-json llga_fp32.json
pytest --ignore_machine_config $models --fuser llga --dtype bf16 --cpu_only --benchmark-json llga_bf16.json
python compare.py llga_fp32.json llga_bf16.json
