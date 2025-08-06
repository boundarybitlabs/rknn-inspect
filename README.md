# rknn-inspect

CLI tool for inspecting RKNN models.

> `rknn-inspect` is a command-line utility for inspecting and profiling Rockchip RKNN models. It helps developers understand model structure, input/output shapes and formats, SDK version info, and performance metrics. Useful during model development, conversion, and deployment on Rockchip NPUs.

## Usage

```shell
rknn-inspect --help
Usage: rknn-inspect [OPTIONS] <MODEL_PATH>

Arguments:
  <MODEL_PATH>  Path to the model file

Options:
      --lib-index <LIB_INDEX>  Which library path to use [default: 0]
  -i, --io                     Show inputs and outputs
  -n, --native-io              Show native input/output information
  -p, --perf                   Enable performance profiling
  -s, --sdk                    Show SDK information
      --npu-cores <NPU_CORES>  Select NPU cores to use [default: auto] [possible values: core0, core1, core2, core-all, core0-1, core0-1-2, auto]
      --markdown               Output in Markdown format
  -h, --help                   Print help
```

## Installation
PyPI Coming Soon.

## Extended Usage

### View the input/output names, shapes, data types, and formats
```sh
rknn-inspect --io model.rknn
```

### View Performance Metrics
```sh
rknn-inspect --perf model.rknn
```

### View Performance Metrics for Specific NPU Core/s
```sh
rknn-inspect --npu-cores core0 --perf model.rknn
```
