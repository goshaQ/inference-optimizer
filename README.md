**inference-optimizer** is a simple tool for optimization of TensorFlow computation graphs. It relies on TensorFlow's built-in realization of optimization algorithms, in particural [Graph Transform Tool](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms). In addition, it provides a convinient way to evaluate the obtained optimized version and compare performance metrics with the unoptimized one using a benchmark. Currently only models with ConvNet architecture performing classification task are supported, so there is no guarantee that this tool will work with any other type of network.

## Getting started
These instructions will get you a copy of the project up and running on your local machine.

### Installing
```
git clone https://github.com/goshaQ/inference-optimizer

cd inference-optimizer/
```

### Running
```
bash scripts/optimize_and_benchmark.sh
```

## Optimization process
Here we provide brief description of different transformations that take place during the optimization process. Please see links provided below for more information.
  1. All nodes that are not used during inference (i.e. training-only) are removed.
  2. The remaining nodes that are used but are useless during inference are removed.
  3. Any sub-graph within the model that always evaluate to constant expression is replaced with these constant.
  4. Redundant multiplications introduced after batch normalization are eliminated.
  5. All floating point constants are converted into eight-bit equivalents.
  6. All calculation nodes are replaced with their eight-bit equivalents (if available).

## Results
We tested our tool on three ConvNet models pre-trained on the ImageNet dataset. For inference we used 10 batches 128 images each selected from the ImageNet evaluation dataset. The statistics shown below were gathered by running the benchmark once on the following system:
  * **OS Platform**: Linux Ubuntu 16.04
  * **TensorFlow version**: 1.9.0
  * **TensorFlow installed from:** Source (with SIMD instructions support)
  * **Python version**: 3.6.6
  * **CPU model & RAM**: Intel Core i5-6200U & 8GB
  * **GPU model**: None

### Inception V3
|                  | Before | After  | Comment              |
|------------------|--------|--------|----------------------|
| Inference Time   | 128.7s | 425.8s | 3.3x slower          |
| Accuracy         | 0.890% | 0.849% | 0.041% less accurate |
| Image per Second | 7.5    | 2.3    | 3.3x less throughput |
| Graph Size       | 95mb   | 25mb   | 3.8x lighter         |

### ResNet V2 152
|                  | Before | After   | Comment              |
|------------------|--------|---------|----------------------|
| Inference Time   | 460.5s | 1532.0s | 3.3x slower          |
| Accuracy         | 0.877% | 0.636%  | 0.241% less accurate |
| Image per Second | 2.1    | 0.6     | 3.5x less throughput |
| Graph Size       | 242mb  | 64mb    | 3.8x lighter         |

### PNASNet-5 Large 331
|                  | Before | After   | Comment              |
|------------------|--------|---------|----------------------|
| Inference Time   | 623.0s | 1686.4s | 2.7x slower          |
| Accuracy         | 0.908% | 0.901%  | 0.007% less accurate |
| Image per Second | 1.5    | 0.6     | 2.5x less throughput |
| Graph Size       | 346mb  | 90mb    | 3.8x lighter         |

Note that we encountered significant inference time degradation. There are already several related issues on TensorFlow's GitHub page (i.e. [#2807](https://github.com/tensorflow/tensorflow/issues/2807), [#13939](https://github.com/tensorflow/tensorflow/issues/13939)) which indicates the shortcomings of Graph Transform Tool.

## For more information
  * [Fixed Point Quantization](https://www.tensorflow.org/performance/quantization)
  * [How to Quantize Neural Networks with TensorFlow](https://petewarden.com/2016/05/03/how-to-quantize-neural-networks-with-tensorflow/)
  * [Graph Transform Tool Documentation](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md)
