<h1 align="center">
<br>
<br>
<a href="https://sinapsis.tech/">
  <img
    src="https://github.com/Sinapsis-AI/brand-resources/blob/main/sinapsis_logo/4x/logo.png?raw=true"
    alt="" width="300">
</a>
<br>
Sinapsis Framework Converter
<br>
</h1>

<h4 align="center">Templates for conversion between deep learning frameworks.</h4>

<p align="center">
<a href="#installation">🐍 Installation</a> •
<a href="#features">🚀 Features</a>
<a href="#documentation">📙 Documentation</a> •
<a href="#license">🔍 License</a>
</p>

The `sinapsis-framework-converter` module allows for the conversion between some of the most popular deep learning frameworks in the community:
- Keras -> Tensorflow
- Tensorflow -> ONNX
- Pytorch -> TensorRT
- Pytorch -> ONNX
- ONNX -> TensorRT






<h2 id="installation">🐍 Installation</h2>

> [!NOTE]
> CUDA-based templates in Sinapsis-framework-converter require NVIDIA driver version to be 550 or higher.

Install using your package manager of choice. We encourage the use of `uv`

Example with <code>uv</code>:

```bash
uv pip install sinapsis-framework-converter --extra-index-url https://pypi.sinapsis.tech
```
or with raw <code>pip</code>:
```bash
pip install sinapsis-framework-converter --extra-index-url https://pypi.sinapsis.tech
```

> [!IMPORTANT]
> Templates in each package may require extra dependencies. For development, we recommend installing the package with all the optional dependencies:
>

Example with <code>uv</code>:

```bash
uv pip install sinapsis-framework-converter[all] --extra-index-url https://pypi.sinapsis.tech
```
or with raw <code>pip</code>:
```bash
pip install sinapsis-framework-converter[all] --extra-index-url https://pypi.sinapsis.tech
```

> [!IMPORTANT]
> To enable tensorflow with cuda support please install `tensorflow` as follows:
>
```bash
uv pip install tensorflow[and-cuda]==2.18.0
```
or
```bash
pip install tensorflow[and-cuda]==2.18.0
```



</details>
<h2 id="features">🚀 Features</h2>

<h3> Templates Supported</h3>

The **Sinapsis Framework Converter** module provides multiple templates for deep learning framework conversion.

- **KerasTensorFlowConverter**: Converts Keras models to TensorFlow.
- **ONNXTRTConverter**: Converts ONNX models to TensorRT.
- **TensorFlowONNXConverter**: Converts TensorFlow models to ONNX.
- **TorchONNXConverter**: Converts PyTorch models to ONNX.
- **TorchTRTConverter**: Converts PyTorch models to TensorRT.


<details>
<summary><strong><span style="font-size: 1.25em;">▶️ Example Usage</span></strong></summary>

The following example demonstrates how to use the **TorchONNXConverter** template to convert a PyTorch model into the ONNX format. The configuration sets up an agent with the necessary templates to load a model, convert it, and store the converted file. Below is the full YAML configuration, followed by a breakdown of each component.

```yaml
agent:
  name: conversion_agent

templates:
- template_name: InputTemplate
  class_name: InputTemplate
  attributes: {}

- template_name: TorchONNXConverter
  class_name: TorchONNXConverter
  template_input: InputTemplate
  attributes:
    model_name: resnet50
    save_model_path: true
    force_compilation: true
    opset_version: 12
    height: 224
    width: 224

```
This configuration defines an **agent** and a sequence of **templates** to perform model conversion.

1. **Input Handling (`InputTemplate`)**: This serves as the initial template.
2. **Model Conversion (`TorchONNXConverter`)**: Loads a PyTorch model (e.g., `resnet50`) and converts it to ONNX format. The template:
   - Uses the **`model_name`** attribute to specify which PyTorch model to convert.
   - Applies the **`opset_version`** attribute to define the ONNX operator set version (e.g., `12`).
   - Adjusts the input tensor dimensions using **`height`** and **`width`**.
   - Enables **`force_compilation`** to ensure the model is recompiled if needed.
3. **Saving the Converted Model**: The **`save_model_path`** attribute is set to `true`, ensuring that the output ONNX model path is saved in the DataContainer.

</details>

<h2 id="documentation">📙 Documentation</h2>

Documentation is available on the [sinapsis website](https://docs.sinapsis.tech/docs)

Tutorials for different projects within sinapsis are available at [sinapsis tutorials page](https://docs.sinapsis.tech/tutorials)

<h2 id="license">🔍 License</h2>

This project is licensed under the AGPLv3 license, which encourages open collaboration and sharing. For more details, please refer to the [LICENSE](LICENSE) file.

For commercial use, please refer to our [official Sinapsis website](https://sinapsis.tech) for information on obtaining a commercial license.