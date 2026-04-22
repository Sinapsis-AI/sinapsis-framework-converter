<h1 align="center">
<br/>
<br/>
<a href="https://sinapsis.tech/">
<img alt="" src="https://github.com/Sinapsis-AI/brand-resources/blob/main/sinapsis_logo/4x/logo.png?raw=true" width="300"/>
</a>
<br/>Sinapsis Framework Converter: Conversión de marcos de trabajo
<br/>
</h1>

<h4 align="center">Plantillas para la conversión entre marcos de trabajo de aprendizaje profundo.</h4>

<p align="center">
<a href="#installation">🐍 Instalación</a> •
<a href="#features">🚀 Características</a>
<a href="#documentation">📙 Documentación</a> •
<a href="#license">🔍 Licencia</a>
</p>

El <code>sinapsis-framework-converter</code> módulo permite la conversión entre algunos de los marcos de trabajo de aprendizaje profundo más populares en la comunidad:

- Keras -> Tensorflow
- Tensorflow -> ONNX*
- Pytorch -> TensorRT*
- Pytorch -> ONNX*
- ONNX -> Propiedad TensorRT

<h2 id="installation">🐍 Instalación</h2>
<blockquote>

[!NOTE]
Las plantillas basadas en CUDA en Sinapsis-framework-converter requieren que la versión de controlador NVIDIA sea 550 o superior.

</blockquote>

Instala el administrador de tu paquete de elección. Alentamos el uso de <code>uv</code>

Ejemplo con <code>uv</code>:

```bash
uv pip install sinapsis-framework-converter --extra-index-url https://pypi.sinapsis.tech
```

o con solo <code>pip</code>:

```bash
pip install sinapsis-framework-converter --extra-index-url https://pypi.sinapsis.tech
```

<blockquote>

[!IMPORTANT]
Las plantillas en cada paquete pueden requerir dependencias adicionales. Para el desarrollo, recomendamos instalar el paquete con todas las dependencias opcionales:

</blockquote>

Ejemplo con <code>uv</code>:

```bash
uv pip install sinapsis-framework-converter[all] --extra-index-url https://pypi.sinapsis.tech
```

o con solo <code>pip</code>:

```bash
pip install sinapsis-framework-converter[all] --extra-index-url https://pypi.sinapsis.tech
```

<blockquote>

[!IMPORTANT]
Para habilitar Tensorflow con soporte cuda por favor instala <code>tensorflow</code> como sigue:

</blockquote>

```bash
uv pip install tensorflow[and-cuda]==2.18.0
```

o

```bash
pip install tensorflow[and-cuda]==2.18.0
```

<h2 id="features">🚀 Características</h2>
<h3> Plantillas soportadas</h3>

El **Sinapsis Framework Converter** El módulo proporciona múltiples plantillas para la conversión del marco de aprendizaje profundo.

* **KerasTensorFlowConverter**: Convierte los modelos Keras en TensorFlow.
* **ONNXTRTConverter**: Convierte modelos ONNX en TensorRT.
* **TensorFlowONNXConverter**: Convertir Tensor Modelos de flujo a ONNX.
* **TorchONNXConverter**: Convierte modelos PyTorch a ONNX.
* **TorchTRTConverter**: Convertir modelos PyTorch en TensorRT.

</details>

<details>
<summary><strong><span style="font-size: 1.25em;">Ejemplo de uso</span></strong></summary>

El siguiente ejemplo demuestra cómo utilizar la plantilla **TorchONNXConverter** para convertir un modelo PyTorch al formato ONNX. La configuración establece un agente con las plantillas necesarias para cargar un modelo, convertirlo y almacenar el archivo convertido. A continuación se encuentra la configuración completa de YAML, seguida de un desglose de cada componente.

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

Esta configuración define un **Agente** y una secuencia de **plantillas** para realizar la conversión de modelo.

1.  **Manejo de entrada (<code>InputTemplate</code>)**: Esto sirve como plantilla inicial.
2. **Conversión modelo (<code>TorchONNXConverter</code>)**:
    - Carga un modelo PyTorch (por ejemplo, <code>resnet50</code>) y lo convierte en formato ONNX. La plantilla:
    - Usa el atributo **<code>model_name</code>** para especificar qué modelo PyTorch para convertir.
    - Aplica el **<code>opset_version</code>**
atributo para definir la versión de configuración del operador ONNX (por ejemplo, <code>12</code>).
    - Ajusta las dimensiones del tensor de entrada utilizando **<code>height</code>**
y **<code>width</code>
**.
    - Habilita **<code>force_compilation</code>**
para asegurar que el modelo se recompile si es necesario.
3.  **Guardar modelo convertido**: El atributo **<code>save_model_path</code>**
se establece <code>true</code>, asegurando que la salida del modelo ONNX se guarda en el contenedor de datos.

<h2 id="documentation">📙 Documentación</h2>

La documentación está disponible en el sitio <a href="https://docs.sinapsis.tech/docs">web de sinapsis</a>

Los tutoriales para diferentes proyectos dentro de sinapsis están disponibles en <a href="https://docs.sinapsis.tech/tutorials">la página de tutoriales de sinapsis</a>

<h2 id="license">🔍 Licencia</h2>

Este proyecto está licenciado bajo la licencia AGPLv3, que fomenta la colaboración abierta y el intercambio. Para más detalles, consulta el <a href="LICENSE">LICENSE</a> archivo.

Para uso comercial, consulta nuestra página <a href="https://sinapsis.tech">Sitio web de Sinapsis</a> para información sobre la obtención de una licencia comercial.

