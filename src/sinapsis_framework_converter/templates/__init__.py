# -*- coding: utf-8 -*-

from typing import Callable


def __getattr__(name: str) -> Callable:
    match name:
        case "KerasTensorFlowConverter":
            from sinapsis_framework_converter.templates.keras_tf_converter_template import (
                KerasTensorFlowConverter as returnModule,
            )
        case "ONNXTRTConverter":
            from sinapsis_framework_converter.templates.onnx_trt_converter_template import (
                ONNXTRTConverter as returnModule,
            )
        case "TensorFlowONNXConverter":
            from sinapsis_framework_converter.templates.tf_onnx_converter_template import (
                TensorFlowONNXConverter as returnModule,
            )
        case "TorchONNXConverter":
            from sinapsis_framework_converter.templates.torch_onnx_converter import (
                TorchONNXConverter as returnModule,
            )
        case "TorchTRTConverter":
            from sinapsis_framework_converter.templates.torch_trt_converter_template import (
                TorchTRTConverter as returnModule,
            )

        case _:
            raise AttributeError(f"module {__name__!r} has no template {name!r}")
    return returnModule


__all__ = [
    "KerasTensorFlowConverter",
    "ONNXTRTConverter",
    "TensorFlowONNXConverter",
    "TorchONNXConverter",
    "TorchTRTConverter",
]
