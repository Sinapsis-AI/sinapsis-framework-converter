# -*- coding: utf-8 -*-
from typing import cast

from sinapsis_framework_converter.framework_converter.framework_converter_trt import (
    FrameworkConverterTRT,
)
from sinapsis_framework_converter.templates.framework_converter_base import (
    FrameworkConverterBase,
)


class ONNXTRTConverter(FrameworkConverterBase):
    """Template to convert between ONNX and TensorRT frameworks
    This module inherits its functionality from the base template
    FrameworkConverterBase.

    Usage example:

    agent:
      name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
      attributes: {}
    - template_name: ONNXTRTConverter
      class_name: ONNXTRTConverter
      template_input: InputTemplate
      attributes:
        model_name: 'name_of_the_model_to_convert'
        save_model_path: false
        force_compilation: true


    """

    _EXPORTER = FrameworkConverterTRT

    def load_model(self) -> None:
        """It returns None as there is no model to be
        loaded prior to the export"""

    def convert_model(self) -> None:
        """Converts model using export_onnx_to_trt method from FrameworkConverterTRT"""
        self.exporter = cast(FrameworkConverterTRT, self.exporter)
        self.exporter.export_onnx_to_trt()
