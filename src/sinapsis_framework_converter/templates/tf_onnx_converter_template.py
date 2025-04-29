# -*- coding: utf-8 -*-
from typing import Optional, cast

from sinapsis_framework_converter.framework_converter.framework_converter_tf import (
    FrameworkConverterTFONNX,
)
from sinapsis_framework_converter.templates.framework_converter_base import (
    FrameworkConverterAttributes,
    FrameworkConverterBase,
)


class TensorFlowONNXConverter(FrameworkConverterBase):
    """Template to convert from TensorFlow to ONNX


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
        opset_version: null

    """

    _EXPORTER = FrameworkConverterTFONNX

    class AttributesBaseModel(FrameworkConverterAttributes):
        """Attributes for TensorFlowONNXConverter template.

        opset_version (Optional[int]) : The operator set version for ONNX."""

        opset_version: Optional[int] = None

    def load_model(self) -> None:
        """It returns None as there is no model to be
        loaded prior to the export"""

    def convert_model(self) -> None:
        """Method to convert the model using the export_tensorflow_to_onxx method
        from FrameworkConverterTFONNX
        """
        self.exporter = cast(FrameworkConverterTFONNX, self.exporter)
        self.exporter.export_tensorflow_to_onnx(self.attributes.opset_version)
