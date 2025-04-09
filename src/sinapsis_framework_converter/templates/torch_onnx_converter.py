# -*- coding: utf-8 -*-
from typing import Optional, cast

from torch import nn
from torchvision import models

from sinapsis_framework_converter.framework_converter.framework_converter_torch import (
    FrameworkConverterTorch,
)
from sinapsis_framework_converter.templates.framework_converter_base import (
    FrameworkConverterAttributes,
    FrameworkConverterBase,
)


class TorchONNXConverter(FrameworkConverterBase):
    """Template to convert from TensorFlow to ONNX
    It inherits the functionality from the Base template

    Usage example:

    agent:
      name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
      attributes: {}
    - template_name: TorchONNXConverter
      class_name: TorchONNXConverter
      template_input: InputTemplate
      attributes:
        model_name: 'name_of_the_model_to_convert'
        save_model_path: false
        force_compilation: true
        opset_version: null
        height: 960
        width: 960

    """

    class AttributesBaseModel(FrameworkConverterAttributes):
        """Attributes to instantiate the template
        Apart from those in FrameworkConverterAttributes:
            opset_version (Optional[int]): The operator set version for ONNX.
            height (int): Height for the dummy tensor used to convert the model.
            width (int): Width of the dummy tensor used to convert the model.
        """

        opset_version: Optional[int] = None
        height: int = 960
        width: int = 960

    _EXPORTER = FrameworkConverterTorch

    def load_model(self) -> nn.Module:
        """Loads the torchvision model to be exported based on attributes model_name.

        Returns:
            nn.Module: Model to be exported.
        """
        model = getattr(models, self.attributes.model_name)
        return model()

    def convert_model(self) -> None:
        """Converts model using export_torch_to_onnx method from FrameworkConverterTorch"""
        self.exporter = cast(FrameworkConverterTorch, self.exporter)
        self.exporter.export_torch_to_onnx(torch_model=self.model, opset_version=self.attributes.opset_version)
