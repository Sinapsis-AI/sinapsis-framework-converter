# -*- coding: utf-8 -*-

from typing import Sequence, cast

from torch import Tensor, nn, randn
from torchvision import models

from sinapsis_framework_converter.framework_converter.framework_converter_torch_trt import (
    FrameworkConverterTorchTRT,
)
from sinapsis_framework_converter.helpers.tags import Tags
from sinapsis_framework_converter.templates.framework_converter_base import (
    FrameworkConverterAttributes,
    FrameworkConverterBase,
)

TorchTRTConverterUIProperties = FrameworkConverterBase.UIProperties
if TorchTRTConverterUIProperties.tags is not None:
    TorchTRTConverterUIProperties.tags.extend([Tags.PYTORCH, Tags.TENSORRT])


class TorchTRTConverter(FrameworkConverterBase):
    """Template to convert from PyTorch to TensorRt
    It inherits the functionality from the Base template

    Usage example:

    agent:
      name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
      attributes: {}
    - template_name: TorchTRTConverter
      class_name: TorchTRTConverter
      template_input: InputTemplate
      attributes:
        model_name: 'name_of_the_model_to_convert'
        save_model_path: false
        force_compilation: true
        optimization_level: 5
        workspace_size: 21474836480
        require_full_compilation: true
        height: 940
        width: 940


    """

    UIProperties = TorchTRTConverterUIProperties
    _EXPORTER = FrameworkConverterTorchTRT

    class AttributesBaseModel(FrameworkConverterAttributes):
        """Attributes to instantiate the framework converter.
        Apart from the attributes defined in the Base class:
           - optimization_level (int): level of optimization for the engine.
            Higher values mean TensorRT spends longer engine building time searching
            for optimization options.
           - workspace_size (int): Maximum size of workspace given to TensorRT
           - require_full_compilation: Whether or not the modules need to be compiled
                end - to - end.
           - height (int): height of the dummy tensor used to make the conversion
                between frameworks
           - width (int) : width of the dummy tensor used to make the conversion
                between frameworks
        """

        optimization_level: int = 5
        workspace_size: int = 20 << 30
        require_full_compilation: bool = True
        height: int = 940
        width: int = 940

    attributes: AttributesBaseModel

    def load_model(self) -> nn.Module:
        """Loads the torchvision model to be exported based on attributes model_name.

        Returns:
            nn.Module: Model to be exported.
        """

        model = getattr(models, self.attributes.model_name)
        return model()

    def convert_model(self) -> None:
        """
        Converts model using the export_torch_to_trt method from FrameworkConverterTorchTRT
        """
        self.exporter = cast(FrameworkConverterTorchTRT, self.exporter)
        self.exporter.export_torch_to_trt(
            torch_model=self.model,
            optimization_level=self.attributes.optimization_level,
            workspace_size=self.attributes.workspace_size,
            require_full_compilation=self.attributes.require_full_compilation,
        )

    def _torch_dummy_image_input(self, device: str = "cuda") -> Sequence[Tensor]:
        """Creates a dummy torch tensor as the image input
        Args:
            device (str) : The device that the tensor is returned with.
            Options are ''cpu'' and ''cuda'' if available
        """
        return [randn(1, 3, self.attributes.height, self.attributes.width).to(device)]
