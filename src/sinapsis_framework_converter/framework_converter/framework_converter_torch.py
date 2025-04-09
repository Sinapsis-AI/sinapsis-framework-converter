# -*- coding: utf-8 -*-
from typing import Any, Sequence

import tensorrt as trt
import torch
from pydantic import StrictStr
from sinapsis_core.utils.logging_utils import sinapsis_logger
from torch_tensorrt.fx import TRTModule

from sinapsis_framework_converter.framework_converter.framework_converter import (
    DLFrameworkConverter,
)


class FrameworkConverterTorch(DLFrameworkConverter):
    """Module to convert from Torch to ONNX framework"""

    def export_torch_to_onnx(
        self,
        torch_model: torch.nn.Module,
        model_inputs: torch.Tensor | Sequence[torch.Tensor] | None = None,
        opset_version: int | None = None,
        **kwargs: dict[str, Any] | None,
    ) -> None:
        """Method to convert from Torch to ONXX
        Args:
            torch_model (torch.nn.Module) : Initialized Torch model to be
                converted
            model_inputs (torch.Tensor | Sequence[torch.Tensor]| None) :  the model inputs
                such that ``model(*args)`` is a valid invocation of the model
            opset_version (int | None) : Version for the operation set. Higher versions
                allow for a longer time searching for optimization methods
            **kwargs : Extra arguments for the export.
        """

        onnx_model_path = self.onnx_model_file_path()
        if not self.force_export(onnx_model_path):
            return
        torch_model.eval().cpu()
        dummy_input = model_inputs if model_inputs else self._torch_dummy_image_input()
        torch.onnx.export(
            torch_model,
            dummy_input,
            str(onnx_model_path.absolute()),
            opset_version=opset_version,
            **kwargs,
        )
        sinapsis_logger.info(f"Converted pytorch model to onnx, saved in: {onnx_model_path.absolute()}")

    def _torch_dummy_image_input(self, device: str = "cpu") -> torch.Tensor:
        """Creates the dummy torch image used during model conversion. The size of dummy image is specified by the
            height and width attributes.

        Args:
            device (str, optional): Device used to store the torch image. Defaults to "cpu".

        Returns:
            torch.Tensor: The resulting random torch image.
        """
        return torch.randn(1, 3, self.attributes.height, self.attributes.width).to(device)

    @classmethod
    def wrap_trt_engine_with_torch_trt(
        cls,
        engine_path: str,
        ordered_input_names: list[StrictStr],
        ordered_output_names: list[StrictStr],
    ) -> TRTModule:
        """Using the tensorRT Runtime, deserializes the cuda engine from host memory
        Args:
            engine_path (str) :  Path to the model to be deserialized
            ordered_input_names (list[StrictStr]): Key values for the model input.
            ordered_outputnames (list[StrictStr]): : Key valyues for the model outputs.
                Order should match that of the input bindings
        """
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        trt_runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        trt_engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return TRTModule(
            trt_engine,
            input_names=ordered_input_names,
            output_names=ordered_output_names,
        )
