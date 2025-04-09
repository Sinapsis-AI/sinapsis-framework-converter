# -*- coding: utf-8 -*-
import typing
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch_tensorrt
from sinapsis_core.utils.logging_utils import sinapsis_logger

from sinapsis_framework_converter.framework_converter.framework_converter import (
    DLFrameworkConverter,
)


class FrameworkConverterTorchTRT(DLFrameworkConverter):
    """Framework converter from PyTorch to TensorRT"""

    def export_torch_to_trt(
        self,
        torch_model: torch.nn.Module,
        model_inputs: torch.Tensor | Sequence[torch.Tensor] | None = None,
        enabled_precision: typing.Iterable[torch.dtype] | None = None,
        optimization_level: int = 5,
        workspace_size: int = 20 << 30,
        require_full_compilation: bool = True,
        **kwargs: dict[str, Any] | None,
    ) -> None:
        """Method to convert from Torch to ONXX
        Args:
            torch_model (torch.nn.Module) : Initialized Torch model to be
                converted
            model_inputs (torch.Tensor | Sequence[torch.Tensor]| None) :  the model inputs
                such that ``model(*args)`` is a valid invocation of the model
            enabled_precision (typing.Iterable[torch.dtype] | None):  The set of datatypes
                that TensorRT can use when selecting kernels
            optimization_level (int)  : Higher values mean TensorRT spends longer engine
                building time searching for optimization options.
            workspace_size (int) :  Maximum size of workspace given to TensorRT
            require_full_compilation (bool): Whether the model should  be compiled even if
                already exists
            **kwargs : Extra arguments for the export.
        """
        enabled_precision = {torch.float16} if enabled_precision is None else set(enabled_precision)

        torch_trt_model_path = self.get_model_path()
        if not self.force_export(torch_trt_model_path):
            return

        torch_model.eval().cuda()
        dummy_input = model_inputs if model_inputs else self._torch_dummy_image_input()
        with torch.no_grad():
            trt_gm = torch_tensorrt.compile(
                torch_model,
                ir="dynamo",
                inputs=dummy_input,
                enabled_precision=enabled_precision,
                optimization_level=optimization_level,
                require_full_compilation=require_full_compilation,
                workspace_size=workspace_size,
                **kwargs,
            )
            self.save_optimised_model(trt_gm, dummy_input)

    def save_optimised_model(
        self,
        optimized_model: (torch.nn.Module | torch.jit.ScriptModule | torch.fx.GraphModule | Callable[..., Any]),
        dummy_input: torch.Tensor | Sequence,
    ) -> None:
        """Method to save the model in a given path
        Args:
            optimized_model ( torch.nn.Module | torch.jit.ScriptModule
                | torch.fx.GraphModule | Callable[..., Any]): Output of the compile method for torch_tensorrt
            dummy_input (torch.Tensor) : Dummy tensor that acts as the input binding for the model
        """
        model_path = self.get_model_path()
        sinapsis_logger.info(f"saving model: {model_path}")
        torch_tensorrt.save(
            optimized_model,
            model_path,
            inputs=dummy_input,
            output_format="exported_program",
        )

    def _torch_dummy_image_input(self, device: str = "cuda") -> list[torch.Tensor]:
        """Creates a dummy torch tensor as the image input
        Args:
            device (str) : The device that the tensor is returned with.
            Options are ''cpu'' and ''cuda'' if available
        """
        return [torch.randn(1, 3, self.attributes.height, self.attributes.width).to(device)]

    @classmethod
    def load_model(cls, engine_path: str) -> torch.nn.Module:
        """loads the a model previously saved using torch.export.save
        Args:
            engine_path (str): model path or bytes-like object containing
            the model
        """
        sinapsis_logger.info(f"loading model: {engine_path}")
        return torch.export.load(engine_path).module()

    def get_model_path(self) -> Path:
        """returns the model path for exported model"""
        return self.torch_trt_model_file_path()


class FrameworkConverterTorchScript(FrameworkConverterTorchTRT):
    """Module to convert from Torch to Torchscript
    Inheris functionality from FrameworkConverterTorchTRT
    """

    def get_model_path(self) -> Path:
        """Returns the model file path for the TorchScript model.

        Returns:
            Path: Model file path for TorchScript model.
        """

        return self.torch_script_model_file_path()

    def save_optimised_model(
        self,
        optimized_model: (torch.nn.Module | torch.jit.ScriptModule | torch.fx.GraphModule | Callable[..., Any]),
        dummy_input: torch.Tensor | Sequence,
    ) -> None:
        """Saves the optimized model in the format ''torchscript''"""
        model_path = self.get_model_path()
        sinapsis_logger.info(f"saving model: {model_path}")
        torch_tensorrt.save(optimized_model, model_path, inputs=dummy_input, output_format="torchscript")

    @classmethod
    def load_model(cls, engine_path: str) -> torch.nn.Module:
        """loads the a model previously saved using torch.jit.save
        Args:
            engine_path (str): model path or bytes-like object containing
            the model
        """
        sinapsis_logger.info(f"loading model: {engine_path}")
        return torch.jit.load(engine_path).cuda()
