# -*- coding: utf-8 -*-
"""TensorRT/Pytorch module Wrapper"""

from dataclasses import dataclass
from typing import Sequence, TypeAlias

import numpy as np
import tensorrt as trt
import torch
from sinapsis_core.utils.logging_utils import sinapsis_logger

if trt.__version__ < "10.1.0":
    raise RuntimeError(
        "This wrapper only works with newer tensorrt 10.+ versions."
        "For older tensorrt versions, please use TRTModule as done in FrameworkConverterTorch"
    )


@dataclass(frozen=True, slots=True)
class TRTModuleInputBinding:
    """
    Attributes for each model input
        name (str): name of the input
        data_type (torch.dtype | np.dtype) : Type of the data in the input. Can be tensor or numpy
        shape (Sequence[int]) : Shape of the tensor/array
    """

    name: str
    data_type: torch.dtype | np.dtype
    shape: Sequence[int]


@dataclass(frozen=True, slots=True)
class TRTModuleOutputBinding(TRTModuleInputBinding):
    """
    Attributes for the output binding.
        data (torch.Tensor | list[torch.Tensor]): The actual data value.
        data_pointer (int): Position in memory for the data value
    """

    data: torch.Tensor
    data_pointer: int


FORWARD_RETURN_TYPE: TypeAlias = dict[str, torch.Tensor] | tuple[torch.Tensor, ...] | torch.Tensor


class TensorrtTorchWrapper(torch.nn.Module):
    """Wrapper class to integrate TensorRT and Pytorch models, allowing the use of TensorRT
    for optimized execution of Pytorch workflows

    The wrapper handles the TensorRT engine load, setting the context,
    managing input and output bindings in memory to run the inference on a CUDA device

    Attributes:
        - device (torch.device) : The device for the engine to run on. By default, is set to CUDA
        - engine (trt.tensorrt.ICudaEngin): Model loaded in the TensorRT engine
        - context(trt.tensorrt.IExecutionContext) : Context for the enfine with the devide memory
        allocation strategy
        - input_bindings (dict[str, TRTModuleInputBinding]): Dictionary with the model input
        bindings attributes
        - output_bindings (dict[str, TRTModuleOutputBinding]): Dictionary with the model output
        attributes with memory allocation
    """

    def __init__(self, engine_path: str, output_as_value_tuple: bool = False) -> None:
        super().__init__()
        self.device = torch.device("cuda")
        self.engine: trt.tensorrt.ICudaEngine = self.load_trt_engine(engine_path)
        self.context: trt.tensorrt.IExecutionContext = self.engine.create_execution_context()
        self.input_bindings, self.output_bindings = self._get_bindings()
        self.output_binding_ptrs: list[int] = [out.data_pointer for out in self.output_bindings.values()]
        self.output_as_value_tuple = output_as_value_tuple

    @staticmethod
    def load_trt_engine(
        engine_path: str,
    ) -> trt.tensorrt.ICudaEngine:
        """Using the tensorRT Runtime, deserializes the cuda engine from host memory"""
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        trt_runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        trt_engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return trt_engine

    def _get_bindings(
        self,
    ) -> tuple[dict[str, TRTModuleInputBinding], dict[str, TRTModuleOutputBinding]]:
        """
        Parses tensorrt engine input and outputs to retrieve needed information for inference,
         i.e. io name, shape, dtype. For outputs, contiguous torch tensors are allocated and
         their data memory pointers stored.

        Returns:
            Dict of IO TRTModuleIOBindings to be used during inference.

        """
        # engine_bindings: list[int | None] = [None] * self.engine.num_io_tensors
        input_bindings: dict[str, TRTModuleInputBinding] = {}
        output_bindings: dict[str, TRTModuleOutputBinding] = {}

        for tensor_idx in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(tensor_idx)
            tensor_dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            tensor_shape = self.engine.get_tensor_shape(tensor_name)
            binding_tensor = torch.from_numpy(np.empty(tensor_shape, dtype=tensor_dtype)).contiguous().to(self.device)
            is_input = self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT
            if is_input:
                input_bindings[tensor_name] = TRTModuleInputBinding(
                    name=tensor_name,
                    data_type=tensor_dtype,
                    shape=tensor_shape,
                )
            else:
                output_bindings[tensor_name] = TRTModuleOutputBinding(
                    name=tensor_name,
                    data_type=tensor_dtype,
                    shape=tensor_shape,
                    data=binding_tensor,
                    data_pointer=int(binding_tensor.data_ptr()),
                )
        return input_bindings, output_bindings

    def _set_input_bindings(self, input_dict_or_tensor: dict[str, torch.Tensor] | torch.Tensor) -> list[int]:
        """
            Note: It is important to make sure that tensors are contiguous.
            Note: Will raise an error if input shapes are not the same as
            the input bindings. Users should ensure that inputs are
            pre-processed before calling forward.
        Args:
            input_dict_or_tensor: Dictionary containing model input names
            and corresponding input tensor, or single input tensor in the
            case of only one input in the model.

        Returns:
            input_binding_ptrs: list of pointers to the input tensors
        """
        # pending: add batch / dynamic shape support
        if isinstance(input_dict_or_tensor, dict):
            if len(input_dict_or_tensor) != len(self.input_bindings):
                raise ValueError(
                    f"This model takes {len(self.input_bindings)} input tensors, only got {len(input_dict_or_tensor)}"
                )
            input_binding_ptrs = []
            for input_binding_name, input_binding_value in self.input_bindings.items():
                try:
                    in_tensor = input_dict_or_tensor[input_binding_name]
                except KeyError as err:
                    sinapsis_logger.error(
                        f"The input {input_binding_name} for the model is not present in the input."
                        f"This model has the following inputs: {self.input_bindings.keys()}"
                    )
                    raise err
                if in_tensor.shape != input_binding_value.shape:
                    raise ValueError(
                        f"Incorrect shape provided for input `{input_binding_name}`. "
                        f"Expected: {input_binding_value.shape} but got: {in_tensor.shape}. "
                        f"This Module currently does not support batching or dynamic shapes."
                    )

                in_tensor = in_tensor.contiguous().to(self.device)
                input_binding_ptrs.append(in_tensor.data_ptr())

            return input_binding_ptrs

        input_binding = next(iter(self.input_bindings.values()))
        if input_binding.shape != input_dict_or_tensor.shape:
            raise ValueError(
                f"Incorrect shape provided for input tensor. "
                f"Expected: {input_binding.shape} but got: {input_dict_or_tensor.shape} "
            )
        input_dict_or_tensor = input_dict_or_tensor.contiguous().to(self.device)
        return [input_dict_or_tensor.data_ptr()]

    def _make_output_dict(self, clone_tensor: bool) -> FORWARD_RETURN_TYPE:
        """
        To be called after a forward pass to gather the output binding tensors.

        Returns:
             return_data: torch.Tensor | tuple[torch.Tensor]: In the case of a
             single output, it will return the value tensor or a tuple with one
             value tensor if output_as_value_tuple was set in the constructor.
                or
             return_data: dict[str, torch.Tensor] | tuple[torch.Tensor]: In the
             case of multiple outputs, returns a dict containing output names
             as keys and value tensors as values. If output_as_value_tuple was set
                it will return a tuple with the value tensors only.
        """

        def _process_output_tensors(
            io_binding: TRTModuleOutputBinding,
        ) -> torch.Tensor:
            if clone_tensor:
                return io_binding.data.clone()
            return io_binding.data

        if len(self.output_bindings) == 1:
            out_binding: TRTModuleOutputBinding = next(iter(self.output_bindings.values()))  # there is only one output
            return_tensor: torch.Tensor = _process_output_tensors(out_binding)

            if self.output_as_value_tuple:
                return (return_tensor,)
            return return_tensor

        if self.output_as_value_tuple:
            return_list_tuple: tuple[torch.Tensor, ...] = tuple(
                _process_output_tensors(out_binding) for out_binding in self.output_bindings.values()
            )
            return return_list_tuple

        return_data: dict[str, torch.Tensor] = {
            out_name: _process_output_tensors(out_binding) for out_name, out_binding in self.output_bindings.items()
        }
        return return_data

    def forward(
        self,
        input_dict_or_tensor: dict[str, torch.Tensor] | torch.Tensor,
        clone_output_tensor: bool = True,
    ) -> FORWARD_RETURN_TYPE:
        """
        Forward pass of the model with tensorrt.IExecutionContext. It will reuse input tensors as input bindings,
        therefore users are expected to pass copies of these tensors if they want to ensure they are not modified.

        Args:
            input_dict_or_tensor: Single tensor or dict of tensors where keys are input names.
                Input tensors are expected to already be pre-processed.
            clone_output_tensor: Flag indicating whether to clone the output tensors. It is strongly
                recommended to use this flag to avoid unintentional manipulation or overriding of the
                output binding tensors.

        Returns:
            tensor, tuple of tensors, or dict of tensors. see _make_output_dict

        Todo:
            Add dynamic shapes and batching support.
            Add support for execute_async_v3
        """
        # with torch.autograd.profiler.record_function("TensorrtTorchWrapper:Forward"):
        input_binding_ptrs = self._set_input_bindings(input_dict_or_tensor)
        all_bindings = input_binding_ptrs + self.output_binding_ptrs
        self.context.execute_v2(all_bindings)
        return self._make_output_dict(clone_output_tensor)
