# -*- coding: utf-8 -*-
from functools import partialmethod
from os.path import join
from pathlib import Path

from pydantic.dataclasses import dataclass
from sinapsis_core.template_base.base_models import TemplateAttributes
from sinapsis_core.utils.env_var_keys import SINAPSIS_CACHE_DIR


@dataclass(frozen=True)
class ModelExtensions:
    """Dataclass containing the model
    extensions for the different frameworks
    """

    TORCH_MODEL_EXTENSION: str = "pth"
    TORCH_TRT_MODEL_EXTENSION: str = ".ep"
    TORCH_SCRIPT_MODEL_EXTENSION: str = ".TS"
    TRT_MODEL_EXTENSION: str = ".engine"
    ONNX_FILE_EXTENSION: str = ".onnx"
    TENSORFLOW_FILE_EXTENSION: str = ".pb"
    KERAS_FILE_EXTENSION: str = ".h5"


class DLFrameworkConverter:
    """Base Module for Deep Learning Frameworks converter.
    This module is designed to allow for the easy conversion
    between the different frameworks for deep learning models

    Args:
        PARENT_SAVE_DIR (str) : The parent directory to save the exported model
        TF_DEFAULT_SAVE_NAME (str) :  The default name of the saved model
    """

    PARENT_SAVE_DIR: str = SINAPSIS_CACHE_DIR
    TF_DEFAULT_SAVE_NAME: str = "saved_model"

    def __init__(self, attributes: TemplateAttributes) -> None:
        self.attributes = attributes

    def model_file_path(self, extension: str | None = None) -> Path:
        """Defines the model path by joining the values for model_name and extension"""
        if extension is None:
            extension = ""
        return Path(join(self.PARENT_SAVE_DIR, f"{self.attributes.model_name}{extension}"))

    onnx_model_file_path = partialmethod(model_file_path, ModelExtensions.ONNX_FILE_EXTENSION)

    trt_model_file_path = partialmethod(model_file_path, ModelExtensions.TRT_MODEL_EXTENSION)

    torch_script_model_file_path = partialmethod(model_file_path, ModelExtensions.TORCH_SCRIPT_MODEL_EXTENSION)

    torch_trt_model_file_path = partialmethod(model_file_path, ModelExtensions.TORCH_TRT_MODEL_EXTENSION)

    def force_export(self, model_path: Path) -> bool:
        """Method that defines whether the model needs to be compiled, even if it exists"""
        if self.attributes.force_compilation:
            return True
        return bool(not model_path.exists())
