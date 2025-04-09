# -*- coding: utf-8 -*-

import subprocess

from sinapsis_core.utils.logging_utils import sinapsis_logger

from sinapsis_framework_converter.framework_converter.framework_converter import (
    DLFrameworkConverter,
)


class FrameworkConverterTFONNX(DLFrameworkConverter):
    """Module to convert from TensorFlow to ONNX"""

    def export_tensorflow_to_onnx(self, opset_version: int | None = None) -> None:
        """Method to export from TensorFlow to ONNX using subprocess
        Args:
            opset_version (int): The version of the operator set."""
        tf_model_path = str(self.model_file_path().absolute())
        onnx_model_path = self.onnx_model_file_path()
        if not self.force_export(onnx_model_path):
            return
        # easier to call sub process than duplicating code from tf2onnx.convert
        command = [
            "python3",
            "-m",
            "tf2onnx.convert",
            "--saved-model",
            tf_model_path,
            "--opset",
            str(opset_version),
            "--output",
            str(onnx_model_path.absolute()),
        ]
        subprocess.call(command)
        sinapsis_logger.info(f"Converted tensorflow model to onnx, saved in: {onnx_model_path.absolute()}")
