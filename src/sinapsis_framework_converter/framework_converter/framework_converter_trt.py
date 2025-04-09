# -*- coding: utf-8 -*-

from pathlib import Path

from polygraphy.backend.trt import (
    CreateConfig,
    SaveEngine,
    engine_from_network,
    network_from_onnx_path,
)
from sinapsis_core.utils.logging_utils import sinapsis_logger

from sinapsis_framework_converter.framework_converter.framework_converter import (
    DLFrameworkConverter,
)


class FrameworkConverterTRT(DLFrameworkConverter):
    """Module for export ONNX to TRT"""

    def export_onnx_to_trt(self) -> None:
        """Method to be called by the exporter to
        convert from ONNX to TRT"""
        trt_model_path: Path = self.trt_model_file_path()
        if not self.force_export(trt_model_path):
            return

        onnx_model_path = str(self.onnx_model_file_path().absolute())
        engine = engine_from_network(
            network_from_onnx_path(onnx_model_path),
            config=CreateConfig(
                fp16=True,
            ),
        )
        SaveEngine(engine, str(trt_model_path.absolute()))()
        sinapsis_logger.info(f"Exported onnx model to tensorrt engine, saved in: {trt_model_path.absolute()}")
