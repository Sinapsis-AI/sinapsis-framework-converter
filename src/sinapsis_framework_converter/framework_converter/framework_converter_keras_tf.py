# -*- coding: utf-8 -*-

from os.path import join
from pathlib import Path

import tensorflow as tf
from keras import Model
from sinapsis_core.utils.logging_utils import sinapsis_logger

from sinapsis_framework_converter.framework_converter.framework_converter import DLFrameworkConverter, ModelExtensions


class FrameworkConverterKerasTF(DLFrameworkConverter):
    """Module to convert from Keras to TensorFlow framework"""

    def export_keras_to_tf(self, keras_model: Model) -> None:
        """Exports the keras model to TensorFlow
        Args:
            keras_model (Model): Keras model already initialized"""
        tf_model_path_dir = self.model_file_path()
        pb_model_path = Path(
            join(
                tf_model_path_dir,
                f"{self.TF_DEFAULT_SAVE_NAME}{ModelExtensions.TENSORFLOW_FILE_EXTENSION}",
            )
        )
        if not self.force_export(pb_model_path):
            return
        tf.saved_model.save(keras_model, str(tf_model_path_dir.absolute()))
        sinapsis_logger.info(f"Converted keras model to tensorflow, saved in:{tf_model_path_dir.absolute()}")
