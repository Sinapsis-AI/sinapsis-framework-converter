# -*- coding: utf-8 -*-
import os
from typing import cast

from keras import Model, models
from sinapsis_core.template_base.base_models import TemplateAttributeType
from sinapsis_core.utils.env_var_keys import SINAPSIS_CACHE_DIR

from sinapsis_framework_converter.framework_converter.framework_converter_keras_tf import (
    FrameworkConverterKerasTF,
)
from sinapsis_framework_converter.helpers.tags import Tags
from sinapsis_framework_converter.templates.framework_converter_base import (
    FrameworkConverterAttributes,
    FrameworkConverterBase,
)

KerasTFConverterUIProperties = FrameworkConverterBase.UIProperties
KerasTFConverterUIProperties.tags.extend([Tags.KERAS, Tags.TENSORFLOW])


class KerasTensorFlowConverter(FrameworkConverterBase):
    """Template to convert between Keras and TensorFlow frameworks
    This module inherits its functionality from the base template
    FrameworkConverterBase
    It receives a model_path targeting to the location for a Keras model

    Usage example:

    agent:
      name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
      attributes: {}
    - template_name: KerasTensorFlowConverter
      class_name: KerasTensorFlowConverter
      template_input: InputTemplate
      attributes:
        model_name: 'name_of_the_model_to_convert'
        save_model_path: false
        force_compilation: true
        model_path: '/path/to/exported/model'


    """

    UIProperties = KerasTFConverterUIProperties

    class AttributesBaseModel(FrameworkConverterAttributes):
        """Additional attributes for the template initialization.
        model_path (str): path where the '.keras' model is saved
        """

        root_dir: str | None = None
        model_path: str

    _EXPORTER = FrameworkConverterKerasTF

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
        self.attributes.root_dir = self.attributes.root_dir or SINAPSIS_CACHE_DIR

    def load_model(self) -> Model:
        """Using the models module from Keras, loads the model from the saved path
        and returns a Model instance"""
        return models.load_model(os.path.join(self.attributes.root_dir, self.attributes.model_path))

    def convert_model(self) -> None:
        """Converts model using export_keras_to_tf method from FrameworkConverterKerasTF"""
        self.exporter = cast(FrameworkConverterKerasTF, self.exporter)
        self.exporter.export_keras_to_tf(self.model)
