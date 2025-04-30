# -*- coding: utf-8 -*-
from abc import abstractmethod
from typing import Any, Type

from sinapsis_core.data_containers.data_packet import DataContainer
from sinapsis_core.template_base import Template
from sinapsis_core.template_base.base_models import OutputTypes, TemplateAttributes, UIPropertiesMetadata
from sinapsis_core.utils.logging_utils import sinapsis_logger

from sinapsis_framework_converter.framework_converter.framework_converter import (
    DLFrameworkConverter,
)


class FrameworkConverterAttributes(TemplateAttributes):
    """Attributes for base FrameworkConverter template.

    model_name (str): The model name for the exported model.
    save_model_path (bool): Flag to enable the storage of the exported model path in DataContainer generic data.
        Defaults to False.
    force_compilation (bool): Flag to enable model compilation even if exported model already exists. Defaults to True.

    """

    model_name: str
    save_model_path: bool = False
    force_compilation: bool = True


class FrameworkConverterBase(Template):
    """
    Base module for model conversion between different frameworks.

    This class defines the foundation for converting models from one
    deep learning framework to another. It relies on specific attributes
    and an exporter to handle the conversion process.

    Attributes:
    AttributesBaseModel (TemplateAttributes): The attribute class that holds
                specific parameters needed for conversion.
    _EXPORTER (DLFrameworkConverter): The class responsible for exporting
                the model to a different framework.

    """

    AttributesBaseModel = FrameworkConverterAttributes
    _EXPORTER: Type[DLFrameworkConverter]
    UIProperties = UIPropertiesMetadata(category="ModelConversion", output_type=OutputTypes.MULTIMODAL)

    def __init__(self, attributes: dict[str, Any]) -> None:
        """
        Initialize the FrameworkConverterBase with given attributes.

        Args:

            attributes (dict[str, Any]) :A dictionary containing
                configuration values for the converter.
        """
        super().__init__(attributes)

        self.model = self.load_model()
        self.exporter = self.load_exporter()

    @abstractmethod
    def load_model(self) -> Any:
        """
        Abstract method for loading the model to be converted.

        This method must be implemented in subclasses to handle loading
        the model based on the specific framework or input format.

        Returns:

            The loaded model instance.
        """

    def load_exporter(self) -> DLFrameworkConverter:
        """
        Load and initialize the model exporter.

        The exporter is responsible for saving the converted model
        in the desired format or framework.

        Returns:

            An initialized exporter instance based on the provided attributes.
        """
        exporter: DLFrameworkConverter = self._EXPORTER(self.attributes)
        return exporter

    @abstractmethod
    def convert_model(self) -> None:
        """
        Abstract method for converting the loaded model to a different framework.

        This method must be implemented in subclasses to define the
        conversion logic specific to the source and target frameworks.
        """

    def execute(self, container: DataContainer) -> DataContainer:
        """
        Execute the model conversion process and update the DataContainer,
        using the convert_model method. If a save path is specified in the
        attributes, the method updates the DataContainer with the path to the
        exported model.

        Args:
        container((DataContainer): Container received by the template

        Returns:
            DataContainer:  The updated DataContainer with information about the exported model.
        """
        sinapsis_logger.debug(f"Attempting to export model using {self.class_name}")
        self.convert_model()
        if self.attributes.save_model_path:
            self._set_generic_data(container, str(self.exporter.model_file_path().absolute()))

        return container
