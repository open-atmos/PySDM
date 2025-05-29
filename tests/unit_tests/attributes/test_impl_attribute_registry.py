"""checks the attribute registry logic mapping attribute names to classes implementing them
and handling different variants of these implementations picked depending on the choice of
dynamics and formulae options"""

import pytest

from PySDM.attributes import Multiplicity
from PySDM.attributes.impl import get_attribute_class, register_attribute


class TestAttributeRegistry:
    """groups multiple tests to facilitate execution"""

    @staticmethod
    def test_get_attribute_class_ok():
        """checks if inquiring for a valid attribute name yields a valid class"""
        assert get_attribute_class("multiplicity") is Multiplicity

    @staticmethod
    def test_get_attribute_class_fail():
        """checks if inquiring for an invalid attribute name raises an exception"""
        with pytest.raises(ValueError):
            get_attribute_class("lorem ipsum")

    @staticmethod
    def test_get_attribute_class_variant_fail():
        """checks if variant logic properly throws an exception if no variant match found"""

        @register_attribute(name="umaminess", variant=lambda _, __: False)
        class Umaminess:  # pylint: disable=unused-variable,too-few-public-methods
            """Dummy class"""

        with pytest.raises(AssertionError):
            get_attribute_class("umaminess")

    @staticmethod
    def test_get_attribute_class_error_message_hints():
        """check if error message thrown on unknown attribute error contains list of valid names"""
        with pytest.raises(ValueError) as excinfo:
            get_attribute_class("XXX")
        assert "multiplicity" in str(excinfo.value)

    @staticmethod
    def test_register_attribute_fail_on_repeated_name():
        """checks if the decorator raises an exception if name olready used"""

        @register_attribute()
        class A:  # pylint: disable=too-few-public-methods
            """Dummy class"""

        with pytest.raises(ValueError) as exception_info:
            register_attribute()(A)
        assert exception_info.match("already exists")
