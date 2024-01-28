""" borrowing everything from unit_tests (this cannot be one level up due to devops_tests) """

import pytest

from ..unit_tests.conftest import backend_class

pytest.fixture(backend_class.__wrapped__)
