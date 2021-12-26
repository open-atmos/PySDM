# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
from PySDM_examples.Arabas_et_al_2015 import Settings
from PySDM_examples.Szumowski_et_al_1998.gui_settings import GUISettings


class TestGUISettings:
    @staticmethod
    def test_instantiate():
        _ = GUISettings(Settings())

    @staticmethod
    def test_stream_function():
        # arrange
        gui_settings = GUISettings(Settings())
        gui_settings.ui_rhod_w_max = None
        failed = False

        # act
        try:
            _ = gui_settings.stream_function(0, 0, 0)
        except AttributeError:
            failed = True

        # assert
        assert failed
