from PySDM_examples.utils.widgets import FloatProgress, display


class ProgBarController:
    def __init__(self, description=""):
        self.progress = FloatProgress(
            value=0.0, min=0.0, max=1.0, description=description
        )
        self.panic = False

    def __enter__(self):
        self.set_percent(0)
        display(self.progress)

    def __exit__(self, *_):
        pass

    def set_percent(self, value):
        self.progress.value = value
