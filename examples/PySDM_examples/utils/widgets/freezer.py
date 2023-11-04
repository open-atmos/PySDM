class Freezer:
    def __init__(self, widgets):
        self.widgets = widgets

    def observe(self, *_):
        pass

    @property
    def value(self):
        return self

    def __enter__(self):
        for widget in self.widgets:
            widget.disabled = True
        return self

    def __exit__(self, *args, **kwargs):
        for widget in self.widgets:
            widget.disabled = False
