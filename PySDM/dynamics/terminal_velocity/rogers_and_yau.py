"""
[Rogers & Yau](https://archive.org/details/shortcourseinclo0000roge_m3k2),
equations: (8.5), (8.6), (8.8)
"""


class RogersYau:  # pylint: disable=too-few-public-methods
    def __init__(self, particulator):
        self.particulator = particulator

    def __call__(self, output, radius):
        self.particulator.backend.rogers_and_yau_terminal_velocity(
            values=output.data,
            radius=radius.data,
        )
