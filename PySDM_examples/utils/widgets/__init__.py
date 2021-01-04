import sys
from ipywidgets import (
    Accordion,
    Box,
    Button,
    Checkbox,
    Dropdown,
    FloatProgress,
    FloatSlider,
    HBox,
    HTML,
    IntProgress,
    IntRangeSlider,
    IntSlider,
    Layout,
    Output,
    Play,
    Select,
    Tab,
    VBox,
    interactive_output,
    jslink
)
from IPython.display import (
    clear_output,
    display,
    FileLink
)

# https://github.com/googlecolab/colabtools/issues/1302
if 'google.colab' in sys.modules:
    display(HTML('''<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css"> '''))