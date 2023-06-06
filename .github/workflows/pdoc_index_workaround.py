"""
workaround for https://github.com/pdoc3/pdoc/issues/101#issuecomment-526957342
extracted from pdoc's cli.py
"""

import inspect
from pdoc import import_module, _render_template


modules = [
    import_module(module, reload=False)
    for module in ('PySDM', 'examples/PySDM_examples')
]

with open('html/index.html', 'w', encoding='utf-8') as index:
    index.write(
        _render_template(
            '/html.mako',
            modules=sorted((module.__name__, inspect.getdoc(module)) for module in modules)
        )
    )
