class Box:
    def __init__(self, _, dv):
        self.dv = dv

    @property
    def n_cell(self):
        return 1

    def ante_step(self): pass
    def post_step(self): pass
