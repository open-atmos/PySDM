"""
Created at 18.12.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""


def global_FisherYates(state, cell_start, u01):
    state.unsort_global(u01, temp=cell_start)
    state.sort_by_cell_id(cell_start)


def local_FisherYates(state, cell_start, u01):
    state.sort_by_cell_id(cell_start)
    state.unsort_local(u01, cell_start)


def global_Shima(particles, cell_start, u01):
    # TODO !
    particles.state.sort_by_cell_id(cell_start)
