import subprocess
import os

from datetime import datetime
import numba  # pylint: disable=unused-import

from PySDM.backends import GPU, CPU
from PySDM.physics import si

from PySDM_examples.Bulenok_2023_MasterThesis.utils import go_benchmark, process_results, plot_processed_results, write_to_file
from PySDM_examples.Bulenok_2023_MasterThesis.setups import setup_coalescence_only_sim, setup_breakup_only_sim, setup_coalescence_breakup_sim

TIMESTAMP = str(datetime.now().strftime("%Y-%d-%m_%Hh-%Mm-%Ss"))

SIM_RUN_FILENAME="env_name_" + TIMESTAMP

assert not os.path.isfile(SIM_RUN_FILENAME)

subprocess.run(['bash', '-c', f'echo NUMBA_DEFAULT_NUM_THREADS: {numba.config.NUMBA_DEFAULT_NUM_THREADS} >> {SIM_RUN_FILENAME}'])
subprocess.run(['bash', '-c', f'echo NUMBA_NUM_THREADS: {numba.config.NUMBA_NUM_THREADS} >> {SIM_RUN_FILENAME}'])
subprocess.run(['bash', '-c', f'lscpu >> {SIM_RUN_FILENAME}'])
subprocess.run(['bash', '-c', f'nvidia-smi >> {SIM_RUN_FILENAME}'])
subprocess.run(['bash', '-c', f'nvidia-smi -L >> {SIM_RUN_FILENAME}'])
subprocess.run(['bash', '-c', f'cat /proc/cpuinfo >> {SIM_RUN_FILENAME}'])

CI = 'CI' in os.environ

exponents = [3, 5, 8, 10, 12, 14, 16, 18, 20, 22, 24] if not CI else [3, 5]
n_sds = [2 ** i for i in exponents]

numba_n_threads = [1, 2, 4, 5, 6, 8, 10] if not CI else [1, 2]

n_realisations = 3 if not CI else 2
seeds = list(range(n_realisations))

n_steps_short = 100 if not CI else 3
n_steps_full = 2048 if not CI else 3

# # Benchmark regular setup (without scaling)

# ### Coalescence-only

res_coalescence_only = go_benchmark(
    setup_coalescence_only_sim, n_sds, n_steps_short, seeds, numba_n_threads=numba_n_threads, double_precision=True, 
    sim_run_filename=SIM_RUN_FILENAME + '-coalescence',
    backends=[CPU, GPU]
)
coalescence_only_processed = process_results(res_coalescence_only)
filename=f"{SIM_RUN_FILENAME}-results-coalescence-double-n_steps{n_steps_short}"
plot_processed_results(
    coalescence_only_processed,
    plot_title=f'coalescence-only (n_steps: {n_steps_short})',
    plot_filename=filename + '.svg'
)
write_to_file(filename=filename + '.txt', d=coalescence_only_processed)

# ### Breakup-only

res_breakup_only = go_benchmark(
    setup_breakup_only_sim, n_sds, n_steps_short, seeds, numba_n_threads=numba_n_threads, double_precision=True,
    sim_run_filename=SIM_RUN_FILENAME + '-breakup',
    backends=[CPU, GPU]
)
breakup_only_processed = process_results(res_breakup_only)
filename=f"{SIM_RUN_FILENAME}-results-breakup-double-n_steps{n_steps_short}"
plot_processed_results(
    breakup_only_processed,
    plot_title=f'breakup-only  (n_steps: {n_steps_short})',
    plot_filename=filename + '.svg'
)
write_to_file(filename=filename + '.txt', d=breakup_only_processed)

# ### Coalescence and Breakup

res_coal_breakup = go_benchmark(
    setup_coalescence_breakup_sim, n_sds, n_steps_full, seeds, numba_n_threads=numba_n_threads, double_precision=True, 
    sim_run_filename=SIM_RUN_FILENAME + '-coal-break',
    backends=[CPU, GPU]
)
coal_breakup_processed = process_results(res_coal_breakup)
filename=f"{SIM_RUN_FILENAME}-results-coal_with_breakup-double-n_steps{n_steps_full}"
plot_processed_results(
    coal_breakup_processed,
    plot_title=f'coalescence+breakup (n_steps: {n_steps_full})',
    plot_filename=filename + '.svg'
)
write_to_file(filename=filename + '.txt', d=coal_breakup_processed)

# # Benchmark setup with scaling

def total_number_from_n_sd(n_sd):
    return n_sd * 1e8

def dv_from_n_sd(n_sd):
    return n_sd * (0.125 * si.m**3)

# ### Coalescence-only

res_coalescence_only_scaled = go_benchmark(
    setup_coalescence_only_sim, n_sds, n_steps_short, seeds, numba_n_threads=numba_n_threads, double_precision=True, 
    sim_run_filename=SIM_RUN_FILENAME + '-coalescence-scaled',
    total_number=total_number_from_n_sd,
    dv=dv_from_n_sd,
    backends=[CPU, GPU]
)
coalescence_only_processed_scaled = process_results(res_coalescence_only_scaled)
filename=f"{SIM_RUN_FILENAME}-results-scaled-coalescence-double-n_steps{n_steps_short}"
plot_processed_results(
    coalescence_only_processed_scaled,
    plot_title=f'coalescence-only with scaling (n_steps: {n_steps_short})',
    plot_filename=filename + '.svg'
)
write_to_file(filename=filename + '.txt', d=coalescence_only_processed_scaled)

# ### Breakup-only

res_breakup_only_scaled = go_benchmark(
    setup_breakup_only_sim, n_sds, n_steps_short, seeds, numba_n_threads=numba_n_threads, double_precision=True,
    sim_run_filename=SIM_RUN_FILENAME + '-breakup-scaled',
    total_number=total_number_from_n_sd,
    dv=dv_from_n_sd,
    backends=[CPU, GPU]
)
breakup_only_processed_scaled = process_results(res_breakup_only_scaled)
filename=f"{SIM_RUN_FILENAME}-results-scaled-breakup-double-n_steps{n_steps_short}"
plot_processed_results(
    breakup_only_processed_scaled,
    plot_title=f'breakup-only with scaling (n_steps: {n_steps_short})',
    plot_filename=filename + '.svg'
)
write_to_file(filename=filename + '.txt', d=breakup_only_processed_scaled)

# ### Coalescence and Breakup

res_coal_breakup_scaled = go_benchmark(
    setup_coalescence_breakup_sim, n_sds, n_steps_full, seeds, numba_n_threads=numba_n_threads, double_precision=True, 
    sim_run_filename=SIM_RUN_FILENAME + '-coal-break-scaled',
    total_number=total_number_from_n_sd,
    dv=dv_from_n_sd,
    backends=[CPU, GPU]
)
coal_breakup_processed_scaled = process_results(res_coal_breakup_scaled)
filename=f"{SIM_RUN_FILENAME}-results-scaled-coal_with_breakup-double-n_steps{n_steps_full}"
plot_processed_results(
    coal_breakup_processed_scaled,
    plot_title=f'coalescence+breakup with scaling (n_steps: {n_steps_full})',
    plot_filename=filename + '.svg'
)
write_to_file(filename=filename + '.txt', d=coal_breakup_processed_scaled)