"""This module compares the number of CPU threads torch uses with the number of available cores.

Torch runs every CPU tensor operation in this package on its intra-op threads.  Torch reads the
thread count from ``OMP_NUM_THREADS`` and ``MKL_NUM_THREADS`` at import and uses the smaller of
the two.  If either variable is set to one, every CPU tensor operation runs on one core.
"""

import os
import warnings

import torch

THREAD_VARIABLES = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS')


def available_cores():
    """This function returns the number of cores this process may run on.  It uses the
    CPU affinity mask when the platform reports one, so it respects a job scheduler's
    allocation.  Otherwise it returns the machine's core count."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def host_thread_report():
    """This function returns the tuple ``(threads, cores, variables)``.  The values are
    torch's intra-op thread count, the number of available cores, and a dict of the
    thread variables set in the environment."""
    variables = {name: os.environ[name] for name in THREAD_VARIABLES if name in os.environ}
    return torch.get_num_threads(), available_cores(), variables


def warn_if_threads_unused():
    """This function issues a warning when torch uses one CPU thread and more cores
    are available.  It returns the warning text, or None when it issues no warning."""
    threads, cores, variables = host_thread_report()
    if threads > 1 or cores <= 1:
        return None
    if variables:
        cause = ('the environment sets ' +
                 ' and '.join(f'{name}={value}' for name, value in variables.items()))
    else:
        cause = 'the thread count was set to one'
    text = (f'torch runs host tensor operations on one thread, and this process may use '
            f'{cores} cores; {cause}.  Host-side work in mbirtorch, such as the consensus '
            f'update of a MACE reconstruction, then runs on one core.  Unset '
            f'{" and ".join(THREAD_VARIABLES)} before starting Python, or call '
            f'torch.set_num_threads({cores}).')
    warnings.warn(text, RuntimeWarning, stacklevel=2)
    return text
