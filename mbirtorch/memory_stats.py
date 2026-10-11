"""This module reports memory statistics for each processor.

``get_memory_stats`` returns one dict for each processor.  The keys of the dict depend on the
type of processor:

- CUDA: ``bytes_in_use``, ``peak_bytes_in_use``, ``reserved_bytes`` (the memory that PyTorch
  holds), ``peak_reserved_bytes``, ``cache_bytes`` (the held memory that no array uses), and
  ``bytes_limit`` (the total memory of the GPU).
- MPS: ``bytes_in_use``, ``driver_bytes_in_use`` (the total memory of the Metal driver), and
  ``bytes_limit`` (the recommended maximum).  MPS does not record a peak.
- CPU: ``bytes_in_use`` is the unique set size, ``peak_bytes_in_use`` is the resident set
  size, and ``bytes_limit`` is the available physical memory.
"""

import os

import psutil
import torch


def get_memory_stats(print_results=True, file=None):
    """This function collects the memory use of each processor and optionally prints it.

    Args:
        print_results (bool, optional): If True, the function prints the statistics of each
            processor in GB.  Defaults to True.
        file (file-like, optional): The stream that receives the printout, such as an
            ``io.StringIO``.  None means stdout.  Defaults to None.

    Returns:
        list of dict: One dict for each processor.  The GPUs come first and the CPU comes last.
        Each dict holds an ``'id'`` and the byte counts described in the module docstring.
    """
    memory_stats_per_processor = []

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_stats = dict()
            memory_stats['id'] = 'GPU ' + str(i)
            memory_stats['bytes_in_use'] = torch.cuda.memory_allocated(i)
            memory_stats['peak_bytes_in_use'] = torch.cuda.max_memory_allocated(i)
            memory_stats['reserved_bytes'] = torch.cuda.memory_reserved(i)
            memory_stats['peak_reserved_bytes'] = torch.cuda.max_memory_reserved(i)
            # This is memory that PyTorch holds for reuse but that no array uses.
            memory_stats['cache_bytes'] = (memory_stats['reserved_bytes']
                                           - memory_stats['bytes_in_use'])
            memory_stats['bytes_limit'] = torch.cuda.get_device_properties(i).total_memory
            memory_stats_per_processor.append(memory_stats)
    elif torch.backends.mps.is_available():
        memory_stats = dict()
        memory_stats['id'] = 'MPS'
        memory_stats['bytes_in_use'] = torch.mps.current_allocated_memory()
        memory_stats['driver_bytes_in_use'] = torch.mps.driver_allocated_memory()
        memory_stats['bytes_limit'] = torch.mps.recommended_max_memory()
        memory_stats_per_processor.append(memory_stats)

    memory_stats = dict()
    current_process = psutil.Process(os.getpid())
    memory_info = current_process.memory_full_info()
    memory_stats['id'] = 'CPU'
    # For the CPU, bytes_in_use is the unique set size, and peak_bytes_in_use is the
    # resident set size.
    memory_stats['bytes_in_use'] = memory_info.uss
    memory_stats['peak_bytes_in_use'] = memory_info.rss
    memory_stats['bytes_limit'] = psutil.virtual_memory().available
    memory_stats_per_processor.append(memory_stats)

    if print_results:
        for memory_stats in memory_stats_per_processor:
            print(memory_stats['id'], file=file)
            for tag, value in memory_stats.items():
                if tag == 'id':
                    continue
                cur_value = value / (1024 ** 3)
                extra_space = ' ' * max(1, 21 - len(tag) - len(str(int(cur_value))))
                print(f'  {tag}:{extra_space}{cur_value:.3f}GB', file=file)

    return memory_stats_per_processor
