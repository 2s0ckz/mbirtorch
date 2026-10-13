"""The host-thread warning: raised when torch has one thread and more cores are
available, naming the variables that set it."""

import pytest
import torch

from mbirtorch import _host_threads


def _report(monkeypatch, threads, cores, variables):
    monkeypatch.setattr(torch, 'get_num_threads', lambda: threads)
    monkeypatch.setattr(_host_threads, 'available_cores', lambda: cores)
    for name in _host_threads.THREAD_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    for name, value in variables.items():
        monkeypatch.setenv(name, value)


def test_one_thread_with_many_cores_warns_and_names_the_variables(monkeypatch):
    _report(monkeypatch, threads=1, cores=8, variables={'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
    with pytest.warns(RuntimeWarning) as record:
        text = _host_threads.warn_if_threads_unused()
    assert len(record) == 1
    assert '8 cores' in text and 'OMP_NUM_THREADS=1' in text and 'MKL_NUM_THREADS=1' in text
    assert 'torch.set_num_threads(8)' in text
