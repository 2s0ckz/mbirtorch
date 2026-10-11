"""This module stores the model parameters.  The user reads and changes the parameters with
``get_params``, ``set_params``, ``verify_valid_params``, and ``print_params``.  The module also
sets up the log for each reconstruction.
"""

import io
import itertools
import logging
import os
import warnings
from collections import namedtuple

import numpy as np

from . import _utils
from ._utils import Param

# This counter gives each model instance a unique logger name.
_instance_counter = itertools.count(1)


class ParameterHandler:
    """This class is used to store and access the model parameters.  ``TomographyModel``
    inherits ``get_params``, ``set_params``, and ``print_params`` from this class."""

    def __init__(self):
        self.params = _utils.get_default_params()
        # Python logging keeps one logger per name, so each instance needs its own name.  The
        # name uses a counter, because Python reuses an object id after the object is freed.
        self.logger_name = 'mbirtorch.{}.{}'.format(type(self).__name__,
                                                    next(_instance_counter))
        # Until a reconstruction calls setup_logger, the logger writes to the console only.
        self.logger = logging.getLogger(self.logger_name)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.INFO)
        self.log_buffer = None
        # The log file of the current reconstruction.  A later call that continues the
        # reconstruction reopens this file.
        self._logfile_path = None

    def setup_logger(self, *, logfile_path: str = "~/.mbirtorch/logs/recon.log", print_logs: bool = True):
        """
        This method sets up the logger and the log buffer for a reconstruction.  The 'verbose'
        parameter sets the logging level.  A value of 0 gives WARNING, 1 gives INFO, and 2 or
        more gives DEBUG.

        Args:
            logfile_path: The path to the log file.  The method expands '~' to the user's home
                directory.  If the path is None or empty, the method writes no file.
            print_logs: If True, the logger also writes to the console.

        Raises:
            Exception: If the directory of logfile_path cannot be created.
        """
        if logfile_path:
            logfile_path = os.path.expanduser(logfile_path)
        verbose = self.get_params('verbose')
        if verbose < 1:
            level = logging.WARNING
        elif verbose < 2:
            level = logging.INFO
        else:
            level = logging.DEBUG

        logger = self.logger
        logger.setLevel(level)
        # The logger writes only to the handlers added below, and not to parent loggers.
        logger.propagate = False
        for h in list(logger.handlers):
            try:
                h.flush()
            finally:
                h.close()
                logger.removeHandler(h)

        self.log_buffer = io.StringIO()
        buffer_handler = logging.StreamHandler(self.log_buffer)
        buffer_handler.setLevel(level)
        buffer_formatter = logging.Formatter('%(message)s')
        buffer_handler.setFormatter(buffer_formatter)
        logger.addHandler(buffer_handler)

        if print_logs:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_formatter = logging.Formatter('%(message)s')
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        # Mode 'w' starts a new log file for each new reconstruction.
        self._logfile_path = logfile_path if logfile_path else None
        if logfile_path:
            self._add_log_file_handler(logfile_path, mode='w', level=level)

    def _add_log_file_handler(self, logfile_path, mode, level):
        """This method attaches a handler that writes the log to ``logfile_path``.  The handler
        creates the file only when the first message is logged.
        """
        from .utilities import makedirs
        makedirs(logfile_path)
        file_handler = logging.FileHandler(logfile_path, mode=mode, delay=True)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(file_handler)

    def close_log_file(self):
        """This method flushes the log file, closes it, and removes its handler.  After this
        call, other code can merge and delete the file.  The log text stays in the log buffer
        for recon_dict['recon_log'].  ``_reopen_log_file`` reopens the file for a later call.
        """
        if self.logger is None:
            return
        for h in list(self.logger.handlers):
            if isinstance(h, logging.FileHandler):
                try:
                    h.flush()
                finally:
                    h.close()
                    self.logger.removeHandler(h)

    def _reopen_log_file(self):
        """This method reopens in append mode the log file that ``close_log_file`` closed.
        The method does nothing if the reconstruction has no log file or if a file handler is
        already attached."""
        if self.logger is None or not self._logfile_path:
            return
        if any(isinstance(h, logging.FileHandler) for h in self.logger.handlers):
            return
        self._add_log_file_handler(self._logfile_path, mode='a',
                                   level=self.logger.level)

    def _log_run_header(self, first_iteration, logfile_path, print_logs):
        """This method sets up the logger for a reconstruction if needed.  Then it logs the
        MBIRTorch version.
        """
        # If the log buffer is None, setup_logger has not run.
        if first_iteration == 0 or self.log_buffer is None:
            self.setup_logger(logfile_path=logfile_path, print_logs=print_logs)
        else:
            self._reopen_log_file()
        from . import __version__
        self.logger.info('MBIRTorch Version = {}'.format(__version__))

    def _log_device_report(self):
        """This method logs the devices that the reconstruction uses.  mbirtorch calls this
        method after it chooses the devices."""
        self.logger.info('Reconstruction devices: {}'.format(
            self._device_report()))

    def _device_report(self):
        """This method returns a summary of the reconstruction devices for the log.  The
        summary has the form 'N x PLATFORM (sharded)'."""
        devices = self.recon_placement.devices
        platform = devices[0].type.upper()
        report = '{} x {} (sharded)'.format(len(devices), platform)
        rejected = getattr(self, 'device_choice_rejections', None)
        automatic = getattr(self, 'device_layout_is_automatic', False)
        if rejected and automatic:
            visible = max([count for count, _why in rejected] + [len(devices)])
            report += ' (using {} of {} {} devices: {})'.format(
                len(devices), visible, platform,
                '; '.join('{} {}, {}'.format(
                    count, 'used' if count == len(devices) else 'rejected',
                    why) for count, why in rejected))
        return report

    # ── access to parameters ──────────────────────────────────────────────────
    def get_params(self, parameter_names):
        """This method returns the values of the named parameters.  If ``parameter_names`` is a
        string, the method returns one value.  If ``parameter_names`` is a list, the method
        returns a list of values."""
        if isinstance(parameter_names, str):
            if parameter_names not in self.params:
                raise NameError(f"'{parameter_names}' not a recognized parameter")
            return self.params[parameter_names].val
        values = []
        for name in parameter_names:
            if name not in self.params:
                raise NameError(f"'{name}' not a recognized parameter")
            values.append(self.params[name].val)
        return values

    @staticmethod
    def normalize_scalar(val):
        """This method converts a numpy scalar to a plain Python scalar.  The method returns
        other values unchanged."""
        if isinstance(val, np.generic):
            return val.item()
        return val

    def set_params(self, no_warning=False, no_compile=False, **kwargs):
        """
        This method sets parameters by keyword, as in ``model.set_params(sharpness=0.7)``.

        When the user sets a geometry parameter, this method rebuilds the projectors.  The
        method does not recompute the reconstruction geometry.  After a change to a detector
        parameter such as ``delta_det_channel``, call ``auto_set_recon_geometry``.

        When the user sets ``sigma_y``, ``sigma_x``, or ``sigma_prox``, this method turns off
        automatic regularization and issues a warning.  When the user sets ``sharpness`` or
        ``snr_db``, this method turns automatic regularization back on.

        Args:
            no_warning (bool, optional): If True, the method stores the values without warnings.
                Setting ``sigma_y``, ``sigma_x``, or ``sigma_prox`` then leaves automatic
                regularization unchanged.  Defaults to False.
            no_compile (bool, optional): If True, the method does not rebuild the projectors.
                Defaults to False.
            **kwargs: The parameter names and values.  An unknown name raises ValueError.

        Example:
            >>> model.set_params(recon_shape=(128, 128, 128), sharpness=0.7)
        """
        recompile = False
        regularization_parameter_change = False
        meta_parameter_change = False

        for key, val in kwargs.items():
            recompile_flag = True
            if key in self.params:
                recompile_flag = self.params[key].recompile_flag
            elif not no_warning:   # Initialization sets no_warning to skip this check.
                error_message = '{} is not a recognized parameter'.format(key)
                error_message += '\nValid parameters are: \n'
                for valid_key in self.params.keys():
                    error_message += '   {}\n'.format(valid_key)
                raise ValueError(error_message)

            clean_val = ParameterHandler.normalize_scalar(val)
            self.params[key] = Param(clean_val, recompile_flag)

            if recompile_flag:
                recompile = True
            elif key in ["sigma_y", "sigma_x", "sigma_prox"]:
                regularization_parameter_change = True
            elif key in ["sharpness", "snr_db"]:
                meta_parameter_change = True

        if regularization_parameter_change:
            if not no_warning:
                self.set_params(auto_regularize_flag=False)
                warnings.warn('You are directly setting regularization parameters, '
                              'sigma_x, sigma_y or sigma_prox. This is an advanced '
                              'feature that will disable auto-regularization.')

        if meta_parameter_change:
            if self.get_params('auto_regularize_flag') is False:
                self.set_params(auto_regularize_flag=True)
                if not no_warning:
                    warnings.warn('You have re-enabled auto-regularization by '
                                  'setting sharpness or snr_db. It was previously '
                                  'disabled')

        if recompile and not no_compile:
            self.refresh_device_bindings()

    # ── methods that TomographyModel and the geometry classes implement ───────
    def create_projectors(self):
        raise NotImplementedError

    def refresh_device_bindings(self):
        self.create_projectors()

    def verify_valid_params(self):
        """This method checks that the parameters are consistent at the start of a
        reconstruction.  Each geometry class extends this method."""
        sinogram_shape = self.get_params('sinogram_shape')
        if len(sinogram_shape) != 3:
            raise ValueError(f'sinogram_shape must be (views, rows, channels); '
                             f'got {sinogram_shape}')

    def print_params(self):
        """
        This method prints the current parameter values of the model.  When ``verbose`` is
        less than 3, the method prints only the shape and dtype of the view parameter array,
        such as the angles.

        Example:
            >>> ct_model = mbirtorch.ParallelBeamModel(sinogram_shape, angles)
            >>> ct_model.set_params(sharpness=0.7)
            >>> ct_model.print_params()
        """
        verbose, view_params_name = self.get_params(['verbose',
                                                     'view_params_name'])
        print('----')
        for key, entry in self.params.items():
            if verbose < 3 and key == view_params_name:
                val = np.asarray(entry.val)
                print(f'{key} = array(shape={val.shape}, '
                      f'dtype={val.dtype})')
            else:
                print(f'{key} = {entry.val}')
        print('----')

    # This dict holds one namedtuple class for each set of field names.  All instances
    # share it.
    _geometry_param_classes = {}

    @classmethod
    def make_geometry_params(cls, names, values):
        key = tuple(names)
        if key not in cls._geometry_param_classes:
            cls._geometry_param_classes[key] = namedtuple('GeometryParams', names)
        return cls._geometry_param_classes[key](*values)
