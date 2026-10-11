"""This module runs the command line program for ``python -m mbirtorch.hsnt``.  The program
itself is defined in ``mbirtorch.hsnt.cli``."""
import sys

from .cli import main

sys.exit(main())
