"""Cross-implementation parity against the reference Fortran AMICA binary."""

from py_amica.benchmark.parity.fortran_io import (  # noqa: F401
    read_fortran_results,
    read_initial_weights,
    write_fdt,
    write_param,
)
