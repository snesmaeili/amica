"""Adapter for Fortran AMICA binary (amica15ub — statically linked, no MPI)."""

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from .base import AmicaAdapter

# amica17_narval: dynamically linked, needs gcc/12.3 + openmpi/4.1.5 + flexiblas
# Load modules before running, or use the Slurm submit script
FORTRAN_BINARY = Path("/home/sesma/refs/sccn-amica/amica17_narval")
# Use a short fixed path to avoid Fortran buffer overflow on long tmpdir paths
FORTRAN_WORKDIR = Path("/tmp/amica_parity")


class FortranAdapter(AmicaAdapter):

    def __init__(self, binary_path=None):
        self._binary = Path(binary_path) if binary_path else FORTRAN_BINARY
        self._available = self._binary.exists()

    @property
    def name(self) -> str:
        return "fortran"

    @property
    def available(self) -> bool:
        return self._available

    def run(self, data, params, n_iters, shared_sphere=None,
            shared_mean=None, log_det_sphere=None):
        if not self._available:
            return None

        n_ch, n_samples = data.shape
        n_comp = params.get("pcakeep", n_ch)

        # Use short fixed path; unique subdir per call to avoid MPI state issues
        import uuid
        run_id = uuid.uuid4().hex[:6]
        workdir = FORTRAN_WORKDIR / run_id
        if workdir.exists():
            shutil.rmtree(workdir)
        workdir.mkdir(parents=True)
        outdir = workdir / "out"
        outdir.mkdir()

        try:
            # Write FDT file (float32, Fortran column-major)
            fdt_path = workdir / "data.fdt"
            data.T.astype(np.float32).tofile(fdt_path)

            # Write param file
            param_path = workdir / "run.param"
            self._write_param_file(
                param_path, fdt_path, outdir,
                n_ch, n_samples, n_comp, params, n_iters,
            )

            # Run Fortran binary — relies on module load in Slurm script
            # for LD_LIBRARY_PATH. Set OMP vars in os.environ directly
            # (matching the working pattern from validate_parity.py).
            os.environ["OMP_NUM_THREADS"] = "4"
            os.environ["OMP_STACKSIZE"] = "512M"

            t0 = time.perf_counter()
            try:
                result = subprocess.run(
                    [str(self._binary), str(param_path)],
                    capture_output=True, text=True, timeout=600,
                )
                elapsed = time.perf_counter() - t0
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
                print(f"Fortran execution error: {e}")
                return None

            if result.returncode != 0:
                # Show first meaningful error line
                err = result.stderr.strip().splitlines()
                msg = next((l for l in err if "error" in l.lower()
                            or "terminate" in l.lower()), err[0] if err else "unknown")
                print(f"Fortran exit code {result.returncode}: {msg}")
                # Show stdout for debugging
                if result.stdout.strip():
                    print(f"Fortran stdout (first 300): {result.stdout[:300]}")
                return None

            # Parse LL trajectory from stdout
            ll_history = self._parse_ll(result.stdout)
            if not ll_history:
                print(f"Fortran: no LL values parsed. stdout (first 500):")
                print(result.stdout[:500])
                return None

            # Read output files
            J = params["num_mix"]
            W = self._read_fortran(outdir / "W", (n_comp, n_comp))
            A = self._read_fortran(outdir / "A", (n_comp, n_comp))
            c = self._read_fortran(outdir / "c", (n_comp,))
            alpha = self._read_fortran(outdir / "alpha", (J, n_comp))
            mu = self._read_fortran(outdir / "mu", (J, n_comp))
            beta = self._read_fortran(outdir / "sbeta", (J, n_comp))
            rho = self._read_fortran(outdir / "rho", (J, n_comp))
            # Fortran writes full (n_ch, n_ch) sphere even with pcakeep
            S_full = self._read_fortran(outdir / "S", (n_ch, n_ch))
            S = S_full[:n_comp, :]  # extract kept components
            mean_vec = self._read_fortran(outdir / "mean", (n_ch,))

            return {
                "W": W,
                "A": A,
                "alpha": alpha,
                "mu": mu,
                "beta": beta,
                "rho": rho,
                "c": c,
                "ll_history": np.array(ll_history),
                "sphere": S,
                "sphere_full": S_full,  # unsliced (n_ch, n_ch) — for L1 diagnosis
                "mean": mean_vec,
                "log_det_sphere": 0.0,
                "elapsed": elapsed,
                "n_iter": len(ll_history),
            }
        finally:
            # Clean up
            if workdir.exists():
                shutil.rmtree(workdir, ignore_errors=True)

    def _write_param_file(self, path, fdt_path, outdir,
                          n_ch, n_samples, n_comp, params, n_iters):
        """Write Fortran AMICA param file matching the validated format.

        Based on the working param file from validate_parity.py.
        """
        # Match the exact format from the working sub01_fortran.param
        path.write_text(f"""files {fdt_path}
outdir {outdir}
num_models 1
num_mix_comps {params['num_mix']}
data_dim {n_ch}
field_dim {n_samples}
num_samples 1
field_blocksize 1
pcakeep {n_comp}
max_threads 4
block_size 256
max_iter {n_iters}
writestep 1
dble_data 0
lrate {params['lrate']}
use_grad_norm {'1' if params.get('use_grad_norm', False) else '0'}
use_min_dll 1
min_grad_norm 0.000001
min_dll {params['min_dll']}
do_approx_sphere 1
do_reject 0
do_newton 1
newt_start {params['newt_start']}
minlrate 0.00000001
lratefact 0.5
rholrate {params['rholrate']}
rho0 {params['rho0']}
minrho {params['minrho']}
maxrho {params['maxrho']}
rholratefact 0.5
newt_ramp {params['newt_ramp']}
newtrate {params['newtrate']}
max_decs {params['max_decs']}
update_A 1
update_c 1
update_gm 1
update_alpha 1
update_mu 1
update_beta 1
do_rho 1
invsigmax {params['invsigmax']}
invsigmin {params['invsigmin']}
do_mean 1
do_sphere 1
doPCA 1
doscaling {'1' if params['doscaling'] else '0'}
scalestep 1
fix_init 0
share_comps 0
""")

    @staticmethod
    def _parse_ll(stdout):
        """Parse LL values from Fortran stdout."""
        lls = []
        for line in stdout.splitlines():
            m = re.search(r"LL\s*=\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)", line)
            if m:
                lls.append(float(m.group(1)))
        return lls

    @staticmethod
    def _read_fortran(path, shape):
        """Read Fortran binary output file (float64, column-major)."""
        if not path.exists():
            return np.zeros(shape)
        raw = np.fromfile(str(path), dtype=np.float64)
        if raw.size == 0:
            return np.zeros(shape)
        return raw.reshape(shape, order="F")
