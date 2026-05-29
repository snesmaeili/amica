# GPU OOM in `apply_sphering` — Fix Options

**Reported:** 2026-05-28  
**Symptom:** JAX RESOURCE_EXHAUSTED crash at `preprocessing.py:305` (`apply_sphering`) when running sub-02 (4,819,701 samples, 64 components, f64).

## Root Cause

`chunk_size` only gates the **E-step inner loop** (`_amica_step_chunked`).  
It does **not** affect preprocessing.

`apply_sphering` is a `@jax.jit` function that computes:

```
data_white = sphere @ (data - mean)
```

with shapes `f64[64, 120] @ f64[120, 4819701] → f64[64, 4819701]`.

During XLA autotuning, the allocator needs input + output + scratch simultaneously:
- `data - mean`: `f64[120, 4819701]` = **3.71 GB**
- `data_white` output: `f64[64, 4819701]` = **2.47 GB**
- Total peak: **~6.2 GB** for one kernel, on an 8 GB GPU

Sub-01 had fewer samples and fit; sub-02 does not.  
`chunk_size` is irrelevant here — `data_white` must fully exist before the iteration loop starts.

---

## Option A — Chunk `apply_sphering` (no JIT over full time axis)

Apply sphere in numpy-looped chunks; each chunk is a JIT call over `f64[64, B]`:

```python
chunks = []
for start in range(0, n_samples, chunk_size):
    chunk = data[:, start:start+chunk_size]
    chunks.append(apply_sphering_chunk(chunk, mean, sphere))  # jit over small chunk
data_white = jnp.concatenate(chunks, axis=1)
```

**Pros:**
- Peak GPU during sphering = `f64[120, B] + f64[64, B]` (small)
- No refactor of solver or E-step
- `chunk_size` already exists in config — reuse it

**Cons:**
- `data_white` is still `f64[64, 4819701]` = 2.47 GB **permanently in VRAM** after assembly
- The E-step loop still needs this full array on GPU — so peak during training is unchanged
- Only reduces peak during the brief preprocessing phase, not during training
- XLA concatenate of 12 chunks may allocate a temporary copy → peak ~4.9 GB at concat

**Verdict:** Fixes the crash at preprocessing but does NOT reduce steady-state VRAM. If GPU only OOMs at sphering, this helps; if it OOMs during training too, this buys nothing.

---

## Option B — Keep `data_white` on CPU; transfer chunks per E-step iteration

Store whitened data as a CPU numpy array. Each E-step, slice + transfer chunk to GPU, compute, pull stats back to CPU:

```python
data_white_cpu = np.asarray(sphere @ (data_cpu - mean_cpu))  # NumPy, stays on CPU

# inside E-step:
chunk_gpu = jnp.asarray(data_white_cpu[:, start:stop])
# ... compute on GPU ...
```

**Pros:**
- Peak GPU VRAM = one chunk only (e.g. `f64[64, 400000]` = 0.2 GB)
- No persistent large array on GPU at all
- Could handle arbitrarily long recordings on small GPUs

**Cons:**
- Major refactor: solver, E-step functions, and `_amica_step_chunked` all assume `data_white` is a JAX (GPU) array
- Host↔device transfers per iteration — latency cost, especially on PCIe (not NVLink)
- If chunk is small and iterations are many (2000–3000), transfer overhead may dominate runtime
- Changes the core compute contract of the solver

**Verdict:** Correct long-term solution for very large datasets on consumer GPUs. High implementation cost. Should be discussed before pursuing.

---

## Option C — Reduce `--n-components`

Pass `--n-components 32` instead of 64:

- `data_white`: `f64[32, 4819701]` = **1.23 GB** (vs 2.47 GB)
- Input `data - mean`: still `f64[120, 4819701]` = 3.71 GB (not reduced)
- Actually the bottleneck is the input data size, not the output

Wait: with 64 components, `sphere` is `f64[64, 120]` so input to GEMM is `f64[120, T]` = 3.71 GB regardless of n_components. Output shrinks but input does not.

**Pros:**
- Zero code change; immediate workaround
- Faster training (smaller model)

**Cons:**
- Loses ICA resolution — paper uses 64 components; reducing changes results
- Input data `f64[120, T]` = 3.71 GB still transferred to GPU regardless
- Does not fix the architectural issue

**Verdict:** Quick workaround for experimentation. Not acceptable for paper-grade runs if paper specifies 64 components.

---

## Option D — Compute sphere on-the-fly per E-step chunk (no stored `data_white`)

Never materialize `data_white`. Instead, store raw centered data and apply sphere inside each E-step chunk:

```python
# preprocessing: compute sphere matrix only (no apply)
# E-step chunk:
raw_chunk = data_centered[:, start:stop]      # f64[120, B]
white_chunk = jnp.dot(sphere, raw_chunk)       # f64[64, B]
# ... run E-step on white_chunk ...
```

**Pros:**
- No `data_white` array ever exists in full — minimal VRAM
- More correct than Option B (keeps GPU computation; avoids host transfer in inner loop)

**Cons:**
- Requires storing `data_centered` (`f64[120, T]` = 3.71 GB on GPU or CPU) OR re-computing mean subtraction per chunk
- If `data_centered` is on CPU: still need host→device per iteration (same problem as B)
- If `data_centered` is on GPU: 3.71 GB, which is the same bottleneck (input > output here)
- Significant refactor of the preprocessing/solver interface

**Verdict:** Only net win if `data_centered` is kept on CPU. Then this collapses to Option B with an inline sphere application. Same trade-offs.

---

## Option E — Use `float32` instead of `float64`

Add `dtype: str = "float32"` to the run command or config:

- `data_white`: `f32[64, 4819701]` = **1.23 GB** (half of f64)
- Input `f32[120, 4819701]` = **1.86 GB** (half of f64)
- Peak during sphering: ~3.1 GB instead of 6.2 GB — fits on 8 GB GPU

**Pros:**
- Zero architectural change; `dtype` is already a config parameter
- 2× reduction in all array sizes across the entire algorithm
- Faster on modern GPUs (Tensor Cores prefer f32)

**Cons:**
- AMICA paper and Fortran reference use f64 — results may differ numerically
- Need to validate convergence quality on f32 vs f64 (may need more iterations)
- Not a permanent fix: even longer recordings or more components would OOM again

**Verdict:** Good medium-term workaround if f32 results are acceptable. Needs validation.

---

## Recommendation Matrix

| Option | VRAM at sphering | VRAM during training | Code change | Paper validity |
|--------|-----------------|---------------------|-------------|---------------|
| A (chunk sphering) | Small | Unchanged (2.47 GB) | Small | OK |
| B (CPU data_white) | Tiny | Tiny (1 chunk) | Large | OK |
| C (32 components) | Smaller | Smaller | None | ❌ changes results |
| D (on-the-fly sphere) | Tiny | Tiny | Large | OK |
| E (float32) | 1.55 GB | 1.23 GB | None | Needs validation |

## Questions for the Team

1. Is the crash only at the sphering phase, or does training also OOM later? (Determines if Option A is sufficient.)
2. Is f32 acceptable for paper-grade results, or is f64 required?
3. Is host↔device transfer per iteration acceptable (Options B/D)? GPU is RTX 4070 on PCIe — not NVLink.
4. Should we target making the algorithm work on 8 GB consumer GPUs as a design constraint, or is H100 the only paper-grade target?
5. Does the paper specify 64 components, or is 32 acceptable for some analyses?

## Immediate Workaround (no code change required)

```bash
# float32 — halves all array sizes
env XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" \
    ... python -m amica_python.benchmark.runner ... --dtype float32

# OR reduce components
... --n-components 32
```

Setting `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false` disables Triton GEMM autotuning
(which is what's failing) and falls back to cuBLAS — may succeed even at f64 on some
allocator configurations, but is not guaranteed.
