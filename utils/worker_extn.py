import gc
import math
import time

import torch


def _stateless_init_process_group(master_address, master_port, rank, world_size, device):
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup
    pg = StatelessProcessGroup.create(
        host=master_address, port=master_port, rank=rank, world_size=world_size
    )
    return PyNcclCommunicator(pg, device=device)


class WorkerExtension:
    """RPC surface used by the ES trainer.

    The extension runs in two modes:

    * **Full-FT mode** (default, unchanged from the original): perturbations,
      restorations, and broadcasts iterate over every model parameter. This
      reproduces the existing behavior for non-LoRA experiments.
    * **LoRA mode**: enabled by calling `init_lora(...)`. Side-state
      `(A, B, W_base)` is maintained per target Linear; the live model
      parameters always equal `W_base + B @ A`. ES perturbations operate on
      `(A, B)` only. An additional `project_lora_orthogonal` RPC enforces
      `A · V.T == 0` after each ES update, where V is the row-orthonormal
      reference subspace loaded via `load_reference_subspaces`.

    Methods used by the ES trainer:
    - perturb_self_weights(seed, sigma_or_scale, negate=False)
    - restore_self_weights(seed, SIGMA)
    - init_inter_engine_group(master_address, master_port, rank, world_size)
    - broadcast_all_weights(src_rank)
    - save_self_weights_to_disk(filepath)

    LoRA-mode-only:
    - init_lora(rank, target_module_substrs, init_seed)
    - load_reference_subspaces(v_ref_path)
    - project_lora_orthogonal()
    - measure_lora_orth_residual()
    """

    # ------------------------------------------------------------------
    # Full-FT path (preserved)
    # ------------------------------------------------------------------

    def perturb_self_weights(self, seed, noise_scale, negate=False):
        if getattr(self, "_lora_mode", False):
            return self._perturb_lora(seed, noise_scale, negate)
        scale = float(noise_scale)
        sign = -1.0 if negate else 1.0
        for _, p in self.model_runner.model.named_parameters():
            gen = torch.Generator(device=p.device)
            gen.manual_seed(int(seed))
            noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
            p.data.add_(sign * scale * noise)
            del noise
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return True

    def restore_self_weights(self, seed, SIGMA):
        if getattr(self, "_lora_mode", False):
            return self._restore_lora(seed, SIGMA)
        for _, p in self.model_runner.model.named_parameters():
            gen = torch.Generator(device=p.device)
            gen.manual_seed(int(seed))
            noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
            p.data.add_(-float(SIGMA) * noise)
            del noise
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return True

    def init_inter_engine_group(self, master_address: str, master_port: int, rank: int, world_size: int):
        self.inter_pg = _stateless_init_process_group(
            master_address, master_port, rank, world_size, self.device
        )
        return True

    def broadcast_all_weights(self, src_rank: int):
        if getattr(self, "_lora_mode", False):
            return self._broadcast_lora(src_rank)
        for _, p in self.model_runner.model.named_parameters():
            self.inter_pg.broadcast(p, src=int(src_rank), stream=torch.cuda.current_stream())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

    def save_self_weights_to_disk(self, filepath):
        state_dict_to_save = {}
        for name, p in self.model_runner.model.named_parameters():
            state_dict_to_save[name] = p.detach().cpu()
        if getattr(self, "_lora_mode", False):
            # Also persist the (A, B) center tensors so the saved checkpoint
            # is reproducible without re-deriving them from the live weights.
            state_dict_to_save["__lora_A__"] = {
                k: v.detach().cpu() for k, v in self._lora_A.items()
            }
            state_dict_to_save["__lora_B__"] = {
                k: v.detach().cpu() for k, v in self._lora_B.items()
            }
        torch.save(state_dict_to_save, filepath)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(0.1)
        return True

    # ------------------------------------------------------------------
    # LoRA mode
    # ------------------------------------------------------------------

    def init_lora(self, rank: int, target_module_substrs, init_seed: int = 42):
        """Build LoRA side-state for every weight whose name contains one of
        `target_module_substrs` (e.g. ["q_proj", "k_proj", ...]).

        Initialization scheme (standard LoRA): A = 0, B ~ N(0, 1/r). With A=0,
        BA=0, so the live weights start exactly at W_base — no behavioral
        perturbation at iteration 0. This matches the constraint that we
        haven't started fine-tuning yet.

        Returns the ordered list of target weight names so the trainer can
        sanity-check the mapping.
        """
        rank = int(rank)
        if rank <= 0:
            raise ValueError(f"init_lora requires rank > 0, got {rank}")

        substrs = tuple(target_module_substrs)
        self._lora_rank = rank
        self._lora_target_names: list[str] = []
        self._lora_param_ref: dict[str, torch.nn.Parameter] = {}
        self._lora_A: dict[str, torch.Tensor] = {}
        self._lora_B: dict[str, torch.Tensor] = {}
        self._W_base: dict[str, torch.Tensor] = {}
        self._V_ref: dict[str, torch.Tensor] = {}

        # Per-tensor seed offsets so noise across tensors is independent
        # rather than sharing a single RNG stream (same idea as the iid-noise
        # variant of the conciseness script). We use a *multiplicative* mix
        # so adding the offset to a different seed cannot collide with
        # `seed + offset` for some other (seed, offset) pair.
        self._seed_stride = 2_000  # generous upper bound on (A,B) tensor count
        self._target_index: dict[str, int] = {}

        # Deterministic init RNG (CPU); cast to per-param device/dtype below.
        cpu_gen = torch.Generator(device="cpu").manual_seed(int(init_seed))

        for name, p in self.model_runner.model.named_parameters():
            if not name.endswith(".weight"):
                continue
            if p.ndim != 2:
                continue
            if not any(sub in name for sub in substrs):
                continue

            d, k = p.shape
            self._target_index[name] = len(self._lora_target_names)
            self._lora_target_names.append(name)
            self._lora_param_ref[name] = p

            # Cache the unperturbed base weight. p is currently the freshly
            # loaded base-model weight, so this is safe.
            self._W_base[name] = p.data.clone()

            A = torch.zeros(rank, k, dtype=p.dtype, device=p.device)
            B_cpu = torch.empty(d, rank, dtype=torch.float32).normal_(
                mean=0.0, std=1.0 / math.sqrt(rank), generator=cpu_gen
            )
            B = B_cpu.to(device=p.device, dtype=p.dtype)

            self._lora_A[name] = A
            self._lora_B[name] = B
            # No refold needed at init: BA = 0 since A = 0.

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self._lora_mode = True
        return list(self._lora_target_names)

    def build_reference_subspaces(
        self,
        energy_threshold: float = 0.9,
        max_rank: int | None = None,
    ):
        """Compute the row-orthonormal V_ref for every LoRA target *from the
        live W_base on this engine*.

        Building V_ref engine-side rather than from a saved HF checkpoint
        sidesteps a naming-mismatch landmine: vLLM fuses Linear layers
        (`qkv_proj`, `gate_up_proj`) so its parameter names diverge from
        HF disk names. By running SVD on `self._W_base[name]`, the keys
        line up with the LoRA targets by construction.

        SVD runs on CPU (vLLM has reserved most of the GPU for the KV
        cache, so a GPU-side SVD workspace OOMs immediately on smaller
        cards like L4). Only the top-`s` rows of Vh are kept and copied
        back to the GPU as fp32, where `s` is the smallest rank such
        that the cumulative squared-singular-value energy reaches
        `energy_threshold`. If `max_rank` is set, `s` is capped there.

        Returns a dict mapping target name -> kept rank (for logging).
        """
        if not getattr(self, "_lora_mode", False):
            raise RuntimeError("build_reference_subspaces requires LoRA mode")
        if not 0 < energy_threshold <= 1.0:
            raise ValueError(f"energy_threshold must be in (0, 1], got {energy_threshold}")

        ranks: dict[str, int] = {}
        for name in self._lora_target_names:
            target_device = self._W_base[name].device
            # Round-trip to CPU for the SVD workspace; W tensors are small
            # enough (a few MB each) that this is fast in aggregate.
            W_cpu = self._W_base[name].detach().to(device="cpu", dtype=torch.float32)
            _, S, Vh = torch.linalg.svd(W_cpu, full_matrices=False)
            sq = S * S
            cum = sq.cumsum(0) / sq.sum().clamp_min(1e-30)
            s = int((cum < float(energy_threshold)).sum().item()) + 1
            if max_rank is not None:
                s = min(s, int(max_rank))
            s = max(s, 1)
            self._V_ref[name] = Vh[:s].contiguous().to(
                device=target_device, dtype=torch.float32
            )
            ranks[name] = s
            del W_cpu, S, Vh

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # One-shot projection: A=0 at init so this is a no-op, but keep the
        # call for symmetry if init ever changes to a non-zero A.
        self.project_lora_orthogonal()
        return ranks

    def project_lora_orthogonal(self):
        """Project A ← A − (A V.T) V for every target whose V_ref is loaded.

        Done in fp32 to avoid drift from repeated low-rank subtractions when
        the model is in fp16. Refolds the live weights at the end.
        """
        if not getattr(self, "_lora_mode", False):
            return False
        if not self._V_ref:
            return False

        for name in self._lora_target_names:
            V = self._V_ref.get(name)
            if V is None:
                continue
            A = self._lora_A[name]
            A_fp32 = A.data.float()
            # A − (A V.T) V; V is row-orthonormal so this is the exact
            # projection onto V's orthogonal complement (in row space).
            A_fp32.sub_(A_fp32 @ V.T @ V)
            A.data.copy_(A_fp32.to(A.dtype))
            self._refold(name)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

    def measure_lora_orth_residual(self) -> float:
        """Return Σ_layer ‖A · V.T‖²_F. Should be ~0 immediately after a
        projection; non-zero before, growing with ES drift."""
        if not getattr(self, "_lora_mode", False) or not self._V_ref:
            return 0.0
        total = 0.0
        for name in self._lora_target_names:
            V = self._V_ref.get(name)
            if V is None:
                continue
            A_fp32 = self._lora_A[name].data.float()
            total += float((A_fp32 @ V.T).pow(2).sum().item())
        return total

    # ------------------------------------------------------------------
    # LoRA-mode internals
    # ------------------------------------------------------------------

    def _refold(self, name: str) -> None:
        """Recompute live weight = W_base + B @ A and write it into the
        model parameter. Caller is responsible for syncing CUDA streams."""
        W_base = self._W_base[name]
        A = self._lora_A[name]
        B = self._lora_B[name]
        # Matmul in the model dtype is fine for r=O(16); the SVD-based
        # projection is the only place that needs fp32.
        self._lora_param_ref[name].data.copy_(W_base + B @ A)

    def _noise_for(self, name: str, seed: int, which: str) -> torch.Tensor:
        """Generate a deterministic noise tensor for an (A or B) target.

        Assigns each (seed, target_idx, which) triple a unique generator
        seed via a multiplicative mix `seed * (2*stride) + offset`. The
        multiplicative spacing prevents collisions across distinct (seed,
        offset) pairs that the additive scheme would alias together when
        seeds are drawn from a wide range relative to the number of
        targets.
        """
        idx = self._target_index[name]
        offset = idx + (self._seed_stride if which == "B" else 0)
        # 2*stride guarantees that bumping seed by 1 cannot land on any
        # other tensor's offset slot for the same seed.
        gen_seed = int(seed) * (2 * self._seed_stride) + int(offset)
        param = self._lora_A[name] if which == "A" else self._lora_B[name]
        gen = torch.Generator(device=param.device)
        gen.manual_seed(gen_seed)
        return torch.randn(
            param.shape, dtype=param.dtype, device=param.device, generator=gen
        )

    def _perturb_lora(self, seed, noise_scale, negate):
        scale = float(noise_scale) * (-1.0 if negate else 1.0)
        for name in self._lora_target_names:
            self._lora_A[name].add_(scale * self._noise_for(name, seed, "A"))
            self._lora_B[name].add_(scale * self._noise_for(name, seed, "B"))
            self._refold(name)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

    def _restore_lora(self, seed, SIGMA):
        sigma = float(SIGMA)
        for name in self._lora_target_names:
            self._lora_A[name].add_(-sigma * self._noise_for(name, seed, "A"))
            self._lora_B[name].add_(-sigma * self._noise_for(name, seed, "B"))
            self._refold(name)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

    def _broadcast_lora(self, src_rank: int):
        # Only (A, B) need to cross engines; W_base is identical everywhere
        # (loaded from the same HF checkpoint). Refold locally afterwards.
        for name in self._lora_target_names:
            self.inter_pg.broadcast(
                self._lora_A[name],
                src=int(src_rank),
                stream=torch.cuda.current_stream(),
            )
            self.inter_pg.broadcast(
                self._lora_B[name],
                src=int(src_rank),
                stream=torch.cuda.current_stream(),
            )
            self._refold(name)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True
