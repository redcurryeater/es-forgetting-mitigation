import argparse
from datetime import datetime
import gc
import json
import os
import random
import shutil
import signal
import sys
import time

import numpy as np
import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.utils import get_ip, get_open_port

from countdown.countdown_task import reward_function

# Default Hyperparameters
SIGMA = 0.001
ALPHA = 0.0005
POPULATION_SIZE = 30
NUM_ENGINES = 4
NUM_ITERATIONS = 1000
EXPERIMENT_DIR = "es-ft-experiment"

# KL behavioral-anchor defaults
KL_BETA = 0.0  # 0 = measurement only (or off entirely if NUM_ANCHOR_PROMPTS=0)
ANCHOR_DATASETS = "mmlu,race,hellaswag"
NUM_ANCHOR_PROMPTS = 0  # 0 = no anchor at all; >0 builds cache and logs KL/iteration
ANCHOR_MAX_TOKENS = 64
ANCHOR_SEED = 1234

def parse_args():
    parser = argparse.ArgumentParser(
        description="ES Fine-tuning for Countdown Task with multi-engine NCCL sync"
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--sigma", type=float, default=SIGMA)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--population_size", type=int, default=POPULATION_SIZE)
    parser.add_argument("--num_engines", type=int, default=NUM_ENGINES)
    parser.add_argument("--num_iterations", type=int, default=NUM_ITERATIONS)
    parser.add_argument("--experiment_dir", type=str, default=EXPERIMENT_DIR)
    parser.add_argument("--cuda_devices", type=str, default="0,1,2,3")
    parser.add_argument('--verbose', action='store_true', help='Print verbose logs')
    parser.add_argument(
        "--global_seed",
        type=int,
        help="Global random seed",
    )
    # KL behavioral anchor (general-capability drift mitigation)
    parser.add_argument(
        "--kl_beta",
        type=float,
        default=KL_BETA,
        help=(
            "KL penalty coefficient for the general-capability anchor. "
            "0 = measurement-only (still logs kl/* if num_anchor_prompts > 0); "
            ">0 = subtract beta * KL_hat from per-candidate reward."
        ),
    )
    parser.add_argument(
        "--anchor_datasets",
        type=str,
        default=ANCHOR_DATASETS,
        help="Comma-separated subset of {mmlu,race,hellaswag} to source anchor prompts from.",
    )
    parser.add_argument(
        "--num_anchor_prompts",
        type=int,
        default=NUM_ANCHOR_PROMPTS,
        help=(
            "Per-generation train-anchor batch size: how many anchor prompts "
            "are scored on every candidate. With --anchor_pool_size > "
            "num_anchor_prompts, a fresh batch is sampled from the pool each "
            "generation (rotation). Within a generation all candidates see "
            "the same batch so ES normalization stays consistent."
        ),
    )
    parser.add_argument(
        "--anchor_pool_size",
        type=int,
        default=0,
        help=(
            "Total pool of anchor prompts pre-scored on the base model at "
            "startup. Each generation samples num_anchor_prompts from the pool. "
            "0 (default) = no pool, fixed anchor (pool_size := num_anchor_prompts). "
            "Larger pool reduces overfitting of the penalty to a specific small "
            "set of prompts."
        ),
    )
    parser.add_argument(
        "--anchor_max_tokens",
        type=int,
        default=ANCHOR_MAX_TOKENS,
        help="Max tokens to sample from the base model for each anchor reference completion.",
    )
    parser.add_argument(
        "--anchor_seed",
        type=int,
        default=ANCHOR_SEED,
        help="Seed used to sample anchor prompts from the source datasets.",
    )
    parser.add_argument(
        "--eval_anchor_seed",
        type=int,
        default=None,
        help=(
            "If set, build a held-out eval anchor (disjoint from --anchor_seed "
            "at the dataset row-index level), measured per-iteration but never "
            "used as a penalty. Required for any clean 'anchor generalizes' claim."
        ),
    )
    parser.add_argument(
        "--num_eval_anchor_prompts",
        type=int,
        default=None,
        help=(
            "Number of held-out eval anchor prompts. If unset, mirrors "
            "--num_anchor_prompts. Only used when --eval_anchor_seed is set."
        ),
    )
    parser.add_argument(
        "--eval_anchor_every",
        type=int,
        default=10,
        help=(
            "Measure held-out eval KL on the (unperturbed) center weights "
            "every N generations. Single scalar per event. 0 disables periodic "
            "measurement (initial sanity check still runs if cache exists)."
        ),
    )
    args = parser.parse_args()
    # Optional: scope host visibility; vLLM actors will ignore it and pick device from PG
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    # set global random seed
    if args.global_seed is not None:
        random.seed(args.global_seed)
        np.random.seed(args.global_seed)
        torch.manual_seed(args.global_seed)
        torch.cuda.manual_seed_all(args.global_seed)

    return args

class ESNcclLLM(LLM):
    def __init__(self, *args, **kwargs):
        # Let Ray/PG determine the actual visible device in the actor
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        super().__init__(*args, **kwargs)

def launch_engines(num_engines, model_name):
    # Strict 1-GPU isolation via PGs
    pgs = [placement_group([{"GPU": 1, "CPU": 0}], lifetime="detached") for _ in range(num_engines)]
    ray.get([pg.ready() for pg in pgs])

    strategies = [
        PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=0,
        )
        for pg in pgs
    ]

    engines = [
        ray.remote(num_cpus=0, num_gpus=0, scheduling_strategy=strategy)(ESNcclLLM).remote(
            model=model_name,
            tensor_parallel_size=1,
            distributed_executor_backend="ray",
            worker_extension_cls="utils.worker_extn.WorkerExtension",
            dtype="float16",
            enable_prefix_caching=False,
            enforce_eager=False,
        )
        for strategy in strategies
    ]
    return engines, pgs

def evaluate_countdown_handle(llm, task_datas):
    prompts = [d["context"] for d in task_datas]
    sampling_params = SamplingParams(
        temperature=0.0,
        seed=42,
        max_tokens=1024,
    )
    handle = llm.generate.remote(prompts, sampling_params, use_tqdm=False)
    return handle, time.time()

# ----------------------------- KL behavioral anchor -----------------------------
#
# We anchor the perturbed policy to the *unperturbed base model* on prompts drawn
# from general-capability benchmarks (MMLU/RACE/HellaSwag). The reference behavior
# is fixed at startup: we sample one greedy completion from the base model for
# each anchor prompt and cache its per-token log-probabilities.
#
# Per ES candidate we then compute log pi_perturbed(ref_completion | prompt)
# via vLLM's `prompt_logprobs=1`, and form the (token-averaged) plug-in estimate
#     KL_hat_i = mean_t [ log pi_ref(y_t | y_<t, x) - log pi_perturbed(y_t | y_<t, x) ]
# i.e. the standard 1-sample estimator of KL(pi_ref || pi_perturbed) for samples
# drawn from pi_ref. The augmented per-candidate reward is:
#     R_tilde_i = R_task_i - kl_beta * KL_hat_i
# Note: under ES with per-generation reward normalization, candidate-independent
# constants in the penalty cancel, so the cached pi_ref logprobs only matter for
# *logging* a meaningful KL number -- the ES update direction is identical if we
# instead used negative log-likelihood of the ref completion as the penalty.

def load_anchor_prompts(dataset_spec, num_total, seed, exclude=None):
    """Sample raw text prompts from MMLU/RACE/HellaSwag.

    Splits `num_total` evenly across the requested datasets (any remainder
    goes to the first listed dataset).

    `exclude`: optional dict mapping dataset name -> set of dataset row indices
    that must NOT be sampled. Used to guarantee the held-out eval anchor is
    disjoint from the training anchor at the row-index level.

    Returns a tuple `(prompts, used_indices)` where `used_indices` is a dict
    mirroring the structure of `exclude` (dataset name -> set of indices),
    suitable for chaining into a follow-up call.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "KL anchor requires the `datasets` library. `pip install datasets`."
        ) from e

    names = [n.strip().lower() for n in dataset_spec.split(",") if n.strip()]
    if not names:
        return [], {}
    per = num_total // len(names)
    remainder = num_total - per * len(names)
    rng = random.Random(seed)
    prompts = []
    used_indices = {}
    exclude = exclude or {}

    def _pick(dataset_obj, name, n_wanted):
        skip = exclude.get(name, set())
        candidates = [i for i in range(len(dataset_obj)) if i not in skip]
        if not candidates:
            return []
        return rng.sample(candidates, min(n_wanted, len(candidates)))

    for idx, name in enumerate(names):
        n = per + (remainder if idx == 0 else 0)
        if n <= 0:
            continue
        if name == "mmlu":
            ds = load_dataset("cais/mmlu", "all", split="test")
            picks = _pick(ds, name, n)
            used_indices[name] = set(picks)
            for i in picks:
                ex = ds[i]
                opts = "\n".join(
                    f"{chr(65 + j)}. {ch}" for j, ch in enumerate(ex["choices"])
                )
                prompts.append(
                    f"Question: {ex['question']}\n{opts}\nAnswer:"
                )
        elif name == "race":
            ds = load_dataset("ehovy/race", "all", split="test")
            picks = _pick(ds, name, n)
            used_indices[name] = set(picks)
            for i in picks:
                ex = ds[i]
                opts = "\n".join(
                    f"{chr(65 + j)}. {ch}" for j, ch in enumerate(ex["options"])
                )
                prompts.append(
                    f"Article: {ex['article']}\n\nQuestion: {ex['question']}\n{opts}\nAnswer:"
                )
        elif name == "hellaswag":
            ds = load_dataset("Rowan/hellaswag", split="validation")
            picks = _pick(ds, name, n)
            used_indices[name] = set(picks)
            for i in picks:
                ex = ds[i]
                ctx = ex.get("ctx") or ex.get("ctx_a", "")
                prompts.append(f"Complete the following passage:\n{ctx}")
        else:
            raise ValueError(f"Unknown anchor dataset: {name}")

    return prompts, used_indices


def build_anchor_cache(reference_engine, anchor_prompts, max_completion_tokens):
    """Generate ref completions on the unperturbed base engine and cache
    per-token logprobs of pi_ref over those completions.

    Each cache entry stores token IDs (not text) so the per-candidate scoring
    pass can bypass tokenization and reuse the exact same token sequence.
    """
    if not anchor_prompts:
        return []

    sampling_params = SamplingParams(
        temperature=0.0,
        seed=42,
        max_tokens=max_completion_tokens,
        logprobs=1,  # return logprob of each generated token
    )
    outputs = ray.get(
        reference_engine.generate.remote(anchor_prompts, sampling_params, use_tqdm=False)
    )

    cache = []
    for prompt_text, out in zip(anchor_prompts, outputs):
        prompt_token_ids = list(out.prompt_token_ids)
        gen = out.outputs[0]
        completion_token_ids = list(gen.token_ids)
        if not completion_token_ids:
            continue

        ref_logprobs = []
        for tok_id, lp_dict in zip(completion_token_ids, gen.logprobs or []):
            if lp_dict is None or tok_id not in lp_dict:
                continue
            ref_logprobs.append(float(lp_dict[tok_id].logprob))

        if not ref_logprobs:
            continue

        cache.append(
            {
                "prompt_text": prompt_text,
                "full_token_ids": prompt_token_ids + completion_token_ids,
                "num_prompt_tokens": len(prompt_token_ids),
                "num_completion_tokens": len(ref_logprobs),
                "ref_logprobs": ref_logprobs,  # length == num_completion_tokens
            }
        )

    return cache


def evaluate_anchor_handle(llm, anchor_cache):
    """Dispatch a (non-blocking) prompt-logprob pass for the anchor sequences.

    Uses `prompt_token_ids` to bypass re-tokenization and guarantee the same
    token boundaries as the cached reference run.
    """
    if not anchor_cache:
        return None
    prompts = [{"prompt_token_ids": item["full_token_ids"]} for item in anchor_cache]
    sampling_params = SamplingParams(
        temperature=0.0,
        seed=42,
        max_tokens=1,
        prompt_logprobs=1,
    )
    return llm.generate.remote(prompts, sampling_params, use_tqdm=False)


def _postprocess_anchor_outputs(outputs, anchor_cache):
    """Compute mean per-token KL(pi_ref || pi_perturbed) across anchor samples.

    Per sample, KL is approximated by the mean over completion tokens of
    `log pi_ref(t) - log pi_perturbed(t)`. Per-prompt KLs are then averaged
    uniformly to produce the candidate's scalar penalty.
    """
    per_prompt_kls = []
    for out, item in zip(outputs, anchor_cache):
        prompt_logprobs = out.prompt_logprobs or []
        n_prompt = item["num_prompt_tokens"]
        n_completion = item["num_completion_tokens"]
        full_token_ids = item["full_token_ids"]
        ref_logprobs = item["ref_logprobs"]

        diffs = []
        for offset in range(n_completion):
            pos = n_prompt + offset
            if pos >= len(prompt_logprobs):
                break
            lp_dict = prompt_logprobs[pos]
            if lp_dict is None:
                continue
            tok_id = full_token_ids[pos]
            lp_obj = lp_dict.get(tok_id)
            if lp_obj is None:
                continue
            diffs.append(ref_logprobs[offset] - float(lp_obj.logprob))

        if diffs:
            per_prompt_kls.append(float(np.mean(diffs)))

    if not per_prompt_kls:
        return 0.0
    return float(np.mean(per_prompt_kls))

def _postprocess_outputs(outputs, task_datas):
    rewards = []
    avg_rewards = []
    for output, data in zip(outputs, task_datas):
        response = output.outputs[0].text
        r = reward_function(response, data["numbers"], data["target"])
        rewards.append(r)
        avg_rewards.append(r["reward"])
    return {
        "rewards": rewards,
        "avg_reward": float(np.mean(avg_rewards)) if avg_rewards else 0.0,
    }

def main(args):
    # Ensure local Ray
    os.environ.pop("RAY_ADDRESS", None)
    os.environ.pop("RAY_HEAD_IP", None)
    os.environ.pop("RAY_GCS_SERVER_ADDRESS", None)
    ray.init(address="local", include_dashboard=False, ignore_reinit_error=True)

    # Logging
    logging_dir = f"{args.experiment_dir}/countdown_nccl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=logging_dir)

    # Prepare an HF checkpoint for vLLM to load
    model_saves_dir = f"{logging_dir}/model_saves"
    os.makedirs(model_saves_dir, exist_ok=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16
    ).to("cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    base_model_path = f"{model_saves_dir}/base_model"
    if os.path.exists(base_model_path):
        shutil.rmtree(base_model_path)
    os.makedirs(base_model_path, exist_ok=True)
    tokenizer.save_pretrained(base_model_path)
    base_model.save_pretrained(base_model_path)
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load data
    data_path = "countdown/data/countdown.json"
    with open(data_path, "r") as f:
        task_datas = json.load(f)
    task_datas = task_datas[:200]

    # Launch engines
    engines, pgs = launch_engines(args.num_engines, base_model_path)

    # Init inter-engine communicator once
    master_address = get_ip()
    master_port = get_open_port()
    ray.get([
        engines[i].collective_rpc.remote(
            "init_inter_engine_group", args=(master_address, master_port, i, args.num_engines)
        )
        for i in range(args.num_engines)
    ])

    def cleanup():
        for llm in engines:
            try:
                ray.kill(llm)
            except Exception:
                pass
        for pg in pgs:
            try:
                remove_placement_group(pg)
            except Exception:
                pass
        ray.shutdown()

    def sig_handler(sig, frame):
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    # Build the KL behavioral-anchor pool while all engines are still at the
    # base-model weights. We do this on engine 0 only and reuse the cached
    # token IDs / pi_ref logprobs throughout training.
    #
    # Build is gated only on `num_anchor_prompts > 0` (NOT kl_beta), so we can
    # run a measurement-only baseline (kl_beta=0) that still logs kl/* curves
    # for apples-to-apples comparison against an anchored run (kl_beta>0).
    #
    # Pool semantics:
    #   - The cache holds `pool_size` sequences total (default = num_anchor_prompts).
    #   - Each generation samples num_anchor_prompts indices from the pool and
    #     scores ONLY that subset on every candidate. All candidates within a
    #     generation see the same subset so ES reward normalization stays clean.
    #   - When pool_size == num_anchor_prompts, sampling is the identity and
    #     behavior matches the original fixed-anchor regime.
    #
    # Also optionally builds a held-out *eval* anchor (disjoint row indices,
    # different sampling seed) measured periodically on the center weights but
    # never used as a penalty target. This is what you compare against to claim
    # the anchor generalizes beyond the prompts it was trained on.
    anchor_cache = []
    eval_anchor_cache = []
    train_anchor_indices = {}
    pool_size = max(args.anchor_pool_size, args.num_anchor_prompts)

    if args.num_anchor_prompts > 0:
        mode = "penalty" if args.kl_beta > 0.0 else "measurement-only"
        rotation_note = (
            f"pool={pool_size}, batch={args.num_anchor_prompts}/gen (rotated)"
            if pool_size > args.num_anchor_prompts
            else f"fixed batch={args.num_anchor_prompts}"
        )
        print(
            f"[anchor] Sampling {pool_size} train prompts from "
            f"{args.anchor_datasets} for KL anchor ({mode}, beta={args.kl_beta}, "
            f"{rotation_note})..."
        )
        anchor_prompts, train_anchor_indices = load_anchor_prompts(
            args.anchor_datasets, pool_size, args.anchor_seed
        )
        print(f"[anchor] Loaded {len(anchor_prompts)} train anchor prompts; sampling base-model refs...")
        anchor_cache = build_anchor_cache(
            engines[0], anchor_prompts, args.anchor_max_tokens
        )
        total_ref_tokens = sum(c["num_completion_tokens"] for c in anchor_cache)
        print(
            f"[anchor] Cached {len(anchor_cache)} train anchor sequences "
            f"({total_ref_tokens} ref completion tokens total)."
        )
    else:
        print("[anchor] KL anchor disabled (num_anchor_prompts=0).")

    if args.eval_anchor_seed is not None and args.num_anchor_prompts > 0:
        n_eval = (
            args.num_eval_anchor_prompts
            if args.num_eval_anchor_prompts is not None
            else args.num_anchor_prompts
        )
        if n_eval > 0:
            print(
                f"[anchor] Sampling {n_eval} held-out eval prompts from "
                f"{args.anchor_datasets} (eval_anchor_seed={args.eval_anchor_seed}, "
                f"disjoint from train anchor indices)..."
            )
            eval_prompts, _ = load_anchor_prompts(
                args.anchor_datasets,
                n_eval,
                args.eval_anchor_seed,
                exclude=train_anchor_indices,
            )
            print(
                f"[anchor] Loaded {len(eval_prompts)} held-out eval anchor prompts; "
                f"sampling base-model refs..."
            )
            eval_anchor_cache = build_anchor_cache(
                engines[0], eval_prompts, args.anchor_max_tokens
            )
            eval_ref_tokens = sum(c["num_completion_tokens"] for c in eval_anchor_cache)
            print(
                f"[anchor] Cached {len(eval_anchor_cache)} held-out eval anchor sequences "
                f"({eval_ref_tokens} ref completion tokens total). "
                f"This anchor is measured but NEVER used as a penalty."
            )
    elif args.eval_anchor_seed is not None:
        print(
            "[anchor] --eval_anchor_seed set but --num_anchor_prompts=0; "
            "skipping eval anchor build."
        )

    # Sanity check: held-out eval KL on the initial (un-fine-tuned) center
    # weights should be ~0 since the cache was built from these exact weights.
    # Logged at step -1 so the curve has a clean "before training" anchor point.
    if eval_anchor_cache:
        eval_handle = evaluate_anchor_handle(engines[0], eval_anchor_cache)
        eval_outputs = ray.get(eval_handle)
        initial_kl_eval = _postprocess_anchor_outputs(eval_outputs, eval_anchor_cache)
        print(
            f"[anchor] Initial held-out eval KL = {initial_kl_eval:.4f} "
            f"(should be ~0 since cache was built against these same weights)"
        )
        writer.add_scalar("kl/eval", initial_kl_eval, -1)

    # Engines start with identical weights (loaded from the same HF checkpoint)
    # For each iteration:
    # - Explore: per-seed add noise -> eval -> subtract noise (GPU-only)
    # - Compute ES update on engine 0 only
    # - Broadcast weights from engine 0 to all engines (NCCL)
    for i in range(args.num_iterations):
        print(f"\n\n=== Generation {i} ===")
        total_iter_start = time.time()

        # Random seeds for population
        seeds = [random.randint(0, 1_000_000) for _ in range(args.population_size)]
        seeds_perf = {}

        # Sample this generation's anchor batch from the pool. All candidates
        # within this generation will be scored on the same batch so that the
        # KL term contributes a consistent baseline to the ES normalization.
        # When pool size == batch size, this is just the full anchor_cache.
        if anchor_cache and len(anchor_cache) > args.num_anchor_prompts:
            anchor_batch = random.sample(anchor_cache, args.num_anchor_prompts)
            if args.verbose:
                print(
                    f"[anchor] Generation {i}: rotated batch of "
                    f"{len(anchor_batch)} prompts from pool of {len(anchor_cache)}"
                )
        else:
            anchor_batch = anchor_cache

        # Round-robin scheduling
        seed_iter = iter(seeds)
        inflight = {}
        results_this_gen = []

        # Kick off an eval on each engine
        for eng_idx, llm in enumerate(engines):
            try:
                seed = next(seed_iter)
            except StopIteration:
                break
            # Add exploration noise
            ray.get(llm.collective_rpc.remote("perturb_self_weights", args=(seed, args.sigma, False)))
            handle, start_ts = evaluate_countdown_handle(llm, task_datas)
            anchor_handle = evaluate_anchor_handle(llm, anchor_batch)
            inflight[handle] = {
                "engine": llm,
                "engine_idx": eng_idx,
                "seed": seed,
                "start_ts": start_ts,
                "anchor_handle": anchor_handle,
            }

        while inflight:
            done, _ = ray.wait(list(inflight.keys()), num_returns=1)
            h = done[0]
            meta = inflight.pop(h)

            outputs = ray.get(h)
            metrics = _postprocess_outputs(outputs, task_datas)

            # KL behavioral anchor: ray.get the prompt-logprob handle (likely
            # already complete since it was dispatched alongside the task call).
            # Postprocess against the same per-generation batch the dispatch
            # used (NOT the full pool), so token offsets line up.
            kl_estimate = 0.0
            if meta["anchor_handle"] is not None:
                anchor_outputs = ray.get(meta["anchor_handle"])
                kl_estimate = _postprocess_anchor_outputs(anchor_outputs, anchor_batch)
            metrics["kl"] = kl_estimate
            metrics["augmented_reward"] = metrics["avg_reward"] - args.kl_beta * kl_estimate

            elapsed = time.time() - meta["start_ts"]

            seeds_perf[meta["seed"]] = metrics
            results_this_gen.append(
                {
                    "seed": meta["seed"],
                    "avg_reward": metrics["avg_reward"],
                    "kl": kl_estimate,
                    "augmented_reward": metrics["augmented_reward"],
                    "time": elapsed,
                }
            )

            llm = meta["engine"]
            # Remove exploration noise
            ray.get(llm.collective_rpc.remote("restore_self_weights", args=(meta["seed"], args.sigma)))

            # Schedule next seed on this engine
            try:
                next_seed = next(seed_iter)
            except StopIteration:
                continue

            ray.get(llm.collective_rpc.remote("perturb_self_weights", args=(next_seed, args.sigma, False)))
            handle, start_ts = evaluate_countdown_handle(llm, task_datas)
            anchor_handle = evaluate_anchor_handle(llm, anchor_batch)
            inflight[handle] = {
                "engine": llm,
                "engine_idx": meta["engine_idx"],
                "seed": next_seed,
                "start_ts": start_ts,
                "anchor_handle": anchor_handle,
            }
            if args.verbose:
                print(f"Scheduled seed {next_seed} on engine {meta['engine_idx']}")

        # Raw task-reward stats (still logged so we can see what the task signal
        # looks like independent of the anchor penalty).
        all_avg_rewards = [v["avg_reward"] for v in seeds_perf.values()]
        mean_reward = float(np.mean(all_avg_rewards)) if all_avg_rewards else 0.0
        std_reward = float(np.std(all_avg_rewards)) if all_avg_rewards else 0.0
        min_reward = float(np.min(all_avg_rewards)) if all_avg_rewards else 0.0
        max_reward = float(np.max(all_avg_rewards)) if all_avg_rewards else 0.0

        # KL penalty stats (training anchor — used in penalty when beta>0)
        all_kls = [v.get("kl", 0.0) for v in seeds_perf.values()]
        mean_kl = float(np.mean(all_kls)) if all_kls else 0.0
        max_kl = float(np.max(all_kls)) if all_kls else 0.0
        min_kl = float(np.min(all_kls)) if all_kls else 0.0

        # Augmented reward = R_task - beta * KL_hat, used to drive the ES update.
        all_aug_rewards = [v["augmented_reward"] for v in seeds_perf.values()]
        mean_aug = float(np.mean(all_aug_rewards)) if all_aug_rewards else 0.0
        std_aug = float(np.std(all_aug_rewards)) if all_aug_rewards else 0.0

        print(
            f"Task reward  mean={mean_reward:.4f}  std={std_reward:.4f}  "
            f"min={min_reward:.4f}  max={max_reward:.4f}"
        )
        if anchor_cache:
            print(
                f"KL anchor    mean={mean_kl:.4f}  min={min_kl:.4f}  max={max_kl:.4f}  "
                f"(beta={args.kl_beta})"
            )
            print(f"Augmented reward (task - beta*KL)  mean={mean_aug:.4f}  std={std_aug:.4f}")

        for k in seeds_perf:
            seeds_perf[k]["norm_reward"] = (
                seeds_perf[k]["augmented_reward"] - mean_aug
            ) / (std_aug + 1e-8)
            if args.verbose:
                print(f"Seed {k} normalized reward: {seeds_perf[k]['norm_reward']}")

        writer.add_scalar("reward/mean", mean_reward, i)
        writer.add_scalar("reward/std", std_reward, i)
        writer.add_scalar("reward/min", min_reward, i)
        writer.add_scalar("reward/max", max_reward, i)
        if anchor_cache:
            writer.add_scalar("kl/mean", mean_kl, i)
            writer.add_scalar("kl/min", min_kl, i)
            writer.add_scalar("kl/max", max_kl, i)
            writer.add_scalar("reward/augmented_mean", mean_aug, i)
            writer.add_scalar("reward/augmented_std", std_aug, i)

        # Compute ES update ONLY on engine 0 (baseline is already current weights)
        per_seed_coeffs = [
            (seed, (args.alpha / args.population_size) * float(seeds_perf[seed]["norm_reward"]))
            for seed in seeds
        ]

        perturb_start = time.time()
        handles = []
        for seed, coeff in per_seed_coeffs:
            # Use sigma_or_scale=1.0 so the applied scale is `coeff`
            handles.append(engines[0].collective_rpc.remote("perturb_self_weights", args=(seed, coeff, False)))
        ray.get(handles)
        if args.verbose:
            print(f"Applied perturbations in {time.time() - perturb_start}s")
        writer.add_scalar("time/perturbation_application", time.time() - perturb_start, i)

        # Broadcast updated weights from engine 0 to all engines (avoid CPU copies)
        broadcast_start = time.time()
        ray.get([e.collective_rpc.remote("broadcast_all_weights", args=(0,)) for e in engines])
        if args.verbose:
            print(f"Broadcasted updated weights in {time.time() - broadcast_start}s")
        writer.add_scalar("time/broadcast", time.time() - broadcast_start, i)

        # Periodic held-out eval anchor measurement on the (unperturbed) center
        # weights. Engine 0 holds the freshly updated weights and is the
        # broadcast source, so it's the canonical place to measure. Done every
        # --eval_anchor_every generations to keep cost negligible. The forced
        # measurement at i == num_iterations - 1 guarantees the curve always
        # has a final data point regardless of period.
        if (
            eval_anchor_cache
            and args.eval_anchor_every > 0
            and (i % args.eval_anchor_every == 0 or i == args.num_iterations - 1)
        ):
            eval_start = time.time()
            eval_handle = evaluate_anchor_handle(engines[0], eval_anchor_cache)
            eval_outputs = ray.get(eval_handle)
            kl_eval = _postprocess_anchor_outputs(eval_outputs, eval_anchor_cache)
            writer.add_scalar("kl/eval", kl_eval, i)
            print(
                f"KL eval (held-out, center weights) = {kl_eval:.4f}  "
                f"[measured in {time.time() - eval_start:.2f}s]"
            )

        # Logging per-result and timing
        if args.verbose:
            for res_idx, res in enumerate(results_this_gen):
                print(f"IDX:{res_idx} Seed {res['seed']} avg_reward: {res['avg_reward']}, time: {res['time']}s")
        total_iter_end = time.time()
        writer.add_scalar("time/iteration", total_iter_end - total_iter_start, i)
        print(f"wall clock time for iteration {i}: {total_iter_end - total_iter_start}s")
        print(f"=== Generation {i} finished ===\n")

    # Save final model weights (all engines are in sync; save from engine 0)
    final_model_path = f"{model_saves_dir}/final_model_iteration_{args.num_iterations}"
    os.makedirs(final_model_path, exist_ok=True)
    ray.get(
        engines[0].collective_rpc.remote(
            "save_self_weights_to_disk", args=(f"{final_model_path}/pytorch_model.pth",)
        )
    )
    print(f"Final model weights saved to {final_model_path}.")

    cleanup()

if __name__ == "__main__":
    args = parse_args()
    main(args)
