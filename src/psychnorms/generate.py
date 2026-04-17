#!/usr/bin/env python3
"""
Psychometric norm generation via offline vLLM inference.

Generates ratings for all cue words × norms using direct GPU access.
Designed for SLURM batch execution with:
  - Manifest-based continuation (skip fully completed model/temp combos)
  - CSV-based per-cue resume (pick up where previous run left off)
  - Multi-stage retry logic (Scale → Std → Temp → Refusal)

Usage:
    # Deterministic (temp=0, 1 repetition)
    python -m psychnorms.generate --model qwen32b --temperature 0.0 --repetitions 1

    # Stochastic  (temp=1, 5 repetitions)
    python -m psychnorms.generate --model qwen32b --temperature 1.0 --repetitions 5
"""

import argparse
import csv
import json
import os
import random
import re
import signal
import sys
import time
from collections import Counter
from pathlib import Path

from psychnorms.registry import MODEL_REGISTRY
from psychnorms.prompt_templates import PROMPT_TEMPLATES

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

# ──────────────────────────────────────────────────────────────────────
# Task preparation helpers
# ──────────────────────────────────────────────────────────────────────

def load_tasks(path: str, norm_filter: list = None):
    """
    Load tasks from an input CSV.

    Supports two modes:
      - **Wide mode** (default): input has a 'word' column → generates
        a task for every configured norm × word.
      - **Targeted mode**: input has both 'word' and 'norm' columns →
        loads specific (word, norm) pairs.
    """
    tasks = []

    valid_norms = sorted(
        [n for n in PROMPT_TEMPLATES.keys() if n != "free_association"]
    )

    # Apply norm filter if provided
    if norm_filter:
        filtered = [n for n in valid_norms if n in norm_filter]
        print(f"DATA: Norm filter active — {len(filtered)}/{len(valid_norms)} norms selected.")
        valid_norms = filtered

    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        headers = rdr.fieldnames or []

        is_targeted = "norm" in headers and "word" in headers

        if is_targeted:
            print("DATA: Targeted mode (input contains 'norm' column).")
            for row in rdr:
                w, n = row.get("word"), row.get("norm")
                if w and n:
                    tasks.append({"word": w, "norm": n})
        else:
            print("DATA: Wide mode. Generating ALL norms for every word.")
            for row in rdr:
                word = row.get("word")
                if not word:
                    continue
                for norm in valid_norms:
                    tasks.append({"word": word, "norm": norm})

    return tasks


def load_completed(output_dir: str, pattern: str, scale_validator=None):
    """
    Scan files matching *pattern* in *output_dir* and return:
      - completed_counts: Counter of (word, norm) → valid-completion count
      - failed: set of (word, norm) tuples with errors

    If *scale_validator* is provided, only count responses that are both
    numeric AND within the norm's valid scale range (matching the
    postprocessing pipeline's ``is_effective_valid`` metric).
    """
    import glob

    completed_counts = Counter()
    failed = set()

    search_path = os.path.join(output_dir, pattern)
    files = glob.glob(search_path)

    if not files and os.path.exists(search_path):
        files = [search_path]

    print(
        f"📂 History: Scanning {len(files)} files matching '{pattern}' …"
    )

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    continue
                for row in reader:
                    word = row.get("word", "").strip()
                    norm = row.get("norm", "").strip()
                    cleaned_text = row.get("cleaned_text", "").strip()
                    cleaned_rating = row.get("cleaned_rating", "").strip()
                    raw_response = row.get("raw_response", "")

                    if (
                        cleaned_text == "EXCEPTION"
                        or cleaned_rating == "NO_NUMBER_FOUND"
                        or "EXCEPTION:" in raw_response
                    ):
                        failed.add((word, norm))
                        continue

                    # Validate scale bounds if validator is provided
                    if scale_validator is not None:
                        try:
                            val = float(cleaned_rating)
                        except (ValueError, TypeError):
                            failed.add((word, norm))
                            continue
                        if scale_validator.is_out_of_scale(norm, val):
                            failed.add((word, norm))
                            continue

                    completed_counts[(word, norm)] += 1
        except Exception as e:
            print(f"⚠️  Warning: Could not read {file_path}: {e}")

    return completed_counts, failed


def _remove_failed_entries(output_csv: str, failed: set):
    """Remove failed entries from the output CSV so they can be rerun."""
    import shutil
    import tempfile

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, newline="", encoding="utf-8"
        ) as temp_file:
            temp_path = temp_file.name
            with open(output_csv, "r", encoding="utf-8") as input_file:
                reader = csv.DictReader(input_file)
                writer = csv.DictWriter(temp_file, fieldnames=reader.fieldnames)
                writer.writeheader()
                for row in reader:
                    word = row.get("word", "").strip()
                    norm = row.get("norm", "").strip()
                    if (word, norm) not in failed:
                        writer.writerow(row)

        shutil.move(temp_path, output_csv)
        print(f"🧹 Removed {len(failed)} failed entries from {output_csv}")
    except Exception as e:
        print(f"⚠️  Warning: Could not clean failed entries: {e}")
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


# ──────────────────────────────────────────────────────────────────────
# Main generation runner (offline vLLM)
# ──────────────────────────────────────────────────────────────────────

def run_offline(
    model_key: str,
    input_csv: str,
    output_csv: str,
    verbose: bool = False,
    shuffle: bool = False,
    repetitions: int = 1,
    gen_kwargs: dict = None,
    shard_id: int = 0,
    num_shards: int = 1,
    write_mode: str = "append",
    max_retries_per_item: int = 3,
    norm_filter: list = None,
):
    """
    Offline vLLM generation with manifest-based continuation and multi-stage retry.

    Steps:
      1. Check manifest — skip if fully complete.
      2. Load CSV history for per-cue continuation.
      3. Batch-generate remaining tasks via OfflineVLLMBackend.
      4. Multi-stage retry (Scale → Parse → Temp → Refusal).
      5. Update manifest.
    """
    from psychnorms.backend import OfflineVLLMBackend
    from psychnorms.progress_manifest import ProgressManifest
    from psychnorms.retry_utils import (
        RefusalClassifier,
        RetryPromptFactory,
        ScaleValidator,
    )

    gen_kwargs = gen_kwargs or {}

    refusal_detector = RefusalClassifier()
    scale_validator = ScaleValidator()

    # 1. Resolve model
    cfg = MODEL_REGISTRY[model_key]
    model_path = cfg.get("model_name")
    if not model_path:
        raise ValueError(f"No 'model_name' in registry for '{model_key}'")
    adapter = cfg["adapter"]

    # Merge adapter request_kwargs with CLI gen_kwargs
    adapter_kwargs = adapter.request_kwargs("offline_vllm")
    gen_kwargs = {**adapter_kwargs, **gen_kwargs}

    # 2. Tensor-parallel size
    tp_size = int(os.getenv("TENSOR_PARALLEL_SIZE", 1))

    # 3. Initialize backend
    print(f"🖥️  Starting Offline Pipeline for {model_key} (TP={tp_size})…")
    try:
        backend = OfflineVLLMBackend(model_path, tensor_parallel_size=tp_size)
    except Exception as e:
        print(f"❌ Failed to initialize vLLM: {e}")
        return

    # 4. Load tasks
    all_tasks = load_tasks(input_csv, norm_filter=norm_filter)

    if num_shards > 1:
        print(f"📦 Shard {shard_id}/{num_shards}")
        tasks = [t for i, t in enumerate(all_tasks) if i % num_shards == shard_id]
        print(f"   Shard content: {len(tasks)} tasks")
    else:
        tasks = all_tasks

    # 5. Manifest check
    p = Path(output_csv)
    output_dir = p.parent
    manifest_path = output_dir / "progress_manifest.json"
    temperature = gen_kwargs.get("temperature", 0.0)

    manifest = ProgressManifest(str(manifest_path))
    if manifest.should_skip(model_key, temperature):
        print(
            f"✅ Manifest: {model_key} @ temp={temperature} is COMPLETE. Skipping."
        )
        return

    # 6. CSV-based per-cue continuation
    #    Scan ALL existing data files for this model + condition to count
    #    already-completed (word, norm) pairs. This covers data from all
    #    legacy formats: vllm_batched, hf_batched, offline, and sharded.

    import glob as _glob

    # Filename patterns (same as postprocess.py)
    _SHARD_RE = re.compile(
        r"(.+?)_(?:vllm|hf|mock)_batched(_hitemp)?_shard(\d+)of(\d+)\.csv"
    )
    _BATCHED_RE = re.compile(
        r"(.+?)_(?:vllm|hf|mock)_batched(_hitemp)?\.csv"
    )
    _OFFLINE_RE = re.compile(r"(.+?)_offline_temp(\d+\.?\d*)\.csv")

    is_stochastic = temperature > 0.0
    condition_label = "stochastic" if is_stochastic else "deterministic"

    all_csvs = sorted(_glob.glob(str(output_dir / "*.csv")))
    history_files = []

    for csv_path in all_csvs:
        fname = os.path.basename(csv_path)
        # Skip non-data files
        if any(x in fname for x in ["missing", "manifest", "merged", "test_"]):
            continue

        file_model = None
        file_is_stochastic = None

        m = _SHARD_RE.match(fname)
        if m:
            file_model = m.group(1).strip("_")
            file_is_stochastic = bool(m.group(2))

        if file_model is None:
            m = _BATCHED_RE.match(fname)
            if m:
                file_model = m.group(1).strip("_")
                file_is_stochastic = bool(m.group(2))

        if file_model is None:
            m = _OFFLINE_RE.match(fname)
            if m:
                file_model = m.group(1).strip("_")
                file_is_stochastic = float(m.group(2)) > 0

        if file_model is None:
            continue

        # Match model AND condition
        if file_model == model_key and file_is_stochastic == is_stochastic:
            history_files.append(csv_path)

    print(
        f"📂 History: Found {len(history_files)} existing {condition_label} "
        f"file(s) for {model_key}:"
    )
    for hf in history_files:
        size_mb = os.path.getsize(hf) / 1e6
        print(f"   → {os.path.basename(hf)} ({size_mb:.1f} MB)")

    completed_counts = Counter()
    failed = set()
    for hf in history_files:
        counts, fails = load_completed(
            str(output_dir), os.path.basename(hf),
            scale_validator=scale_validator,
        )
        completed_counts += counts
        failed.update(fails)

    total_existing = sum(completed_counts.values())
    print(f"📂 History: {total_existing:,} existing valid completions found")

    # Remove from failed set any items that already have enough completions
    for key in list(failed):
        if completed_counts[key] >= repetitions:
            failed.discard(key)

    final_tasks = []
    for t in tasks:
        key = (t["word"], t["norm"])
        current_count = completed_counts[key]
        needed = max(0, repetitions - current_count)
        for _ in range(needed):
            final_tasks.append(t)

    tasks = final_tasks
    failed_tasks = [{"word": word, "norm": norm} for word, norm in failed]
    tasks.extend(failed_tasks)

    if shuffle:
        random.shuffle(tasks)

    print(f"📊 Offline tasks to run: {len(tasks)}")
    if not tasks:
        manifest.mark_complete(model_key, temperature)
        manifest.save()
        print("✅ No tasks remaining. Marked complete in manifest.")
        return

    # 7. Batch generation
    BATCH_SIZE = 100_000
    header = [
        "model_key",
        "backend",
        "endpoint_url",
        "norm",
        "word",
        "raw_response",
        "cleaned_text",
        "cleaned_rating",
        "temperature",
        "retry_attempt",
        "attempt_type",
    ]

    if write_mode == "new" and os.path.exists(output_csv):
        timestamp = int(time.time())
        p = Path(output_csv)
        output_csv = str(p.parent / f"{p.stem}_recovery_{timestamp}{p.suffix}")
        print(f"🔄 Write mode 'new': Output → {output_csv}")
        file_exists = False
        mode_flag = "w"
    else:
        file_exists = os.path.exists(output_csv)
        mode_flag = "a" if file_exists else "w"

    completed_total = 0
    start_time = time.time()

    with open(output_csv, mode_flag, newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(header)

        def chunked(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        for chunk in chunked(tasks, BATCH_SIZE):
            print(f"🧠 Preparing batch of {len(chunk)} conversations…")

            conversations = []
            meta_data = []
            for t in chunk:
                norm, word = t["norm"], t["word"]
                msgs = adapter.build_messages(norm, word)
                conversations.append(msgs)
                meta_data.append((norm, word))

            gen_out = backend.chat_batch(conversations, **gen_kwargs)

            rows = []
            for i, raw_text in enumerate(gen_out):
                norm, word = meta_data[i]
                cleaned = adapter.clean_text(raw_text)
                rating = adapter.parse_rating(cleaned)
                temp = gen_kwargs.get("temperature", 0.0)
                row = [
                    model_key,
                    "offline_vllm",
                    "OFFLINE_GPU",
                    norm,
                    word,
                    json.dumps(raw_text),
                    cleaned,
                    rating,
                    temp,
                    0,
                    "zero_shot",
                ]
                rows.append(row)

            w.writerows(rows)
            f.flush()
            completed_total += len(rows)
            print(f"💾 Wrote {len(rows)} rows. Total: {completed_total}/{len(tasks)}")

    print(f"🏁 Initial pass complete in {int(time.time() - start_time)}s")

    # ── Retry passes ──────────────────────────────────────────────────

    print(f"\n🔄 Starting retry passes (max {max_retries_per_item} per item)…")

    # Track cumulative retry attempts per (word, norm) across all stages
    retry_attempt_counts = Counter()

    def get_failures(filepath, model_key, validator, detector):
        fail_map = {}
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["model_key"] != model_key:
                    continue
                w, n = row["word"], row["norm"]
                rating = row.get("cleaned_rating", "")
                raw = row.get("raw_response", "")
                try:
                    val = float(rating)
                    is_valid_num = True
                except (ValueError, TypeError):
                    val = None
                    is_valid_num = False

                if is_valid_num:
                    if validator.is_out_of_scale(n, val):
                        fail_map[(w, n)] = ("scale", raw)
                    else:
                        # Valid and in-scale → NOT a failure, remove if previously flagged
                        fail_map.pop((w, n), None)
                else:
                    if detector.is_refusal(raw):
                        fail_map[(w, n)] = ("refusal", raw)
                    else:
                        fail_map[(w, n)] = ("parse", raw)
        return fail_map

    task_lookup = {(t["word"], t["norm"]): t for t in all_tasks}

    for stage in [1, 2, 3, 4]:
        current_failures = get_failures(
            output_csv, model_key, scale_validator, refusal_detector
        )
        if not current_failures:
            print("✅ No failures remaining. Retries complete.")
            break

        retry_tasks = []
        retry_meta = []
        skipped_cap = 0

        for (w, n), (f_type, raw_text) in current_failures.items():
            if (w, n) not in task_lookup:
                continue

            # Enforce per-item retry cap
            if retry_attempt_counts[(w, n)] >= max_retries_per_item:
                skipped_cap += 1
                continue

            should_run = False
            new_prompt_modifier = None
            temp_mod = 0.0

            if stage == 1 and f_type == "scale":
                should_run = True
                new_prompt_modifier = "scale"
            elif stage == 2 and f_type == "parse":
                should_run = True
            elif stage == 3 and f_type in ["parse", "scale"]:
                should_run = True
                temp_mod = 0.1
            elif stage == 4 and f_type == "refusal":
                should_run = True
                new_prompt_modifier = "refusal"

            if should_run:
                orig_msgs = adapter.build_messages(n, w)
                if new_prompt_modifier == "scale":
                    rule = scale_validator.scales.get(n, {})
                    new_msgs = RetryPromptFactory.get_scale_prompt(
                        orig_msgs, rule.get("min", 0), rule.get("max", 5)
                    )
                elif new_prompt_modifier == "refusal":
                    if refusal_detector.is_safety_refusal(raw_text):
                        new_msgs = RetryPromptFactory.get_safety_prompt(orig_msgs)
                    else:
                        new_msgs = RetryPromptFactory.get_roleplay_prompt(orig_msgs)
                else:
                    new_msgs = orig_msgs

                retry_tasks.append(new_msgs)
                retry_meta.append((w, n, temp_mod))

        if skipped_cap > 0:
            print(f"⏭️  Stage {stage}: Skipped {skipped_cap} items (hit retry cap of {max_retries_per_item})")

        if not retry_tasks:
            print(f"⏩ Stage {stage}: No applicable tasks.")
            continue

        print(f"🚀 Stage {stage} retry: {len(retry_tasks)} tasks…")

        current_temp = gen_kwargs.get("temperature", 0.0)
        stage_temp = current_temp + (0.1 if stage == 3 else 0.0)

        retry_kwargs = {**gen_kwargs, "temperature": stage_temp}
        gen_out = backend.chat_batch(retry_tasks, **retry_kwargs)

        rows = []
        for i, raw_text in enumerate(gen_out):
            w, n, _ = retry_meta[i]
            cleaned = adapter.clean_text(raw_text)
            rating = adapter.parse_rating(cleaned)
            raw_log = f"[RETRY_S{stage}] " + json.dumps(raw_text)
            retry_type_map = {1: "retry_s1_scale", 2: "retry_s2_parse", 3: "retry_s3_temp", 4: "retry_s4_refusal"}
            row = [
                model_key,
                "offline_retry",
                "OFFLINE_GPU",
                n,
                w,
                raw_log,
                cleaned,
                rating,
                stage_temp,
                stage,
                retry_type_map.get(stage, f"retry_s{stage}"),
            ]
            rows.append(row)
            # Track retry attempt for this item
            retry_attempt_counts[(w, n)] += 1

        with open(output_csv, "a", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerows(rows)
        print(f"💾 Saved {len(rows)} retries.")

    # Report items that exhausted retries
    exhausted = sum(1 for v in retry_attempt_counts.values() if v >= max_retries_per_item)
    if exhausted > 0:
        print(f"\n⚠️  {exhausted} items exhausted all {max_retries_per_item} retries and remain failed.")

    # 8. Update manifest
    total_unique_tasks = len(all_tasks)
    manifest.update_progress(
        model_key, temperature, total_unique_tasks, completed_total, repetitions
    )
    manifest.save()


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Generate psychometric norms via offline vLLM inference."
    )
    ap.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model key from registry",
    )
    ap.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "resources" / "psychNorms_vocab.csv"),
        help="Input vocabulary CSV",
    )
    ap.add_argument("--output", default=None, help="Output CSV path")
    ap.add_argument(
        "--write-mode",
        choices=["append", "new"],
        default="append",
        help="'append' adds to existing file; 'new' creates a fresh file.",
    )
    ap.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature"
    )
    ap.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")
    ap.add_argument(
        "--repetition-penalty", type=float, default=1.0, help="Repetition penalty"
    )
    ap.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Number of repetitions per cue (1 for deterministic, 5 for stochastic)",
    )
    ap.add_argument("--shuffle", action="store_true", help="Randomise task order")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--shard-id", type=int, default=0, help="Shard ID (0-indexed)"
    )
    ap.add_argument(
        "--num-shards", type=int, default=1, help="Total number of shards"
    )
    ap.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Tensor parallel size (overrides TENSOR_PARALLEL_SIZE env var)",
    )
    ap.add_argument(
        "--max-retries-per-item",
        type=int,
        default=3,
        help="Max retry attempts per failed (word, norm) pair before giving up (default: 3)",
    )
    ap.add_argument(
        "--norms",
        type=str,
        default=None,
        help="Comma-separated list of norms to generate (default: all). "
             "E.g. --norms auditory_lancaster,gustatory_lancaster",
    )

    args = ap.parse_args()

    # Tensor-parallel override
    if args.tensor_parallel_size is not None:
        os.environ["TENSOR_PARALLEL_SIZE"] = str(args.tensor_parallel_size)

    gen_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
    }

    # Output path
    output_dir = PROJECT_ROOT / "outputs" / "raw_behavior" / "model_norms"
    output_dir.mkdir(parents=True, exist_ok=True)

    default_out = f"{args.model}_offline_temp{args.temperature}.csv"
    output = args.output or str(output_dir / default_out)

    # Sharding suffix
    if args.num_shards > 1:
        p = Path(output)
        suffix_str = f"_shard{args.shard_id}of{args.num_shards}"
        if not p.stem.endswith(suffix_str):
            output = str(p.parent / f"{p.stem}{suffix_str}{p.suffix}")
        print(f"📦 Sharded output path: {output}")

    # Ctrl+C handler
    def _sigint(sig, frame):
        print("\nInterrupted. Exiting…")
        sys.exit(130)

    signal.signal(signal.SIGINT, _sigint)

    # Run
    run_offline(
        model_key=args.model,
        input_csv=args.input,
        output_csv=output,
        verbose=args.verbose,
        shuffle=args.shuffle,
        repetitions=args.repetitions,
        gen_kwargs=gen_kwargs,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        write_mode=args.write_mode,
        max_retries_per_item=args.max_retries_per_item,
        norm_filter=args.norms.split(",") if args.norms else None,
    )


if __name__ == "__main__":
    main()
