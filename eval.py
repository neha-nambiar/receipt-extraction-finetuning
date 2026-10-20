"""Run a model on a split, save predictions and scores to results/<tag>__<split>/.

python eval.py --model base|api|<adapter dir> --split test|val [--limit N]
For GRPO pass the adapter path from runs/grpo/selected.txt (or the .txt file itself).
torch/unsloth are only imported in the local-model path, never for --model api.
"""
import argparse
import base64
import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import config
from config import (API_KEY_ENV, API_MODEL, API_PRICE_IN, API_PRICE_OUT, DATA_DIR, EVAL_BATCH_SIZE,
                    FUZZY_THRESHOLD, GEN_TIMEOUT_S, LOAD_IN_4BIT, MAX_NEW_TOKENS, MAX_PIXELS,
                    MIN_PIXELS, MODEL_NAME, POD_PRICE_PER_H, PROMPT, RAW_IMAGES, RESULTS_DIR, RUNS_DIR)
from scorer import parse_output, clean_pred, write_results


def model_tag(model):
    # base, api_<API_MODEL>, or the run name of an adapter (runs/<run>/...)
    if model in ("base", "api"):
        return "base" if model == "base" else f"api_{API_MODEL}"
    p = Path(model).resolve()
    try:
        return p.relative_to(RUNS_DIR.resolve()).parts[0]
    except ValueError:
        return p.parent.name if p.name == "adapter" else p.name


def image_path(base):
    # Raw image for a base name (.jpg or .JPG)
    for ext in (".jpg", ".JPG"):
        p = RAW_IMAGES / f"{base}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"no image for {base}")


def make_api_generator():
    # OpenAI Chat Completions
    if not os.environ.get(API_KEY_ENV):
        print(f"{API_KEY_ENV} is not set (shell or .env)")
        sys.exit(1)
    from openai import OpenAI
    # SDK retries off, my own 2/4/8 s retries below
    client = OpenAI(api_key=os.environ[API_KEY_ENV], timeout=GEN_TIMEOUT_S, max_retries=0)

    def api_generate(img):
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=95)
        url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": PROMPT}]}]
        waits = [2, 4, 8]
        for attempt in range(len(waits) + 1):
            try:
                t0 = time.perf_counter()
                resp = client.chat.completions.create(
                    model=API_MODEL, messages=messages, temperature=0,
                    max_tokens=MAX_NEW_TOKENS, timeout=GEN_TIMEOUT_S)
                latency = time.perf_counter() - t0
                u = resp.usage
                return dict(raw_text=resp.choices[0].message.content or "", latency_s=latency,
                            in_tokens=u.prompt_tokens if u else None,
                            out_tokens=u.completion_tokens if u else None, failed=False)
            except Exception as e:
                print(f"  API error ({type(e).__name__}), attempt {attempt + 1}/{len(waits) + 1}")
                if attempt < len(waits):
                    time.sleep(waits[attempt])
        return dict(raw_text="", latency_s=None, in_tokens=None, out_tokens=None, failed=True)

    return api_generate


def make_local_generator(model):
    # Base model or a LoRA adapter dir with Unsloth / HF generate, greedy, batched (pod only).
    # Returns generate(list of images) -> list of result dicts.
    if model == "base":
        name = MODEL_NAME
    else:
        p = Path(model)
        if p.is_file():
            # e.g. runs/grpo/selected.txt holds the selected adapter path
            p = Path(p.read_text(encoding="utf-8").strip())
        if not p.is_dir():
            print(f"adapter directory not found: {p}")
            sys.exit(1)
        name = str(p)

    # Heavy imports only here, never for --model api
    from unsloth import FastVisionModel
    import torch

    print(f"loading {name}")
    model_, tokenizer = FastVisionModel.from_pretrained(
        name, load_in_4bit=LOAD_IN_4BIT, max_seq_length=4096)
    tokenizer.image_processor.size = {"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS}
    FastVisionModel.for_inference(model_)

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    # Left padding so every prompt in a batch ends at the same position
    tokenizer.tokenizer.padding_side = "left"
    pad_id = tokenizer.tokenizer.pad_token_id
    eos = model_.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])

    def failed_row():
        return dict(raw_text="", latency_s=None, in_tokens=None, out_tokens=None, failed=True)

    def run_batch(imgs):
        inputs = tokenizer(images=imgs, text=[prompt_text] * len(imgs), add_special_tokens=False,
                           padding=True, return_tensors="pt").to("cuda")
        n_pad = inputs["input_ids"].shape[1]
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model_.generate(**inputs, do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
                                  max_time=GEN_TIMEOUT_S, use_cache=True, pad_token_id=pad_id)
        wall = time.perf_counter() - t0
        rows = []
        for i in range(len(imgs)):
            gen = out[i][n_pad:].tolist()
            # Tokens up to and including the first EOS; the rest is padding
            end = next((k for k, t in enumerate(gen) if t in eos_ids), None)
            gen = gen[:end + 1] if end is not None else [t for t in gen if t != pad_id]
            text = tokenizer.decode(gen, skip_special_tokens=True)
            n_in = int(inputs["attention_mask"][i].sum())
            # max_time cuts the whole batch at 120 s without raising. A receipt that had not
            # reached EOS by then timed out; receipts that finished are kept.
            if wall >= GEN_TIMEOUT_S and end is None:
                rows.append(dict(raw_text=text, latency_s=None, in_tokens=n_in,
                                 out_tokens=len(gen), failed=True))
            else:
                # Per-page latency = batch wall time shared over the batch (throughput time)
                rows.append(dict(raw_text=text, latency_s=wall / len(imgs), in_tokens=n_in,
                                 out_tokens=len(gen), failed=False))
        return rows

    def local_generate(imgs):
        try:
            return run_batch(imgs)
        except Exception as e:
            print(f"  batch error ({type(e).__name__}: {e}), retrying one by one")
            torch.cuda.empty_cache()
        rows = []
        for img in imgs:
            try:
                rows += run_batch([img])
            except Exception as e:
                print(f"  generation error ({type(e).__name__}: {e})")
                torch.cuda.empty_cache()
                rows.append(failed_row())
        return rows

    return local_generate


def make_generator(model):
    # Returns (generate(list of images) -> list of dicts, backend, batch_size); each dict has
    # raw_text, latency_s, in_tokens, out_tokens, failed
    if model == "api":
        api = make_api_generator()
        return (lambda imgs: [api(im) for im in imgs]), "openai_chat_completions", 1
    return make_local_generator(model), "hf_generate", EVAL_BATCH_SIZE


def load_done(pred_file):
    # Rows already saved, keyed by base name
    done = {}
    if pred_file.exists():
        with open(pred_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[r["base"]] = r
    return done


def cost_block(rows, model):
    # Latency and cost per page over receipts that did not fail
    ok = [r for r in rows if not r.get("failed") and r.get("latency_s") is not None]
    lat = [r["latency_s"] for r in ok]
    is_api = model == "api"
    if not ok:
        cost = 0.0
    elif is_api:
        cost = float(np.mean([(r.get("in_tokens") or 0) * API_PRICE_IN +
                              (r.get("out_tokens") or 0) * API_PRICE_OUT for r in ok]))
    else:
        cost = POD_PRICE_PER_H / 3600 * sum(lat) / len(ok)
    return dict(mean_latency_s=round(float(np.mean(lat)), 3) if lat else None,
                p90_latency_s=round(float(np.percentile(lat, 90)), 3) if lat else None,
                mean_cost_per_page_usd=cost,
                n_timed=len(ok), n_failed=sum(1 for r in rows if r.get("failed")),
                pod_price_per_h=None if is_api else POD_PRICE_PER_H,
                api_price_in=API_PRICE_IN if is_api else None,
                api_price_out=API_PRICE_OUT if is_api else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="base, api, or an adapter directory")
    ap.add_argument("--split", required=True, choices=["test", "val"])
    ap.add_argument("--limit", type=int, default=None, help="only the first N receipts (partial run)")
    ap.add_argument("--stop-pod", action="store_true", help="stop (not terminate) the pod at exit")
    ap.add_argument("--suffix", default="", help="appended to the results folder name, e.g. _bs1check")
    args = ap.parse_args()

    config.check_settings()
    config.print_spend()
    if args.stop_pod and config.check_stop_pod():
        config.stop_pod_at_exit()
    start = datetime.now()

    with open(DATA_DIR / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    if args.split == "test":
        bases = [r["base"] for r in manifest["receipts"] if r["split"] == "test"]
    else:
        bases = [r["base"] for r in manifest["receipts"] if r["val"]]
    partial = args.limit is not None and args.limit < len(bases)
    if args.limit is not None:
        bases = bases[:args.limit]

    generate, backend, batch_size = make_generator(args.model)

    tag = model_tag(args.model) + args.suffix
    run_dir = RESULTS_DIR / f"{tag}__{args.split}"
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_file = run_dir / "predictions.jsonl"
    done = load_done(pred_file)
    todo = [b for b in bases if b not in done]
    print(f"{tag} on {args.split}: {len(bases)} receipts, {len(bases) - len(todo)} already done")

    # Batch receipts of similar image size together so less padding is wasted
    if batch_size > 1:
        px = {r["base"]: r["width"] * r["height"] for r in manifest["receipts"]}
        todo.sort(key=lambda b: px[b])
    batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]

    # Local model: one warm-up batch, output discarded, not in latency/cost
    if args.model != "api" and todo:
        print(f"warm-up on {len(batches[0])} receipts (discarded)")
        generate([config.load_image(image_path(b)) for b in batches[0]])

    n = 0
    with open(pred_file, "a", encoding="utf-8", newline="\n") as f:
        for batch in batches:
            outs = generate([config.load_image(image_path(b)) for b in batch])
            for base, out in zip(batch, outs):
                n += 1
                obj = None if out["failed"] else parse_output(out["raw_text"])
                row = dict(base=base, raw_text=out["raw_text"],
                           pred=clean_pred(obj) if obj is not None else None,
                           parse_failed=obj is None, failed=out["failed"],
                           latency_s=out["latency_s"], in_tokens=out.get("in_tokens"),
                           out_tokens=out.get("out_tokens"))
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                done[base] = row
                lat = f"{out['latency_s']:.1f}s" if out["latency_s"] is not None else "-"
                print(f"[{n}/{len(todo)}] {base} {lat}"
                      f"{' FAILED' if out['failed'] else ''}{' parse-fail' if obj is None else ''}")
            f.flush()

    rows = [done[b] for b in bases]
    labels = {}
    for b in bases:
        with open(DATA_DIR / "labels" / f"{b}.json", encoding="utf-8") as lf:
            labels[b] = json.load(lf)
    meta = dict(model=tag, api_model=API_MODEL if args.model == "api" else None,
                split=args.split, partial=partial, max_pixels=MAX_PIXELS,
                fuzzy_threshold=FUZZY_THRESHOLD, backend=backend, batch_size=batch_size,
                cost=cost_block(rows, args.model))
    s = write_results(run_dir, rows, labels, meta)
    print(f"overall F1 {s['overall']['f1']:.3f}, parse failures {s['parse_failures']} "
          f"({s['parse_failure_rate']}%){' [partial]' if partial else ''}")
    config.log_run(f"eval_{tag}__{args.split}", start, datetime.now())


if __name__ == "__main__":
    main()
