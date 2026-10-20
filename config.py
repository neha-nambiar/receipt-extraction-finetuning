"""Shared settings and small helpers. No torch import, runs on Windows CPU and the pod."""
import atexit
import csv
import math
import os
import sys
from pathlib import Path

from PIL import Image

# Paths
ROOT = Path(__file__).resolve().parent
RAW_IMAGES = ROOT / "dataset" / "images"
RAW_JSONS = ROOT / "dataset" / "jsons"
DATA_DIR = ROOT / "data"          # Derived_Data_Folder
RUNS_DIR = ROOT / "runs"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"
RUN_LOG = RUNS_DIR / "run_log.csv"

# Fields
SCORED_FIELDS = ["std_start_time", "orig_start_time", "std_invoice_time", "orig_invoice_time",
                 "place", "std_curr", "orig_curr", "std_total", "orig_total",
                 "seller_name", "seller_address", "invoice_number", "tax_number", "detail"]
DROPPED_FIELDS = ["type", "departure", "arrival", "std_end_time", "orig_end_time"]
LIST_FIELDS = ["orig_curr", "seller_name", "seller_address"]

# Data / inference
SEED = 3407
MAX_PIXELS = 1_048_576            
MIN_PIXELS = 65_536               
FUZZY_THRESHOLD = 0.85
MAX_NEW_TOKENS = 1536
GEN_TIMEOUT_S = 120
EVAL_BATCH_SIZE = 8               

# Model
MODEL_NAME = "unsloth/Qwen3-VL-4B-Instruct"
LOAD_IN_4BIT = False
LAYERS = "language+vision"        # or "language-only"

# Cost
POD_PRICE_PER_H = 0.89            
API_MODEL = "gpt-4.1-mini"
API_KEY_ENV = "OPENAI_API_KEY"
RUNPOD_KEY_ENV = "RUNPOD_API_KEY"  
RUNPOD_STOP_URL = "https://rest.runpod.io/v1/pods/{}/stop"
API_PRICE_IN = 0.40 / 1e6         # USD per input token
API_PRICE_OUT = 1.60 / 1e6        # USD per output token
BUDGET_USD = 25.0
BUDGET_WARN_USD = 20.0
ARITH_BONUS = 0.2                 # GRPO reward bonus, read by scorer.reward

PROMPT = """Read this purchase receipt and return one JSON object with exactly these keys, in this order. Reply with the JSON only.

std_start_time: purchase date as YYYY-MM-DD
orig_start_time: purchase date exactly as printed
std_invoice_time: invoice date as YYYY-MM-DD
orig_invoice_time: invoice date exactly as printed
place: "Country-City" where the purchase happened
std_curr: ISO 4217 currency code, e.g. "USD"
orig_curr: list with the currency symbol as printed (if any) and the city of purchase, e.g. ["$", "New York"]
std_total: total paid as a number string with 2 decimals, e.g. "169.82"
orig_total: total exactly as printed
seller_name: list of seller names
seller_address: list with the seller location as "Country-City" (usually the same as place), not the street address
invoice_number: receipt or invoice number
tax_number: seller tax or VAT number
detail: list of line items, each {"content": item text, "amount": amount as printed, "ifTax": true for tax lines, else false}

Use "" for a missing text value and [] for a missing list. Do not invent values that are not on the receipt."""


def load_env():
    p = ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        if k == RUNPOD_KEY_ENV and v:
            os.environ[k] = v
        elif k:
            os.environ.setdefault(k, v)


load_env()


def check_settings():
    # Stop before any model load if MAX_PIXELS is missing or invalid
    v = globals().get("MAX_PIXELS")
    if v is None or not isinstance(v, int) or isinstance(v, bool) or not 262_144 <= v <= 2_097_152:
        print(f"invalid setting MAX_PIXELS={v!r}: must be an int from 262,144 to 2,097,152")
        sys.exit(1)


def resize_dims(w, h, max_pixels):
    # Scale down to at most max_pixels, keep aspect ratio; floor keeps under the cap
    if w * h <= max_pixels:
        return w, h
    s = math.sqrt(max_pixels / (w * h))
    return max(1, math.floor(w * s)), max(1, math.floor(h * s))


def load_image(path):
    # RGB image, downscaled only when over MAX_PIXELS
    with Image.open(path) as im:
        img = im.convert("RGB")
    size = resize_dims(img.width, img.height, MAX_PIXELS)
    if size != img.size:
        img = img.resize(size, Image.LANCZOS)
    return img


def on_pod():
    return "RUNPOD_POD_ID" in os.environ


def log_run(name, start, end):
    # Append one line per run; start/end are datetimes
    minutes = (end - start).total_seconds() / 60
    cost = minutes / 60 * POD_PRICE_PER_H if on_pod() else 0.0
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not RUN_LOG.exists()
    with open(RUN_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["run_name", "start_iso", "end_iso", "minutes", "cost_usd"])
        w.writerow([name, start.isoformat(), end.isoformat(), f"{minutes:.2f}", f"{cost:.4f}"])


def print_spend():
    # Cumulative estimated spend from the run log
    total = 0.0
    if RUN_LOG.exists():
        with open(RUN_LOG, newline="", encoding="utf-8") as f:
            total = sum(float(r["cost_usd"] or 0) for r in csv.DictReader(f))
    print(f"cumulative estimated spend: ${total:.2f} of ${BUDGET_USD:.2f}")
    if total >= BUDGET_WARN_USD:
        print(f"WARNING: spend is ${total:.2f}, the ${BUDGET_USD:.0f} budget is close to exhausted")
    return total


def check_stop_pod():
    if not on_pod():
        print("not on a RunPod pod, --stop-pod ignored")
        return False
    if not os.environ.get(RUNPOD_KEY_ENV):
        print("RUNPOD_API_KEY is not set (shell or .env)")
        sys.exit(1)
    return True


def stop_pod_at_exit():
    # Stop (never terminate) the pod when the process exits, so /workspace is kept
    import urllib.request

    def _stop():
        pod_id = os.environ["RUNPOD_POD_ID"]
        print(f"stopping pod {pod_id}", flush=True)
        req = urllib.request.Request(
            RUNPOD_STOP_URL.format(pod_id), method="POST",
            headers={"Authorization": f"Bearer {os.environ[RUNPOD_KEY_ENV]}",
                     "User-Agent": "curl/8.0"})
        try:
            urllib.request.urlopen(req, timeout=30).close()
        except Exception as e:
            print(f"pod stop failed: {type(e).__name__}: {e}")

    atexit.register(_stop)
