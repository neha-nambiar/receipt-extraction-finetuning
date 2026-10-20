"""Scoring of predicted receipt JSON against ground truth. No torch import."""
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from dateutil import parser as dateparser
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

from config import ARITH_BONUS, DATA_DIR, FUZZY_THRESHOLD, LIST_FIELDS, SCORED_FIELDS

TOTAL_FIELDS = ("std_total", "orig_total")


def parse_output(text):
    # First JSON object found in the text (skips code fences and chatter text), None if none
    dec = json.JSONDecoder()
    i = text.find("{")
    while i != -1:
        try:
            obj, _ = dec.raw_decode(text, i)
            if isinstance(obj, dict):
                return obj
        except ValueError:
            pass
        i = text.find("{", i + 1)
    return None


def is_scalar(v):
    return isinstance(v, (str, int, float)) and not isinstance(v, bool)


def clean_pred(obj):
    # Keep only scored fields; a wrong-typed field becomes empty
    out = {}
    for f in SCORED_FIELDS:
        v = obj.get(f)
        if f == "detail":
            if isinstance(v, list) and all(isinstance(x, dict) for x in v):
                out[f] = [{"content": str(x["content"]) if is_scalar(x.get("content")) else "",
                           "amount": str(x["amount"]) if is_scalar(x.get("amount")) else "",
                           "ifTax": x.get("ifTax") if isinstance(x.get("ifTax"), bool) else False}
                          for x in v]
            else:
                out[f] = []
        elif f in LIST_FIELDS:
            if is_scalar(v):
                out[f] = [str(v)]
            elif isinstance(v, list) and all(is_scalar(x) for x in v):
                out[f] = [str(x) for x in v]
            else:
                out[f] = []
        else:
            out[f] = str(v) if is_scalar(v) else ""
    return out


def parse_amount(v):
    # Number from an amount value, or None when unparseable
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if not isinstance(v, str):
        return None
    s = re.sub(r"[^0-9,.\-]", "", v)
    # Last separator followed by 1-2 final digits is the decimal point
    m = re.search(r"[,.](\d{1,2})$", s)
    if m:
        s = re.sub(r"[,.]", "", s[:m.start()]) + "." + m.group(1)
    else:
        s = re.sub(r"[,.]", "", s)
    try:
        return float(s)
    except ValueError:
        return None


def is_blank(x):
    return x is None or (isinstance(x, str) and not x.strip())


def is_empty(field, v):
    # Empty: None, blank string, [], list of only blank values, or a zero total
    if is_blank(v):
        return True
    if isinstance(v, list):
        return all(is_blank(x) for x in v)
    if field in TOTAL_FIELDS:
        return parse_amount(v) == 0
    return False


def to_str(v):
    return "" if v is None else str(v)


def norm_text(s):
    # Lowercase, drop punctuation (keeps CJK and other letters), collapse whitespace
    s = "".join(c for c in to_str(s).lower() if c.isalnum() or c.isspace())
    return " ".join(s.split())


def text_sim(a, b):
    return fuzz.ratio(norm_text(a), norm_text(b)) / 100


def parse_date(s):
    # Date (plus time if present) or None; same rule for both sides
    s = to_str(s).replace("年", "-").replace("月", "-").replace("日", "")
    try:
        return dateparser.parse(s.strip(), default=datetime(1900, 1, 1))
    except Exception:
        return None


def amount_match(a, b):
    x, y = parse_amount(a), parse_amount(b)
    return x is not None and y is not None and abs(x - y) <= 0.01


def match_lists(pred_vals, gt_vals, pair_ok, pair_sim):
    # Hungarian pairing: maximize valid pairs first, then similarity.
    # Returns (n_valid_pairs, mean_sim_of_valid_pairs)
    if not pred_vals or not gt_vals:
        return 0, 0.0
    ok = np.array([[1.0 if pair_ok(p, g) else 0.0 for g in gt_vals] for p in pred_vals])
    sim = np.array([[pair_sim(p, g) for g in gt_vals] for p in pred_vals])
    rows, cols = linear_sum_assignment(-(ok + 0.001 * sim))
    sims = [sim[r, c] for r, c in zip(rows, cols) if ok[r, c]]
    return len(sims), (float(np.mean(sims)) if sims else 0.0)


def as_list(v):
    # Raw labels may hold a plain string where a list is expected
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def list_values(v):
    # Normalized, deduped, non-blank values
    out = []
    for x in as_list(v):
        n = norm_text(x) if is_scalar(x) else ""
        if n and n not in out:
            out.append(n)
    return out


def item(x):
    # (content, amount, ifTax) from a raw or cleaned Line_Item
    if not isinstance(x, dict):
        return "", "", "false"
    return to_str(x.get("content")), x.get("amount"), str(x.get("ifTax", False)).lower()


def detail_ok(p, g):
    return text_sim(p[0], g[0]) >= FUZZY_THRESHOLD and amount_match(p[1], g[1]) and p[2] == g[2]


def field_match(field, pred, gt):
    # (matched, sim) for two non-empty values
    if field in TOTAL_FIELDS:
        return (amount_match(pred, gt), 1.0)
    if field in ("std_start_time", "std_invoice_time"):
        return (to_str(pred).strip() == to_str(gt).strip(), 1.0)
    if field in ("orig_start_time", "orig_invoice_time"):
        a, b = parse_date(pred), parse_date(gt)
        if a is not None and b is not None:
            return (a == b, 1.0)
        return (norm_text(pred) == norm_text(gt), 1.0)
    if field == "place":
        s = text_sim(pred, gt)
        return (s >= FUZZY_THRESHOLD, s)
    if field in ("invoice_number", "tax_number", "std_curr"):
        return ("".join(to_str(pred).lower().split()) == "".join(to_str(gt).lower().split()), 1.0)
    if field in LIST_FIELDS:
        p, g = list_values(pred), list_values(gt)
        n, s = match_lists(p, g, lambda a, b: text_sim(a, b) >= FUZZY_THRESHOLD, text_sim)
        if not p and not g:
            return (True, 1.0)
        return (len(p) == len(g) and n == len(g), s)
    raise ValueError(f"unknown field {field}")


def compare_field(field, pred, gt):
    # One outcome per field per receipt, with TP/FP/FN counts and a similarity
    pe, ge = is_empty(field, pred), is_empty(field, gt)
    if field == "detail":
        p = [item(x) for x in as_list(pred)] if not pe else []
        g = [item(x) for x in as_list(gt)] if not ge else []
        if ge and pe:
            return dict(outcome="correct", tp=0, fp=0, fn=0, sim=1.0)
        if ge:
            return dict(outcome="Hallucination", tp=0, fp=len(p), fn=0, sim=0.0)
        if pe:
            return dict(outcome="Missing", tp=0, fp=0, fn=len(g), sim=0.0)
        n, s = match_lists(p, g, detail_ok, lambda a, b: text_sim(a[0], b[0]))
        ok = n == len(p) == len(g)
        return dict(outcome="correct" if ok else "Misread", tp=n, fp=len(p) - n, fn=len(g) - n, sim=s)
    if ge and pe:
        return dict(outcome="correct", tp=0, fp=0, fn=0, sim=1.0)
    if ge:
        return dict(outcome="Hallucination", tp=0, fp=1, fn=0, sim=0.0)
    if pe:
        return dict(outcome="Missing", tp=0, fp=0, fn=1, sim=0.0)
    ok, s = field_match(field, pred, gt)
    if ok:
        return dict(outcome="correct", tp=1, fp=0, fn=0, sim=s)
    return dict(outcome="Misread", tp=0, fp=1, fn=1, sim=s)


def score_receipt(pred, gt):
    # None (parse failure) scores as an all-empty prediction
    pred = clean_pred(pred) if isinstance(pred, dict) else {}
    return {f: compare_field(f, pred.get(f), gt.get(f)) for f in SCORED_FIELDS}


def row_pred(row):
    # Re-parse from raw_text so re-scoring matches; failed requests are parse failures
    if row.get("failed"):
        return None
    raw = row.get("raw_text")
    if raw is None:
        return row.get("pred") if isinstance(row.get("pred"), dict) else None
    return parse_output(raw)


def score_rows(rows, labels):
    # base -> per-field compare results
    return {r["base"]: score_receipt(row_pred(r), labels[r["base"]]) for r in rows}


def prf(tp, fp, fn):
    # 1.0 for all three when nothing to count
    if tp + fp + fn == 0:
        return 1.0, 1.0, 1.0
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def aggregate(results):
    # Per-field and overall micro P/R/F1 from a list of score_receipt results
    fields = {}
    tot = dict(tp=0, fp=0, fn=0)
    for f in SCORED_FIELDS:
        c = dict(tp=0, fp=0, fn=0, correct=0, missing=0, hallucination=0, misread=0)
        for res in results:
            x = res[f]
            for k in ("tp", "fp", "fn"):
                c[k] += x[k]
            c[x["outcome"].lower()] += 1
        p, r, f1 = prf(c["tp"], c["fp"], c["fn"])
        fields[f] = dict(precision=p, recall=r, f1=f1, **c)
        for k in tot:
            tot[k] += c[k]
    p, r, f1 = prf(tot["tp"], tot["fp"], tot["fn"])
    return dict(overall=dict(precision=p, recall=r, f1=f1, **tot), fields=fields)


def score_run(rows, labels):
    # Scores dict for a run: overall, fields, parse failures
    per = score_rows(rows, labels)
    fails = sorted(r["base"] for r in rows if row_pred(r) is None)
    n = len(rows)
    out = dict(n_receipts=n, parse_failures=len(fails),
               parse_failure_rate=round(100 * len(fails) / n, 1) if n else 0.0,
               parse_failure_bases=fails)
    out.update(aggregate(list(per.values())))
    return out


def error_examples(rows, labels, per_receipt):
    # Up to 20 receipts with the most wrong fields, ties by base name
    by_base = {r["base"]: r for r in rows}
    wrong = []
    for b, res in per_receipt.items():
        errs = {f: x["outcome"] for f, x in res.items() if x["outcome"] != "correct"}
        if errs:
            wrong.append((-len(errs), b, errs))
    wrong.sort()
    out = []
    for neg, b, errs in wrong[:20]:
        r = by_base[b]
        out.append(dict(base=b, n_wrong=-neg, raw_text=r.get("raw_text"), pred=row_pred(r),
                        target=labels[b], errors=errs))
    if len(out) < 10:
        print(f"only {len(out)} receipts with errors saved")
    return out


def write_results(run_dir, rows, labels, meta):
    # Write scores.json (meta + scores) and errors.json; returns the scores dict
    run_dir = Path(run_dir)
    scores = score_run(rows, labels)
    per = score_rows(rows, labels)
    fails = set(scores["parse_failure_bases"])
    errors = dict(examples=error_examples(rows, labels, per),
                  parse_failures=[dict(base=r["base"], raw_text=r.get("raw_text"))
                                  for r in rows if r["base"] in fails])
    with open(run_dir / "scores.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({**meta, **scores}, f, ensure_ascii=False, indent=1)
    with open(run_dir / "errors.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(errors, f, ensure_ascii=False, indent=1)
    return scores


SCORE_KEYS = ("n_receipts", "parse_failures", "parse_failure_rate", "parse_failure_bases",
              "overall", "fields")


def rescore(run_dir):
    # Re-score a saved run, keeping non-score metadata from an existing scores.json
    run_dir = Path(run_dir)
    with open(run_dir / "predictions.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    labels = {}
    for r in rows:
        with open(DATA_DIR / "labels" / f"{r['base']}.json", encoding="utf-8") as f:
            labels[r["base"]] = json.load(f)
    meta = {}
    if (run_dir / "scores.json").exists():
        with open(run_dir / "scores.json", encoding="utf-8") as f:
            meta = {k: v for k, v in json.load(f).items() if k not in SCORE_KEYS}
    s = write_results(run_dir, rows, labels, meta)
    print(f"{run_dir.name}: {s['n_receipts']} receipts, overall F1 {s['overall']['f1']:.3f}, "
          f"parse failures {s['parse_failures']} ({s['parse_failure_rate']}%)")


def sums_ok(label):
    # Line_Item amounts sum to std_total within 0.01 (1+ items, all amounts parse)
    items = [x for x in as_list(label.get("detail")) if isinstance(x, dict)]
    total = parse_amount(label.get("std_total"))
    amounts = [parse_amount(x.get("amount")) for x in items]
    return (len(items) >= 1 and total is not None and None not in amounts
            and abs(sum(amounts) - total) <= 0.01)


def arith_bonus(pred, gt):
    # Bonus only when both ground truth and prediction sum up, so the ~10% of labels
    # whose items do not add up never push GRPO away from the correct answer
    return ARITH_BONUS if sums_ok(gt) and sums_ok(pred) else 0.0


def reward(text, gt):
    # Rule-based GRPO reward
    obj = parse_output(text)
    if obj is None:
        return -1.0
    pred = clean_pred(obj)
    scores = []
    for f in SCORED_FIELDS:
        r = compare_field(f, pred[f], gt.get(f))
        if r["outcome"] == "correct":
            scores.append(0.3 if is_empty(f, gt.get(f)) else r["sim"])
        elif r["outcome"] == "Hallucination":
            scores.append(-0.5)
        else:
            scores.append(0.0)
    total = sum(scores) / len(scores) + arith_bonus(pred, gt)
    assert -0.5 <= total <= 1.2, total
    return total


def selfcheck():
    # Labels must score perfectly against themselves
    labels = {}
    for p in sorted((DATA_DIR / "labels").glob("*.json")):
        with open(p, encoding="utf-8") as f:
            labels[p.stem] = json.load(f)
    problems = []
    for case in ("self", "detail reversed"):
        res = []
        for gt in labels.values():
            pred = dict(gt)
            if case != "self" and isinstance(gt.get("detail"), list):
                pred["detail"] = gt["detail"][::-1]
            res.append(score_receipt(pred, gt))
        for f, x in aggregate(res)["fields"].items():
            if x["f1"] < 1.0:
                problems.append(f"{case}: {f} F1 {x['f1']:.4f}")
    # Own Target_Text should earn a solid reward
    for b, gt in labels.items():
        r = reward(json.dumps(gt, ensure_ascii=False, separators=(",", ":")), gt)
        if r < 0.3:
            problems.append(f"reward {b}: {r:.3f} < 0.3")
    if reward("no json", next(iter(labels.values()))) != -1.0:
        problems.append("reward on 'no json' is not -1.0")

    for a, b in (("205,50", "$205.50"), ("1.234,50", "1,234.50")):
        if not amount_match(a, b) or compare_field("std_total", a, b)["outcome"] != "correct":
            problems.append(f"amount {a!r} vs {b!r} not equal")
    print(f"{len(labels)} labels checked")
    if problems:
        print("SELFCHECK FAIL")
        for p in problems:
            print(" ", p)
        raise SystemExit(1)
    print("SELFCHECK PASS")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--selfcheck"]:
        selfcheck()
        raise SystemExit(0)
    if len(args) != 1:
        raise SystemExit("usage: python scorer.py results/<run_folder> | --selfcheck")
    rescore(args[0])
