"""Data prep: pair, verify, filter, group and split the raw receipts into data/.

Run: python prep.py
"""
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from config import DATA_DIR, DROPPED_FIELDS, RAW_IMAGES, RAW_JSONS, ROOT, SCORED_FIELDS, SEED
from scorer import is_empty

RAW_ROOT = ROOT / "dataset"

TEST_TARGET, TEST_MIN, TEST_MAX = 300, 280, 320
SUBSETS = [100, 250, 500, 1000]
N_VAL = 50


def guard_output_dir():
    out, raw = DATA_DIR.resolve(), RAW_ROOT.resolve()
    if out == raw or out.is_relative_to(raw):
        print(f"ERROR: output location {out} is inside the Raw_Dataset ({raw})")
        sys.exit(1)


def fingerprint():
    # Sorted (relative name, sha256) for every file in dataset/images and dataset/jsons
    out = []
    for folder in (RAW_IMAGES, RAW_JSONS):
        for p in sorted(folder.iterdir()):
            if p.is_file():
                out.append((p.relative_to(RAW_ROOT).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest()))
    return out


def count_raw(fp):
    n_img = sum(1 for name, _ in fp if name.startswith("images/"))
    return n_img, len(fp) - n_img


def pair_files(image_names, json_names):
    # Match .jpg and .json files 1:1 by case-sensitive base name; extensions compared lowercased.
    # Returns (pairs, skipped): pairs = sorted [(base, image_name, json_name)],
    # skipped = sorted [(base, reason, detail)].
    imgs, jsons = {}, {}
    for names, ext, d in ((image_names, ".jpg", imgs), (json_names, ".json", jsons)):
        for n in names:
            p = Path(n)
            if p.suffix.lower() == ext:
                d.setdefault(p.stem, []).append(n)
    pairs, skipped = [], []
    for base in sorted(set(imgs) | set(jsons)):
        i, j = imgs.get(base, []), jsons.get(base, [])
        if len(i) > 1 or len(j) > 1:
            skipped.append((base, "duplicate", ", ".join(sorted(i + j))))
        elif not i or not j:
            skipped.append((base, "unpaired", (i or j)[0]))
        else:
            pairs.append((base, i[0], j[0]))
    return pairs, skipped


def check_image(path):
    with Image.open(path) as img:
        img.load()
        return img.width, img.height


def read_label(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"top-level JSON value is {type(obj).__name__}, not an object")
    return obj


def filter_label(raw):
    # Keep the 14 Scored_Fields in order with raw values; raises ValueError on bad keys or items
    missing = [k for k in SCORED_FIELDS if k not in raw]
    unknown = sorted(k for k in raw if k not in SCORED_FIELDS and k not in DROPPED_FIELDS)
    if missing or unknown:
        raise ValueError(f"missing keys {missing}, unknown keys {unknown}")
    label = {k: raw[k] for k in SCORED_FIELDS}
    if not isinstance(label["detail"], list):
        raise ValueError(f"detail is {type(label['detail']).__name__}, not a list")
    items = []
    for i, it in enumerate(label["detail"]):
        if not isinstance(it, dict):
            raise ValueError(f"detail[{i}] is {type(it).__name__}, not an object")
        lack = [k for k in ("content", "amount", "ifTax") if k not in it]
        if lack:
            raise ValueError(f"detail[{i}] missing keys {lack}")
        items.append({"content": it["content"], "amount": it["amount"], "ifTax": it["ifTax"]})
    label["detail"] = items
    return label


def target_text(label):
    # Compact JSON, keys in SCORED_FIELDS order, non-ASCII kept as is
    return json.dumps({k: label[k] for k in SCORED_FIELDS}, ensure_ascii=False, separators=(",", ":"))


def filter_rows(rows, skipped):
    # Adds "label" to each row; drops rows whose label fails filtering
    out = []
    for r in rows:
        try:
            r["label"] = filter_label(r["raw"])
        except ValueError as e:
            skip(skipped, r["base"], "bad_keys", e)
            continue
        out.append(r)
    return out


def write_labels(rows):
    # Write data/labels/<base>.json = Target_Text after a round-trip check
    labels_dir = DATA_DIR / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    for p in labels_dir.glob("*.json"):
        p.unlink()
    for r in rows:
        text = target_text(r["label"])
        if json.loads(text) != r["label"]:
            print(f"ERROR: Target_Text round trip differs for {r['base']}")
            sys.exit(1)
        with open(labels_dir / f"{r['base']}.json", "w", encoding="utf-8", newline="\n") as f:
            f.write(text)


def skip(skipped, base, reason, detail):
    # Record a skipped receipt and print it right away
    detail = " ".join(str(detail).split())  # keep the tsv one line per entry
    skipped.append((base, reason, detail))
    print(f"skip {base}: {reason} ({detail})")


def write_lines(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(line + "\n" for line in lines)


def verify_pairs(pairs, skipped):
    # Open every image and label; returns sorted [{base, width, height, raw}] of valid pairs
    rows = []
    for base, img_name, json_name in pairs:
        try:
            w, h = check_image(RAW_IMAGES / img_name)
        except Exception as e:
            skip(skipped, base, "bad_image", f"{type(e).__name__}: {e}")
            continue
        try:
            raw = read_label(RAW_JSONS / json_name)
        except Exception as e:
            skip(skipped, base, "bad_json", f"{type(e).__name__}: {e}")
            continue
        rows.append({"base": base, "width": w, "height": h, "raw": raw})
    return rows


def first_seller(label):
    # First seller_name value, lowercased and trimmed; "" when the list is empty
    names = label["seller_name"]
    if isinstance(names, list):
        return str(names[0]).strip().lower() if names else ""
    return str(names).strip().lower()


def group_receipts(rows):
    # Union-find over shared keys; returns numbered by first member name
    bases = sorted(r["base"] for r in rows)
    parent = {b: b for b in bases}

    def find(b):
        while parent[b] != b:
            parent[b] = parent[parent[b]]
            b = parent[b]
        return b

    nums = {}
    for b in bases:
        m = re.match(r"^(\d+)_", b)
        nums[b] = m.group(1) if m else None
    use_num = len({n for n in nums.values() if n is not None}) >= 2
    if not use_num:
        print("leading file number is constant; grouping by seller_name only")

    owner = {}  
    for r in sorted(rows, key=lambda r: r["base"]):
        b = r["base"]
        keys = []
        seller = first_seller(r["label"])
        if seller:
            keys.append("seller:" + seller)
        if use_num and nums[b] is not None:
            keys.append("num:" + nums[b])
        for k in keys:
            if k in owner:
                ra, rb = find(owner[k]), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
            else:
                owner[k] = b

    members = {}
    for b in bases:
        members.setdefault(find(b), []).append(b)
    groups = {}
    for i, ms in enumerate(sorted(members.values(), key=lambda ms: ms[0])):
        for b in ms:
            groups[b] = f"g{i:04d}"
    return groups


def make_split(rows, groups):
    # Template-aware test/pool split, nested subsets and validation
    rows = sorted(rows, key=lambda r: r["base"])
    bases = [r["base"] for r in rows]

    by_group = {}
    for b in bases:
        by_group.setdefault(groups[b], []).append(b)
    glist = sorted(by_group.items())
    random.Random(SEED).shuffle(glist)

    # Add whole groups in shuffled order until the target size is reached
    test, used = [], set()
    for gid, ms in glist:
        if len(test) >= TEST_TARGET:
            break
        if len(test) + len(ms) <= TEST_MAX:
            test.extend(ms)
            used.add(gid)

    spanning = []
    if len(test) < TEST_MIN:
        rest = sorted([g for g in glist if g[0] not in used], key=lambda g: -len(g[1]))  # stable: seeded order on ties
        for gid, ms in rest:
            if len(test) >= TEST_MIN:
                break
            take = ms[:TEST_MIN - len(test)]
            test.extend(take)
            used.add(gid)
            if len(take) < len(ms):
                spanning.append(gid)
    clean = not spanning
    if spanning:
        print(f"WARNING: split is not cleanly separated; groups in both sets: {', '.join(spanning)}")

    test_set = set(test)
    pool = sorted(b for b in bases if b not in test_set)
    if len(pool) < SUBSETS[-1] + N_VAL:
        print(f"ERROR: Training_Pool has {len(pool)} receipts, need at least {SUBSETS[-1] + N_VAL}")
        sys.exit(1)
    random.Random(SEED + 1).shuffle(pool)
    subset_of = {}
    for size in reversed(SUBSETS):  
        for b in pool[:size]:
            subset_of[b] = size
    val = set(random.Random(SEED + 2).sample(sorted(pool[SUBSETS[-1]:]), N_VAL))


    pool_set = set(pool)
    assert not (test_set & pool_set), "test and pool overlap"
    assert test_set | pool_set == set(bases), "test + pool do not cover all valid receipts"
    assert TEST_MIN <= len(test_set) <= TEST_MAX, f"test size {len(test_set)}"
    for small, big in zip(SUBSETS, SUBSETS[1:]):
        assert set(pool[:small]) <= set(pool[:big])
    for size in SUBSETS:
        assert len({b for b, s in subset_of.items() if s <= size}) == size, f"subset {size} size"
    assert len(val) == N_VAL and val <= pool_set and not (val & set(pool[:SUBSETS[-1]]))
    if clean:
        test_groups = {groups[b] for b in test_set}
        assert not test_groups & {groups[b] for b in pool_set}, "group in both sets"

    receipts = []
    for r in rows:
        b = r["base"]
        receipts.append({
            "base": b, "split": "test" if b in test_set else "pool", "subset": subset_of.get(b),
            "val": b in val, "group": groups[b],
            "width": r["width"], "height": r["height"], "n_items": len(r["label"]["detail"]),
        })
    return {"seed": SEED, "clean_split": clean, "spanning_groups": spanning, "receipts": receipts}


def write_manifest(manifest):
    path = DATA_DIR / "manifest.json"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
        f.write("\n")


def print_data_stats(rows, skipped, n_bases):
    # Pair counts, image sizes, field fill rates and Line_Item counts over valid pairs
    reasons = {}
    for _, r, _ in skipped:
        reasons[r] = reasons.get(r, 0) + 1
    print(f"valid pairs {len(rows)} + skipped {len(skipped)} = {len(rows) + len(skipped)} "
          f"(distinct base names: {n_bases})")
    print("skipped by reason: " + (", ".join(f"{r} {n}" for r, n in sorted(reasons.items())) or "none"))
    mp = [r["width"] * r["height"] / 1e6 for r in rows]
    print(f"megapixels: median {np.median(mp):.2f}, p90 {np.percentile(mp, 90):.2f}, max {max(mp):.2f}")
    print("fill rate per field (% non-empty):")
    for f in SCORED_FIELDS:
        filled = sum(not is_empty(f, r["label"][f]) for r in rows)
        print(f"  {f:<18} {100 * filled / len(rows):6.1f}")
    n_items = [len(r["label"]["detail"]) for r in rows]
    print(f"Line_Items per receipt: mean {sum(n_items) / len(n_items):.2f}, max {max(n_items)}")
    print(f"labels written: {len(rows)}, skipped: {len(skipped)}")


def print_split_stats(manifest):
    # Group counts and receipt counts per set
    rec = manifest["receipts"]
    sizes = {}
    for r in rec:
        sizes[r["group"]] = sizes.get(r["group"], 0) + 1
    test_groups = {r["group"] for r in rec if r["split"] == "test"}
    pool_groups = {r["group"] for r in rec if r["split"] == "pool"}
    print(f"split: {len(sizes)} Template_Groups, largest {max(sizes.values())}, "
          f"spanning both sets {len(test_groups & pool_groups)}, clean_split {manifest['clean_split']}")
    sets = [("Test_Set", lambda r: r["split"] == "test"),
            ("Training_Pool", lambda r: r["split"] == "pool"),
            ("Validation_Split", lambda r: r["val"])]
    sets += [(f"subset {s}", lambda r, s=s: r["subset"] is not None and r["subset"] <= s) for s in SUBSETS]
    for name, keep in sets:
        print(f"  {name:<17} {sum(keep(r) for r in rec):5d}")


def main():
    guard_output_dir()
    before = fingerprint()
    try:
        image_names = [p.name for p in RAW_IMAGES.iterdir()]
        json_names = [p.name for p in RAW_JSONS.iterdir()]
        pairs, skipped = pair_files(image_names, json_names)
        n_bases = len({Path(n).stem for n in image_names if Path(n).suffix.lower() == ".jpg"}
                      | {Path(n).stem for n in json_names if Path(n).suffix.lower() == ".json"})
        for base, reason, detail in skipped:
            print(f"skip {base}: {reason} ({detail})")
        rows = verify_pairs(pairs, skipped)
        rows = filter_rows(rows, skipped)

        skipped.sort()
        write_lines(DATA_DIR / "skipped.tsv", [f"{b}\t{r}\t{d}" for b, r, d in skipped])
        if not rows:
            print("ERROR: no valid pairs found")
            sys.exit(1)
        write_labels(rows)
        write_lines(DATA_DIR / "valid_pairs.txt", sorted(r["base"] for r in rows))

        (DATA_DIR / "manifest.json").unlink(missing_ok=True)
        groups = group_receipts(rows)
        manifest = make_split(rows, groups)
        write_manifest(manifest)

        print_data_stats(rows, skipped, n_bases)
        print_split_stats(manifest)
    finally:
        after = fingerprint()
        if after != before:
            changed = sorted(set(before) ^ set(after))
            print(f"ERROR: Raw_Dataset changed during the run: {sorted({n for n, _ in changed})}")
            sys.exit(1)
        n_img, n_json = count_raw(after)
        print(f"Raw_Dataset unchanged ({n_img} images, {n_json} jsons)")


if __name__ == "__main__":
    main()
