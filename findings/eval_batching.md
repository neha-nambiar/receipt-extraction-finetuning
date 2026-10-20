# Batched local inference

## Summary

Generating 8 receipts per `generate()` call instead of 1 cut local eval time on the 300 Test_Set receipts from **22.4 s to 4.9 s per receipt (4.6× faster)** on the RTX 4090. A full test eval went from 111 min ($1.65) to 26 min ($0.39). Greedy decoding stays the same, but batched results aren't bit-identical. Over the full Test_Set, overall F1 was 0.602 at batch 1 and 0.605 at batch 8, and no field moved by more than 0.017. The saving decides the budget: at batch size 1, evaluation would have cost more than all of the training.

## Why batch size 1 was slow

Autoregressive decoding of a 4B model at batch 1 is memory-bound. Every new token reads all ~9 GB of weights from GPU memory to do a small amount of arithmetic, so the 4090 managed about 20 tokens/s and its compute sat mostly idle. With 8 sequences decoding together, each weight read serves 8 tokens. The time per step barely rises, so throughput can rise almost 8×.

## Why the gain is 4.6×, not 8×

A batch finishes only when its longest output finishes. Receipts that stop early sit idle while the longest one keeps generating, so the batch runs at the speed of its slowest member. Larger images also mean longer prompts, which cost more per step. Batches are sorted by image size, and the effect shows in the per-receipt throughput: about 3.6 s for mid-size receipts and 7–9 s for the largest. An early 16-receipt sample measured 3.3 s per receipt because it didn't include the long tail, so it overstated the gain.

## How it's done

- **Left padding:** prompts are left-padded so every sequence's last prompt token sits at the same position. Each output is then cut at its first end-of-sequence token, and the leftover padding is dropped.
- **Size-sorted batches:** receipts are sorted by image size before batching. Images of similar size produce similar prompt lengths, so less padding is wasted.
- **Throughput latency:** per-receipt time is reported as the batch time divided by the batch size. That's the right number for cost per page. It isn't the latency of a single request, which is still about 20 s for one receipt alone. So it can't be compared directly with the API's per-request latency.
- **Timeout:** the 120 s limit now applies to the whole batch. A receipt counts as timed out only if it hadn't finished when the batch hit the limit, so a receipt that finished in time is kept even if another one in its batch ran away.
- **Errors:** if a batch fails (e.g. out of memory), its receipts are retried one at a time, so one bad receipt can't lose a whole batch.

## Does batching change the answers?

Padding changes the shape of the matrix maths. In bf16 that shifts logits by tiny amounts, and when two candidate tokens are nearly tied, greedy decoding can then take a different path. One changed token can alter the rest of a long output (e.g. 391 → 327 tokens).

On a 16-receipt check (base model, both greedy):

| | Count |
|---|---|
| receipts with identical raw output | 10 / 16 |
| identical field values | 214 / 224 |
| identical field outcomes (correct / misread / …) | **221 / 224 (98.7%)** |

Over the full Test_Set, per-field F1 changed by between −0.012 and +0.017 and overall F1 by +0.003. The one batch-1 parse failure (a runaway repetition) parsed at batch 8. The changes don't favour either setting, but they mean F1 values from batch 1 and batch 8 shouldn't be compared at the third decimal place. So every local model in the final tables, base included, is evaluated at batch 8.

## Cost impact

| Item (at $0.89/h) | Batch 1 | Batch 8 |
|---|---|---|
| Test eval, 300 receipts | 111 min, $1.65 | 26 min, $0.39 |
| Val eval, 50 receipts | about 19 min, $0.28 | about 5 min, $0.07 |
| Evals for the 4-size SFT curve | about 8.6 h, $7.70 | about 2 h, $1.80 |
| Cost per receipt | $0.0055 | $0.0012 |

At batch 8, a self-hosted 4B model on a $0.89/h secure-cloud GPU costs about 1.25× gpt-4.1-mini per receipt ($0.00097), down from 5.7× at batch 1. On a $0.34/h community-cloud GPU it would cost about half as much as the API.
