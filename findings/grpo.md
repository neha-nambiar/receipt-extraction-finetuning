# GRPO on top of SFT

## Setup

GRPO starts from the SFT adapter with the best Validation_Split F1, `sft_lv_500` (val 0.751, test 0.750). It trains on the 500 receipts of the 1,000-receipt subset that this adapter never saw. These have no overlap with the SFT-500 receipts, the Validation_Split or the Test_Set. So GRPO and `sft_lv_1000` (test 0.767) use exactly the same 1,000 labelled receipts. The comparison answers a practical question: are the second 500 labels worth more as extra SFT data or as RL prompts with a rule-based reward?

Each step samples 4 answers to one receipt at temperature 0.9 and scores each with the scorer-based reward:
- per-field credit
- −0.5 per hallucinated field
- a bonus when the item amounts sum to the total
- −1.0 when nothing parses

The advantage of each answer is its reward minus the group mean. A KL penalty (β 0.04) ties the policy to a frozen reference model. The run stops early if KL goes above 0.5.

## The reference model must be the SFT model, not the base model

Training the SFT adapter directly gives the wrong reference. With a LoRA model, TRL computes the reference by switching the adapter off. When the adapter being trained *is* the SFT adapter, the reference becomes the untuned base model. The KL penalty then pulls the policy back toward the base model, the opposite of what's wanted. This showed up immediately: KL was **0.24 at step 1, with a learning rate of 0**, so before any update had happened.

The fix is to merge the SFT adapter into the weights once, then train a fresh LoRA on top. The adapter-off reference is then the SFT model itself. After the fix, step-1 KL was 0.00003 and the step-1 rewards were identical to the unmerged run (1.025). That confirms the merged model behaves exactly like the SFT adapter. The same trap applies to any recipe that runs DPO or GRPO "on top of" a LoRA SFT checkpoint without merging.

## A learning rate of 5e-6 doesn't move a LoRA policy

The first run used learning rate 5e-6 with cosine decay, a common GRPO setting for full fine-tuning. After 115 of 300 steps (460 sampled answers) it had changed almost nothing:

| Metric over 115 steps | Value |
|---|---|
| Mean reward, first 50 / last 50 steps | 0.755 / 0.745 |
| Linear reward trend | +0.004 per 100 steps |
| Mean KL to the SFT model | 0.0005 (max 0.009; stop limit 0.5) |
| Median within-group reward std | 0.082 |
| Steps where all 4 answers scored the same | 5 of 115 |
| Answers cut off at the token limit | 0 |

The learning signal was there: on 110 of 115 steps the 4 answers got different rewards, so every step had non-zero advantages. The problem was step size. A KL of 0.0005 means the policy's token probabilities were practically those of the SFT model. LoRA trains about 0.9% of the weights, through a low-rank update whose effective size scales with the learning rate. Learning rates suited to full fine-tuning are typically about 10× too small for LoRA, which is why LoRA SFT here used 2e-4. With reward noise this large between receipts (single-step rewards from 0.03 to 1.13), a reward trend can only appear once the policy actually moves. The run was stopped and restarted at 2e-5.

Two more observations from this run:

- **The reward is dominated by which receipt is drawn.** Each step trains on one receipt, so the per-step mean reward mostly measures how hard that receipt is. The 6 steps with mean reward under 0.3 were hard receipts, not bad updates. Progress has to be judged on held-out F1, not on the reward curve.
- **The SFT model is already near the reward ceiling on easy receipts.** On 28% of steps the 4 answers differed by less than 0.05 in reward. On those receipts the samples agree, so there is almost nothing to learn. The useful signal comes from the minority of receipts where the samples disagree.

## A learning rate of 2e-5 moves the policy, but not toward better answers

The second run used learning rate 2e-5 (4× higher) and ran all 300 steps. That's 1,200 sampled answers over 300 of the 500 GRPO receipts. Both runs draw the receipts in the same order, so the 20-step blocks can be compared directly:

| Steps | Reward 5e-6 | Reward 2e-5 | KL 5e-6 | KL 2e-5 |
|---|---|---|---|---|
| 1–20 | 0.761 | 0.765 | 0.00016 | 0.00013 |
| 21–40 | 0.775 | 0.756 | 0.00039 | 0.00050 |
| 41–60 | 0.708 | 0.755 | 0.00053 | 0.00058 |
| 61–80 | 0.769 | 0.746 | 0.00082 | 0.00214 |
| 81–100 | 0.662 | 0.703 | 0.00055 | 0.00125 |
| 101–300 (2e-5 only) | – | 0.657–0.800 per block | – | 0.0009–0.0030 per block |

| Metric over 300 steps (2e-5) | Value |
|---|---|
| Mean reward, first 50 / last 50 steps | 0.758 / 0.731 |
| Linear reward trend | −0.012 per 100 steps |
| Mean KL to the SFT model | 0.0017 (max 0.017; stop limit 0.5) |
| Steps where all 4 answers scored the same | 17 of 300 |
| Steps with reward std under 0.05 | 104 of 300 |

KL rose steadily, from about 0.0001 in the first block to 0.002–0.003 from step 160 on. That's about 3× the 5e-6 run at the same steps. So the policy did move, but the reward didn't follow. The trend is flat to slightly negative, and the block means are again ruled by which receipts were drawn.

## Held-out results: GRPO adds nothing measurable

| Model (same 1,000 labels for the last two) | Val F1 (50) | Test F1 (300) | Test precision / recall | Test parse failures |
|---|---|---|---|---|
| `sft_lv_500` (GRPO's start) | **0.751** | 0.750 | 0.744 / 0.755 | 1 |
| SFT 500 + GRPO on 500 more | 0.746 | 0.754 | 0.751 / 0.757 | 0 |
| `sft_lv_1000` (SFT on all 1,000) | 0.747 | **0.767** | 0.770 / 0.765 | 1 |

GRPO moved test F1 by +0.004 and val F1 by −0.005. Both shifts are within noise (on 50 val receipts, one field on one receipt changes F1 by about 0.001–0.002). Under the project's rule of selecting by val F1, GRPO would not replace the SFT model. Using the same 500 extra receipts as SFT data instead gave +0.017 on test. **For this task and budget, extra labels are worth about 4× more as supervised examples than as RL prompts.**

GRPO did change the model's behaviour. Under greedy decoding, 151 of 300 test outputs differ from the SFT model's, and on average the changes cancel out. Per-field test F1 shows where they went:

| Field | SFT 500 | GRPO | Change | Missing (SFT → GRPO) | Hallucinated (SFT → GRPO) |
|---|---|---|---|---|---|
| place | 0.658 | 0.624 | −0.035 | 13 → 29 | 10 → 5 |
| seller_address | 0.654 | 0.615 | −0.040 | 13 → 28 | 16 → 10 |
| tax_number | 0.565 | 0.600 | +0.035 | 8 → 9 | 7 → 7 |
| detail (line items) | 0.645 | 0.658 | +0.013 | – | – |
| times, totals, seller_name | – | – | ±0.011 | – | – |
| All fields | – | – | – | 110 → 139 | 142 → 132 |

## Why: the reward teaches the model to skip uncertain fields

The clearest change is that the model leaves `place` and `seller_address` empty more often: on 78 test receipts instead of 56. This follows from the reward's asymmetry:
- A hallucinated field (output where the label is empty) costs −0.5.
- A missing field (empty where the label has a value) costs nothing beyond lost credit.

For a field the model is unsure about, skipping it is the safer bet under this reward. GRPO found that and halved the hallucinations on these two fields (26 → 15). But each skipped field that did have a label is a missed field. Recall on `place` fell from 0.654 to 0.593 with no gain in precision (0.663 → 0.658), so F1 dropped.

The lesson: a rule-based reward gets optimised literally. Any asymmetry in the penalties becomes the policy's strategy, even at a KL of 0.002, long before the overall reward visibly improves. Here the penalty matched what the scorer counts as a hallucination, but F1 weights missed and wrong fields the same. So the reward and the metric disagree about what an empty field is worth. A reward closer to F1 (e.g. −0.5 for a missing field too, or the per-receipt F1 itself) would remove this incentive.

## Why GRPO had little to work with

- **The SFT model already sits near its ceiling on most receipts.** On a third of the steps (104 of 300), the 4 samples scored within 0.05 of each other. On those receipts the samples agree, so the advantages are near zero and the step teaches almost nothing.
- **One receipt per step is a very noisy gradient.** Each update comes from 4 answers to a single receipt. Across 300 steps the model saw 300 receipts once, while `sft_lv_1000` saw 1,000 receipts × 3 epochs with an exact target for every token. GRPO was also the more expensive option: 85 minutes of training against 25 minutes for SFT on 1,000, because it generates 4 full answers per step.
- **The errors that are left are mostly perception errors.** Misreads (a wrong value where the label has one) are the largest error type for every model (678 for SFT 500 on test). Sampling at temperature 0.9 rarely produces the correct reading of a smudged digit or an unusual address layout. If no sample in the group gets it right, there is no positive advantage to learn from. SFT shows the correct answer directly.

## Note on latency

The GRPO eval measured 2.36 s per receipt against 4.5 s for the SFT adapters. Output length was the same (252 vs 257 tokens on average), and so was the LoRA structure (r 16, same layers). So the difference is very likely the eval host, which changed between these runs, not the model. Only compare latencies measured on the same host.
