# Prompt wording vs label conventions

## Summary

Rewording two lines of the extraction prompt raised gpt-4.1-mini's overall Test_Set F1 from **0.625 to 0.687**. The two fields those lines describe went from almost always wrong to mostly right: orig_curr 0.109 → 0.785, seller_address 0.012 → 0.565. Nothing else changed: same model, images, scorer and the other 12 prompt lines. The model's reading didn't improve. It was finally told what the labels mean.

![Per-field F1, old vs new prompt](img/prompt_change_f1.png)

## The change

| Field | Old prompt | New prompt |
|---|---|---|
| orig_curr | `list of currency text as printed` | `list with the currency symbol as printed (if any) and the city of purchase, e.g. ["$", "New York"]` |
| seller_address | `list of seller addresses` | `list with the seller location as "Country-City" (usually the same as place), not the street address` |

## Why the old wording failed

The old lines described the fields by their names, and the labels mean something else. seller_address is a `Country-City` location (no label contains a digit, and 77% of Test_Set labels equal `place`), and orig_curr is a currency symbol plus the purchase city. Told only the names, the model did the sensible thing and copied what was printed:

| Field | Label | Old prediction | New prediction |
|---|---|---|---|
| seller_address | `United States-Miami Beach` | `300 72th Street Miami Beach fl 33141` | `USA-Miami Beach` |
| seller_address | `USA-Carmichael` | `6505 Fair Oaks Blvd Carmichael, CA 95608` | `USA-Carmichael` |
| orig_curr | `["$", "Carmichael"]` | `["$"]` | `["$", "Carmichael"]` |
| orig_curr | `["Miami Beach"]` | `["$"]` | `["$", "Miami Beach"]` |

## Outcomes per field (300 receipts)

| Field | F1 old → new | Correct | Missing | Hallucination | Misread |
|---|---|---|---|---|---|
| orig_curr | 0.109 → 0.785 | 48 → 233 | 74 → 7 | 5 → 9 | 173 → 51 |
| seller_address | 0.012 → 0.565 | 32 → 174 | 7 → 3 | 31 → 29 | 230 → 94 |
| std_curr | 0.870 → 0.894 | 240 → 254 | 48 → 32 | 0 → 0 | 12 → 14 |
| overall | 0.625 → 0.687 | precision 0.601 → 0.655 | recall 0.651 → 0.724 | | |

All 12 other fields moved by less than 0.025. Cost and latency were the same: about $0.001 and 4 s per receipt.

## What the numbers say

- **Misreads turned into correct answers:**
  - orig_curr: the old prompt gave a bare symbol where the label wants symbol + city (173 misreads). Where no symbol was printed it gave nothing, but the label holds just the city (74 missing). Asking for the city fixed both.
  - seller_address: misreads fell by 136 (230 → 94) and correct answers rose by 142, as street addresses became `Country-City` locations.
- **Hallucinations barely moved (31 → 29 for seller_address):** those are receipts where the label is empty. Wording can change what kind of value the model gives, not whether the annotators filled the field.
- **seller_address stops at 0.565 because of naming, not reading:** the remaining misreads are mostly country spelling ("USA" vs "United States", "Singapore-Singapore" vs "Singapore - Singapore city"). The labels use both spellings with no visible rule, so a zero-shot model is wrong about half the time whichever it picks.
- **std_curr gained as a side effect (16 fewer missing codes):** asking for the purchase city seems to help the model infer the currency when no symbol is printed.

## Lessons

1. **On a labelled benchmark, the prompt has to describe the labels, not the concept.** A field name is a hint about meaning, not a specification. Two vague lines cost 0.06 overall F1 and made two fields look unreadable.
2. **Check per-field outcomes before trusting a baseline number.** Overall F1 0.625 looked plausible. Only the near-zero F1 on two fields and the error breakdown (misreads, not missing) showed the model was answering a different question.
3. **There's a limit to what prompting can fix:** it can describe a convention that's consistent (location, not address). It can't teach an inconsistent one, like which country spelling a given label uses. That part is left for fine-tuning.
4. **The comparison stays fair because the fix went into the prompt, not the scorer:** every model uses the new prompt and the scorer is unchanged, so all models are judged against the same labels by the same rules. The old-prompt run is kept only for this comparison.
