# What the labels actually encode

## Summary

The ReceiptBench PurchaseTicket labels (1,492 receipts after filtering) aren't a literal transcription of each receipt. Several fields follow annotation habits that differ from what their names suggest, a few are noisy, and two are empty most of the time. That shapes the whole project. A zero-shot model that reads a receipt perfectly can still score badly, because it gives the "obvious" answer instead of the dataset's answer. Learning these habits from examples is most of what fine-tuning has to do.

## Conventions the field names don't reveal

| Field | What the name suggests | What the labels contain | How consistent |
|---|---|---|---|
| seller_address | the printed street address | a `Country-City` location, e.g. `["United States-Miami Beach"]` | no label contains a digit; 1,094 of 1,492 equal `place` exactly |
| orig_curr | the currency symbol as printed | symbol plus purchase city, e.g. `["£", "London"]`, or just the city | about 1,100 labels contain a city |
| orig_start_time, orig_invoice_time | the date/time "exactly as printed" | the date part only, in the printed format | no label in the dataset contains a time, though most receipts print one |
| place | `Country-City` | `Country-City`, with no fixed country spelling | "USA" 410, "United States" 213, "United Kingdom" 94; 334 empty, 35 without a dash |
| detail.ifTax | "this line is a tax line" | that, **or** "this price includes tax" | 217 receipts (15%) mark every line true |

The `ifTax` split is worth spelling out. In 797 receipts, `true` marks only the tax lines ("Sales Tax", "CGST@2.5%", "TX"). But in 217 receipts every line is `true`, including "Bacon Naan Roll" and "Chai". These lean towards EUR (77), GBP (34) and AUD (13), where shelf prices include VAT/GST. So the same flag means "is tax" on some receipts and "tax-inclusive price" on others. A prompt can't convey that rule cleanly. A model has to infer it from the receipt's region and layout.

## Fields where empty is often the right answer

| Field | Empty labels | Share |
|---|---|---|
| tax_number | 1,117 | 75% |
| invoice_number | 686 | 46% |
| seller_address | 356 | 24% |
| place | 338 | 23% |

Receipts are full of numbers: order IDs, table numbers, register IDs, terminal IDs. When the label is empty, a model that "helpfully" picks one counts as a hallucination. A zero-shot model can't know which numbers the annotators chose to ignore.

## Label noise

- **Missing standardised totals:** 113 receipts (7.6%) have `std_total` empty or `0.00` while `orig_total` holds the printed amount (e.g. `orig_total "4864.00"`, `std_total "0.00"`). A correct prediction on these counts as a hallucination.
- **Line items that don't add up:** on about 10% of receipts, the line amounts don't sum to `std_total` (within 0.01). The cause can be missing lines, discounts, tips or tax handling. This is why the GRPO arithmetic bonus is only applied when the ground truth itself adds up. Otherwise the reward would push the model towards wrong labels.
- **Food marked as tax:** a few labels in mixed receipts mark food items as tax, e.g. "Cold water lobster tail" (12 times) and "Porterhouse steak". These are too rare to matter for training but cost a little score.

## Shape of the data

- 38 currencies. USD dominates (955), followed by EUR 125, GBP 106, INR 70, MYR 47, CAD 44, SGD 43.
- Line items per receipt: median 5, mean 6.0, max 41.
- Images: median 0.45 MP, 90th percentile 1.77 MP, max 12.78 MP, height/width up to 3.9. The 1 MP cap leaves most receipts untouched and only shrinks the long tail.
- The start date equals the invoice date in 1,461 receipts (98%), so the four date fields behave like two.

## What this means for the experiments

| Pattern | Zero-shot behaviour | What fine-tuning should learn |
|---|---|---|
| orig dates are date-only | copies the printed date and time → misread | drop the time |
| place / seller_address country spelling | picks one spelling ("USA") → misread about half the time | the most likely spelling for each region and layout |
| empty invoice/tax numbers | fills in a plausible number → hallucination | which kinds of number the annotators keep |
| ifTax on tax-inclusive receipts | marks only tax lines → misread | the regional tax-inclusive convention |
| zero std_total | gives the real total → hallucination | nothing useful; this is label noise and caps every model's score |

The scorer wasn't changed to forgive any of this. Every model is judged against the same labels, so fine-tuned and zero-shot models compete on equal terms, and the gap between them measures how much of these conventions they learned.
