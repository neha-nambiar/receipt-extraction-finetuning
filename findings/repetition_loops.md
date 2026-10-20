# Repetition loops after fine-tuning on few examples

## Summary

The adapter trained on 100 receipts produces unparseable output on **8 of 300 Test_Set receipts (2.7%)**. The base model fails on none of them at eval batch 8. Seven of the eight are **runaway repetition**: the output starts as valid JSON and then repeats one token, phrase or whole item line until generation is cut off. Each failure scores as a fully missed receipt. Without them, the adapter would score **0.681 instead of 0.670** overall F1. That is most of the 0.017 gap that remains to gpt-4.1-mini (0.687). The loops also hold up their whole batch for the full 120 s timeout, which accounts for about 14 of the 31 minutes the test eval took.

More data almost removes the problem. With 500 and with 1,000 training examples there is **one parse failure each (0.3%)**, and it's the same receipt in both. That receipt really does contain many near-identical item lines (`1 Super Drink Line 55.00`), so repeating the line is partly correct and the model can't tell when to stop.

| Training examples | Parse failures (test) | Of which loops | Longest parsed output |
|---|---|---|---|
| 100 | 8 (2.7%) | 7 | n/a |
| 500 | 1 (0.3%) | 1 | 758 tokens |
| 1,000 | 1 (0.3%) | 1 | 673 tokens |

## What the loops look like

| Receipt | Field where it loops | Repeated unit |
|---|---|---|
| 393, 79 | `invoice_number` | the digit `1` |
| 752 | `detail` amount | the digit `6` |
| 736 | `detail` content | `100% ` |
| 1460 | `detail` content | a made-up phrase (`100% COW, 100% CHEESE, 100% TOMATO SAUCE, … 100% SATISFYING`), then the whole item line again and again |
| 1112 | `detail` | one real item line, `Super Drink Line 25.99`, emitted over and over |
| 355 | `detail` | one real item line, `BATH TOWEL 7.97`, emitted over and over |

The other fields are usually right before the loop starts. Receipt 393, for example, has the correct date, city, currency and total, and then writes `1111…` into `invoice_number`. The loops start after 300–1,100 characters of normal output. They all ran until the batch's 120 s timeout stopped them, at 1,430–1,450 output tokens (the cap is 1,536). A normal parsed adapter output has a median of 236 tokens.

The eighth failure is a different kind. On receipt 1036 the model stopped after 75 tokens with `"detail"}`, a key with no value. It also filled both standard dates with `1900-01-01` although the receipt shows no date. That's a collapsed output, not a loop.

## Why it happens

- **Greedy decoding has no way out of a loop.** Once a token has been repeated a few times, the repeated pattern itself becomes the strongest evidence for what comes next. Language models are good at continuing patterns they see in their context. Under greedy decoding the most likely token is always chosen, so nothing breaks the cycle.
- **The cause is too little variety, not a low training loss.** The 100-example run ended at a training loss of 0.055. That first looked like memorisation, but the 500- and 1,000-example runs reach the same loss within their first epoch, and they barely loop. With only 100 receipts, the adapter has seen few long receipts and few cases where a run of similar lines ends. So on an unfamiliar long receipt its sense of when to stop is weak. Once it repeats something, the repetition becomes the best evidence for the next token. The base model, trained on far more varied data, only looped once in 300 receipts at batch 1, and not at all at batch 8.
- **The loops are not copied from the labels.** No training label repeats a line dozens of times or has a 1,000-digit invoice number. The loop is a decoding failure that the narrowly trained model falls into, not something it learned.
- **Long and repetitive receipts trigger it.** The failed receipts have a median of 9 item lines in their labels against 5 for the whole Test_Set. Receipts with many similar lines, or with long digit strings, give the model several places where repeating the previous line looks plausible. The one receipt that still loops at 500 and 1,000 examples is the extreme case, with many identical drink lines.

## Why the obvious fixes don't fit

- **A repetition penalty or n-gram blocking at decode time** would fight the output format itself. Every item in `detail` repeats the same keys (`"content":`, `"amount":`, `"ifTax":`). Real receipts also repeat item lines and digits, as in a receipt with two identical drinks or the invoice number `0001`. A penalty strong enough to stop the loops would also distort correct outputs. It would also change the decoding for every model, so the base and API comparisons would need re-running.
- **Falling back to the base model's output on failed receipts** gives almost nothing: overall F1 moves from 0.6699 to 0.6701. These are hard receipts, and the base model only scores 0.383 F1 on them. Its fallback answers add 68 true positives but also 132 false positives. A parse failure costs recall but no precision, so a poor fallback answer is barely better than no answer.
- **Fewer epochs on small sets** might help a little. But the training-size curve needs the same settings at every size to stay comparable, and the larger sizes show that data, not epochs, is what matters.

The real fix is more varied training data: 500 examples cut the failure rate from 2.7% to 0.3%. Beyond that, a cheap guard is worth having in a deployment: stop generation as soon as the last few hundred characters are periodic, and retry or flag the receipt. That turns a 120 s stall into an instant, visible failure.

## Val vs test

The 50-receipt Validation_Split had no parse failures. At a 2.7% rate, the expected number of failures in 50 receipts is 1.3, and seeing none has about a 25% chance. So the clean val run is consistent with the test rate. 50 receipts are too few to measure a failure this rare, so the loop rate has to be judged on the Test_Set.
