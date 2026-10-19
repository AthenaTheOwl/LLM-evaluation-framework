# Does a reviewer inspect the handoff evidence?

Four synthetic cases isolate a failed test, stale revision approval, a missing
write effect, and a matching clean control. Each answer must name a receipt and
give the expected readiness verdict. The first case deliberately changes its mock
verdict on attempt two.

```powershell
llm-evals validate examples/handoff_eval
llm-evals run examples/handoff_eval --provider mock --stage deterministic --repeat 3 --min-pass-hat-k 1.0
```

The repeated mock run should exit 1: first-attempt and any-attempt success are
100%, while all-attempt reliability is 75%. This checks reporting and regression
plumbing. It provides no measurement of a real model's review ability.

For a later authorized model experiment, hold the raw task and receipts fixed.
Compare a reviewer that receives the author's conclusion immediately with one
that receives it only after an evidence assessment. Keep model, effort, tool
budget and grader fixed; record changed verdicts, false blocks, time and total
cost including retries. Use held-out cases before adopting a prompt change.

This local experiment design follows the question raised by
[Audit Without Verification](https://arxiv.org/abs/2609.07680).
The exact-string receipts here cannot establish factual support in free text or
replace independent human labels. Keep those claims separate from fixture pass rate.
