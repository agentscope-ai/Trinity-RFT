# M2PO on GSM8K with asynchronous rollouts

This example runs [M2PO](https://openreview.net/forum?id=IIgl5MWelz), an ICLR 2026
algorithm for stable reinforcement fine-tuning with stale rollout data. M2PO removes only
the largest second-moment outliers in PPO's active trust-region quadrants and keeps the
remaining stale-token updates.

The trainer and explorer are intentionally separate so that the trainer consumes rollouts
from a lagging behavior policy. Start both processes with:

```bash
bash examples/m2po_gsm8k/run.sh
```

The paper uses `m2_threshold: 0.04`, which is also the default in Trinity-RFT. During
training, monitor:

- `actor/m2po/m2_before` and `actor/m2po/m2_after`
- `actor/m2po/masked_fraction`
- `actor/m2po/trust_region_fraction`

This is a lightweight integration example, not an exact reproduction of the paper's
`s = 256` experiments. Exact reproduction additionally requires matching the paper's model,
DeepScaleR data, update-indexed staleness schedule, batch sizes, and evaluation suite.
