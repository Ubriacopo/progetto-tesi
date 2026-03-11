> So your trial-level aggregation strategy is correct for all four datasets.

Basically: I have samples that are annotated. These annotations are on the full clip only. <br>
This makes splitting and chunking very hard so here is a possible solution:

- Keep 1 experiment = 1 sample
- Chunk samples in 32s windows (or 16s)
    - TODO: Check how many train samples have more than 16s in my data and how many > 24s and
      how many exactly 32. This gives insight in what we are gonna do later (see if I trained for long sequences).
- Call our model as frozen encoder on each chunk
- Aggregate the CLS tokens (Mean or Attention) + build predictor (MLP)
- For comparison build same predictor on CBraMod as frozen backbone to see if my model with fusion helps.

Check chunk size=8 only if bad performance.