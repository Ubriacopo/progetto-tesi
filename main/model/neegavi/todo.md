# Temporal video design notes

The current Perceiver resampler compresses spatial video tokens. \
A second, causal resampler could produce one compact video token per second.

## Candidate design

1. Sample video features at 1 Hz.
2. Spatially compress each frame to a small token set.
3. Keep the last 3–5 seconds in a rolling buffer.
4. Apply a causal temporal resampler to the buffer.
5. Pass the resulting token to gated cross-attention.

For a video step `t`, keys and values must come from times `<= t`. Short initial windows may use the available history or masked padding.

This factorized design keeps the frozen video backbone. Its memory cost scales with the temporal window and compressed token count, not with all overlapping clips.

## Evaluation

- Compare 2-, 3-, and 5-second context windows.
- Measure memory use and preprocessing time.
- Verify that no future frames enter causal tokens.
- Compare against the current fixed-window baseline.
- Add temporal regularization only if attention collapses or becomes diffuse.
