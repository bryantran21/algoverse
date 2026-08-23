# Degenerate patching run -- do not use

This is the **unrestricted** patching run: layer L's output was replaced at *all*
token positions rather than at a restricted span.

Recovery is 1.000 at every layer including layer 0. That is guaranteed by
construction, not a finding: replacing layer 0's full output means every
subsequent layer computes on clean values, so the corrupt run is simply
overwritten.

Kept for provenance. The usable result is `../patch_masked.npz`, which patches
either the vision-token span or the answer position.
