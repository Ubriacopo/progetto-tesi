Per epoch:

shuffle shard order

for each shard, shuffle indices within that shard

read that shard sequentially (or in contiguous runs)

That gives you good mixing while keeping disk reads fast.

If “similar samples are close” worries you

Do a one-time offline shuffle at write time (recommended):

when you build the HDF5, write samples in a randomized order (or interleave sources/classes)

then per-epoch you only shuffle indices (cheap)

Concrete default settings

shard size: 16–32GB

chunk0: >= batch_size (often 4–8× batch_size)

DataLoader: open .h5 per worker; keep workers modest

So yes: HDF5 + (one-time) reorder + per-epoch shard-aware permutation + sequential streaming is the best bet for your I/O bottleneck.