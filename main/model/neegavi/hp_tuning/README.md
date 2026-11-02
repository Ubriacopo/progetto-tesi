# Hyperparameter tuning

Successive halving limits the training budget spent on weak configurations.

1. A broad search is conducted with a small epoch budget.
2. Only the strongest candidates are retained.
3. Those candidates are trained for 10–15 epochs.
4. The final candidates receive the full training budget.
5. Selection is based on validation metrics. The chosen run is evaluated once on the test set.

A fixed data subset can accelerate exploration. The subset and its seed remain constant across trials. The final comparison uses the full training data.

The approach follows [Hyperband](https://jmlr.org/papers/v18/16-558.html).
