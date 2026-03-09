### Rules for faster tuning

- Work on a subset of the data?
    - This will help with speed. How many?
      If we have 100k samples I'd say around 20-30% of the data.
- Limit the steps?

Best approach:
> Hyperband / Successive Halving (deep learning standard) <br>
> Li et al., “Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization”

Core idea:

- Train many configurations with small training budgets (few epochs).
- Only promising models continue training.

#### Simple approach:

- Fixed number of grid (not all) → 5 epochs each
- Take best k → 10-15 epochs on these
- Bet 1 → Full training