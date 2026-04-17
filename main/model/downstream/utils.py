def print_parameter_summary_by_module(model):
    summary = {}

    for name, p in model.named_parameters():
        top = name.split(".")[0]
        if top not in summary:
            summary[top] = {"trainable": 0, "frozen": 0}

        if p.requires_grad:
            summary[top]["trainable"] += p.numel()
        else:
            summary[top]["frozen"] += p.numel()

    for module, counts in summary.items():
        total = counts["trainable"] + counts["frozen"]
        print(
            f"{module:20s} "
            f"trainable={counts['trainable']:>12,d} "
            f"frozen={counts['frozen']:>12,d} "
            f"total={total:>12,d}"
        )


def print_trainable_parameters(model):
    trainable = 0
    frozen = 0

    for name, p in model.named_parameters():
        n = p.numel()
        status = "TRAIN" if p.requires_grad else "FROZEN"
        print(f"{status:7s} | {n:>12,d} | {name}")
        if p.requires_grad:
            trainable += n
        else:
            frozen += n

    total = trainable + frozen
    pct = 100.0 * trainable / total if total > 0 else 0.0

    print("\nSummary")
    print(f"Trainable: {trainable:,}")
    print(f"Frozen:    {frozen:,}")
    print(f"Total:     {total:,}")
    print(f"% trainable: {pct:.2f}%")
