def run(cost_tracker=None, **kwargs) -> str:
    if cost_tracker is None:
        return "Cost tracking not enabled."
    return cost_tracker.summary()