"""Token usage and cost tracking."""

PRICING = {
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-8": (15.00, 75.00),
    "claude-haiku-4-5": (0.80, 4.00),
    "gpt-4o": (2.50, 10.00),
    "llama3.2": (0.0, 0.0),
    "qwen3.5:4b": (0.0, 0.0),
}


class CostTracker:
    def __init__(self, model: str):
        self.model = model
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.turn_count = 0

    def add_usage(self, input_tokens: int, output_tokens: int):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        input_price, output_price = PRICING.get(self.model, (0.0, 0.0))
        cost = (input_tokens / 1_000_000 * input_price) + \
               (output_tokens / 1_000_000 * output_price)
        self.total_cost += cost

    def new_turn(self):
        self.turn_count += 1

    def summary(self) -> str:
        return (
            f"Model: {self.model}\n"
            f"Turns: {self.turn_count}\n"
            f"Tokens: {self.total_input_tokens:,} in / {self.total_output_tokens:,} out\n"
            f"Estimated cost: ${self.total_cost:.4f}"
        )