# Python Testing

When writing Python tests:
- Use pytest, not unittest
- Use fixtures over setUp/tearDown
- Name test files test_*.py
- Use parametrize for multiple test cases
- Run with: pytest -q --tb=short
- Always test edge cases: empty input, None, negative numbers