# Contributing

Contributions are welcome! Here's how to help.

## Development Setup

```bash
git clone https://github.com/Kareemfarid/nested-learning-optimizer.git
cd nested-learning-optimizer
pip install -e ".[dev]"
```

## Code Style

- Use `black` for formatting
- Use `isort` for import sorting
- Type hints are encouraged

```bash
black nested_learning_optimizer/
isort nested_learning_optimizer/
```

## Testing

```bash
pytest tests/ -v
```

## Pull Requests

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run `black` and `isort`
5. Submit PR

## Issues

- Bug reports: Include TensorFlow version, minimal reproducible example
- Feature requests: Describe use case and expected behavior

