# Nested Learning Optimizer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.10+](https://img.shields.io/badge/tensorflow-2.10+-orange.svg)](https://www.tensorflow.org/)

A TensorFlow/Keras optimizer implementing concepts from Google Research's paper ["Nested Learning: The Illusion of Deep Learning Architectures"](https://arxiv.org/abs/2402.09747) (Behrouz et al., 2024), **extended with multi-timescale associative memory, attention-based gradient retrieval, and depth-aware update scheduling**.

**In practice, this optimizer has shown significant improvements in performance, convergence speed, and training stability compared to Adam, AdamW, NAdam, and SGD across various architectures.**

## Additional Components Beyond the Paper

This implementation extends the theoretical insights from the Nested Learning paper with:

1. **Continuum Memory System** - Three-tier gradient memory (short/medium/long-term) instead of single momentum
2. **Attention-Based Retrieval** - Dynamic memory weighting using scaled dot-product attention
3. **Depth-Based Update Scheduling** - Automatic layer-wise update frequencies based on network depth
4. **Gradient Accumulation** - No information loss for slow-updating layers
5. **Per-Epoch LR Schedule** - Built-in warmup → hold → decay learning rate scheduling

## Core Concept

The Nested Learning paper reveals that standard optimizers like Adam and SGD can be viewed as **associative memory systems** that compress gradient information over time. This optimizer extends that insight by implementing:

1. **Explicit Multi-Timescale Memory**: Instead of a single momentum term, maintains three gradient memories with different decay rates (short/medium/long-term)
2. **Attention-Based Retrieval**: Dynamically weights memory contributions based on current gradient similarity
3. **Depth-Aware Update Scheduling**: Different network depths update at different frequencies

## Key Innovations Beyond the Paper

| Feature | Standard Optimizers | This Implementation |
|---------|---------------------|---------------------|
| Gradient Memory | Single EMA (momentum) | Three-tier continuum memory |
| Memory Access | Fixed weighted sum | Attention-based retrieval |
| Update Frequency | Uniform across layers | Depth-based scheduling |
| Gradient Handling | Discard between updates | Accumulation for slow layers |

### Continuum Memory System

```
Short-term  (decay=0.94)   → Fast adaptation, recent gradients
Medium-term (decay=0.994)  → Intermediate stability
Long-term   (decay=0.9999) → Stable, historical direction
```

The optimizer blends these memories using one of three modes:
- **fixed**: Static weights (default: 0.1/0.3/0.6)
- **adaptive**: Learned weights that adjust during training
- **attention**: Query-based retrieval using current gradient as query

### Depth-Based Update Scheduling

Earlier layers in deep networks tend to learn more general features that stabilize early. This optimizer leverages that by:

- Assigning update intervals based on layer depth
- Supporting multiple schedule curves: `linear`, `exponential`, `cosine`
- Accumulating gradients for slow-updating layers (no information loss)

## Installation

```bash
pip install nested-learning-optimizer
```

Or from source:

```bash
git clone https://github.com/kareemfarid/nested-learning-optimizer.git
cd nested-learning-optimizer
pip install -e .
```

## Quick Start

### Basic Usage

```python
from nested_learning_optimizer import NestedLearningOptimizer

optimizer = NestedLearningOptimizer(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse')
model.fit(x_train, y_train, epochs=10)
```

### With Depth-Based Scheduling

```python
optimizer = NestedLearningOptimizer(
    learning_rate=0.001,
    auto_schedule=True,
    max_interval=6,
    schedule_curve="cosine",
)

# Analyze model structure before compiling
optimizer.compute_depths_from_model(model)
model.compile(optimizer=optimizer, loss='categorical_crossentropy')
```

### With Per-Epoch LR Schedule

```python
steps_per_epoch = len(train_dataset)

optimizer = NestedLearningOptimizer(
    learning_rate=0.001,
    steps_per_epoch=steps_per_epoch,
    lr_warmup_fraction=0.25,   # 25% warmup
    lr_hold_fraction=0.25,     # 25% hold at peak
    lr_decay_fraction=0.50,    # 50% decay
)
```

### Manual Layer Depth Registration

```python
optimizer = NestedLearningOptimizer(auto_schedule=True)

# Register specific layers with custom depths
optimizer.register_layer_depth(model.layers[0], depth=0)  # Slowest updates
optimizer.register_layer_depth(model.layers[-1], depth=10)  # Fastest updates

model.compile(optimizer=optimizer, loss='mse')
```

### Pattern-Based Intervals (Manual Mode)

```python
optimizer = NestedLearningOptimizer(
    auto_schedule=False,
    layer_update_intervals={
        "embedding": 8,      # Very slow updates
        "encoder": 4,        # Moderate
        "decoder": 2,        # Faster
        "output": 1,         # Every step
    },
    default_interval=1,
)
```

## API Reference

### Constructor Parameters

#### Memory System
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `short_term_decay` | float | 0.94 | Decay rate for short-term memory |
| `medium_term_decay` | float | 0.994 | Decay rate for medium-term memory |
| `long_term_decay` | float | 0.9999 | Decay rate for long-term memory |
| `memory_blend_mode` | str | "attention" | How to combine memories: "fixed", "adaptive", "attention" |
| `short_term_weight` | float | 0.1 | Fixed weight for short-term (when mode="fixed") |
| `medium_term_weight` | float | 0.3 | Fixed weight for medium-term |
| `long_term_weight` | float | 0.6 | Fixed weight for long-term |

#### Depth Scheduling
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `auto_schedule` | bool | True | Auto-compute intervals from layer depth |
| `max_interval` | int | 6 | Maximum update interval for slowest layers |
| `schedule_curve` | str | "cosine" | Interval curve: "linear", "exponential", "cosine" |
| `reverse_depth_order` | bool | True | If True, deeper layers update slower |
| `warmup_steps` | int | 0 | Steps where all layers update every step |
| `accumulate_gradients` | bool | True | Accumulate gradients for slow layers |

#### Gradient Compression
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_gradient_compression` | bool | False | Enable low-rank gradient approximation |
| `compression_rank` | int | None | Rank for SVD compression |

#### Adam Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `beta_1` | float | 0.9 | First moment decay |
| `beta_2` | float | 0.999 | Second moment decay |
| `epsilon` | float | 1e-7 | Numerical stability |
| `amsgrad` | bool | False | Use AMSGrad variant |

#### LR Schedule
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `steps_per_epoch` | int | None | Steps per epoch (enables schedule) |
| `lr_warmup_fraction` | float | 0.25 | Fraction of epoch for warmup |
| `lr_hold_fraction` | float | 0.25 | Fraction to hold at peak |
| `lr_decay_fraction` | float | 0.50 | Fraction for decay |
| `lr_min_fraction` | float | 0.01 | Min LR as fraction of base |

### Methods

#### `compute_depths_from_model(model)`
Analyze model structure to assign depths automatically. Call before `model.compile()`.

```python
depths = optimizer.compute_depths_from_model(model)
```

#### `register_layer_depth(layer, depth)`
Manually set depth for a layer's variables.

```python
optimizer.register_layer_depth(model.get_layer("encoder"), depth=2)
```

#### `register_variable_depth(variable, depth)`
Manually set depth for a specific variable.

```python
optimizer.register_variable_depth(my_variable, depth=5)
```

#### `set_steps_per_epoch(steps)`
Configure LR schedule after initialization.

```python
optimizer.set_steps_per_epoch(1000)
```

#### `print_memory_stats()`
Print debug information about optimizer state.

```python
optimizer.print_memory_stats()
```

### Properties

- `iterations`: Current global step
- `current_learning_rate`: Current scheduled LR

## Examples

See the [examples/](examples/) directory for complete examples:

- `basic_usage.py`: Simple training example
- `depth_scheduling.py`: Using automatic depth-based scheduling
- `custom_memory.py`: Configuring memory blend modes

## How It Works

### Memory Update (every step)
```
short_mem  = 0.94 * short_mem  + 0.06 * gradient
medium_mem = 0.994 * medium_mem + 0.006 * gradient  
long_mem   = 0.9999 * long_mem + 0.0001 * gradient
```

### Memory Retrieval (attention mode)
```python
scores = [dot(gradient, mem) / sqrt(dim) for mem in memories]
weights = softmax(scores)
combined = sum(w * mem for w, mem in zip(weights, memories))
effective_grad = 0.5 * gradient + 0.5 * combined
```

### Weight Update (Adam-style)
```python
v = beta_2 * v + (1 - beta_2) * effective_grad²
update = lr * effective_grad / (sqrt(v / bias_correction) + epsilon)
weights -= update
```

## Citation

If you use this optimizer in your research, please cite:

```bibtex
@article{behrouz2024nested,
  title={Nested Learning: The Illusion of Deep Learning Architectures},
  author={Behrouz, Ali and others},
  journal={arXiv preprint arXiv:2402.09747},
  year={2024}
}
```

## Author

**Kareem Farid**
- GitHub: [@kareemfarid](https://github.com/kareemfarid)
- Twitter: [@kareemfarid](https://twitter.com/kareemfarid)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

