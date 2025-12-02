# ============================================================================
# NESTED LEARNING OPTIMIZER - Deep Optimizer with Associative Memory
# Based on: "Nested Learning: The Illusion of Deep Learning Architectures"
# (Behrouz et al., Google Research, 2024)
#
# Copyright (c) 2024 Kareem Farid (https://github.com/kareemfarid)
# Licensed under the MIT License
# ============================================================================

"""
Nested Learning Optimizer implementation.

This optimizer extends concepts from the Nested Learning paper by treating
the optimizer itself as an associative memory module that compresses gradients
across multiple timescales.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="NestedLearning", name="NestedLearningOptimizer")
class NestedLearningOptimizer(tf.keras.optimizers.Optimizer):
    """
    Nested Learning Optimizer with multi-timescale associative memory.
    
    Key innovations beyond standard optimizers:
    
    1. **Continuum Memory System**: Maintains gradient history at three timescales
       (short/medium/long-term) with different decay rates, enabling the optimizer
       to balance between fast adaptation and stable long-term learning.
       
    2. **Associative Gradient Retrieval**: Uses attention mechanism to retrieve
       relevant gradient information from memory, similar to how transformers
       attend to different positions.
       
    3. **Automatic Depth-Based Scheduling**: Earlier layers in deep networks
       update less frequently than later layers, reflecting the empirical
       observation that early features stabilize faster.
       
    4. **Gradient Accumulation**: Layers that skip updates accumulate gradients,
       preventing information loss in slow-updating layers.
    
    Example:
        ```python
        optimizer = NestedLearningOptimizer(
            learning_rate=0.001,
            auto_schedule=True,
            max_interval=6,
        )
        optimizer.compute_depths_from_model(model)
        model.compile(optimizer=optimizer, loss='mse')
        ```
    
    Args:
        learning_rate: Base learning rate.
        short_term_decay: Decay for short-term memory (default: 0.94).
        medium_term_decay: Decay for medium-term memory (default: 0.994).
        long_term_decay: Decay for long-term memory (default: 0.9999).
        memory_blend_mode: How to combine memories - "fixed", "adaptive", or "attention".
        short_term_weight: Weight for short-term memory in fixed mode.
        medium_term_weight: Weight for medium-term memory in fixed mode.
        long_term_weight: Weight for long-term memory in fixed mode.
        use_gradient_compression: Enable low-rank gradient approximation.
        compression_rank: Rank for low-rank approximation (None = no compression).
        auto_schedule: Automatically compute update intervals from layer depth.
        max_interval: Maximum update interval for slowest layers.
        schedule_curve: Interval curve type - "linear", "exponential", or "cosine".
        reverse_depth_order: If True, deeper layers update slower; if False, earlier layers update slower.
        warmup_steps: Steps where all layers update every step.
        accumulate_gradients: Accumulate gradients for slow-updating layers.
        layer_update_intervals: Manual interval overrides by layer name pattern.
        default_interval: Default interval when no match found (manual mode).
        beta_1: Adam first moment decay.
        beta_2: Adam second moment decay.
        epsilon: Numerical stability constant.
        amsgrad: Use AMSGrad variant.
        steps_per_epoch: Steps per epoch for LR schedule.
        lr_warmup_fraction: Fraction of epoch for LR warmup.
        lr_hold_fraction: Fraction of epoch to hold peak LR.
        lr_decay_fraction: Fraction of epoch for LR decay.
        lr_min_fraction: Minimum LR as fraction of base LR.
        lr_warmup_start_fraction: Starting LR as fraction of base LR.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        # Continuum Memory System parameters
        short_term_decay: float = 0.94,
        medium_term_decay: float = 0.994,
        long_term_decay: float = 0.9999,
        # Memory combination weights
        memory_blend_mode: str = "attention",
        short_term_weight: float = 0.1,
        medium_term_weight: float = 0.3,
        long_term_weight: float = 0.6,
        # Gradient compression
        use_gradient_compression: bool = False,
        compression_rank: Optional[int] = None,
        # Automatic depth-based scheduling
        auto_schedule: bool = True,
        max_interval: int = 6,
        schedule_curve: str = "cosine",
        reverse_depth_order: bool = True,
        warmup_steps: int = 0,
        accumulate_gradients: bool = True,
        # Manual override
        layer_update_intervals: Optional[Dict[str, int]] = None,
        default_interval: int = 1,
        # Adam parameters
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        amsgrad: bool = False,
        # Per-epoch LR schedule
        steps_per_epoch: Optional[int] = None,
        lr_warmup_fraction: float = 0.25,
        lr_hold_fraction: float = 0.25,
        lr_decay_fraction: float = 0.50,
        lr_min_fraction: float = 0.01,
        lr_warmup_start_fraction: float = 0.01,
        **kwargs: Any
    ) -> None:
        super().__init__(learning_rate=learning_rate, **kwargs)
        
        # Validate parameters
        if memory_blend_mode not in ("fixed", "adaptive", "attention"):
            raise ValueError(f"memory_blend_mode must be 'fixed', 'adaptive', or 'attention', got '{memory_blend_mode}'")
        if schedule_curve not in ("linear", "exponential", "cosine"):
            raise ValueError(f"schedule_curve must be 'linear', 'exponential', or 'cosine', got '{schedule_curve}'")
        
        # Memory decay rates
        self._short_term_decay = short_term_decay
        self._medium_term_decay = medium_term_decay
        self._long_term_decay = long_term_decay
        
        # Memory blending
        self._memory_blend_mode = memory_blend_mode
        self._fixed_short_weight = short_term_weight
        self._fixed_medium_weight = medium_term_weight
        self._fixed_long_weight = long_term_weight
        
        # Gradient compression
        self._use_gradient_compression = use_gradient_compression
        self._compression_rank = compression_rank
        
        # Depth-based scheduling
        self._auto_schedule = auto_schedule
        self._max_interval = max_interval
        self._schedule_curve = schedule_curve
        self._reverse_depth_order = reverse_depth_order
        self._warmup_steps = warmup_steps
        self._accumulate_gradients = accumulate_gradients
        
        # Manual override
        self._layer_update_intervals = layer_update_intervals or {}
        self._default_interval = default_interval
        
        # Adam parameters
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._epsilon = epsilon
        self._amsgrad = amsgrad
        
        # LR schedule
        self._steps_per_epoch = steps_per_epoch
        self._lr_warmup_fraction = lr_warmup_fraction
        self._lr_hold_fraction = lr_hold_fraction
        self._lr_decay_fraction = lr_decay_fraction
        self._lr_min_fraction = lr_min_fraction
        self._lr_warmup_start_fraction = lr_warmup_start_fraction
        self._base_lr = float(learning_rate)
        
        # State tracking
        self._global_step: Optional[tf.Variable] = None
        self._is_built = False
        
        # Computed during build (keyed by id(var))
        self._var_intervals: Dict[int, int] = {}
        self._var_depths: Dict[int, int] = {}
        self._registered_depths: Dict[int, int] = {}
        self._max_detected_depth = 0
        self._accumulated_grads: Dict[int, tf.Variable] = {}
        self._accumulation_count: Dict[int, tf.Variable] = {}
        
        # Memory stores (keyed by id(var))
        self._short_term_memory: Dict[int, tf.Variable] = {}
        self._medium_term_memory: Dict[int, tf.Variable] = {}
        self._long_term_memory: Dict[int, tf.Variable] = {}
        self._second_moment: Dict[int, tf.Variable] = {}
        self._max_second_moment: Dict[int, tf.Variable] = {}
        self._blend_weights: Dict[int, tf.Variable] = {}
        
    def _sanitize_name(self, name: str) -> str:
        """Sanitize variable name for optimizer variable names."""
        return name.replace('/', '_').replace(':', '_')
    
    def _compute_depth_intervals(self, var_list: List[tf.Variable]) -> None:
        """Compute update intervals based on variable depth."""
        total_vars = len(var_list)
        if total_vars == 0:
            return
        
        if self._registered_depths:
            self._apply_registered_depths(var_list)
            return
        
        # Fallback: position-based depths
        self._max_detected_depth = total_vars - 1 if total_vars > 1 else 1
        
        for idx, var in enumerate(var_list):
            if total_vars == 1:
                relative_depth = 0.5
            else:
                relative_depth = idx / (total_vars - 1)
            
            interval = self._compute_interval_from_depth(relative_depth)
            self._var_depths[id(var)] = idx
            self._var_intervals[id(var)] = interval
        
        self._print_depth_summary(var_list, "by variable order")
    
    def _compute_interval_from_depth(self, relative_depth: float) -> int:
        """
        Compute update interval from relative depth (0.0=earliest, 1.0=deepest).
        
        Default: earlier layers (depth=0) get slower updates (higher interval)
        Reversed: deeper layers (depth=1) get slower updates (higher interval)
        """
        if self._reverse_depth_order:
            relative_depth = 1.0 - relative_depth
        
        if self._schedule_curve == "linear":
            interval = int(self._max_interval * (1 - relative_depth) + 1)
        elif self._schedule_curve == "exponential":
            interval = int(self._max_interval * np.exp(-3 * relative_depth) + 1)
        elif self._schedule_curve == "cosine":
            interval = int(self._max_interval * 0.5 * (1 + np.cos(np.pi * relative_depth)) + 1)
        else:
            interval = 1
        return max(1, interval)
    
    def _apply_registered_depths(self, var_list: List[tf.Variable]) -> None:
        """Apply pre-registered depths from model analysis."""
        max_depth = self._max_detected_depth or 1
        if max_depth == 0:
            max_depth = max(self._registered_depths.values()) if self._registered_depths else 1
        
        for var in var_list:
            var_id = id(var)
            depth = self._registered_depths.get(var_id, max_depth)
            relative_depth = depth / max_depth if max_depth > 0 else 1.0
            interval = self._compute_interval_from_depth(relative_depth)
            self._var_depths[var_id] = depth
            self._var_intervals[var_id] = interval
        
        self._print_depth_summary(var_list, "from model structure")
    
    def _print_depth_summary(self, var_list: List[tf.Variable], source: str) -> None:
        """Print depth schedule summary."""
        total_vars = len(var_list)
        print("\n" + "="*70)
        print(f"NESTED LEARNING - DEPTH SCHEDULE ({source})")
        print("="*70)
        print(f"Total variables: {total_vars}")
        print(f"Max interval: {self._max_interval}, Curve: {self._schedule_curve}")
        print(f"Warmup steps: {self._warmup_steps}")
        print("-"*70)
        
        depth_groups: Dict[int, List[tf.Variable]] = {}
        for var in var_list:
            depth = self._var_depths.get(id(var), -1)
            if depth not in depth_groups:
                depth_groups[depth] = []
            depth_groups[depth].append(var)
        
        for depth in sorted(depth_groups.keys())[:10]:
            vars_at_depth = depth_groups[depth]
            interval = self._var_intervals.get(id(vars_at_depth[0]), 1)
            mem_type = "long-term" if interval >= 5 else "medium" if interval >= 2 else "short-term"
            bar = "█" * min(interval, 15)
            print(f"  Depth {depth:3d}: {len(vars_at_depth):4d} vars, interval={interval:3d} [{mem_type:10s}] {bar}")
        
        if len(depth_groups) > 10:
            print(f"  ... ({len(depth_groups) - 10} more depth levels) ...")
        print("="*70 + "\n")
    
    def compute_depths_from_model(self, model: tf.keras.Model) -> Dict[int, int]:
        """
        Analyze model structure to compute layer depths.
        
        Call this BEFORE model.compile() to enable automatic depth-based
        update scheduling based on actual model architecture.
        
        Args:
            model: Keras model to analyze.
            
        Returns:
            Dictionary mapping variable id to depth.
            
        Example:
            ```python
            optimizer = NestedLearningOptimizer(auto_schedule=True)
            optimizer.compute_depths_from_model(model)
            model.compile(optimizer=optimizer, loss='mse')
            ```
        """
        self._registered_depths = {}
        self._max_detected_depth = 0
        
        all_layers = self._collect_all_layers_recursive(model)
        self._assign_position_based_depths(all_layers)
        
        # Print summary
        print("\n" + "="*70)
        print("NESTED LEARNING - MODEL DEPTH ANALYSIS")
        print("="*70)
        
        depth_counts: Dict[int, int] = {}
        for d in self._registered_depths.values():
            depth_counts[d] = depth_counts.get(d, 0) + 1
        
        print(f"Total trainable vars: {len(self._registered_depths)}")
        print(f"Unique depths: {len(depth_counts)}")
        print(f"Max depth: {self._max_detected_depth}")
        mode = "REVERSED (deeper=slower)" if self._reverse_depth_order else "STANDARD (earlier=slower)"
        print(f"Mode: {mode}")
        print("-"*70)
        
        for depth in sorted(depth_counts.keys()):
            interval = self._compute_interval_from_depth(depth / max(1, self._max_detected_depth))
            mem_type = "long-term" if interval >= 5 else "medium" if interval >= 2 else "short-term"
            print(f"  Depth {depth}: {depth_counts[depth]:4d} variables, interval={interval:3d} [{mem_type}]")
        print("="*70 + "\n")
        
        return self._registered_depths
    
    def _collect_all_layers_recursive(
        self, 
        layer_or_model: tf.keras.layers.Layer, 
        collected: Optional[List[tf.keras.layers.Layer]] = None,
        depth: int = 0
    ) -> List[tf.keras.layers.Layer]:
        """Recursively collect all layers including nested ones."""
        if collected is None:
            collected = []
        
        if layer_or_model in collected:
            return collected
        
        collected.append(layer_or_model)
        
        if hasattr(layer_or_model, 'layers'):
            try:
                for sub_layer in layer_or_model.layers:
                    self._collect_all_layers_recursive(sub_layer, collected, depth + 1)
            except Exception:
                pass
        
        return collected
    
    def _assign_position_based_depths(self, layers: List[tf.keras.layers.Layer]) -> None:
        """Assign depths to layers based on position."""
        layers_with_weights = [l for l in layers if len(l.trainable_weights) > 0]
        
        if not layers_with_weights:
            return
        
        num_layers = len(layers_with_weights)
        
        for idx, layer in enumerate(layers_with_weights):
            depth = idx if num_layers > 1 else 0
            self._max_detected_depth = max(self._max_detected_depth, depth)
            
            for weight in layer.trainable_weights:
                if id(weight) not in self._registered_depths:
                    self._registered_depths[id(weight)] = depth
    
    def register_layer_depth(self, layer: tf.keras.layers.Layer, depth: int) -> None:
        """
        Manually register depth for a specific layer's variables.
        
        Use this for fine-grained control over update frequencies.
        
        Args:
            layer: Keras layer.
            depth: Integer depth (0 = slowest updates, higher = faster).
        """
        for weight in layer.trainable_weights:
            self._registered_depths[id(weight)] = depth
        self._max_detected_depth = max(self._max_detected_depth, depth)
    
    def register_variable_depth(self, variable: tf.Variable, depth: int) -> None:
        """
        Manually register depth for a specific variable.
        
        Args:
            variable: TensorFlow variable.
            depth: Integer depth (0 = slowest updates, higher = faster).
        """
        self._registered_depths[id(variable)] = depth
        self._max_detected_depth = max(self._max_detected_depth, depth)
    
    def build(self, var_list: List[tf.Variable]) -> None:
        """Build optimizer state for all variables."""
        if self._is_built:
            return
            
        super().build(var_list)
        
        self._global_step = self.add_variable(
            shape=(),
            dtype=tf.int64,
            initializer="zeros",
            name="global_step"
        )
        
        if self._auto_schedule:
            self._compute_depth_intervals(var_list)
        
        for var in var_list:
            var_id = id(var)
            
            # Multi-timescale gradient memory
            self._short_term_memory[var_id] = self.add_variable_from_reference(
                var, name="short_term_memory"
            )
            self._medium_term_memory[var_id] = self.add_variable_from_reference(
                var, name="medium_term_memory"
            )
            self._long_term_memory[var_id] = self.add_variable_from_reference(
                var, name="long_term_memory"
            )
            
            # Second moment for Adam-style updates
            self._second_moment[var_id] = self.add_variable_from_reference(
                var, name="second_moment"
            )
            
            if self._amsgrad:
                self._max_second_moment[var_id] = self.add_variable_from_reference(
                    var, name="max_second_moment"
                )
            
            # Gradient accumulation buffer
            if self._accumulate_gradients:
                self._accumulated_grads[var_id] = self.add_variable_from_reference(
                    var, name="accumulated_grad"
                )
                safe_name = self._sanitize_name(var.name)
                self._accumulation_count[var_id] = self.add_variable(
                    shape=(),
                    dtype=tf.int32,
                    initializer="zeros",
                    name=f"accum_count_{safe_name}"
                )
            
            # Adaptive blend weights
            if self._memory_blend_mode == "adaptive":
                safe_name = self._sanitize_name(var.name)
                self._blend_weights[var_id] = self.add_variable(
                    shape=(3,),
                    dtype=var.dtype,
                    initializer=tf.constant_initializer([
                        self._fixed_short_weight,
                        self._fixed_medium_weight,
                        self._fixed_long_weight
                    ]),
                    name=f"blend_weights_{safe_name}"
                )
        
        self._is_built = True
    
    def _get_update_interval(self, var: tf.Variable) -> int:
        """Get update interval for a variable."""
        var_id = id(var)
        
        if self._auto_schedule and var_id in self._var_intervals:
            return self._var_intervals[var_id]
        
        for pattern, interval in self._layer_update_intervals.items():
            if pattern in var.name:
                return interval
        return self._default_interval
    
    def _compute_should_update(self, var: tf.Variable) -> tf.Tensor:
        """Compute whether variable should update this step."""
        interval = self._get_update_interval(var)
        
        if interval <= 1:
            return tf.constant(True)
        
        in_warmup = self._global_step < self._warmup_steps
        is_update_step = tf.equal(self._global_step % interval, 0)
        
        return tf.logical_or(in_warmup, is_update_step)
    
    def _compute_scheduled_lr(self) -> tf.Tensor:
        """Compute learning rate based on per-epoch schedule."""
        if self._steps_per_epoch is None or self._steps_per_epoch <= 0:
            return tf.constant(self._base_lr, dtype=tf.float32)
        
        steps_per_epoch = tf.cast(self._steps_per_epoch, tf.float32)
        global_step = tf.cast(self._global_step, tf.float32)
        
        step_in_epoch = tf.math.mod(global_step - 1.0, steps_per_epoch)
        
        warmup_steps = self._lr_warmup_fraction * steps_per_epoch
        hold_end = (self._lr_warmup_fraction + self._lr_hold_fraction) * steps_per_epoch
        
        min_lr = self._base_lr * self._lr_min_fraction
        start_lr = self._base_lr * self._lr_warmup_start_fraction
        peak_lr = self._base_lr
        
        # Warmup
        warmup_progress = tf.minimum(step_in_epoch / tf.maximum(warmup_steps, 1.0), 1.0)
        warmup_lr = start_lr + warmup_progress * (peak_lr - start_lr)
        
        # Hold
        hold_lr = peak_lr
        
        # Decay
        decay_start = hold_end
        decay_steps = self._lr_decay_fraction * steps_per_epoch
        decay_progress = tf.minimum(
            (step_in_epoch - decay_start) / tf.maximum(decay_steps, 1.0), 1.0
        )
        decay_lr = peak_lr - decay_progress * (peak_lr - min_lr)
        
        lr = tf.where(step_in_epoch < warmup_steps, warmup_lr,
              tf.where(step_in_epoch < hold_end, hold_lr, decay_lr))
        
        return lr
    
    def set_steps_per_epoch(self, steps_per_epoch: int) -> None:
        """
        Set steps per epoch for LR schedule.
        
        Args:
            steps_per_epoch: Number of training steps per epoch.
        """
        self._steps_per_epoch = steps_per_epoch
        print(f"[NestedLearningOptimizer] LR schedule configured:")
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Warmup: {self._lr_warmup_fraction*100:.0f}% ({int(steps_per_epoch*self._lr_warmup_fraction)} steps)")
        print(f"  Hold: {self._lr_hold_fraction*100:.0f}% ({int(steps_per_epoch*self._lr_hold_fraction)} steps)")
        print(f"  Decay: {self._lr_decay_fraction*100:.0f}% ({int(steps_per_epoch*self._lr_decay_fraction)} steps)")
    
    def _compress_gradient(self, gradient: tf.Tensor) -> tf.Tensor:
        """Compress gradient using low-rank approximation if enabled."""
        if not self._use_gradient_compression or self._compression_rank is None:
            return gradient
        
        if len(gradient.shape) < 2:
            return gradient
        
        original_shape = tf.shape(gradient)
        grad_2d = tf.reshape(gradient, [original_shape[0], -1])
        
        rank = min(self._compression_rank, min(grad_2d.shape[0], grad_2d.shape[1]))
        s, u, v = tf.linalg.svd(grad_2d)
        
        s_truncated = s[:rank]
        u_truncated = u[:, :rank]
        v_truncated = v[:, :rank]
        
        compressed = tf.matmul(u_truncated * s_truncated, v_truncated, transpose_b=True)
        return tf.reshape(compressed, original_shape)
    
    def _update_memory_stores(self, var: tf.Variable, gradient: tf.Tensor) -> None:
        """Update all memory timescales with new gradient."""
        var_id = id(var)
        grad_to_store = self._compress_gradient(gradient)
        
        self._short_term_memory[var_id].assign(
            self._short_term_decay * self._short_term_memory[var_id] + 
            (1 - self._short_term_decay) * grad_to_store
        )
        self._medium_term_memory[var_id].assign(
            self._medium_term_decay * self._medium_term_memory[var_id] + 
            (1 - self._medium_term_decay) * grad_to_store
        )
        self._long_term_memory[var_id].assign(
            self._long_term_decay * self._long_term_memory[var_id] + 
            (1 - self._long_term_decay) * grad_to_store
        )
    
    def _compute_attention_weights(
        self, 
        query_grad: tf.Tensor, 
        short_mem: tf.Tensor, 
        medium_mem: tf.Tensor, 
        long_mem: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Compute attention weights for memory retrieval."""
        query_flat = tf.reshape(query_grad, [-1])
        short_flat = tf.reshape(short_mem, [-1])
        medium_flat = tf.reshape(medium_mem, [-1])
        long_flat = tf.reshape(long_mem, [-1])
        
        scale = tf.sqrt(tf.cast(tf.size(query_flat), tf.float32) + 1e-8)
        
        score_short = tf.reduce_sum(query_flat * short_flat) / scale
        score_medium = tf.reduce_sum(query_flat * medium_flat) / scale
        score_long = tf.reduce_sum(query_flat * long_flat) / scale
        
        scores = tf.stack([score_short, score_medium, score_long])
        weights = tf.nn.softmax(scores)
        
        return weights[0], weights[1], weights[2]
    
    def _retrieve_from_memory(self, var: tf.Variable, current_gradient: tf.Tensor) -> tf.Tensor:
        """Retrieve combined gradient from multi-timescale memory."""
        var_id = id(var)
        
        short_mem = self._short_term_memory[var_id]
        medium_mem = self._medium_term_memory[var_id]
        long_mem = self._long_term_memory[var_id]
        
        if self._memory_blend_mode == "fixed":
            w_s = self._fixed_short_weight
            w_m = self._fixed_medium_weight
            w_l = self._fixed_long_weight
            
        elif self._memory_blend_mode == "adaptive":
            raw_weights = self._blend_weights[var_id]
            normalized = tf.nn.softmax(raw_weights)
            w_s, w_m, w_l = normalized[0], normalized[1], normalized[2]
            
        elif self._memory_blend_mode == "attention":
            w_s, w_m, w_l = self._compute_attention_weights(
                current_gradient, short_mem, medium_mem, long_mem
            )
        else:
            w_s, w_m, w_l = 0.5, 0.3, 0.2
        
        memory_contribution = w_s * short_mem + w_m * medium_mem + w_l * long_mem
        combined_gradient = 0.5 * current_gradient + 0.5 * memory_contribution
        
        return combined_gradient
    
    def _update_blend_weights(
        self, 
        var: tf.Variable, 
        gradient: tf.Tensor, 
        combined_gradient: tf.Tensor
    ) -> None:
        """Update adaptive blend weights."""
        if self._memory_blend_mode != "adaptive":
            return
            
        var_id = id(var)
        
        short_mem = self._short_term_memory[var_id]
        medium_mem = self._medium_term_memory[var_id]
        long_mem = self._long_term_memory[var_id]
        
        grad_flat = tf.reshape(gradient, [-1])
        grad_norm = tf.norm(grad_flat) + 1e-8
        
        align_short = tf.reduce_sum(grad_flat * tf.reshape(short_mem, [-1])) / grad_norm
        align_medium = tf.reduce_sum(grad_flat * tf.reshape(medium_mem, [-1])) / grad_norm
        align_long = tf.reduce_sum(grad_flat * tf.reshape(long_mem, [-1])) / grad_norm
        
        current_weights = self._blend_weights[var_id]
        weight_lr = 0.01
        
        delta = tf.stack([align_short, align_medium, align_long])
        new_weights = current_weights + weight_lr * delta
        
        self._blend_weights[var_id].assign(new_weights)
    
    def update_step(
        self, 
        gradient: tf.Tensor, 
        variable: tf.Variable, 
        learning_rate: Union[float, tf.Tensor]
    ) -> None:
        """Update a single variable using the Nested Learning approach."""
        var_id = id(variable)
        should_update = self._compute_should_update(variable)
        
        if self._steps_per_epoch is not None and self._steps_per_epoch > 0:
            learning_rate = self._compute_scheduled_lr()
        
        self._update_memory_stores(variable, gradient)
        
        if self._accumulate_gradients:
            count = tf.cast(self._accumulation_count[var_id], gradient.dtype)
            avg_grad = tf.cond(
                count > 0,
                lambda: (self._accumulated_grads[var_id] + gradient) / (count + 1),
                lambda: gradient
            )
            effective_gradient = tf.cond(should_update, lambda: avg_grad, lambda: gradient)
        else:
            effective_gradient = gradient
        
        combined_gradient = self._retrieve_from_memory(variable, effective_gradient)
        
        new_second_moment = (
            self._beta_2 * self._second_moment[var_id] + 
            (1 - self._beta_2) * tf.square(combined_gradient)
        )
        
        if self._amsgrad:
            new_max_moment = tf.maximum(self._max_second_moment[var_id], new_second_moment)
            v_hat = new_max_moment
        else:
            v_hat = new_second_moment
        
        step = tf.cast(self._global_step + 1, variable.dtype)
        bias_correction = 1.0 - tf.pow(self._beta_2, step)
        v_corrected = v_hat / bias_correction
        
        weight_update = learning_rate * combined_gradient / (tf.sqrt(v_corrected) + self._epsilon)
        
        def do_update():
            self._second_moment[var_id].assign(new_second_moment)
            if self._amsgrad:
                self._max_second_moment[var_id].assign(new_max_moment)
            variable.assign_sub(weight_update)
            if self._accumulate_gradients:
                self._accumulated_grads[var_id].assign(tf.zeros_like(gradient))
                self._accumulation_count[var_id].assign(0)
            if self._memory_blend_mode == "adaptive":
                self._update_blend_weights(variable, effective_gradient, combined_gradient)
            return tf.constant(0.0)
        
        def do_accumulate():
            if self._accumulate_gradients:
                self._accumulated_grads[var_id].assign_add(gradient)
                self._accumulation_count[var_id].assign_add(1)
            return tf.constant(0.0)
        
        tf.cond(should_update, do_update, do_accumulate)
    
    def apply_gradients(
        self, 
        grads_and_vars: List[Tuple[tf.Tensor, tf.Variable]], 
        **kwargs: Any
    ) -> tf.Variable:
        """Apply gradients with nested learning approach."""
        grads_and_vars = list(grads_and_vars)
        
        if not self._is_built:
            var_list = [v for g, v in grads_and_vars if g is not None]
            self.build(var_list)
        
        self._global_step.assign_add(1)
        
        for grad, var in grads_and_vars:
            if grad is None:
                continue
            self.update_step(grad, var, self.learning_rate)
        
        return self._global_step
    
    def get_config(self) -> Dict[str, Any]:
        """Serialize optimizer configuration."""
        config = super().get_config()
        config.update({
            "short_term_decay": self._short_term_decay,
            "medium_term_decay": self._medium_term_decay,
            "long_term_decay": self._long_term_decay,
            "memory_blend_mode": self._memory_blend_mode,
            "short_term_weight": self._fixed_short_weight,
            "medium_term_weight": self._fixed_medium_weight,
            "long_term_weight": self._fixed_long_weight,
            "use_gradient_compression": self._use_gradient_compression,
            "compression_rank": self._compression_rank,
            "auto_schedule": self._auto_schedule,
            "max_interval": self._max_interval,
            "schedule_curve": self._schedule_curve,
            "reverse_depth_order": self._reverse_depth_order,
            "warmup_steps": self._warmup_steps,
            "accumulate_gradients": self._accumulate_gradients,
            "layer_update_intervals": self._layer_update_intervals,
            "default_interval": self._default_interval,
            "beta_1": self._beta_1,
            "beta_2": self._beta_2,
            "epsilon": self._epsilon,
            "amsgrad": self._amsgrad,
            "steps_per_epoch": self._steps_per_epoch,
            "lr_warmup_fraction": self._lr_warmup_fraction,
            "lr_hold_fraction": self._lr_hold_fraction,
            "lr_decay_fraction": self._lr_decay_fraction,
            "lr_min_fraction": self._lr_min_fraction,
            "lr_warmup_start_fraction": self._lr_warmup_start_fraction,
        })
        return config
    
    @property
    def iterations(self) -> Optional[tf.Variable]:
        """Return global step."""
        return self._global_step
    
    @property
    def current_learning_rate(self) -> tf.Tensor:
        """Return current scheduled learning rate."""
        if self._steps_per_epoch is not None and self._steps_per_epoch > 0:
            return self._compute_scheduled_lr()
        return tf.constant(self._base_lr, dtype=tf.float32)
    
    def print_memory_stats(self) -> None:
        """Print memory statistics for debugging."""
        print("\n" + "="*70)
        print("NESTED LEARNING OPTIMIZER - STATUS")
        print("="*70)
        step = self._global_step.numpy() if self._global_step is not None else 0
        print(f"Global step: {step}")
        print(f"In warmup: {step < self._warmup_steps} (warmup ends at step {self._warmup_steps})")
        print(f"Memory blend mode: {self._memory_blend_mode}")
        print(f"Decay rates - Short: {self._short_term_decay}, "
              f"Medium: {self._medium_term_decay}, Long: {self._long_term_decay}")
        print(f"Auto-schedule: {self._auto_schedule}, Max interval: {self._max_interval}")
        print(f"Gradient accumulation: {self._accumulate_gradients}")
        
        if self._steps_per_epoch is not None:
            current_lr = self.current_learning_rate.numpy() if hasattr(self.current_learning_rate, 'numpy') else self._base_lr
            step_in_epoch = (step - 1) % self._steps_per_epoch if step > 0 else 0
            warmup_end = int(self._lr_warmup_fraction * self._steps_per_epoch)
            hold_end = int((self._lr_warmup_fraction + self._lr_hold_fraction) * self._steps_per_epoch)
            if step_in_epoch < warmup_end:
                phase = "WARMUP"
            elif step_in_epoch < hold_end:
                phase = "HOLD"
            else:
                phase = "DECAY"
            print(f"LR schedule: {phase} phase, step {step_in_epoch}/{self._steps_per_epoch}, LR={current_lr:.6f}")
        print("-"*70)
        
        if self._var_depths:
            print("\nDepth-based update intervals (sample):")
            depth_samples: Dict[int, int] = {}
            for ref, depth in self._var_depths.items():
                if depth not in depth_samples:
                    depth_samples[depth] = self._var_intervals[ref]
            for depth in sorted(depth_samples.keys())[:10]:
                interval = depth_samples[depth]
                mem_type = "long-term" if interval >= 5 else "medium" if interval >= 2 else "short-term"
                print(f"  Depth {depth}: interval={interval} [{mem_type}]")
        
        if self._short_term_memory:
            print("\nMemory norms (sample of first 5 variables):")
            for i, (ref, mem) in enumerate(list(self._short_term_memory.items())[:5]):
                short_norm = tf.norm(mem).numpy()
                med_norm = tf.norm(self._medium_term_memory[ref]).numpy()
                long_norm = tf.norm(self._long_term_memory[ref]).numpy()
                print(f"  Var {i}: Short={short_norm:.4f}, Med={med_norm:.4f}, Long={long_norm:.4f}")
        
        print("="*70 + "\n")

