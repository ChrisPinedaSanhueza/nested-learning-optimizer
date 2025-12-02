"""Tests for NestedLearningOptimizer."""

import pytest
import numpy as np
import tensorflow as tf
from nested_learning_optimizer import NestedLearningOptimizer


class TestNestedLearningOptimizer:
    """Test cases for the optimizer."""
    
    def test_basic_instantiation(self):
        """Test optimizer can be instantiated with defaults."""
        opt = NestedLearningOptimizer()
        assert opt is not None
        assert opt._memory_blend_mode == "attention"
    
    def test_custom_parameters(self):
        """Test optimizer with custom parameters."""
        opt = NestedLearningOptimizer(
            learning_rate=0.01,
            short_term_decay=0.9,
            memory_blend_mode="fixed",
            max_interval=8,
        )
        assert opt._short_term_decay == 0.9
        assert opt._memory_blend_mode == "fixed"
        assert opt._max_interval == 8
    
    def test_invalid_blend_mode_raises(self):
        """Test that invalid blend mode raises ValueError."""
        with pytest.raises(ValueError):
            NestedLearningOptimizer(memory_blend_mode="invalid")
    
    def test_invalid_schedule_curve_raises(self):
        """Test that invalid schedule curve raises ValueError."""
        with pytest.raises(ValueError):
            NestedLearningOptimizer(schedule_curve="invalid")
    
    def test_simple_training(self):
        """Test optimizer works with simple model."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(4, input_shape=(2,)),
            tf.keras.layers.Dense(1)
        ])
        
        opt = NestedLearningOptimizer(learning_rate=0.01, auto_schedule=False)
        model.compile(optimizer=opt, loss='mse')
        
        x = np.random.randn(10, 2).astype(np.float32)
        y = np.random.randn(10, 1).astype(np.float32)
        
        history = model.fit(x, y, epochs=2, verbose=0)
        assert len(history.history['loss']) == 2
    
    def test_depth_computation(self):
        """Test compute_depths_from_model."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(8, input_shape=(4,)),
            tf.keras.layers.Dense(4),
            tf.keras.layers.Dense(2)
        ])
        
        opt = NestedLearningOptimizer(auto_schedule=True)
        depths = opt.compute_depths_from_model(model)
        
        assert len(depths) > 0
        assert opt._max_detected_depth >= 0
    
    def test_serialization(self):
        """Test get_config and from_config."""
        opt = NestedLearningOptimizer(
            learning_rate=0.005,
            max_interval=10,
            memory_blend_mode="fixed"
        )
        
        config = opt.get_config()
        assert config['max_interval'] == 10
        assert config['memory_blend_mode'] == "fixed"
        
        opt2 = NestedLearningOptimizer.from_config(config)
        assert opt2._max_interval == 10
    
    def test_lr_schedule(self):
        """Test learning rate scheduling."""
        opt = NestedLearningOptimizer(
            learning_rate=0.001,
            steps_per_epoch=100,
            lr_warmup_fraction=0.2,
        )
        
        # Build with dummy variable
        var = tf.Variable([1.0, 2.0])
        opt.build([var])
        
        # Check LR computation doesn't crash
        lr = opt.current_learning_rate
        assert lr is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

