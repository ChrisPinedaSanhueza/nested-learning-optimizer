"""Basic usage example for NestedLearningOptimizer."""

import tensorflow as tf
from nested_learning_optimizer import NestedLearningOptimizer

# Create a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Create optimizer with depth-based scheduling
optimizer = NestedLearningOptimizer(
    learning_rate=0.001,
    auto_schedule=True,
    max_interval=4,
    schedule_curve="cosine",
    memory_blend_mode="attention",
)

# Analyze model structure for depth-based scheduling
optimizer.compute_depths_from_model(model)

# Compile model
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# Train
model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=32,
    validation_data=(x_test, y_test)
)

# Print optimizer stats
optimizer.print_memory_stats()

