# Tensorflow_Tutorial

# How to install Tensorflow GPU
Check version: https://www.tensorflow.org/install/source#gpu

# Note
<details>
<summary>Visualize model</summary>

```
from tensorflow.keras.utils import plot_model
plot_model(model, to_file="my_model.png", show_shapes=True)
```

</details>

<details>
<summary>initialize weights at any layer</summary>

[API](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#set_weights)

```
# initialize parameters
model.layers[0].set_weights([np.array([[-0.34]]), np.array([0.04])])

# declare optimization method and loss function
model.compile(optimizer='adam', loss='mean_squared_error')

# training
model.fit(X, y, 4, epochs=100)

# parameters after one epoch
print('weight-bias: \n', model.layers[0].get_weights())
```

</details>

<details>
<summary>Get weight at any layer</summary>

```
print(model.layers[0].get_weights())
```

</details>

<details>
<summary>Predict</summary>

```
y_hat = model.predict(X_testing)
```

</details>

<details>
<summary>Save-Load weights</summary>

[API](https://www.tensorflow.org/tutorials/keras/save_and_load#manually_save_weights)
```
# Save the weights
model.save_weights('./checkpoints/my_checkpoint')

# Create a new model instance
model = create_model()

# Restore the weights
model.load_weights('./checkpoints/my_checkpoint')

# Evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
```
</details>

<details>
<summary>Save-Load model</summary>

[API](https://www.tensorflow.org/tutorials/keras/save_and_load#save_the_entire_model)
```
# Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model to a HDF5 file.
# The '.h5' extension indicates that the model should be saved to HDF5.
model.save('my_model.h5')

# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model('my_model.h5')

# Show the model architecture
new_model.summary()

loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
```
</details>