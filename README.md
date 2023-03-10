# Machine Learning Basic
<details>
  <summary><b> What is batch_size?</b></summary>
    - Batch size refers to the number of samples or data points that a machine learning algorithm uses in one iteration or training step. In other words, it determines how many examples are processed at once by the algorithm during training. For instance, if a dataset contains 1000 training examples, and the batch size is set to 32, the algorithm would take 32 examples at a time and update the weights of the model accordingly. The process of updating the weights after processing each batch of data is called stochastic gradient descent (SGD).The batch size can affect the accuracy and speed of the training process. A larger batch size can speed up the training process, but it can also cause the model to generalize poorly. A smaller batch size can lead to slower training times but may improve the accuracy of the model. Choosing the appropriate batch size is a trade-off between these factors and depends on the specific problem being addressed.
</details>
<details>
  <summary><b> What is class_mode?</b></summary>
    - <code>class_mode</code> is a parameter in Keras ImageDataGenerator class that determines how the labels are returned for the image dataset during training or testing.

There are different options available for class_mode:

  - <code>class_mode='categorical': </code> This mode is used for multi-class classification problems, where the labels are one-hot encoded vectors.
  - <code>class_mode='binary': </code> This mode is used for binary classification problems, where the labels are binary values (0 or 1).
  - <code>class_mode='sparse': </code> This mode is used for multi-class classification problems, where the labels are integers representing the class index.
  - <code>class_mode=None: </code> This mode is used when you do not have any labels for the images.

The choice of class_mode depends on the type of problem you are trying to solve and how your labels are encoded. It is important to choose the correct class_mode to ensure that your model is trained properly and can accurately predict the correct labels.
</details>


