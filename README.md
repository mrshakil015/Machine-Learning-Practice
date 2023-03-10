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


<details>
  <summary><b>Overview of the layers typically used in CNNs:</b></summary>
  Convolutional neural networks (CNNs) are a type of deep learning neural network that are specifically designed for processing images and other high-dimensional data. Here's an overview of the layers typically used in CNNs:

   - <b>Convolutional Layer:</b> This layer performs convolution operations on the input data using a set of filters to produce a set of feature maps. The filters are learned during training and can detect various types of features such as edges, corners, and textures.
   - <b>Activation Layer:</b> This layer applies an activation function to the output of the convolutional layer. Common activation functions used in CNNs include ReLU, sigmoid, and tanh.
   - <b>Pooling Layer:</b> This layer reduces the spatial dimensions of the feature maps produced by the convolutional layer by selecting the maximum or average value within small regions of the feature maps.
   - <b>Dropout Layer:</b> This layer randomly drops out a percentage of neurons in the previous layer during training to prevent overfitting.
   - <b>Flatten Layer:</b> This layer flattens the output of the previous layer into a 1D vector to be passed on to the fully connected layers.
   - <b>Fully Connected Layer:</b> This layer performs computations on the flattened output of the previous layer using a set of weights and biases to produce an output vector. This layer is similar to the fully connected layers used in traditional neural networks.
   - <b>Output Layer:</b> This layer produces the output of the network. The number of neurons in this layer depends on the type of problem being solved. For example, in a binary classification problem, there would be one output neuron, while in a multi-class classification problem, there would be multiple output neurons, one for each class.
</details>

<details>
  <summary><b>Overview of the layers typically used in ANN:</b></summary>
  Artificial neural networks (ANN) consist of multiple layers of interconnected neurons that process and transform input data to generate output. There are several types of layers that can be used in an ANN. Here is an overview of the most commonly used layers:

   - <b>Input Layer:</b> This layer is the first layer of the network and takes in the input data. It does not perform any computation on the input data, but rather passes it on to the next layer.
   
   - <b>Hidden Layer:</b> These are the layers in between the input and output layers. They perform computations on the input data by applying a set of weights and biases to the inputs and passing the result through an activation function. The number of hidden layers and the number of neurons in each layer are determined by the complexity of the problem being solved.
   
   - <b>Ouptput Layer:</b> This layer produces the output of the network. The number of neurons in the output layer depends on the type of problem being solved. For example, in a binary classification problem, there would be one output neuron, while in a multi-class classification problem, there would be multiple output neurons, one for each class.
   
   - <b>Fully Connected Layer:</b> A fully connected layer is a type of hidden layer where each neuron is connected to every neuron in the previous layer. This layer is used to learn complex relationships between inputs and outputs.
   
   - <b>Recurrent Layer:</b> A recurrent layer is a type of layer used in recurrent neural networks (RNNs) that allows the network to process sequences of data by retaining information about previous inputs. This layer is commonly used in natural language processing and speech recognition tasks.
   
   - <b>Fully Connected Layer:</b> This layer performs computations on the flattened output of the previous layer using a set of weights and biases to produce an output vector. This layer is similar to the fully connected layers used in traditional neural networks.
   - <b>Convolutional Layer:</b> A convolutional layer is a type of layer used in convolutional neural networks (CNNs) that applies a set of filters to the input data to extract features from it. This layer is commonly used in image and video processing tasks.
   
   - <b>Dropout Layer:</b> A dropout layer is a regularization technique that randomly drops out a percentage of neurons in the previous layer during training. This helps prevent overfitting and improves the generalization of the model.
</details>

