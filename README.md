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

<details>
  <summary><b> What is kernel_size?</b></summary>
  
  
  - A kernel refers to a small matrix of weights that is used to extract features from an input image or signal. The kernel slides over the input data, performing a dot product at each position, which generates a new output feature map.
  - The kernel size, also known as the filter size, determines the size of the receptive field of the convolutional neural network (CNN) layer. The receptive field refers to the area of the input data that is taken into account by the kernel during the convolution operation.
  - The kernel size is typically set as a hyperparameter of the CNN and is usually a square matrix, with the most common sizes being 3x3, 5x5, and 7x7. The choice of kernel size depends on the specific task and the characteristics of the input data. Smaller kernel sizes are used to capture local features, while larger kernel sizes can capture more global features.
</details>


<details>
  <summary><b> What is pool_size?</b></summary>
  
  
  - In deep learning, pooling refers to a downsampling operation that reduces the spatial size (width and height) of the input feature map while retaining important features. Pooling is often used after convolutional layers in a convolutional neural network (CNN) to reduce the size of the feature maps and to help control overfitting.
  - The pool size, also known as the pooling kernel size, determines the size of the pooling window that slides over the input feature map. The most common pool size is 2x2, although other sizes such as 3x3 or 4x4 can also be used.
  - During the pooling operation, the pool window slides over the feature map and performs an operation such as maximum or average pooling, which takes the maximum or average value of the pixels in the window, respectively. This reduces the size of the feature map while retaining the most important information.
</details>

<details>
  <summary><b> What is Flatten Layer?</b></summary>
  
  
  - In deep learning, a flatten layer is a type of layer that transforms a multi-dimensional input tensor into a one-dimensional vector. This is often done in preparation for passing the data through a fully connected neural network layer.
  - The flatten layer takes the input tensor, which can have multiple dimensions such as height, width, and depth (or channels), and rearranges it into a one-dimensional vector by concatenating all the elements of the input tensor in a single row. The resulting vector has a length equal to the product of the original tensor dimensions.
  - The purpose of the flatten layer is to convert the feature map generated by the convolutional layers into a format that can be processed by a fully connected layer, which requires a one-dimensional input vector. By flattening the feature map, the spatial relationships between the input pixels are lost, but the features extracted from the image are retained.
  - The flatten layer is typically used in the later stages of a convolutional neural network (CNN), after one or more convolutional and pooling layers. The output of the flatten layer is then passed to one or more fully connected layers, which can perform classification or regression tasks.
</details>

<details>
  <summary><b> What is Filter?</b></summary>
  
  
  - The term "filters" refers to the number of convolutional kernels that are applied to the input image. Each filter is a small matrix of weights that slides over the input image and performs element-wise multiplication and summation to produce a single output value in the output feature map.
  - Suppose, we defined a Conv2D layer with 32 filters. This means that 32 separate convolutional kernels are applied to the input image, each producing a separate output feature map. The output feature maps are then stacked together to form the output volume of the Conv2D layer.
  - Filter
    | 1 | 0 | 1 |
    | - | - | - |
    | 0 | 1 | 0 |
    | 1 | 0 | 1 |
    
</details>

<details>
  <summary><b> What is Activation Function?</b></summary>
  An activation function is a non-linear function that is applied to the output of a neural network layer to introduce non-linearity into the model. It allows the neural network to learn complex, non-linear relationships between the input and output, which would be impossible with a linear model.
  
  
  - <strong>ReLU: </strong>which is one of the most commonly used activation functions in deep learning. The ReLU function applies the element-wise function <code>f(x) = max(0, x)</code> to the output of the previous layer. In other words, it sets all negative values in the output to zero and leaves all positive values unchanged. This has the effect of introducing non-linearity into the model and can help prevent the vanishing gradient problem during training.
  - <strong>Sigmoid: </strong><code>f(x) = 1 / (1 + exp(-x))</code>, which squashes the output to a range between 0 and 1 and is often used in binary classification problems.
  - <strong>Softmax: </strong><code>f(x_i) = exp(x_i) / sum(exp(x_j))</code>, which converts the output of the previous layer to a probability distribution over a set of mutually exclusive classes and is often used as the final activation function in classification problems.
  - <strong>Tanh: </strong>f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)), which squashes the output to a range between -1 and 1 and is often used in multi-class classification problems.

</details>

<details>
  <summary><b> What is Padding?</b></summary>
  Padding is a technique used in convolutional neural networks to preserve the spatial dimensions of the input image after convolution, by adding zeros around the input image before convolution. This is done to ensure that the output of the convolution operation has the same shape as the input image, which is important for building deeper networks that can extract complex features from larger images.
  
  
  - <strong>Padding- Same: </strong>Which means that we added just enough padding to the input image so that the output feature map has the same spatial dimensions as the input image. 
  - <strong>Padding- Valid: </strong>Which means that no padding is added to the input image and the output feature map is smaller than the input image. In this case, the padding added to each side of the input image would be 0.

</details>

<details>
  <summary><b> What is Stride?</b></summary>
  
  
  - Stride is a parameter used in convolutional neural networks to control the amount of sliding that the convolutional kernel moves across the input image. It determines the number of pixels that the kernel shifts at a time 
  - <strong>Strides=1,</strong> which means that the convolutional kernel moves one pixel at a time in both the horizontal and vertical directions. This is the default stride value, and it is commonly used in many convolutional neural networks.
  - However, it is possible to set the stride to a value greater than 1, which means that the kernel skips pixels during the convolution operation. 
  - For example, if the stride is set to 2, the kernel would move two pixels at a time, effectively reducing the spatial dimensions of the output feature map by a factor of 2.
  - Increasing the stride can have the effect of reducing the computational cost of the convolutional operation and can help prevent overfitting. However, it also reduces the amount of spatial information in the output feature map, which may lead to a loss of performance in some applications.

</details>

<details>
  <summary><b> What is input shape?</b></summary>
  
  
  - In convolutional neural networks, the input shape typically refers to the size of the input image or volume, including the number of channels.
  - For example, in the Conv2D layer that I provided earlier as an example, the input_shape parameter was set to (32, 32, 3), which means that the input images are 32x32 pixels in size and have 3 color channels (red, green, and blue). This input shape is appropriate for many computer vision tasks, including image classification, object detection, and segmentation.

</details>

<details>
  <summary><b> What is Dense Layer?</b></summary>
  
  
  - The Dense layer is a type of neural network layer commonly used in deep learning models for a variety of tasks, such as image classification, language processing, and time series analysis. The Dense layer is a fully connected layer, meaning that each neuron in the layer is connected to every neuron in the previous layer.
  - The Dense layer takes as input a matrix of activations from the previous layer, and applies a linear transformation followed by a non-linear activation function to produce a new matrix of activations. The linear transformation involves computing a dot product between the input matrix and a weight matrix, and adding a bias vector to the result. The activation function is then applied element-wise to the resulting matrix.
  - Example: <code>model.add(Dense(128, activation='relu'))</code> here adding a Dense layer with 128 neurons and ReLU activation function.

</details>



