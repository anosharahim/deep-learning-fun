In this project, I build a fully-connected neural network from scratch on MNIST data. The challenge is to build it with only Python and NumPy and no other libraries. 

## Step 1: Load Dataset
I used the MNIST dataset of 70,000 handwritten digit images from sklearn and split it into training (45,000 images), validation (15,000) and test set (10,000 images). The class labels are one-hot encoded in order for the neural network to interpret categorical variables.

## Step 2: Define Neural Net Functions
#### Layers 
The neural network has 2 fully-connected layers. The first layer i.e. input has 784 nodes of pixels from the MNIST images, which are connected to the first hidden layer, containing 64 nodes, and the output layer, which has 10 nodes for the 10 mnist digits to be classified. 

#### Activation Function 
For the activation function, I have used a sigmoid function which keeps the output of each neuron between 0 and 1:

$\sigma(z) = \frac{1}{1+e^{-z}} $ 

Since we are doing multi-class classification, I also used a Softmax function on the final layer. Softmax is a takes a vector as input and outputs a probability distribution whose total sums to one. As such , it gives the probability of an input belonging to any particular class. 

$S(\vec{x})_i = \frac{e^{x_i}}{\Sigma_{j=1}^{K}e^{x_j}} $

#### Loss Functions 
We used a binary cross-entropy function as our loss function in conjunction with softmax. Where softmax outputs probabilities, cross entropy takes those probabilities and measures their distance from the truth value. 

$L(Y,\hat{Y})=-\frac{1}{m}(y^i log(\hat{y}^i+ (1-y)log(1-\hat{y}^i))$

## Step 3: Gradient Descent 

### Gradient Descent 

For the gradient descent step, I have randomly initialized weights to be really small numbers, and biases to be zero. Weights cannot be initialized to zero because it needs to be multiplied by the neuron's value, and it would not start training. On the other hand, biases can be initialized to 0 because they just need to be added to the neuron's output. In the function below, weights and biases are globally initialized so that the updated values can be saved and reused for testing. 

I started out with batch gradient descent, and at a learning rate of 1 it was still learning very slowly since it was performing the entire gradient descent for the entire dataset each time. In order to improve the learning, I implemented mini-batch gradient descent, as well as increased the learning rate to 3, which significantly improved performance on both training and validation set.  

#### Forward Pass
In forward pass, the input values are fed through the neural network where weights w are multiplied by the input values X along with adding the bias. This is done for each layer as per the following formulas. 

$ \hat{y} = \sigma (z) $\
$ z = w^T x +b $ \
$ \sigma(z) = \frac{1}{1+e^{-z}} $ 


#### Backward Pass 
In backward pass, the main thing we need to is find out how sensitive the loss is to different components of the weights and bias matrices. In order to do this, we take the derivative of the loss function using the chain rule as follows: 

$L(Y,\hat{Y})=-\frac{1}{m}(y^i \log(\hat{y}^i+ (1-y)\log(1-\hat{y}^i))$

We need to find the derivative of the loss function with respect of biases and weights, for which we apply the chain rule as follows: 

$\frac{\partial L}{\partial w_j}=\frac{\partial L}{\partial \hat{y}}.\frac{\partial \hat{y}}{\partial z}.\frac{\partial z}{\partial w_j}$

$ \frac{\partial L}{\partial b} =\frac{1}{m} \Sigma_{i=1}^m(\hat{y}^i-y^i)  $ 

$\frac{\partial L}{\partial \hat{y}} = \frac{\hat{y}-y}{\hat{y}(1-\hat{y})}$

$\frac{\partial \hat{y}}{\partial z} = \hat{y}(1-\hat{y})$

$\frac{\partial z}{\partial w_j} = \frac{\partial (w_0x_0+w_1 x_1+w_2 x_2+....+w_n x_n+b)}{\partial w_j}$

This is the main logic behind the gradient descent function. Using this chain rule, we get the gradients of the loss function with respect to the biases and weights, which are then used to update the weights and biases. Full derivations for the above equations can be found on Jonathan Weisberg's blogpost on [Building Neural Networks from Scratch](https://jonathanweisberg.org/post/A%20Neural%20Network%20from%20Scratch%20-%20Part%201/).

## Step 4: Training

When applying this neural network to the validation set, I set backpropagation off to compute loss and accuracy. Using mini-batches helped increase the speed of training and did not require holding the entire dataset in memory, as opposed to batch gradient descent. However, mini-batch still runs the risk of getting trapped at local minima. Moreover, increasing the learning rate to 5 resulted in 95% accuracy after 230 epochs, as opposed to staying at 94% after 300 epochs at a learning rate of 4. The learning can be improved in the future by adding other optimization methods such as by incrementally and adaptively increasing learning rate as needed, so as to not overshoot. So far, the neural network can predict handwritten digits with about 94.9% accuracy on the validation data as well as the test data, which shows evidence for little overfitting.  

## Step 5: Performance 