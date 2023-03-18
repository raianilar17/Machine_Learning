# Neural--Networks-2

Implement Back-propagation on  Recognize Hand-Written Digits 

# Neural Networks Learning

In this project, I implemented the backpropagation algorithm for neural networks and apply it to the task of hand-written digit recognition. 

To get started with the project, you will need to download the code and unzip its contents to the directory where you wish to run the project. If needed, use the cd command in Octave to change to this directory before starting this exercise.

This code could also run on MATLAB(you can try). In future, I will try to excute this code on MATLAB also.

This code is successfully implemented on octave version 4.2.1

# Environment Setup Instructions

## Instructions for installing Octave 

The Project use Octave (Octave is a free alternative to MATLAB) a high-level programming language well-suited for numerical computations. If you do not have Octave installed, please install.

[Download](https://www.gnu.org/software/octave/download.html)

[Octave_for_GNU/Linux](http://wiki.octave.org/Octave_for_GNU/Linux)

[Octave_for_Microsoft_Windows](http://wiki.octave.org/Octave_for_Microsoft_Windows)

[Octave_for_macOS#Homebrew](http://wiki.octave.org/Octave_for_macOS#Homebrew)

Documentation for Octave functions can be found at the [Octave documentation pages](http://www.gnu.org/software/octave/doc/interpreter/).

# Files included in this project

ex4.m - Octave/MATLAB script that steps you through the project

ex4data1.mat - Training set of hand-written digits

ex4weights.mat - Neural network parameters(Pre-trained)

displayData.m - Function to help visualize the dataset

fmincg.m - Function minimization routine (similar to fminunc)

sigmoid.m - Sigmoid function

computeNumericalGradient.m - Numerically compute gradients

checkNNGradients.m - Function to help check your gradients

debugInitializeWeights.m - Function for initializing weights

predict.m - Neural network prediction function

sigmoidGradient.m - Compute the gradient of the sigmoid function

randInitializeWeights.m - Randomly initialize weights

nnCostFunction.m - Neural network cost function

Throughout the project, you will be using the script [ex4.m](ex4.m). These scripts set up the dataset for the problems and make calls to functions.

The project use Octave(Octave is a free alternative to MATLAB), a high-level programming language well-suited for numerical computations. If you do not have Octave installed, please refer to above the installation instructions in the “Environment Setup Instructions”.

# Neural Networks

In the previous project [Neural-Network-1](https://github.com/raianilar17/Neural-Networks-1),I implemented feedforward propagation for neural networks and used it to predict handwritten digits with the weights(Pre-trained weights). In this project, I implemented the backpropagation algorithm to learn the parameters for the neural network.

## Visualizing the data

In the first part of [ex4.m](ex4.m), the code will load the data and display it on a 2-dimensional plot (Figure 1) by calling the function [displayData](displayData).

Figure 1: Examples from the dataset

![](https://github.com/raianilar17/Neural-Networks-2/blob/master/Figure_01.jpg)

There are 5000 training examples in [ex3data1.mat](ex3data1.mat), where each training example is a 20 pixel by 20 pixel grayscale image of the digit. Each pixel is represented by a floating point number indicating the grayscale intensity at that location. The 20 by 20 grid of pixels is “unrolled” into a 400-dimensional vector. Each of these training examples becomes a single row in our data matrix X. This gives us a 5000 by 400 matrix X where every row is a training example for a handwritten digit image.

The second part of the training set is a 5000-dimensional vector y that contains labels for the training set. To make things more compatible with Octave/MATLAB indexing, where there is no zero index,I have mapped the digit zero to the value ten. Therefore, a “0” digit is labeled as “10”, while the digits “1” to “9” are labeled as “1” to “9” in their natural order.

## Model representation

It has 3 layers – an input layer, a hidden layer and an output layer. Recall that our inputs are pixel values of digit images. Since the images are of size 20 × 20, this gives us 400 input layer units (not counting the extra bias unit which always outputs +1). The training data will be loaded into the variables X and y by the [ex4.m](ex4.m) script.

In [ex4weights.mat](ex4weights.mat) stored a set of network parameters (Θ^(1) , Θ^(2) ) already trained by researchers and will be loaded by [ex4.m](ex4.m) into Theta1 and Theta2. The parameters have dimensions that are sized for a neural network with 25 units in the second layer and 10 output units (corresponding to the 10 digit classes).

## Feedforward and cost function

Now I implemented the cost function and gradient for the neural network in the code [nnCostFunction.m](nnCostFunction.m) to return the cost and grad.

whereas the original labels (in the variable y) were 1, 2, ..., 10, for the purpose of training a neural network, we need to recode the labels as vectors containing only values 0 or 1.

For example, if Training examples(x^(i)) is an image of the digit 5, then the corresponding label(y^(i)) (that you should use with the cost function) should be a 10-dimensional vector with fifth_row(y_5) = 1, and the other elements equal to 0.

Now, I implemented the feedforward computation that computes Hypothesis function(h_θ(x^(i))) for every example i and sum the cost over all examples.My code also work for a dataset of any size, with any number of labels (assume that there are always at least K ≥ 3 labels). where K is number of units in output layer.

## Implementation Note: 

The matrix X contains the examples in rows (i.e., X(i,:)’ is the i-th training example x^(i) , expressed as a n × 1 vector.)  The code in [nnCostFunction.m](nnCostFunction.m), you will need to add the column of 1’s to the X matrix. The parameters for each unit in the neural network is represented in Theta1 and Theta2 as one row. Specifically, the first row of Theta1 corresponds to the first hidden unit in the second layer.

Now, [ex4.m](ex4.m) will call your [nnCostFunction](nnCostFunction.m) using the loaded set of parameters for Theta1 and Theta2. You should see that the cost is about 0.287629.

## Regularized cost function

In the code [ex4.m](ex4.m), I also implented the cost function for neural networks with regularization.

I assume that the neural network will only have 3 layers – an input layer, a hidden layer and an output layer. My code work for any number of input units, hidden units and outputs units. My code also work with Theta1(Θ^(1)) and Theta2(Θ^(2)) of any size.

Note that you should not be regularizing the terms that correspond to the bias. For the matrices Theta1 and Theta2, This corresponds to the first column of each matrix. You should now add regularization to your cost function. Notice that you can first compute the unregularized cost function J using your existing [nnCostFunction.m](nnCostFunction.m) and then later add the cost for the regularization terms.

Now, [ex4.m](ex4.m) will call your [nnCostFunction](nnCostFunction.m) using the loaded set of parameters for Theta1 and Theta2, and λ = 1. You should see that the cost is about 0.383770.

# Backpropagation

In this part of the section, I implemented the backpropagation algorithm to compute the gradient for the neural network cost function in the code the [nnCostFunction.m](nnCostFunction.m) so that it returns an appropriate value for grad. Once I have computed the gradient, I able to train the neural network by minimizing the cost function J(Θ) using an advanced optimizer such as [fmincg](fmincg.m).

First implemented the backpropagation algorithm to compute the gradients for the parameters for the (unregularized) neural network. After I verified that my gradient computation for the unregularized case is correct, Then, I implemented the gradient for the regularized neural network.

## Sigmoid gradient

To help you get started with this part of the segment, I first implement the sigmoid gradient function. The gradient for the sigmoid function can be computed as:

g'(z) = g(z)(1 - g(z))

where

sigmoid(z) = g(z) = 1 / (1 + e^-z)

e^-z = exponential of z

z = matrix or vector or real number

I tried testing a few values by calling [sigmoidGradient(z)](sigmoidGradient.m) at the Octave command line. For large values (both positive and negative) of z, the gradient is close to 0. When z = 0, the gradient is exactly 0.25. My code also work with vectors and matrices. For a matrix, My function perform the sigmoid gradient function on every element.

## Random initialization

When training neural networks, it is important to randomly initialize the parameters for symmetry breaking. One effective strategy for random initialization is to randomly select values for Θ^(l) uniformly in the range [−Epsilon_init ,Epsilon_init]. This range of values ensures that the parameters are kept small and makes the learning more efficient.

One effective strategy for choosing Epsilon_init is to base it on the number of units in the network.

A good choice of Epsilon_init is:

Epsilon_init = squareroot(6)/squareroot(L_in + L_out) 

where:

L_in = s_l number of units in layer l.

L out = s_(l+1) are the number of units in the layers adjacent to Θ^(l).

The code is in [randInitializeWeights.m](randInitializeWeights.m) to initialize the weights for Theta(Θ).

## Backpropagation

Now, I implement the backpropagation algorithm. The intuition behind the backpropagation algorithm is as follows. Given a
training example (x^(t) , y^(t) ),First run a “forward pass” to compute all the activations throughout the network, including the output value of the hypothesis h_Θ(x). Then, for each node j in layer l, we would like to compute an “error term” (δ^l)_j that measures how much that node was “responsible” for any errors in our output.

For an output node, we can directly measure the difference between the network’s activation and the true target value, and use that to define (δ^(3)_j) (since layer 3 is the output layer). For the hidden units, you will compute (δ^(l)_j) based on a weighted average of the error terms of the nodes in layer (l + 1).

In detail,The backpropagation algorithm coming soon...

## Octave Tip: 

You should implement the backpropagation algorithm only after you have successfully completed the feedforward and
cost functions. While implementing the backpropagation algorithm, it is often useful to use the size function to print out the sizes of the variables you are working with if you run into dimension mismatch errors (“nonconformant arguments” errors in Octave).

After I implemented the backpropagation algorithm, the script [ex4.m](ex4.m) will proceed to run gradient checking on my implementation. The gradient check will allow you to increase your confidence that your code is computing the gradients correctly.

## Gradient checking

In your neural network, you are minimizing the cost function J(Θ). To perform gradient checking on your parameters, you can imagine “unrolling” the parameters Θ^(1) , Θ^(2) into a long vector θ. By doing so, you can think of the cost function being J(θ) instead and use the gradient checking procedure in code [checkNNGradients.m](checkNNGradients.m).

I implemented the function to compute the numerical gradient in [computeNumericalGradient.m](computeNumericalGradient.m). Then, In the next step of [ex4.m](ex4.m), it will run the function [checkNNGradients.m](checkNNGradients.m) which will create a small neural network and dataset that will be used for checking gradients. If backpropagation implementation is correct,you should see a relative difference that is less than 1e-9.

## Practical Tip:

When performing gradient checking, it is much more efficient to use a small neural network with a relatively small number
of input units and hidden units, thus having a relatively small number of parameters. Each dimension of θ requires two evaluations of the cost function and this can be expensive. In the function [checkNNGradients](checkNNGradients.m), The code creates a small random model and dataset which is used with [computeNumericalGradient](computeNumericalGradient.m) for gradient checking. Furthermore, after you are confident that your gradient computations are correct, you should turn off gradient checking before running your learning algorithm.

Gradient checking works for any function where you are computing the cost and the gradient. Concretely, you can use the same
[computeNumericalGradient.m](computeNumericalGradient.m) function to check if your gradient implementations for the other project are correct too (e.g., logistic regression’s cost function).

## Regularized Neural Networks

After I successfully implemeted the backpropagation algorithm, I add regularization to the gradient. To account for regularization, it turns out that you can add this as an additional term after computing the gradients using backpropagation.

Note that you should not be regularizing the first column of Θ^(l) which is used for the bias term. Furthermore, in the parameters (Θ^(l)_i,j) , i is indexed starting from 1, and j is indexed starting from 0.

Somewhat confusingly, indexing in Octave starts from 1 (for both i and j), thus Theta1(2, 1) actually corresponds to (Θ^(l)_ 2,0)(i.e., the entry in the second row, first column of the matrix Θ^(1)).

Now, I modify my code that computes grad in [nnCostFunction](nnCostFunction.m) to account for regularization. After, the [ex4.m](ex4.m) script will proceed to run gradient checking on your implementation. If your code is correct, you should
expect to see a relative difference that is less than 1e-9.

## Learning parameters using fmincg

After I have successfully implemented the neural network cost function and gradient computation, the next step of the [ex4.m](ex4.m) script will use [fmincg](fmincg.m) to learn a good set parameters.

After the training, the [ex4.m](ex4.m) script will proceed to report the training accuracy of my classifier by computing the percentage of examples it got correct. If your implementation is correct, you should see a reported training accuracy of about 95.3% (this may vary by about 1% due to the random initialization). It is possible to get higher training accuracies by training the neural network for more iterations. I encourage you to try training the neural network for more iterations (e.g., set MaxIter to 500) and also vary the regularization parameter λ. With the right learning settings, it is possible to get the neural network to perfectly fit the training set.

I also plot graph between  Number of iterations(X-axis) and cost function(y-axis) to see optimazation [fmincg](fmincg.m) function working correctly.

Figure looks like...

![](https://github.com/raianilar17/Neural-Networks-2/blob/master/Figure_02.jpg)

## Visualizing the hidden layer

One way to understand what your neural network is learning is to visualize what the representations captured by the hidden units. Informally, given a particular hidden unit, one way to visualize what it computes is to find an input x that will cause it to activate (that is, to have an activation value (a^(l)_i) close to 1). For the neural network you trained, notice that the i th row of Θ^(1) is a 401-dimensional vector that represents the parameter for the i th hidden unit. If we discard the bias term, we get a 400 dimensional vector that represents the weights from each input pixel to the hidden unit.

Thus, one way to visualize the “representation” captured by the hidden unit is to reshape this 400 dimensional vector into a 20 × 20 image and display it(It turns out that this is equivalent to finding the input that gives the highest activation
for the hidden unit, given a “norm” constraint on the input (i.e., ||x||_2 ≤ 1)). The next step of [ex4.m](ex4.m) does this by using the [displayData](displayData.m) function and it will show you an image (similar to Figure 3) with 25 units,each corresponding to one hidden unit in the network.

Figure 3: Visualization of Hidden Units.

![](https://github.com/raianilar17/Neural-Networks-2/blob/master/Figure_03.jpg)

Figure 4: Visualization of Ouput Units

![](https://github.com/raianilar17/Neural-Networks-2/blob/master/Figure_04.jpg)

In your trained network, you find that the hidden units corresponds roughly to detectors that look for strokes and other patterns in the input.

## Prediction

After training the neural network, I call [predict function](predict.m) using the loaded set of parameters for Theta1 and Theta2. you see that the accuracy is about 95.16%(with lambda = 1). After that, an interactive sequence will launch displaying images from the training set one at a time, while the console prints out the predicted label for the displayed image. To stop the image sequence, press Ctrl-C.

The Test_images looks like...

![](https://github.com/raianilar17/Neural-Networks-2/blob/master/Figure_05.jpg)

![](https://github.com/raianilar17/Neural-Networks-2/blob/master/Figure_06.jpg)

![](https://github.com/raianilar17/Neural-Networks-2/blob/master/Figure_07.jpg)

![](https://github.com/raianilar17/Neural-Networks-2/blob/master/Figure_08.jpg)

![](https://github.com/raianilar17/Neural-Networks-2/blob/master/Figure_09.jpg)

![](https://github.com/raianilar17/Neural-Networks-2/blob/master/Figure_10.jpg)

![](https://github.com/raianilar17/Neural-Networks-2/blob/master/Figure_11.jpg)


## Experiment

In this part of the section, I tried out different learning settings for the neural network to see how the performance of the neural network varies with the regularization parameter λ and number of training steps (the MaxIter option when using fmincg).

Neural networks are very powerful models that can form highly complex decision boundaries. Without regularization, it is possible for a neural network to “overfit” a training set so that it obtains close to 100% accuracy on the training set but does not as well on new examples that it has not seen before. You can set the regularization λ to a smaller value and the MaxIter parameter to a higher number of iterations to see this for youself.You will also be able to see for yourself the changes in the visualizations of the hidden units when you change the learning parameters λ and MaxIter.

After setting differnt lambda and MaxIter value, I seen differnt Training accuracy and cost with respect to lambda and MaxIter.

The results looks like...

| lambda | MaxIter | Cost | Training Accuracy |
| ---    | ---     |  --- |    ---            |
|   0 | 50 | 0.303|96.12%|
|1 |50 | 0.481| 95.16%|
|3| 50| 0.674|94.28%| 
|0|400|0.827|100%|
|1|400|0.321|99.48%|
|3|400|0.567|97.64%|

The ouput of [ex4.m](ex4.m) script

Type on octave_cli

ex4

[output looks like](output.txt)


# Future Work

Add mathematical equations (such as cost function, gradient descent...)

Explanation in more deeply...

# Work in Progress...
