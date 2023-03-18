# Neural-Networks-1
Implement one-vs-all Logistic Regression and Neural Networks to Recognize Hand-Written Digits


# Multi-class Classification and Neural Networks

In this Project, I implemented one-vs-all logistic regression and neural networks to recognize hand-written digits.

To get started with the Project, you will need to download the code and unzip its contents to the directory where you wish to run the project. If needed, use the cd command in Octave/MATLAB to change to this directory before starting this projects.

# installing Octave
code sucessfully excuted on octave version 4.2.1

install octave

[Download](https://www.gnu.org/software/octave/download.html)

[Octave_for_GNU/Linux](http://wiki.octave.org/Octave_for_GNU/Linux)

[Octave_for_Microsoft_Windows](http://wiki.octave.org/Octave_for_Microsoft_Windows)

[Octave_for_macOS#Homebrew](http://wiki.octave.org/Octave_for_macOS#Homebrew)


## Files included in this project

[ex3.m](ex3.m) - Octave/MATLAB script that steps you through Multi-class Classification

[ex3_nn.m](ex3_nn.m) - Octave/MATLAB script that steps you through Neural Networks

[ex3data1.mat](ex3data1.mat) - Training set of hand-written digits

[ex3weights.mat](ex3weights.mat) - Initial weights for the neural network

[displayData.m](displayData.m) -  visualize the dataset

[fmincg.m](fmincg.m) - Function minimization routine (similar to fminunc)

[sigmoid.m](sigmoid.m) - Sigmoid function

[lrCostFunction.m](lrCostFunction.m) - Logistic regression cost function

[oneVsAll.m](oneVsAll.m) - Train a one-vs-all multi-class classifier

[predictOneVsAll.m](predictOneVsAll.m) - Predict using a one-vs-all multi-class classifier

[predict.m](predict.m) - Neural network prediction function

Throughout the Project, you will be using the scripts [ex3.m](ex3.m) and [ex3_nn.m](ex3_nn.m). These scripts set up the dataset for the problems and make calls to functions.

## Multi-class Classification

Used logistic regression to recognize handwritten digits (from 0 to 9). 
 
Automated handwritten digit recognition is widely used today - from recognizing zip codes (postal codes) on mail envelopes to recognizing amounts written on bank checks. 

### Datasets

Data set in ex3data1.mat that contains 5000 training examples of handwritten digits.This is a subset of the [MNIST handwritten digit dataset ](http://yann.lecun.com/exdb/mnist/). The .mat format means that the data has been saved in a native Octave/MATLAB matrix format, instead of a text (ASCII) format like a csv-file. These matrices can be read directly into your program by using the load command. After loading, matrices of the correct dimensions and values will appear in your program’s memory. The matrix will already be named, so you do not need to assign names to them.

There are 5000 training examples in [ex3data1.mat](ex3data1.mat), where each training example is a 20 pixel by 20 pixel grayscale image of the digit. Each pixel is represented by a floating point number indicating the grayscale intensity at that location. The 20 by 20 grid of pixels is “unrolled” into a 400-dimensional vector. Each of these training examples becomes a single row in our data matrix X. This gives us a 5000 by 400 matrix X where every row is a training example for a handwritten digit image.


The second part of the training set is a 5000-dimensional vector y that contains labels for the training set. To make things more compatible with Octave/MATLAB indexing, where there is no zero index, we have mapped the digit zero to the value ten. Therefore, a “0” digit is labeled as “10”, while the digits “1” to “9” are labeled as “1” to “9” in their natural order.

### Visualizing the data

Visualizing a subset of the training set. 

The [ex3.m](ex3.m),the code randomly selects 100 rows from X and passes those rows to the [displayData function](displayData.m). This function maps each row to a 20 pixel by 20 pixel grayscale image and displays the images together.
After you run this step, you should see an image like 
[Figure1](output_input_samples.png)

![](https://github.com/raianilar17/Neural-Networks-1/blob/master/output_input_samples.png)

#### Vectorizing Logistic Regression

Used multiple one-vs-all logistic regression models to build a multi-class classifier. Since there are 10 classes, I need to train 10 separate logistic regression classifiers. To make this training efficient, it is important to ensure that code is well vectorized. In this section, I implementad a vectorized version of logistic regression that does not employ any for loops.

##### Follow the following steps to vectorization:

1. Vectorizing the cost function

2. Vectorizing the gradient

3. Vectorizing regularized logistic regression

#### One-vs-all Classification

In this part of the section, I implemented one-vs-all classification by training multiple regularized logistic regression classifiers, one for each of the K classes in our dataset (Figure 1). In the handwritten digits dataset,K = 10, but my code  work for any value of K.

The code should return all the classifier parameters in a matrix Θ ∈ R(K×(N +1)) , where each row of Θ corresponds to the learned logistic regression parameters for one class. You can do this with a “for”loop from 1 to K, training each classifier independently.

Note that the y argument to this function is a vector of labels from 1 to 10, where we have mapped the digit “0” to the label 10 (to avoid confusions with indexing).

When training the classifier for class k ∈ {1, ..., K}, you will want a m-dimensional vector of labels y, where y (j) ∈ {0, 1} indicates whether the j-th training instance belongs to class k (y (j) = 1), or if it belongs to a different
class (y (j) = 0). 

Furthermore, I used fmincg for this exercise (instead of fminunc). fmincg works similarly to fminunc, but it is more efficient for dealing with a large number of parameters.

#### One-vs-all Prediction

After training your one-vs-all classifier, you can now use it to predict the digit contained in a given image. For each input, you should compute the “probability” that it belongs to each class using the trained logistic regression
classifiers. Your one-vs-all prediction function will pick the class for which the corresponding logistic regression classifier outputs the highest probability and return the class label (1, 2,..., or K) as the prediction for the input example.

You should see that the training set accuracy is about 94.9% (i.e., it classifies 94.9% of the examples in the training set correctly).

The output of ex3.m shows:

Type on octave_cli 

 ex3

![](https://github.com/raianilar17/Neural-Networks-1/blob/master/output_ex3.png)

## Neural Networks

In the previous part of this section, I implemented multi-class logistic regression to recognize handwritten digits. However, logistic regression cannot form more complex hypotheses as it is only a linear classifier.You could add more features (such as polynomial features) to logistic regression, but that can be very expensive to train.

In this part of the section, I implemented a neural network to recognize handwritten digits using the same training set as before. The neural network will be able to represent complex models that form non-linear hypotheses. For this,I used parameters from a neural network that researchers have already trained. Here, goal is to implement the feedforward propagation algorithm to use our weights for prediction. In next Project,I will write the backpropagation algorithm for learning the neural network parameters.

The provided script, [ex3_nn.m](ex3_nn.m), will help you step through this section.

### Model representation

It has 3 layers – an input layer, a hidden layer and an output layer. Recall that our inputs are pixel values of digit images. Since the images are of size 20×20, this gives us 400 input layer units (excluding the extra bias unit which always outputs +1). As before, the training data will be loaded into the variables X and y.

You have been provided with a set of network parameters (Θ (1) , Θ (2) ) already trained by researchers. These are stored in [ex3weights.mat](ex3weights.mat) and will be loaded by [ex3_nn.m](ex3_nn.m) into Theta1 and Theta2 The parameters have dimensions that are sized for a neural network with 25 units in the second layer and 10 output units (corresponding to the 10 digit classes).

### Feedforward Propagation and Prediction

Now I implemented feedforward propagation for the neural network. The code in [predict.m](predict.m) to return the neural network’s prediction.

Now, I implement the feedforward computation that computes hθ(x(i)) for every example i and returns the associated predictions. Similar to the one-vs-all classification strategy, the prediction from the neural network will
be the label that has the largest output (hθ(x))k .

Once I have done, [ex3_nn.m](ex3_nn.m) will call your predict function using the loaded set of parameters for Theta1 and Theta2. you see that the accuracy is about 97.5%. After that, an interactive sequence will launch displaying images from the training set one at a time, while the console prints out the predicted label for the displayed image. To stop the image sequence,
press Ctrl-C.

the Test_images are like:
![](https://github.com/raianilar17/Neural-Networks-1/blob/master/output_tes_images2.png)
![](https://github.com/raianilar17/Neural-Networks-1/blob/master/output_test_images.png)


The output ex3_nn.m shows like this:

Type on octave_cli 

 ex3
 
![](https://github.com/raianilar17/Neural-Networks-1/blob/master/output_ex3_nn.png)


# FUTURE WORK

Apply on Different feild...



# Work In Progress...



