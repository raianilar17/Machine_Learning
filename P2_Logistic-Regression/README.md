# Logistic-Regression

classification: Discrete-valued output

code is successfully excuted on Octave version 4.2.1

## Logistic Regression

In this Project, I implemented logistic regression and apply it to two different datasets.

To get started with the code, you will need to download code and unzip its contents to the directory where you wish to Run the code. If needed, use the cd command in Octave to change to this directory before starting this code. This code could also run on MATLAB(you can try). In future, I will try to excute this code on MATLAB also.

## Install Octave

The Project use Octave (Octave is a free alternative to MATLAB) a high-level programming language well-suited for numerical computations. If you do not have Octave installed, please install.

[Download](https://www.gnu.org/software/octave/download.html)

[Octave_for_GNU/Linux](http://wiki.octave.org/Octave_for_GNU/Linux)

[Octave_for_Microsoft_Windows](http://wiki.octave.org/Octave_for_Microsoft_Windows)

[Octave_for_macOS#Homebrew](http://wiki.octave.org/Octave_for_macOS#Homebrew)

Further documentation for Octave functions can be found at the [Octave documentation pages](http://www.gnu.org/software/octave/doc/interpreter/).

## Files included in this Project

ex2.m - Octave/MATLAB script that steps you through the Logistic Regression

ex2 reg.m - Octave/MATLAB script that steps you through the Regularized Logistic Regression

ex2data1.txt - Training set for the Logistic Regression

ex2data2.txt - Training set for the Regularized Logistic Regression

mapFeature.m - Function to generate polynomial features

plotDecisionBoundary.m - Function to plot classifier’s decision boundary

plotData.m - Function to plot 2D classification data

sigmoid.m - Sigmoid Function

costFunction.m - Logistic Regression Cost Function

predict.m - Logistic Regression Prediction Function

costFunctionReg.m - Regularized Logistic Regression Cost

Throughout the project, you will be using the scripts [ex2.m](ex2.m) and [ex2_reg.m](ex2_reg.m) These scripts set up the dataset for the problems and make calls to functions.

# Logistic-Regression

In this part of the section, you will build a logistic regression model to predict whether a student gets admitted into a university.

Suppose that you are the administrator of a university department and you want to determine each applicant’s chance of admission based on their results on two exams. You have historical data from previous applicants that you can use as a training set for logistic regression. For each training example, you have the applicant’s scores on two exams and the admissions decision.

Here my Goal is to build a classification model that estimates an applicant’s probability of admission based the scores from those two exams. This outline and the framework code in [ex2.m](ex2.m) will guide you through the section.

## Visualizing the data

Before starting to implement any learning algorithm, it is always good to visualize the data if possible. In the first part of [ex2.m](ex2.m), the code will load the data and display it on a 2-dimensional plot by calling the function plotData.

The code in [plotData](plotData.m) so that it displays a figure like Figure 1, where the axes are the two exam scores, and the positive and negative examples are shown with different markers.

Figure 1: Scatter plot of training data

![](https://github.com/raianilar17/Logistic-Regression/blob/master/ex2_Figure_1.jpg)

## Implementation

### sigmoid function

Before you start with the actual cost function, recall that the logistic regression hypothesis is defined as:

h_θ(x) = g((θ^T)*x),

Theta_parameter = θ

θ^T = Transpose of Theta_parameter 

x = vector or matrix

where function g is the sigmoid function. The sigmoid function is defined as:

g(z) = 1/(1 + exp^-z)

exp^-z = exponential of z

z = vector or matrix

First step is to implement this function in [sigmoid.m](sigmoid.m) so it can be called by the rest of your program. When you are finished, try testing a few values by calling sigmoid(x) at the Octave command line. For large positive values of x, the sigmoid should be close to 1, while for large negative values, the sigmoid should be close to 0. Evaluating sigmoid(0) should give you exactly 0.5. My code also work with vectors and matrices. For a matrix, my function  perform the sigmoid function on every element.

## Cost function and gradient

Now I implement the cost function and gradient for logistic regression.

The code in [costFunction.m](costFunction.m) to return the cost and gradient.

Note that while this gradient looks identical to the linear regression gradient, the formula is actually different because linear and logistic regression have different definitions of hθ(x).

The [ex2.m](ex2.m) will call your costFunction using the initial parameters of θ. You should see that the cost is about 0.693.

## Learning parameters using fminunc

In the previous Project, I got the optimal parameters of a linear regression model by implementing gradent descent. I wrote a cost function and calculated its gradient, then took a gradient descent step accordingly. This time, instead of taking gradient descent steps, I used an Octave/MATLAB built-in function called fminunc.

Octave/MATLAB’s fminunc is an optimization solver that finds the minimum of an unconstrained(Constraints in optimization often refer to constraints on the parameters, for example,constraints that bound the possible values θ can take (e.g., θ ≤ 1). Logistic regression does not have such constraints since θ is allowed to take any real value.) function. For logistic regression, you want to optimize the cost function J(θ) with parameters θ.

Concretely, you are going to use fminunc to find the best parameters θ for the logistic regression cost function, given a fixed dataset (of X and y values). You will pass to fminunc the following inputs:

1. The initial values of the parameters we are trying to optimize.

2. A function that, when given the training set and a particular θ, computes the logistic regression cost and gradient with    respect to θ for the dataset(X, y).

In this [code](ex2.m) snippet, we first defined the options to be used with fminunc. Specifically, we set the GradObj option to on, which tells fminunc that our function returns both the cost and the gradient. This allows fminunc to use the gradient when minimizing the function. Furthermore, we set the MaxIter option to 400, so that fminunc will run for at most 400 steps before it terminates.

To specify the actual function we are minimizing, we use a “short-hand” for specifying functions with the @(t) (costFunction(t, X, y) ) . This creates a function, with argument t, which calls your costFunction. This
allows us to wrap the costFunction for use with fminunc.

If you have completed the costFunction correctly, fminunc will converge on the right optimization parameters and return the final values of the cost and θ. Notice that by using fminunc, you did not have to write any loops yourself, or set a learning rate like we did in gradient descent. This is all done by fminunc: you only needed to provide a function calculating the cost and the gradient.

Once fminunc completes, [ex2.m](ex2.m) will call your costFunction function using the optimal parameters of θ. You should see that the cost is about 0.203.

This final θ value will then be used to plot the decision boundary on the training data, resulting in a figure similar to Figure 2. The code in [plotDecisionBoundary.m](plotDecisionBoundary.m)  plot such a boundary using the θ values.

Figure 2: Training data with decision boundary

![](https://github.com/raianilar17/Logistic-Regression/blob/master/ex2_Figure_2.jpg)

## Evaluating logistic regression

After learning the parameters,I use the model to predict whether a particular student will be admitted. For a student with an Exam 1 score of 45 and an Exam 2 score of 85, you should expect to see an admission probability of 0.776.

Another way to evaluate the quality of the parameters I found is to see how well the learned model predicts on our training set. In this part, I Wrote the code in [predict.m](predict.m). The predict function will produce “1” or “0” predictions given a dataset and a learned parameter vector θ.

Now, [ex2.m](ex2.m) script will proceed to report the training accuracy of your classifier by computing the percentage of examples it got correct.

Type on octave_cli

ex2
 
The output of [ex2.m script](ex2.m) looks like [ex2_output](ex2_output.txt).


# Regularised-Logistic-Regression

In this part of the segment, I implemented regularized logistic regression to predict whether microchips from a fabrication plant passes quality assurance (QA). During QA, each microchip goes through various tests to ensure it is functioning correctly.

Suppose you are the product manager of the factory and you have the test results for some microchips on two different tests. From these two tests, you would like to determine whether the microchips should be accepted or rejected. To help you make the decision, you have a dataset of test results on past microchips, from which you can build a logistic regression model.

You will use another script, [ex2_reg.m](ex2_reg.m) to complete this portion of the exercise.

## Visualizing the data

Similar to the previous parts of this section, [plotData](plotData.m) is used to generate a figure like Figure 3, where the axes are the two test scores, and the positive (y = 1, accepted) and negative (y = 0, rejected) examples are shown with different markers.

Figure 3: Plot of training data

![](https://github.com/raianilar17/Logistic-Regression/blob/master/ex2_reg_Figure_1.jpg)

Figure 3 shows that our dataset cannot be separated into positive and negative examples by a straight-line through the plot. Therefore, a straight forward application of logistic regression will not perform well on this dataset since logistic regression will only be able to find a linear decision boundary.

## Feature mapping

One way to fit the data better is to create more features from each data point. In the function [mapFeature](mapFeature.m), I map the features into all polynomial terms of x_1 and x_2 up to the sixth power.


As a result of this mapping, our vector of two features (the scores on two QA tests) has been transformed into a 28-dimensional vector. A logistic regression classifier trained on this higher-dimension feature vector will have a more complex decision boundary and will appear nonlinear when drawn in our 2-dimensional plot.

While the feature mapping allows us to build a more expressive classifier, it also more susceptible to overfitting. In the next parts of the project([Neural-Networks-1](https://github.com/raianilar17/Neural-Networks-1)) , I implemented regularized logistic regression to fit the data and also see for how regularization can help combat the overfitting problem.

## Cost function and gradient

Now I implemented code to compute the cost function and gradient for regularized logistic regression.

The code in [costFunctionReg.m](costFunctionReg.m) to return the cost and gradient.

Note that you should not regularize the parameter Theta_zero(θ_0) . In Octave,The indexing starts from 1, hence, you should not be regularizing the theta(1) parameter (which corresponds to θ_0 ) in the code. 

Now, [ex2_reg.m](ex2_reg.m) will call your costFunctionReg function using the initial value of θ (initialized to all zeros). You should see that the cost is about 0.693.

## Learning parameters using fminunc

Similar to the previous parts, I use fminunc to learn the optimal parameters θ. If you have completed the cost and gradient for regularized logistic regression (costFunctionReg.m) correctly, you should be able to step through the next part of 
[ex2_reg.m](ex2_reg.m) to learn the parameters θ using fminunc.

## Plotting the decision boundary

To visualize the model learned by this classifier, The function [plotDecisionBoundary.m](plotDecisionBoundary.m) which plots the (non-linear) decision boundary that separates the positive and negative examples. In [plotDecisionBoundary.m](plotDecisionBoundary.m), I plot the non-linear decision boundary by computing the classifier’s predictions on an evenly spaced grid and then and drew a contour plot of where the predictions change from y = 0 to y = 1. 

After learning the parameters θ, the next step in [ex_reg.m](ex_reg.m) will plot a decision boundary similar to Figure 4.

Figure 4: Training data with decision boundary (λ = 1)

![](https://github.com/raianilar17/Logistic-Regression/blob/master/ex2_reg_Figure_2.jpg)

In this part of the section, I get to try out different regularization parameters for the dataset to understand how regularization prevents overfitting.

Notice the changes in the decision boundary as you vary λ. With a small λ, you should find that the classifier gets almost every training example correct, but draws a very complicated boundary, thus overfitting the data(Figure 5). This is not a good decision boundary: for example, it predicts that a point at x = (−0.25, 1.5) is accepted (y = 1), which seems to be an incorrect decision given the training set.

Figure 5: No regularization (Overfitting) (λ = 0)

![](https://github.com/raianilar17/Logistic-Regression/blob/master/ex2_reg_Figure_4.jpg)

With a larger λ, you should see a plot that shows an simpler decision boundary which still separates the positives and negatives fairly well. However, if λ is set to too high a value, you will not get a good fit and the decision boundary will not follow the data so well, thus underfitting the data(Figure 6).

Figure 6: Too much regularization (Underfitting) (λ = 100)

![](https://github.com/raianilar17/Logistic-Regression/blob/master/ex2_reg_Figure_3.jpg)

Figure 6:Very long regularization (Underfitting) (λ = 150)

![](https://github.com/raianilar17/Logistic-Regression/blob/master/ex2_reg_Figure_5.jpg)

Type on octave_cli

ex2_reg
 
The output of [ex2_reg.m script](ex2_reg.m) looks like [ex2_reg_output](ex2_reg_output.txt).

# Work in progress...
