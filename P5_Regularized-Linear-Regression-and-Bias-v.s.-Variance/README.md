# Regularized Linear Regression and Bias v.s. Variance

Implement regularized linear regression and use it to study models with different bias-variance properties

This code is successfully implemented on octave version 4.2.1

To get started with the project, you will need to download the code and unzip its contents to the directory where you wish to run the project. If needed, use the cd command in Octave to change to this directory before starting this exercise.

This code could also run on MATLAB(you can try). In future, I will try to excute this code on MATLAB also.

# Environment Setup Instructions

## Instructions for installing Octave 

The Project use Octave (Octave is a free alternative to MATLAB) a high-level programming language well-suited for numerical computations. If you do not have Octave installed, please install.

[Download](https://www.gnu.org/software/octave/download.html)

[Octave_for_GNU/Linux](http://wiki.octave.org/Octave_for_GNU/Linux)

[Octave_for_Microsoft_Windows](http://wiki.octave.org/Octave_for_Microsoft_Windows)

[Octave_for_macOS#Homebrew](http://wiki.octave.org/Octave_for_macOS#Homebrew)

Documentation for Octave functions can be found at the [Octave documentation pages](http://www.gnu.org/software/octave/doc/interpreter/).

# Files included in the project

[ex5.m](ex5.m) - Octave script that steps you through the project

[ex5data1.mat](ex5data1.mat) - Dataset

[featureNormalize.m](featureNormalize.m) - Feature normalization function

[fmincg.m](fmincg.m) - Function minimization routine (similar to fminunc)

[plotFit.m](plotFit.m) - Plot a polynomial fit

[trainLinearReg.m](trainLinearReg.m) - Trains linear regression using your cost function

[linearRegCostFunction.m](linearRegCostFunction.m ) - Regularized linear regression cost function

[learningCurve.m](learningCurve.m) - Generates a learning curve

[polyFeatures.m](polyFeatures.m) - Maps data into polynomial feature space

[validationCurve.m](validationCurve.m) - Generates a cross validation curve


Throughout the project, you will be using the script [ex5.m](ex5.m). These scripts set up the dataset for the problems and make calls to functions.

# Regularized Linear Regression

In the first half of the project, I implement regularized linear regression to predict the amount of water flowing out of a dam using the change of water level in a reservoir. In the next half, I gone through some diagnostics of debugging learning algorithms and examine the effects of bias v.s. variance.

## Visualizing the dataset

I begin by visualizing the dataset containing historical records on the change in the water level, x, and the amount of water flowing out of the dam, y.

This dataset is divided into three parts:

1. A training set that your model will learn on: X, y

2. A cross validation set for determining the regularization parameter: Xval, yval

3. A test set for evaluating performance. These are “unseen” examples which your model did not see during training: Xtest,      ytest

The next step of [ex5.m](ex5.m) will plot the training data [Figure 1](https://github.com/raianilar17/Regularized-Linear-Regression-and-Bias-v.s.-Variance/blob/master/Figure%201.png). In the following parts, I implement linear regression and use that to fit a straight line to the data and plot learning curves. Following that, I implement polynomial regression to find a better fit to the data.

Figure 1: Data

![](https://github.com/raianilar17/Regularized-Linear-Regression-and-Bias-v.s.-Variance/blob/master/Figure%201.png)

## Regularized linear regression cost function

Where regularization parameter which controls the degree of regularization (thus, help preventing overfitting). The regularization term puts a penalty on the overal cost. As the magnitudes of the model parameters increase, the penalty increases as well. 

The code in the file [linearRegCostFunction.m](linearRegCostFunction.m) return cost and gradient. I vectorize my code and avoid writing loops. The next part of [ex5.m](ex5.m) will run cost function using theta initialized at [1; 1]. You should expect to see an output of 303.993.

## Regularized linear regression gradient

In [linearRegCostFunction.m](linearRegCostFunction.m), code added to calculate the gradient, returning it in the variable grad.The next part of [ex5.m](ex5.m) will run your gradient function using theta initialized at [1; 1]. You should expect to see a gradient of [-15.30; 598.250].

## Fitting linear regression

The next part of [ex5.m](ex5.m) will run the code in [trainLinearReg.m](trainLinearReg.m) to compute the optimal values
of θ. This training function uses [fmincg](fmincg.m) to optimize the cost function.I also plot the graph to check optimization working correctly(cost error reduce in every iterations).

![](https://github.com/raianilar17/Regularized-Linear-Regression-and-Bias-v.s.-Variance/blob/master/Figure%203.png)

In this part, I set regularization parameter lambda to zero. Because my current implementation of linear regression is trying to fit a 2-dimensional θ, regularization will not be incredibly helpful for a θ of such low dimension. In
the later parts of the project, I will be using polynomial regression with regularization.

Finally, the [ex5.m](ex5.m) script should also plot the best fit line, resulting in an image similar to [Figure 2](https://github.com/raianilar17/Regularized-Linear-Regression-and-Bias-v.s.-Variance/blob/master/Figure%202.png). The best fit line tells us that the model is not a good fit to the data because the data has a non-linear pattern. While visualizing the best fit as shown is one possible way to debug your learning algorithm, it is not always easy to visualize the data and model. In the next section, I will implement a function to generate learning curves that can help in debug of learning algorithm even if it is not easy to visualize the data.

Figure 2: Linear Fit

![](https://github.com/raianilar17/Regularized-Linear-Regression-and-Bias-v.s.-Variance/blob/master/Figure%202.png)

# Bias-variance

An important concept in machine learning is the bias-variance tradeoff. Models with high bias are not complex enough for the data and tend to underfit, while models with high variance overfit to the training data.

In this part of the project,I will plot training and test errors on a learning curve to diagnose bias-variance problems.

## Learning curves

Now, I implement code to generate the learning curves that will be useful in debugging learning algorithms. Learning curve plots training and cross validation error as a function of training set size. The code [learningCurve.m](learningCurve.m)  returns a vector of errors for the training set and cross validation set.

To plot the learning curve, we need a training and cross validation set error for different training set sizes. To obtain different training set sizes, I use different subsets of the original training set(X).

I use the [trainLinearReg](trainLinearReg.m) function to find the theta(θ) parameters. Note that the lambda is passed as a parameter to the [learningCurve](learningCurve.m) function. After learning the Theta(θ) parameters, I compute the error on the training and cross validation sets.

In particular, note that the training error does not include the regularization term. One way to compute the training error is to use your existing cost function and set regularization parameter lambda(λ) to 0 only when using it to compute the training error and cross validation error. When you are computing the training set error, make sure you compute it on the training subset (i.e., X(1:n,:) and y(1:n)) (instead of the entire training set). However, for the cross validation error,
you should compute it over the entire cross validation set. You should store the computed errors in the vectors error train and error val.

Now, [ex5.m](ex5.m) wil print the learning curves and produce a plot similar to [Figure 3](https://github.com/raianilar17/Regularized-Linear-Regression-and-Bias-v.s.-Variance/blob/master/Figure%204.png).

Figure 3: Linear regression learning curve

![](https://github.com/raianilar17/Regularized-Linear-Regression-and-Bias-v.s.-Variance/blob/master/Figure%204.png)


In Figure 3, you can observe that both the train error and cross validation error are high when the number of training examples is increased. This reflects a high bias problem in the model – the linear regression model is too simple and is unable to fit our dataset well. In the next section,I will implement polynomial regression to fit a better model for this dataset.


# Polynomial regression

The problem with my linear model was that it was too simple for the data and resulted in underfitting (high bias). In this part of the project, I will address this problem by adding more features.

For use polynomial regression, My hypothesis has the form:

h_θ(x) = θ_0 + θ_1 ∗ (waterLevel) + θ_2 ∗ (waterLevel)^2 + · · · + θ_p ∗ (waterLevel)^p

       = θ_0 + θ_1 *x_1 + θ_2*x_2 + ... + θ_p*x_p .

where :

h_θ(x) = Hypothesis

θ_anynumber = theta parameter

x_anynumber(waterlevel) = feature

Notice that by defining x_1 = (waterLevel), x_2 = (waterLevel)^2 , . . . , x_p =(waterLevel)^p , we obtain a linear regression model where the features are the various powers of the original value (waterLevel).


Now, you will add more features using the higher powers of the existing feature x in the dataset. I wrote the code in
[polyFeatures.m](polyFeatures.m) so that the function maps the original training set X of size m × 1 into its higher powers. Specifically, when a training set X of size m × 1 is passed into the function, the function should return a m×p matrix X_poly, where column 1 holds the original values of X, column 2 holds the values of X.^2, column 3 holds the values of X.^3, and so on. Note that you don’t have to account for the zero-eth power in this function.

Now you have a function that will map features to a higher dimension,and next Part of [ex5.m](ex5.m) will apply it to the training set, the test set, and the cross validation set.

## Learning Polynomial Regression

Next, the ex5.m script will proceed to train polynomial regression using your linear regression cost function.

Keep in mind that even though we have polynomial terms in our feature vector, we are still solving a linear regression optimization problem. The polynomial terms have simply turned into features that we can use for linear regression. I am using the same cost function and gradient that I wrote for the earlier part of this project.


For this part of the project, I use a polynomial of degree 8. It turns out that if I run the training directly on the projected data, will not work well as the features would be badly scaled (e.g., an example with x = 40 will now have a feature x^8 = 40^8 = 6.5 × 10^12 ). Therefore, you will need to use [feature normalization](featureNormalize.m).

Before learning the parameters Theta(θ) for the polynomial regression, [ex5.m](ex5.m) will first call [featureNormalize](featureNormalize.m) and normalize the features of the training set, storing the mu, sigma parameters separately.

After learning the parameters θ, you should see two plots (Figure 4,5) generated for polynomial regression with λ = 0.

Figure 4: Polynomial fit, λ = 0

![](https://github.com/raianilar17/Regularized-Linear-Regression-and-Bias-v.s.-Variance/blob/master/Figure%205.png)

From [Figure 4](https://github.com/raianilar17/Regularized-Linear-Regression-and-Bias-v.s.-Variance/blob/master/Figure%205.png), you should see that the polynomial fit is able to follow the datapoints very well - thus, obtaining a low training error. However, the polynomial fit is very complex and even drops off at the extremes. This is an indicator that the polynomial regression model is overfitting the training data and will not generalize well.

Figure 5: Polynomial learning curve, λ = 0

![](https://github.com/raianilar17/Regularized-Linear-Regression-and-Bias-v.s.-Variance/blob/master/Figure%206.png)

To better understand the problems with the unregularized (λ = 0) model, you can see that the learning curve [(Figure 5)](https://github.com/raianilar17/Regularized-Linear-Regression-and-Bias-v.s.-Variance/blob/master/Figure%206.png) shows the same effect where the low training error is low, but the cross validation error is high. There is a gap between the training and cross validation errors, indicating a high variance problem.

One way to combat the overfitting (high-variance) problem is to add regularization to the model. In the next section, I will get to try different λ parameters to see how regularization can lead to a better model.

## Adjusting the regularization parameter

In this section, you will get to observe how the regularization parameter affects the bias-variance of regularized polynomial regression. I modify the the lambda parameter in the [ex5.m](ex5.m) and try λ = 1, 100. For each of these values, the script should generate a polynomial fit to the data and also a learning curve.

For λ(lambda) = 1, you should see a polynomial fit that follows the data trend well (Figure 6) and a learning curve (Figure 7) showing that both the cross validation and training error converge to a relatively low value. This shows the λ = 1 regularized polynomial regression model does not have the high-bias or high-variance problems. In effect, it achieves a good trade-off between bias and variance.

Figure 6: Polynomial fit, λ = 1

![](https://github.com/raianilar17/Regularized-Linear-Regression-and-Bias-v.s.-Variance/blob/master/Figure%209.png)

Figure 7: Polynomial learning curve, λ = 1

![](https://github.com/raianilar17/Regularized-Linear-Regression-and-Bias-v.s.-Variance/blob/master/Figure%2010.png)


For λ = 100, you should see a polynomial fit (Figure 8) that does not follow the data well and a learning curve (Figure 9) showing that both the cross validation and training error have high value. In this case, there is too much regularization and the model is unable to fit the training data.

Figure 8: Polynomial fit, λ = 100

![](https://github.com/raianilar17/Regularized-Linear-Regression-and-Bias-v.s.-Variance/blob/master/Figure%2011.png)

Figure 9: Polynomial learning curve, λ = 100

![](https://github.com/raianilar17/Regularized-Linear-Regression-and-Bias-v.s.-Variance/blob/master/Figure%2012.png)

## Selecting lambda(λ) using a cross validation set

From the previous section of the project, you observed that the value of λ can significantly affect the results of regularized polynomial regression on the training and cross validation set. In particular, a model without regularization (λ = 0) fits the training set well, but does not generalize. Conversely,a model with too much regularization (λ = 100) does not fit the training set and testing set well. A good choice of λ (e.g., λ = 1) can provide a good fit to the data.


In this section, I implement an automated method to select the λ parameter. Concretely, I use a cross validation set to evaluate how good each λ value is. After selecting the best λ value using the cross validation set,Then evaluate the model on the test set to estimate how well the model will perform on actual unseen data.

I wrote the code in [validationCurve.m](validationCurve.m). Specifically, I used the [trainLinearReg](trainLinearReg) function to train the model using different values of λ and compute the training error and cross validation error.

I try λ in the following range: {0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10}.

Now, the next part of [ex5.m](ex5.m) will run your function can plot a cross validation curve of error v.s. λ that allows you select which λ parameter to use. You should see a plot similar to Figure 10. In this figure, we can see that the best value of λ is around 3. Due to randomness in the training and validation splits of the dataset, the cross validation error
can sometimes be lower than the training error.

Figure 10: Selecting λ using a cross validation set

![](https://github.com/raianilar17/Regularized-Linear-Regression-and-Bias-v.s.-Variance/blob/master/Figure%207.png)

## Computing test set error

In the previous part of the project, I implemented code to compute the cross validation error for various values of the regularization parameter λ. However, to get a better indication of the model’s performance in the real world, it is important to evaluate the “final” model on a test set that was not used in any part of training (that is, it was neither used to select the λ parameters, nor to learn the model parameters θ).

I compute the test error using the best value of λ. In my cross validation,I obtained a test error of 3.8599 for λ = 3.

## Plotting learning curves with randomly selected examples

In practice, especially for small training sets, when you plot learning curves to debug your algorithms, it is often helpful to average across multiple sets of randomly selected examples to determine the training error and cross validation error.

Concretely, to determine the training error and cross validation error for i examples, I first randomly select i examples from the training set and i examples from the cross validation set.I then learn the parameters θ using the randomly chosen training set and evaluate the parameters θ on the randomly chosen training set and cross validation set. The above steps should then be repeated multiple times (say 50) and the averaged error should be used to determine the training error and cross validation error for i examples.

For this section, you can implement the above strategy for computing the learning curves. For reference, figure 11 shows the learning curve I obtained for polynomial regression with λ = 3.00. Your figure may differ slightly due to the random selection of examples.

Figure 11: Learning curve with randomly selected examples

![](https://github.com/raianilar17/Regularized-Linear-Regression-and-Bias-v.s.-Variance/blob/master/Figure%208.png)


The ouput of [ex5.m](ex5.m) script

Type on octave_cli

[ex5](ex5.m)

[output looks like](output.txt)

# Future work
 Add mathematical equation 
 
 github readme doesn't support mathelatex , so I have to find different method.
 
 Improve the code
 
 Add the Research papers
 
 Add more description
 
 # Work in progress
 


