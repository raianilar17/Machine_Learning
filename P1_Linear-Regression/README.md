# Linear-Regression

Predict continous value (real-valued)...

code is successfully excuted on Octave version 4.2.1

# Linear Regression

In this Project, I implemented linear regression and get to see it work on data.

To get started with the code, you will need to download code and unzip its contents to the directory where you wish to Run the code. If needed, use the cd command in Octave to change to this directory before starting this code. This code could also run on MATLAB(you can try). In future, I will try to excute this code on MATLAB also.

## Install Octave

The Project use Octave (Octave is a free alternative to MATLAB) a high-level programming language well-suited for numerical computations. If you do not have Octave installed, please install.

[Download](https://www.gnu.org/software/octave/download.html)

[Octave_for_GNU/Linux](http://wiki.octave.org/Octave_for_GNU/Linux)

[Octave_for_Microsoft_Windows](http://wiki.octave.org/Octave_for_Microsoft_Windows)

[Octave_for_macOS#Homebrew](http://wiki.octave.org/Octave_for_macOS#Homebrew)

Further documentation for Octave functions can be found at the [Octave documentation pages](http://www.gnu.org/software/octave/doc/interpreter/).

## Files included in this project

ex1.m - Octave/MATLAB script that steps you through the Univariate_Linear_Regression

ex1 multi.m - Octave/MATLAB script for the later parts of the Multivariate_Linear_Regression

ex1data1.txt - Dataset for linear regression with one variable(Univariate_Linear_Regression)

ex1data2.txt - Dataset for linear regression with multiple variables(Multivariate_Linear_Regression)


warmUpExercise.m - Simple example function in Octave/MATLAB

plotData.m - Function to display the dataset

computeCost.m - Function to compute the cost of linear regression

gradientDescent.m - Function to run gradient descent

computeCostMulti.m - Cost function for multiple variables

gradientDescentMulti.m - Gradient descent for multiple variables

featureNormalize.m - Function to normalize features

normalEqn.m - Function to compute the normal equations

Throughout the Project, you will be using the scripts [ex1.m](ex1.m) and [ex1_multi.m](ex1_multi.m). These scripts set up the dataset for the problems and make calls to functions.

## Simple Octave function

The first part of [ex1.m](ex1.m ) gives you practice with Octave syntax. The function [warmUpExercise.m](warmUpExercise.m), return a 5 x 5 identity matrix by filling in the following code:

A = eye(5);

# Univariate_Linear_Regression(Linear regression with one variable)

In this part of this section, I implemented linear regression with one variable to predict profits for a food truck. Suppose you are the CEO of a restaurant franchise and are considering different cities for opening a new outlet. The chain already has trucks in various cities and you have data for profits and populations from the cities. You would like to use this data to help you select which city to expand to next.

The file [ex1data1.txt](ex1data1.txt) contains the dataset for our linear regression problem. The first column is the population of a city and the second column is the profit of a food truck in that city. A negative value for profit indicates a loss.

The [ex1.m](ex1.m) script has already been set up to load this data for you.

### Plotting the Data

Before starting on any task, it is often useful to understand the data by visualizing it. For this dataset, you can use a scatter plot to visualize the data, since it has only two properties to plot (profit and population). (Many other problems that you will encounter in real life are multi-dimensional and can’t be plotted on a 2-d plot.)

In [ex1.m](ex1.m), the dataset is loaded from the data file into the variables X and y 

Next, the script calls the [plotData](plotData.m) function to create a scatter plot of the data.

Now, when you continue to run [ex1.m](ex1.m), our end result should look like [Figure 1](datasets_vis.png), with the  red “x” markers and axis labels.

Figure 1: Scatter plot of training data looks like:

![](https://github.com/raianilar17/Linear-Regression/blob/master/datsets_vis.png)

### Gradient Descent
In this part, you will fit the linear regression parameters θ to our dataset using gradient descent.

#### Update Equations:

The objective of linear regression is to minimize the cost function

J(θ) = (1/2*m)*(sum_{i=1}^{m}((h(θ)X^i) - (y^i))^2);

where:

θ = theta(Hypothesis parameter)

m = Total training examples

^ = power

h(θ) = hpothesis

sum_{i=1}^{m} = Summation of (i = 1 to m)

X(Upper_case) = matrix of features

x(lower_case) = paticular feature in matrix X(upper_case), it's vector 

where the hypothesis hθ(x) is given by the linear model

hθ(x) = (θ^T)*X = θ(0) + θ(1)*(x(1))

(θ^T) = Transpose of theta

θ(0) = Theta_zero parameter
 
θ(1) = Theta_one parameter
  
x(1) = feature_1
  
But I used different hypothesis hθ(x) for the linear model:
  
hθ(x) = X*(θ) = θ(0) + θ(1)*(x(1))
  
Because Dimesion size of feature x after add extra feature(x_zero = 1) is m *(n+1) and Dimesion size of theta(θ) is (n+1) *1  so hypothesis hθ(x) dimension size become m *1
  
where n = number of fearture in training example
 
The parameters of model are the θ(j) values. These are the values need to adjust to minimize cost J(θ). One way to do this is to use the batch gradient descent algorithm. In batch gradient descent, each iteration performs the update.

θ(j) = θ(j) - (alpha * (1/m)*(sum_{i=1}^{m}((h(θ)X^i) - (y^i))^2 * X^i(j)))

(simultaneously update θ j for all j)

where j = 1...n

alpha = learning rate

Implementation Note: We store each example as a row in the the X matrix. To take into account the intercept term Theta_zero(θ(0)), we add an additional first column to X and set it to all ones. This allows us to treat Theta_zero(θ(0)) as simply another ‘feature’.

### Implementation

In [ex1.m](ex1.m), I have already set up the data for linear regression. In the following lines, I add another dimension to our data to accommodate the Theta_zero(θ(0)) intercept term. I also initialize the initial parameters to Theta(0) and the
learning rate alpha to 0.01.

### Computing the cost J(θ)

As you perform gradient descent to learn minimize the cost function J(θ),it is helpful to monitor the convergence by computing the cost. In this section, I implemented a function to calculate J(θ) so I can check the convergence of our gradient descent implementation.

The file [computeCost.m](computeCost.m), which is a function that computes J(θ). For doing this, remember that the
variables X and y are not scalar values, but matrices whose rows represent the examples from the training set.

You should expect to see a cost of 32.07

### Gradient descent

Next, I implemented gradient descent in the file [gradientDescent.m](gradientDescent.m) through vectorization method.

As you run program, make sure you understand what you are trying to optimize and what is being updated. Keep in mind that the cost J(θ) is parameterized by the vector θ, not X and y. That is, we minimize the value of J(θ) by changing the values of the vector θ, not by changing X or y.

A good way to verify that gradient descent is working correctly is to look at the value of J(θ) and check that it is decreasing with each step. The starter code for [gradientDescent.m](gradientDescent.m) calls computeCost on every iteration
and prints the cost. 

The ouput looks like:

![](https://github.com/raianilar17/Linear-Regression/blob/master/cost_function.png)

After you are finished, [ex1.m](ex1.m) will use your final parameters to plot the linear fit. The result should look something like [Figure 2](prediction.png):

[Figure 2](prediction.png): Training data with linear regression fit looks like:

![](https://github.com/raianilar17/Linear-Regression/blob/master/prediction.png)

The final values for Theta(θ) will also be used to make predictions on profits in areas of 35,000 and 70,000 people. Note the way that the following lines in [ex1.m](ex1.m) uses matrix multiplication, rather than explicit summation or looping, to calculate the predictions. This is an example of code vectorization in Octave.

## Debugging

Here are some things to keep in mind as you implement gradient descent:

1. Octave array indices start from one, not zero. If you’re storing theta_zero(θ(0)) and theta_one(θ(1)) in a vector called    theta, the values will be theta(1) and theta(2).

2. If you are seeing many errors at runtime, inspect your matrix operations to make sure that you’re adding and multiplying    matrices of compatible dimensions. Printing the dimensions of variables with the size command will help you debug.

3. By default, Octave interprets math operators to be matrix operators. This is a common source of size incompatibility        errors. If you don’t want matrix multiplication, you need to add the “dot” notation to specify this to Octave/MATLAB. For    example, A*B does a matrix multiply, while A.*B does an element-wise multiplication.

## Visualizing J(θ)

To understand the cost function J(θ) better, you will now plot the cost over a 2-dimensional grid of θ(0) and θ(1) values.

The script ex1.m will then produce surface and contour plots of J(θ) using the surf and contour commands. The plots should look something like Figure 3:

Figure 3: Cost function J(θ)

Figure 3.a: Cost function J(θ) using surf commands

![](https://github.com/raianilar17/Linear-Regression/blob/master/surf.png)

Figure 3.b: Cost function J(θ) using contour commands

![](https://github.com/raianilar17/Linear-Regression/blob/master/contour.png)

The purpose of these graphs is to show you that how J(θ) varies with changes in θ(0) and θ(1) . The cost function J(θ) is bowl-shaped and has a global mininum. (This is easier to see in the contour plot than in the 3D surface plot). This minimum is the optimal point for θ(0) and θ(1) , and each step of gradient descent moves closer to this point.

The output of ex1 looks like:

Type on octave_cli:

ex1

![](https://github.com/raianilar17/Linear-Regression/blob/master/output_ex1.png)


# Multivariate_Linear_Regression(Linear regression with multiple variables)

In this section, I implemented linear regression with multiple variables to predict the prices of houses. Suppose you are selling your house and you want to know what a good market price would be. One way to do this is to first collect information on recent houses sold and make a model of housing prices.

The file [ex1data2.txt](ex1data2.txt) contains a training set of housing prices in Portland, Oregon. The first column is the size of the house (in square feet), the second column is the number of bedrooms, and the third column is the price of the house.

The [ex1_multi.m](ex1_multi.m) script has been used in this problem.

## Feature Normalization

The [ex1_multi.m](ex1_multi.m) script will start by loading and displaying some values from this dataset. By looking at the values, note that house sizes are about 1000 times the number of bedrooms. When features differ by orders of magnitude, first performing feature scaling can make gradient descent converge much more quickly.

[Feature Normalization](featureNormalize.m) Algorithm :

1. Subtract the mean value of each feature from the dataset.

2. After subtracting the mean, additionally scale (divide) the feature values by their respective “standard deviations.”

The standard deviation is a way of measuring how much variation there is in the range of values of a particular feature (most data points will lie within ±2 standard deviations of the mean); this is an alternative to taking the range of values (max-min). In Octave, you can use the “std” function to compute the standard deviation. For example, inside [featureNormalize.m](featureNormalize.m), the quantity X(:,1) contains all the values of x_1 (house sizes) in the training
set, so std(X(:,1)) computes the standard deviation of the house sizes. At the time that featureNormalize.m is called, the extra column of 1’s corresponding to x_0 = 1 has not yet been added to X (see [ex1_multi.m](ex1_multi.m) for details).

### Implementation Note: 

When normalizing the features, it is important to store the values used for normalization - the mean value and the standard deviation used for the computations. After learning the parameters from the model, we often want to predict the prices of houses we have not seen before. Given a new x value (living room area and number of bedrooms), we must first normalize x using the mean and standard deviation that we had previously computed from the training set.

## Gradient Descent

Previously, I implemented gradient descent on a univariate regression problem. The only difference now is that there is one more feature in the matrix X. The hypothesis function and the batch gradient descent update rule remain unchanged.

The code in [computeCostMulti.m](computeCostMulti.m) and [gradientDescentMulti.m](gradientDescentMulti.m)
to implement the cost function and gradient descent for linear regression with multiple variables.

My code supports any number of features and is well-vectorized. You can use ‘size(X, 2)’ to find out how many features are present in the dataset.

The vectorized version is efficient when you’re working with numerical computing tools like Octave/MATLAB.

## Selecting learning rates

In this part of the section, I tried out different learning rates for the dataset and find a learning rate that converges quickly. You can change the learning rate by modifying [ex1_multi.m](ex1_multi.m) and changing the part of the code that sets the learning rate.

The next phase in [ex1_multi.m](ex1_multi.m) will call your [gradientDescent.m](gradientDescent.m) function and run gradient descent for about 400 iterations at the chosen learning rate(0.01). The function should also return the history of J(θ) values in a vectorJ. After the last iteration, the [ex1_multi.m](ex1_multi.m) script plots the J values against the number of the iterations.

If you picked a learning rate within a good range, your plot look similar
Figure 4.

Figure 4: Convergence of gradient descent with an appropriate learning rate looks like:

![](https://github.com/raianilar17/Linear-Regression/blob/master/cost_function.png)

If your graph looks very different, especially if your value of J(θ) increases or even blows up, adjust your learning rate and try again. We recommend trying values of the learning rate α on a log-scale, at multiplicative steps of about 3 times the previous value (i.e., 0.3, 0.1, 0.03, 0.01 and so on).You may also want to adjust the number of iterations you are running if that will help you see the overall trend in the curve.

### Implementation Note:

If your learning rate is too large, J(θ) can diverge and ‘blow up’, resulting in values which are too large for computer
calculations. In these situations, Octave/MATLAB will tend to return NaNs. NaN stands for ‘not a number’ and is often caused by undefined operations that involve −∞ and +∞.

### Octave/MATLAB Tip:

To compare how different learning rates affect convergence, it’s helpful to plot J for several learning rates
on the same figure. In Octave/MATLAB, this can be done by performing gradient descent multiple times with a ‘hold on’ command between plots. Concretely, if you’ve tried three different values of alpha (you should probably try more values than this) and stored the costs in J1, J2 and J3, you can use the following commands to plot them on the same figure:

plot(1:400, J1(1:400), ‘b’);

hold on;

plot(1:400, J2(1:400), ‘r’);

plot(1:400, J3(1:400), ‘k’);

The final arguments ‘b’, ‘r’, and ‘k’ specify different colors for the plots.

Notice the changes in the convergence curves as the learning rate changes. With a small learning rate, you should find that gradient descent takes a very long time to converge to the optimal value. Conversely, with a large learning rate, gradient descent might not converge or might even diverge!

Using the best learning rate that you found, run the [ex1_multi.m](ex1_multi.m) script to run gradient descent until convergence to find the final values of θ. Next, use this value of θ to predict the price of a house with 1650 square feet and 3 bedrooms. You will use value later to check your implementation of the normal equations. Don’t forget to normalize your features when you make this prediction!

## Normal Equations

θ = (pinv((X^T)*X))*((X^T)*y)

Where:

pinv = pseudo inverse function

X = feature matrix

X^T = transpose of matrix X

y = Target/label vector

Using this formula does not require any feature scaling, and you will get an exact solution in one calculation: there is no “loop until convergence” like in gradient descent.

The code in [normalEqn.m](normalEqn.m) to use the formula above to calculate θ. Remember that while you don’t need to scale your features, we still need to add a column of 1’s to the X matrix to have an intercept term (θ(0)).The code in ex1.m will add the column of 1’s to X for you.

Now, once you have found θ using this method, use it to make a price prediction for a 1650-square-foot house with
3 bedrooms. You should find that gives the same predicted price as the value you obtained using the model fit with gradient descent.

The output of [ex1_multi.m](ex1_multi.m) looks like :

Type on octave_cli 

ex1_multi

![](https://github.com/raianilar17/Linear-Regression/blob/master/output_ex1_multi.png)

## Future work

Apply this algorithm on more real world Problem.

Describe more about the algorithm stength.

# Work in progress...
