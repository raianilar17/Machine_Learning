# Anomaly Detection and Recommender Systems

In the first part , I implement the Anomaly Detection Algorithm and apply it to detect failing servers on a network. In the second part, I use Collaborative Filtering Algorithm to build a recommender system for movies.

# Introduction

In this project,Firstly I implement the Anomaly Detection Algorithm and apply it to detect failing servers on a network. In the second part, I use Collaborative Filtering Algorithm to build a recommender system for movies.

This code is successfully implemented on octave version 4.2.1

To get started with the project, you will need to download the code and unzip its contents to the directory where you wish to run the project. If needed, use the cd command in Octave to change to this directory before starting this project.

This code could also run on MATLAB(you can try). In future, I will try to excute this code on MATLAB also.

# Environment Setup Instructions

## Instructions For Installing Octave 

The Project use Octave (Octave is a free alternative to MATLAB) a high-level programming language well-suited for numerical computations. If you do not have Octave installed, please install.

[Download](https://www.gnu.org/software/octave/download.html)

[Octave_for_GNU/Linux](http://wiki.octave.org/Octave_for_GNU/Linux)

[Octave_for_Microsoft_Windows](http://wiki.octave.org/Octave_for_Microsoft_Windows)

[Octave_for_macOS#Homebrew](http://wiki.octave.org/Octave_for_macOS#Homebrew)

Documentation for Octave functions can be found at the [Octave documentation pages](http://www.gnu.org/software/octave/doc/interpreter/).

# Files Included in this Project

[ex8.m](ex8.m) - Octave/MATLAB script for Anamoly Detection Algorithm

[ex8_cofi.m](ex8_cofi.m) - Octave/MATLAB script for Collaborative Filtering Algorithm

[ex8data1.mat](ex8data1.mat) - First example Dataset for anomaly detection

[ex8data2.mat](ex8data2.mat) - Second example Dataset for anomaly detection

[ex8_movies.mat](ex8_movies.mat) - Movie Review Dataset

[ex8_movieParams.mat](ex8_movieParams.mat) - Parameters provided for debugging

[multivariateGaussian.m](multivariateGaussian.m) - Computes the probability density function for a Gaussian distribution

[visualizeFit.m](visualizeFit.m) - 2D plot of a Gaussian distribution and a dataset

[checkCostFunction.m](checkCostFunction.m) - Gradient checking for collaborative filtering

[computeNumericalGradient.m](computeNumericalGradient.m) - Numerically compute gradients

[fmincg.m](fmincg.m) - Function minimization routine (similar to fminunc)

[loadMovieList.m](loadMovieList.m) - Loads the list of movies into a cell-array

[movie_ids.txt](movie_ids.txt) - List of movies

[normalizeRatings.m](normalizeRatings.m) - Mean normalization for collaborative filtering

[estimateGaussian.m](estimateGaussian.m) - Estimate the parameters of a Gaussian distribution with a diagonal covariance matrix

[selectThreshold.m](selectThreshold.m) - Find a threshold for anomaly detection

[cofiCostFunc.m](cofiCostFunc.m) - Implement the cost function for collaborative filtering


Throughout the first part of the project (anomaly detection) I use the script [ex8.m](ex8.m). For the second part of collaborative filtering, I use [ex8_cofi.m](ex8_cofi.m). These scripts set up the dataset for the problems and
make calls to functions.


# Anomaly Detection


In this part, I implement an anomaly detection algorithm to detect anomalous behavior in server computers. The features measure the throughput (mb/s) and latency (ms) of response of each server. While servers were operating,I collected m = 307 examples of how they were behaving, and thus have an unlabeled dataset {x^(1) , . . . , x^(m) }. I suspect that the vast majority of these examples are “normal” (non-anomalous) examples of the servers operating normally, but there might also be some examples of servers acting anomalously within this dataset.

I use a Gaussian model to detect anomalous examples in my dataset. I first start on a 2D dataset that will allow me to visualize what the algorithm is doing. On that dataset I fit a Gaussian distribution and then find values that have very low probability and hence can be considered anomalies. After that, I apply the anomaly detection algorithm to a larger dataset with many dimensions. I use [ex8.m](ex8.m) for this part of the project.

The first part of [ex8.m](ex8.m) will visualize the dataset as shown in Figure 1.

Figure 1: The first dataset.

![](https://github.com/raianilar17/Anomaly-Detection-and-Recommender-Systems/blob/master/ex8_figure1.png)

## Gaussian Distribution

To perform anomaly detection, I first need to fit a model to the data’s distribution.

Given a training set {x^(1) , ..., x^(m) } (where x^(i) ∈ R^n), I estimate the Gaussian distribution for each of the features x_i . For each feature i = 1 . . . n, I find parameters μ_i and (σ_i)^2 that fit the data in the
i-th dimension {x^(1)_i , ..., x^(m)_i } (the i-th dimension of each example).

The Gaussian distribution is given by :

     p(x; μ, σ^2 ) = ((1/sqrt(2πσ^2)) * (exp(-((x-μ)^2/2σ^2))));
     
     where μ is the mean and σ^2 controls the variance.
    
## Estimating Parameters For a Gaussian  

I estimate the parameters, (μ_i , (σ_i)^2 ), of the i-th feature by using the following equations.

            To estimate the mean, I use:

             μ_i = (1/m)*(Σ{j=1}{m}(x^(j)_i));
             
             and for the variance I use:
             
             (σ_i)^2 = (1/m)* (Σ{j=1}{m}((x_i)^j - μ_i)^2);
             
             
I wrote the code in [estimateGaussian.m](estimateGaussian.m). This function takes as input the data matrix X and output an n-dimension vector mu that holds the mean of all the n features and another n-dimension vector sigma2 that holds the variances of all the features. I implement this using  a vectorized implementation because it is  more efficient than for-loop. Note that in Octave/MATLAB, the var function will (by default) use (1/m−1), instead of 1/m , when computing (σ_i)^2 .            
Once completion the code in [estimateGaussian.m](estimateGaussian.m), The next part of [ex8.m](ex8.m) will visualize the contours of the fitted Gaussian distribution.I get a plot similar to Figure 2. From my plot, I see that most of the examples are in the region with the highest probability, while the anomalous examples are in the regions with lower probabilities.     
     
Figure 2: The Gaussian distribution contours of the distribution fit to the dataset.
     
![](https://github.com/raianilar17/Anomaly-Detection-and-Recommender-Systems/blob/master/ex8_figure2.png)
     

## Selecting the threshold, ε

Now that I have estimated the Gaussian parameters, I can investigate which examples have a very high probability given this distribution and which examples have a very low probability. The low probability examples are more likely to be the anomalies in our dataset. One way to determine which examples are anomalies is to select a threshold based on a cross validation set. In this part of the segment, I implement an algorithm to select the threshold ε using the F_1 score on a cross validation set.

I wrote the code in [selectThreshold.m](selectThreshold.m). For this, I use a cross validation set {((x_cv)^(1) , (y_cv)^(2) ), . . . , ((x_cv)^m_cv , (y_cv)^m_cv )}, where the label y = 1 corresponds to an anomalous example, and y = 0 corresponds
to a normal example. For each cross validation example,I compute p(x_cv^(i) ). The vector of all of these probabilities p(x_cv^(1) ), . . . , p(x_cv^m_cv) ) is passed to [selectThreshold.m](selectThreshold.m) in the vector pval. The corresponding labels (y_cv^1 , . . . , y_cv^m_cv) is passed to the same function in the vector yval.


The function [selectThreshold.m](selectThreshold.m) return two values; the first is the selected threshold ε. If an example x has a low probability p(x) < ε, Then it is considered to be an anomaly. The function  also return the F_1 score, which tells  how well I'm doing on finding the ground truth anomalies given a certain threshold. For many different values of ε, I compute the resulting F_1 score by computing how many examples the current threshold classifies correctly and incorrectly.


    The F_1 score is computed using precision (prec) and recall (rec):
                  
                         F_1 =  (2 · prec · rec ) / (prec + rec);
                         
    I compute precision and recall by:
                        
                        prec = tp / (tp + fp);
                        
                        rec =  tp  / (tp + fn);
                        
    where :
    • tp is the number of true positives: the ground truth label says it’s an anomaly and our algorithm correctly classified it as an anomaly.
    • fp is the number of false positives: the ground truth label says it’s not an anomaly, but our algorithm incorrectly classified it as an anomaly.
    • fn is the number of false negatives: the ground truth label says it’s an anomaly, but our algorithm incorrectly classified it as not being anomalous.      
                        
                        
                  
In the code [selectThreshold.m](selectThreshold.m), I tried many different values of ε and select the best ε based on the 
F_1 score.

I wrote the code selectThreshold in [selectThreshold.m](selectThreshold.m). I implement the computation of the F_1 score using a for-loop over all the cross validation examples (to compute the values tp, fp, fn). The output of this function, I see a value for epsilon of about 8.99e-05.

## Implementation Note:

In order to compute tp, fp and fn, I use a vectorized implementation rather than loop over all the examples. This can be implemented by Octave/MATLAB’s equality test between a vector and a single number. If you have several binary values
in an n-dimensional binary vector v ∈ {0, 1}^n , you can find out how many values in this vector are 0 by using:
sum(v == 0). You can also apply a logical and operator to such binary vectors. For instance, let cvPredictions be a binary vector of the size of your number of cross validation set, where the i-th element is 1 if your algorithm considers x_cv^(i) an anomaly, and 0 otherwise. You can then, for example, compute the number of false positives using: 
fp = sum((cvPredictions == 1) &(yval == 0)).

After completion the code in [selectThreshold.m](selectThreshold.m), the next step in [ex8.m](ex8.m) will run my anomaly detection code and circle the anomalies in the plot (Figure 3).


Figure 3: The classified anomalies.

![](https://github.com/raianilar17/Anomaly-Detection-and-Recommender-Systems/blob/master/ex8_figure3.png)


## High Dimensional Dataset

The last part of the script [ex8.m](ex8.m) will run the anomaly detection algorithm. I implemented on a more realistic and much harder dataset. In this dataset, each example is described by 11 features, capturing many more properties of your compute servers.

The script will use my code to estimate the Gaussian parameters (μ_i and σ_i^2), evaluate the probabilities for both the training data X from which I estimated the Gaussian parameters, and do so for the the cross-validation set Xval. Finally, It will use selectThreshold to find the best threshold ε. The output,I see a value epsilon of about 1.38e-18, and 117 anomalies found.


The output of ex8.m octave script

Type on Octave_CLI

     >> ex8

[Output Looks like](ex8_output.txt)

# Recommender Systems

In this part of the segment, I implement the collaborative filtering learning algorithm and apply it to a dataset of movie ratings([MovieLens 100k Dataset ](http://www.grouplens.org/node/73/) from GroupLens Research). This dataset consists of ratings on a scale of 1 to 5. The dataset has n_u = 943 users, and n_m = 1682 movies. For this part of the segment, I work with the script [ex8_cofi.m](ex8_cofi.m).

In the next parts of this segment, I implement the function [cofiCostFunc.m](cofiCostFunc.m) that computes the collaborative filtering objective function and gradient. After implementing the cost function and gradient, I use [fmincg.m](fmincg.m) to learn the parameters for collaborative filtering.

## Movie Ratings Dataset

The first part of the script [ex8_cofi.m](ex8_cofi.m) will load the dataset [ex8_movies.mat](ex8_movies.mat), providing the variables Y and R in your Octave environment.

The matrix Y (a num movies × num users matrix) stores the ratings y^(i,j)(from 1 to 5). The matrix R is an binary-valued indicator matrix, where R(i, j) = 1 if user j gave a rating to movie i, and R(i, j) = 0 otherwise. 

    The objective of collaborative filtering is to predict movie ratings for the movies
    that users have not yet rated, that is, the entries with R(i, j) = 0. 
    This will allow us to recommend the movies with the highest predicted ratings to the user.

To help understand the matrix Y, the script [ex8_cofi.m](ex8_cofi.m) will compute the average movie rating for the first movie (Toy Story) and output the average rating to the screen.

 "visualize" the ratings matrix by plotting it with imagesc:

![](https://github.com/raianilar17/Anomaly-Detection-and-Recommender-Systems/blob/master/ex8_cofi_figure1.png)

Throughout this part of the segment, I also work with the matrices, X and Theta. The i-th row of X corresponds to the feature vector x^(i) for the i-th movie, and the j-th row of Theta corresponds to one parameter vector θ^(j) , for the
j-th user. Both x^(i) and θ^(j) are n-dimensional vectors. For the purposes of this section, I use n = 100, and therefore, x^(i) ∈ R^100 and θ^(j) ∈ R^100 . Correspondingly, X is a n_m × 100 matrix and Theta is a n_u × 100 matrix.


## Collaborative Filtering Learning Algorithm

Now, I implement the collaborative filtering learning algorithm. I start by implementing the cost function (without regularization).

The collaborative filtering algorithm in the setting of movie recommendations considers a set of n-dimensional parameter vectors x^(1) , ..., x^(n_m ) and θ^(1) , ..., θ^(n_u ) , where the model predicts the rating for movie i by user j as
y^(i,j) = (θ^(j))T x^(i) . Given a dataset that consists of a set of ratings produced by some users on some movies,I wish to learn the parameter vectors x^(1) , ..., x^(n_m) , θ^(1) , ..., θ^(n_u) that produce the best fit (minimizes the squared
error).


I wrote the code in [cofiCostFunc.m](cofiCostFunc.m) to compute the cost function and gradient for collaborative filtering. Note that the parameters to the function (i.e., the values that I am trying to learn) are X and Theta. In order to use an off-the-shelf minimizer such as [fmincg](fmincg.m), the cost function has been set up to unroll the parameters into a single vector params. 

## Collaborative Filtering Cost Function

![](https://github.com/raianilar17/Anomaly-Detection-and-Recommender-Systems/blob/master/fun_1.png)

I now modify [cofiCostFunc.m](cofiCostFunc.m) to return this cost in the variable J. Note that I accumulate the cost for user j and movie i only if R(i, j) = 1.

After completion the function,The script [ex8_cofi.m](ex8_cofi.m) will run cost function.I expect to see an output of 22.22.


## Implementation Note:

I use a vectorized implementation to compute J, since it will later by called many times by the optimization package [fmincg](fmincg.m). As usual, It might be easiest to first write a non-vectorized implementation (to make sure you have the right answer), and the modify it to become a vectorized implementation (checking that the vectorization steps don’t change your algorithm’s output). To come up with a vectorized implementation, the following tip might be helpful: You can use the R matrix to set selected entries to 0. For example, R .* M will do an element-wise multiplication between M and R; since R only has elements with values either 0 or 1, this has the effect of setting the elements of M to 0 only when the corresponding value in R is 0. Hence, sum(sum(R.*M)) is the sum of all the elements of M for which the corresponding element in R equals 1.

## Collaborative Filtering Gradient

Now, I implement the gradient (without regularization). Specifically, I wrote the code in [cofiCostFunc.m](cofiCostFunc.m) to return the variables X_grad and Theta_grad. Note that X_grad should be a matrix of the same size as X and similarly, Theta_grad is a matrix of the same size as Theta.

![](https://github.com/raianilar17/Anomaly-Detection-and-Recommender-Systems/blob/master/fun_2.png)

Note that the function returns the gradient for both sets of variables by unrolling them into a single vector. After completion the code to compute the gradients, The script [ex8_cofi.m](ex8_cofi.m) will run a gradient check
([checkCostFunction](checkCostFunction.m)) to numerically check the implementation of your gradients.  If implementation is correct, Then the analytical and numerical gradients match up closely.


## Implementation Note: 

If your code use Non-vectorized implementation, Then code will run much more slowly (a large number of hours), and so I recommend that you try to vectorize your implementation.

To get started, you can implement the gradient with a for-loop over movies (for computing ∂J/∂x_k^(i) ) and a for-loop over users (for computing ∂J/∂θ_k^(j) ). When you first implement the gradient, you might start with an unvectorized
version, by implementing another inner for-loop that computes each element in the summation. After you have completed the gradient computation this way, you should try to vectorize your implementation (vectorize the inner for-loops), so that you’re left with only two for-loops (one for looping over movies to compute ∂J/∂x_k^(i) for each movie, and one for looping
over users to compute ∂J/∂θ_k^(j) for each user).

To perform the vectorization, you might find this helpful: You should come up with a way to compute all the derivatives
associated with x_1^(i) , x_2^(i) , . . . , x_n^(i) (i.e., the derivative terms associated with the feature vector x^(i)) at the same time. Let us define the derivatives for the feature vector of the i-th movie as:

![](https://github.com/raianilar17/Anomaly-Detection-and-Recommender-Systems/blob/master/fun_3.png)

To vectorize the above expression, you can start by indexing into Theta and Y to select only the elements of interests (that is, those with r(i, j) = 1). Intuitively, when you consider the features for the i-th movie, you only need to be concern about the users who had given ratings to the movie, and this allows you to remove all the other users from Theta and Y.

Concretely, you can set idx = find(R(i, :)==1) to be a list of all the users that have rated movie i. This will allow you to create the temporary matrices Theta_temp = Theta(idx, :) and Y_temp = Y(i, idx) that index into Theta and Y to give you only the set of users which have rated the i-th movie. This will allow you to write the derivatives as:

![](https://github.com/raianilar17/Anomaly-Detection-and-Recommender-Systems/blob/master/fun_4.png)

After you have vectorized the computations of the derivatives with respect to x^(i) , you should use a similar method to vectorize the derivatives with respect to θ^(j) as well.

## Regularized Cost Function

The cost function for collaborative filtering with regularization is given by

![](https://github.com/raianilar17/Anomaly-Detection-and-Recommender-Systems/blob/master/fun_5.png)

Now , I add regularization to my original computations of the cost function, J. After done, the script [ex8_cofi.m](ex8_cofi.m) will run regularized cost function, and expected output to see a cost of about 31.34.


## Regularized Gradient

Now that I implemented the regularized cost function,Then I proceed to implement regularization for the gradient. I add to my implementation in [cofiCostFunc.m](cofiCostFunc.m) to return the regularized gradient by adding the contributions from the regularization terms. Note that the gradients for the regularized cost function is given by:

![](https://github.com/raianilar17/Anomaly-Detection-and-Recommender-Systems/blob/master/fun_6.png)

This means that I just need to add λx^(i) to the X_grad(i,:) variable described earlier, and add λθ^(j) to the Theta_grad(j,:) variable described earlier.


## Learning Movie Recommendations

After I finished implement the collaborative filtering cost function and gradient, Now I start training my algorithm to make movie recommendations for myself. In the next part of the [ex8_cofi.m](ex8_cofi.m) script, I enter my own movie preferences, so that later when the algorithm runs, I get my own movie recommendations! I have filled out some values according to my own preferences, but you should change this according to your own tastes. The list of all movies and their number in the
dataset can be found listed in the file [movie_idx.txt](movie_idx.txt).


## Recommendations

Figure 4: Movie recommendations

      Top recommendations for you:
      
      Predicting rating 5.0 for movie Someone Else's America (1995)
      Predicting rating 5.0 for movie Marlene Dietrich: Shadow and Light (1996)
      Predicting rating 5.0 for movie Prefontaine (1997)
      Predicting rating 5.0 for movie Santa with Muscles (1996)
      Predicting rating 5.0 for movie Aiqing wansui (1994)
      Predicting rating 5.0 for movie They Made Me a Criminal (1939)
      Predicting rating 5.0 for movie Entertaining Angels: The Dorothy Day Story (1996)
      Predicting rating 5.0 for movie Star Kid (1997)
      Predicting rating 5.0 for movie Saint of Fort Washington, The (1993)
      Predicting rating 5.0 for movie Great Day in Harlem, A (1994)


      Original ratings provided:
      
      Rated 4 for Toy Story (1995)
      Rated 3 for Twelve Monkeys (1995)
      Rated 5 for Usual Suspects, The (1995)
      Rated 4 for Outbreak (1995)
      Rated 5 for Shawshank Redemption, The (1994)
      Rated 3 for While You Were Sleeping (1995)
      Rated 5 for Forrest Gump (1994)


After the additional ratings have been added to the dataset, The script will proceed to train the collaborative filtering model. This will learn the parameters X and Theta. To predict the rating of movie i for user j,I need to compute 
(θ^(j)T x^(i)). The next part of the script computes the ratings for all the movies and users and displays the movies that it recommends (Figure 4), according to ratings that were entered earlier in the script. Note that It might obtain a different set of the predictions due to different random initializations.


The output of ex8_cofi.m octave script

Type on Octave_CLI

ex8_cofi

[Output Looks like](ex8_cofi_output.txt)

# Future Work

Apply this algorithm on different field 

Add more description about code



# WORK IN PROGRESS.....
