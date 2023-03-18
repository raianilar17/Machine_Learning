# K-means Clustering and Principal Component Analysis

Implement the K-means clustering algorithm and apply it to compress an image. Use principal component analysis to find a low-dimensional representation of face images.


# Introduction

In this project,In the first section,I implement the K-means clustering algorithm and apply it to compress an image. 
In the second section, I use principal component analysis to find a low-dimensional representation of face images.

This code is successfully implemented on octave version 4.2.1

To get started with the project, you will need to download the code and unzip its contents to the directory where you wish to run the project. If needed, use the cd command in Octave to change to this directory before starting this exercise.

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

[ex7.m](ex7.m) - Octave/MATLAB script for the first section on K-means

[ex7_pca.m](ex7_pca.m) - Octave/MATLAB script for the second section on PCA

[ex7data1.mat](ex7data1.mat) - Example Dataset for PCA

[ex7data2.mat](ex7data2.mat) - Example Dataset for K-means

[ex7faces.mat](ex7faces.mat) - Faces Dataset

[bird_small.png](bird_small.png) - Example Image

[displayData.m](displayData.m) - Displays 2D data stored in a matrix

[drawLine.m](drawLine.m) - Draws a line over an exsiting figure

[plotDataPoints.m](plotDataPoints.m) - Initialization for K-means centroids

[plotProgresskMeans.m](plotProgresskMeans.m) - Plots each step of K-means as it proceeds

[runkMeans.m](runkMeans.m) - Runs the K-means algorithm

[pca.m](pca.m) - Perform principal component analysis

[projectData.m](projectData.m) - Projects a data set into a lower dimensional space

[recoverData.m](recoverData.m) - Recovers the original data from the projection

[findClosestCentroids.m](findClosestCentroids.m) - Find closest centroids (used in K-means)

[computeCentroids.m](computeCentroids.m) - Compute centroid means (used in K-means)

[kMeansInitCentroids.m](kMeansInitCentroids.m) - Initialization for K-means centroids


Throughout the first segment of the project, I use the script [ex7.m](ex7.m), for the second segment I use [ex7_pca.m](ex7_pca.m). These scripts set up the dataset for the problems and make calls to functions.

# K-means Clustering

In this segment, I implement the K-means algorithm and use it for image compression. I first start on an example 2D dataset that will help me gain an intuition of how the K-means algorithm works. After that, I use the K-means algorithm for image compression by reducing the number of colors that occur in an image to only those that are most common in that image. I use script [ex7.m](ex7.m) for this part of the segment.

## Implementing K-means

The K-means algorithm is a method to automatically cluster similar data examples together. Concretely,Given a training set {x^(1)(first training examples) , ..., x^(m)(last training examples)} (where x^(i) ∈ R^n ), and want to group the data into a few cohesive “clusters”. The intuition behind K-means is an iterative procedure that starts by guessing the initial centroids, and then refines this guess by repeatedly assigning examples to their closest centroids and then recomputing the centroids based on the assignments.

### The K-means Algorithm is as follows:

                % Initialize centroids

                centroids = [kMeansInitCentroids(X, K)](kMeansInitCentroids.m);

                for iter = 1:iterations

                    % Cluster assignment step: Assign each data point to the

                    % closest centroid. idx(i) corresponds to c^(i), the index

                    % of the centroid assigned to example i

                    idx = [findClosestCentroids(X, centroids)](findClosestCentroids.m);

                    % Move centroid step: Compute means based on centroid

                    % assignments

                    centroids = [computeMeans(X, idx, K)](computeCentroids.m);

                end


The inner-loop of the algorithm repeatedly carries out two steps: (i) Assigning each training example x^(i) to its closest centroid, and (ii) Recomputing the mean of each centroid using the points assigned to it. The K-means algorithm will always converge to some final set of means for the centroids. Note that the converged solution may not always be ideal and depends on the initial setting of the centroids. Therefore, In practice the K-means algorithm is usually run a few times with different random initializations. One way to choose between these different solutions from different [random initializations](kMeansInitCentroids.m) is to choose the one with the lowest cost function value (distortion).


I implement the two phases of the K-means algorithm separately in this sections.

## Finding Closest Centroids

In the “cluster assignment” phase of the K-means algorithm, The algorithm assigns every training example x^(i) to its closest centroid, given the current positions of centroids. Specifically, for every example i we set

        c^(i) := j that minimizes ||x^(i) − μ_j ||^2
        
where c^(i) is the index of the centroid that is closest to x^(i) , and μ_j is the position (value) of the j’th centroid. Note that c^(i) corresponds to idx(i) in the [code](findClosestCentroids.m).        

I wrote the code in [findClosestCentroids.m](findClosestCentroids.m).This function takes the data matrix X and the locations of all centroids inside centroids and output a one-dimensional array idx that holds the index (a value in {1, ..., K}, where K is total number of centroids) of the closest centroid to every training example.

After completion of code in [findClosestCentroids.m](findClosestCentroids.m),The script [ex7.m](ex7.m) will run code and  see the output [1 3 2] corresponding to the centroid assignments for the first 3 examples.

## Computing Centroid Means

Given assignments of every point to a centroid, the second phase of the algorithm recomputes, for each centroid, the mean of the points that were assigned to it. Specifically, for every centroid k we set 

          μ_k = (1/|C_k|)* Σ{i=c_k} X^(i) 
          
where C_k is the set of examples that are assigned to centroid k. Concretely,if two examples say x^(3) and x^(5) are assigned to centroid k = 2, then update μ_2 = (1/2)*(x^(3) + x^(5)). 

For this, I wrote the code in [computeCentroids.m](computeCentroids.m).I Implement this function using a loop over the centroids. We can also use a loop over the examples; but if we can use a vectorized implementation that does not use such a loop, our code may run faster.

After, The completion the code in [computeCentroids.m](computeCentroids.m), The script [ex7.m](ex7.m) will run code and output the centroids after the first step of K-means.

## K-means on example dataset


![](https://github.com/raianilar17/K-means-Clustering-and-Principal-Component-Analysis/blob/master/ex7_Figure4.png)

![](https://github.com/raianilar17/K-means-Clustering-and-Principal-Component-Analysis/blob/master/ex7_Figure1.png)


Figure 1: The expected output.


After completion the two functions ([findClosestCentroids](findClosestCentroids.m) and [computeCentroids](computeCentroids)), The next step in [ex7.m](ex7.m) will run the K-means algorithm on a toy 2D dataset to help understand how K-means works. My functions are called from inside the [runKmeans.m](runKmeans.m) script. 

When I run the next step, The [K-means code](runKmeans.m) will produce a visualization that steps through the progress of the algorithm at each iteration. Press enter multiple times to see how each step of the K-means algorithm changes the centroids and cluster assignments. At the end, Last iteration should look as the one displayed in Figure 1.


## Random Initialization

In practice, a good strategy for initializing the centroids is to select random examples from the training set.

In this part of the segment,I wrote the function [kMeansInitCentroids.m](kMeansInitCentroids.m) with the following code:

             % Initialize the centroids to be random examples
             % Randomly reorder the indices of examples
               randidx = randperm(size(X, 1));
             % Take the first K examples as centroids
               centroids = X(randidx(1:K), :);


The code above first randomly permutes the indices of the examples (using randperm). Then, it selects the first K examples based on the random permutation of the indices. This allows the examples to be selected at random without the risk of selecting the same example twice.

## Image Compression with K-means

  ![](https://github.com/raianilar17/K-means-Clustering-and-Principal-Component-Analysis/blob/master/bird_small.png)

Figure 2: The original 128x128 image.

In this segment, I apply K-means to image compression. In a straightforward 24-bit color representation of an image, each pixel is represented as three 8-bit unsigned integers (ranging from 0 to 255) that specify the red, green and blue intensity values. This encoding is often refered to as the RGB encoding. This image contains thousands of colors, and in this part of the segment, I reduce the number of colors to 16 colors.


By making this reduction, It is possible to represent (compress) the photo in an efficient way. Specifically, I only need to store the RGB values of the 16 selected colors, and for each pixel in the image I now need to only store the index of the color at that location (where only 4 bits are necessary to represent 16 possibilities).

In this part, I use the K-means algorithm to select the 16 colors that will be used to represent the compressed image. Concretely, I treat every pixel in the original image as a data example and use the K-means algorithm to find the 16 colors that best group (cluster) the pixels in the 3-dimensional RGB space. Once I computed the cluster centroids on the image, I then use the 16 colors to replace the pixels in the original image.

## K-means on pixels

#### Instructions to  Installing Packages(OCTAVE)

To install a package from the Octave Forge, at the Octave prompt type [pkg install -forge package_name](https://octave.org/doc/v4.2.1/Installing-and-Removing-Packages.html)

### Image Processing package(OCTAVE)

The Octave-forge Image package provides functions for processing images. The package also provides functions for feature extraction, image statistics, spatial and geometric transformations, morphological operations, linear filtering, and much more.

Type on Octave prompt

     pkg install image-2.6.1.tar.gz
     
     OR

     pkg install image-2.10.0.tar.gz

[For more Information](https://octave.sourceforge.io/image/index.html)


In Octave/MATLAB, images can be read in as follows:

     % Load 128x128 color image (bird small.png)
       A = imread('bird small.png');
    % You will need to have installed the image package to used
    % imread.If you don't have installed the image package follow the above 
    % Instructions to  Installing Packages(OCTAVE) OR    
    % you should instead change the following line to
    %
    %
      load('bird small.mat'); % Loads the image into the variable A

This creates a three-dimensional matrix A whose first two indices identify a pixel position and whose last index represents red, green, or blue. For example, A(50, 33, 3) gives the blue intensity of the pixel at row 50 and column 33.

The code inside [ex7.m](ex7.m) first loads the image, and then reshapes it to create an m × 3 matrix of pixel colors (where m = 16384 = 128 × 128), and calls K-means function on it.

After finding the top K = 16 colors to represent the image, Now assign each pixel position to its closest centroid using the [findClosestCentroids function](findClosestCentroids.m). This allows to represent the original image using the centroid assignments of each pixel. Notice that I have significantly reduced the number of bits that are required to describe the image. The original image required 24 bits for each one of the 128×128 pixel locations, resulting in total size of 128 × 128 × 24 = 393, 216 bits. The new representation requires some overhead storage in form of a dictionary of 16 colors, each of which require 24 bits, but the image itself then only requires 4 bits per pixel location. The final number of bits used is therefore 16 × 24 + 128 × 128 × 4 = 65, 920 bits,which corresponds to compressing the original image by about a factor of 6.


Figure 3: Original and reconstructed image (when using K-means to compress the image).

![](https://github.com/raianilar17/K-means-Clustering-and-Principal-Component-Analysis/blob/master/ex7_Figure2.png)

Finally,I view the effects of the compression by reconstructing the image based only on the centroid assignments. Specifically, I replace each pixel location with the mean of the centroid assigned to it. Figure 3 shows the reconstruction I obtained. Even though the resulting image retains most of the characteristics of the original, I also see some compression artifacts.

## Use Different image

In this part, Modify the code I have supplied to run on one of my own images. Note that if Image is very large, then K-means can take a long time to run. Therefore, I recommend that you resize your images to managable sizes before running the code. I also try to vary K to see the effects on the compression.

The Results Shown Below:

![](https://github.com/raianilar17/K-means-Clustering-and-Principal-Component-Analysis/blob/master/ex7_Figure3.png)

![](https://github.com/raianilar17/K-means-Clustering-and-Principal-Component-Analysis/blob/master/ex7_Figure3a.png)

![](https://github.com/raianilar17/K-means-Clustering-and-Principal-Component-Analysis/blob/master/ex7_Figure3b.png)


The output of ex7.m script

type on Octave_cli

      >> ex7

[The Results Looks Like](ex7_output.txt)

# Principal Component Analysis

In this segment,I use principal component analysis (PCA) to perform dimensionality reduction. I first experiment with an example 2D dataset to get intuition on how PCA works, and then use it on a bigger dataset of 5000 face image dataset.

The script, [ex7_pca.m](ex7_pca.m), will help step through the first half of the second segment.

## Example Dataset

To help understand how PCA works, I first start with a 2D dataset which has one direction of large variation and one of smaller variation. The script [ex7_pca.m](ex7_pca.m) will plot the training data (Figure 4). In this part of the second segment, I visualize what happens when I use PCA to reduce the data from 2D to 1D. In practice,We might want to reduce data from 256 to 50 dimensions, say; but using lower dimensional data in this example allows us to visualize the algorithms better.


Figure 4: Example Dataset 1

![](https://github.com/raianilar17/K-means-Clustering-and-Principal-Component-Analysis/blob/master/ex7_pca_Figure1.png)

## Implementing PCA

In this part of the segment, I implement PCA. PCA consists of two computational steps: First, I compute the covariance matrix of the data. Then, I use Octave SVD function to compute the eigenvectors U_1 , U_2 , . . . , U_n . These will correspond to the principal components of variation in the data.

Before using PCA, it is important to first normalize the data by subtracting the mean value of each feature from the dataset, and scaling each dimension so that they are in the same range. In the script [ex7_pca.m](ex7_pca.m),This normalization has been performed for you using the [featureNormalize function](featureNormalize.m).

After normalizing the data, I run PCA to compute the principal components. I wrote the code in pca.m to compute the prin-
cipal components of the dataset. First,I compute the covariance matrix of the data, which is given by:

      Σ = (1/m)*(X'*X)

where X is the data matrix with examples in rows, and m is the number of examples. Note that Σ is a n × n matrix and not the summation operator.

After computing the covariance matrix, I run SVD on it to compute the principal components. In Octave, you can run SVD with the following command: [U, S, V] = svd(Sigma), where U will contain the principal components and S will contain a diagonal matrix.

Figure 5: Computed eigenvectors of the dataset

![](https://github.com/raianilar17/K-means-Clustering-and-Principal-Component-Analysis/blob/master/ex7_pca_Figure2.png)


After completion [pca.m](pca.m), The [ex7_pca.m](ex7_pca.m) script will run PCA on the example dataset and plot the corresponding principal components found (Figure 5). The script will also output the top principal component (eigenvector) found, and I see an output of about [-0.707 -0.707]. (It is possible that Octave/MATLAB may instead output the negative of this, since U_1 and −U_1 are equally valid choices for the first principal component.)


## Dimensionality Reduction with PCA

After computing the principal components, I use them to reduce the feature dimension of your dataset by projecting each example onto a lower dimensional space, x^(i) → z^(i) (e.g., projecting the data from 2D to 1D). In this part of the segment, I use the eigenvectors returned by PCA and project the example dataset into a 1-dimensional space.

In practice, if you were using a learning algorithm such as linear regression or perhaps neural networks, you could now use the projected data instead of the original data. By using the projected data, you can train your model faster as there are less dimensions in the input.

## Projecting the Data onto the Principal Components

I wrote the code in [projectData.m](projectData.m). Specifically,Given a dataset X, the principal components U, and the desired number of dimensions to reduce to K. I project each example in X onto the top K components in U. Note that the top K components in U are given by the first K columns of U, that is U_reduce = U(:, 1:K).

After completion of the code in [projectData.m](projectData.m), [ex7_pca.m](ex7_pca.m) will project the first example onto the first dimension and see a value of about 1.481 (or possibly -1.481, if you got −U_1 instead of U_1).

## Reconstructing an Approximation of the Data

After projecting the data onto the lower dimensional space, I approximately recover the data by projecting them back onto the original high dimensional space. I wrote the code in [recoverData.m](recoverData.m) to project each example in Z back onto the original space and return the recovered approximation in X_rec.

After, completion the code in [recoverData.m](recoverData.m), [ex7_pca.m](ex7_pca.m) will recover an approximation of the first example and see a value of about [-1.047 -1.047].

## Visualizing the Projections

Figure 6: The normalized and projected data after PCA

![](https://github.com/raianilar17/K-means-Clustering-and-Principal-Component-Analysis/blob/master/ex7_pca_Figure3.png)

After completing both [projectData](projectData.m) and [recoverData](recoverData.m), [ex7_pca.m](ex7_pca.m) will
now perform both the projection and approximate reconstruction to show how the projection affects the data. In Figure 6, the original data points are indicated with the blue circles, while the projected data points are indicated with the red circles. The projection effectively only retains the information in the direction given by U_1 .

## Face Image Dataset

In this part of the segment, I run PCA on face images to see how it can be used in practice for dimension reduction. The dataset [ex7faces.mat](ex7faces.mat) contains a dataset(This dataset was based on a [cropped version](http://itee.uq.edu.au/~conrad/lfwcrop/) of the [labeled faces in the wild ](http://vis-www.cs.umass.edu/lfw/) dataset.) X of face images, each 32 × 32 in grayscale. Each row of X corresponds to one face image (a row vector of length 1024). The next step in [ex7_pca.m](ex7_pca.m) will load and visualize the first 100 of these face images (Figure 7).

Figure 7: Faces dataset

![](https://github.com/raianilar17/K-means-Clustering-and-Principal-Component-Analysis/blob/master/ex7_pca_Figure4.png)


## PCA on Faces

To run PCA on the face dataset, I first normalize the dataset by subtracting the mean of each feature from the data matrix X. The script [ex7_pca.m](ex7_pca.m) will do this and then run your PCA code. After running PCA, I obtain the principal components of the dataset. Notice that each principal component in U (each row) is a vector of length n (where for the face dataset, n = 1024). It turns out that we can visualize these principal components by reshaping each of them into a 32 × 32 matrix that corresponds to the pixels in the original dataset. The script [ex7_pca.m](ex7_pca.m) displays the first 36 principal components that describe the largest variations (Figure 8). If we want, we can also change the code to display more principal components to see how they capture more and more details.

Figure 8: Principal components on the face dataset

![](https://github.com/raianilar17/K-means-Clustering-and-Principal-Component-Analysis/blob/master/ex7_pca_Figure5.png)


## Dimensionality Reduction

Now that I have computed the principal components for the face dataset,I can use it to reduce the dimension of the face dataset. This allows me to use my learning algorithm with a smaller input size (e.g., 100 dimensions) instead of the original 1024 dimensions. This can help speed up my learning algorithm.


Figure 9: Original images of faces and ones reconstructed from only the top 100 principal components.

![](https://github.com/raianilar17/K-means-Clustering-and-Principal-Component-Analysis/blob/master/ex7_pca_Figure6.png)

![](https://github.com/raianilar17/K-means-Clustering-and-Principal-Component-Analysis/blob/master/ex7_pca_Figure7.png)


The next part in [ex7_pca.m](ex7_pca.m) will project the face dataset onto only the first 100 principal components. Concretely, each face image is now described by a vector z^(i) ∈ R^100.

To understand what is lost in the dimension reduction, I recover the data using only the projected dataset. In [ex7_pca.m](ex7_pca.m), an approximate recovery of the data is performed and the original and projected face images are displayed  (Figure 9). From the reconstruction, I observe that the general structure and appearance of the face are kept while the fine details are lost. This is a remarkable reduction (more than 10×) in the dataset size that can help speed up my learning algorithm significantly. For example, if you were training a neural network to perform person recognition (gven a face image, predict the identitfy of the person), you can use the dimension reduced input of only a 100 dimensions instead of the original pixels.


## PCA for Visualization

Figure 10: Original data in 3D

![](https://github.com/raianilar17/K-means-Clustering-and-Principal-Component-Analysis/blob/master/ex7_pca_Figure8.png)

In the earlier K-means image compression first segment, I used the K-means algorithm in the 3-dimensional RGB space. In the last part of the [ex7_pca.m](ex7_pca.m) script, I wrote code to visualize the final pixel assignments in this 3D space using the scatter3 function. Each data point is colored according to the cluster it has been assigned to. I drag my mouse on the figure to rotate and inspect this data in 3 dimensions.

It turns out that visualizing datasets in 3 dimensions or greater can be cumbersome. Therefore, it is often desirable to only display the data in 2D even at the cost of losing some information.

     In practice, PCA is often used to reduce the dimensionality of data for visualization purposes. 

In the next part of [ex7_pca.m](ex7_pca.m), the script will apply my implementation of PCA to the 3-dimensional data to reduce it to 2 dimensions and visualize the result in a 2D scatter plot. The PCA projection can be thought of as a rotation that selects the view that maximizes the spread of the data, which often corresponds to the “best” view.

Figure 11: 2D visualization produced using PCA

![](https://github.com/raianilar17/K-means-Clustering-and-Principal-Component-Analysis/blob/master/ex7_pca_Figure9.png)



The output of ex7_pca.m script

Type on Octave_cli

      >> ex7_pca

[The Results Looks Like](ex7_pca_output.txt)


# Future work

more description(technology terminology)

Apply this algorithm on different datasets

apply this algorithm on different field

try to upcome up with better approach

# Work In Progress
