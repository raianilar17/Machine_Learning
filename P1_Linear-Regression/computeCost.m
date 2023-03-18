function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== MY CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


%Non_vectorization_approach
%Prediction = ((theta(1,1)*X(:,1)) + (theta(2,1)*X(:,2)));
%square_sum = 0;
%for i = 1:m;
   %square_sum = square_sum + (Prediction(i) - y(i))^2;
 %end;
 %J = (1/(2*m)) * square_sum;

%vectorization_approach
%J = (1/(2*m))*((X*theta) -y)'*((X*theta) - y);

%another_vectorization_methods
Prediction = (X*theta);
J = (1/(2*m))*(Prediction - y)'*(Prediction - y);

% =========================================================================

end
