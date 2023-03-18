function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== IMPORTANT CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%X = [ones(size(X, 1), 1) X]; % add x_0 feature
Hypothesis = X *theta;
Regularized = (lambda/(2*m)) *(theta(2:end, 1)'*theta(2:end, 1));
J = ((1/(2*m))*((Hypothesis - y)'*(Hypothesis - y))) + Regularized;

grad(1,1) = (1/m) * (X(:,1)' * (Hypothesis - y));
Regul_grad = (lambda/m) *(theta(2:end, 1));
grad(2:end, 1) = ((1/m) * (X(:, 2:end)' * (Hypothesis - y))) + Regul_grad;









% =========================================================================

grad = grad(:);

end
