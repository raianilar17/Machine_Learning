function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== MY CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

%Non_vectorization_approach
%prediction = X*theta;
%theta_zero = theta(1,1) - (alpha*(1/m)*sum((prediction -y).*X(:, 1))); 
%theta_one = theta(2,1) - (alpha*(1/m)*sum((prediction -y).*X(:, 2)));

%theta(1,1) = theta_zero;
%theta(2,1) = theta_one;

%vectorization_approach
delta = (1/m) * (X'*((X*theta) - y));
theta = theta - (alpha * delta);

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end
figure;
iters = 1:num_iters;
%plot(iters, J_history);
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J(theta)');
%plot(iters, J_history,'rx', 'MarkerSize', 10);
end
