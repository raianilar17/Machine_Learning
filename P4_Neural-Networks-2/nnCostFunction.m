function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== IMPORTANT CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

eye_matrix = eye(num_labels);
y_label = eye_matrix(y, :);

%% "Feedforward Pass"

A_1 = X;                    % Activation function of layer_1
A_01 = [ones(size(X,1), 1), A_1]; % add bias unit in layer_1(unit_zero);
Z_2 = A_01 * Theta1';
A_2 = sigmoid(Z_2);         % Activation function of layer_2
A_02 = [ones(size(A_2, 1), 1), A_2]; % add bias unit in layer_2(unit_zero);
Z_3 = A_02 * Theta2';
A_3 = sigmoid(Z_3);         % Activation function of layer_3
%[Value, label] = max(A_3, [], 2);
%p = label;


%y_l = ones(m, num_labels);

%for i=1:m,
%  for j =1:num_labels,
%    y_l(i,j) = (find(i, j)== y(i));
%  end;
%end;

%% "Regularized Cost Function"

Regul_cost = (lambda/(2*m)) * ((sum(sum(Theta1(:,2:end) .* Theta1(:,2:end)))) + ...
                                (sum(sum(Theta2(:,2:end) .* Theta2(:,2:end)))));
                               
J = -((1/(m))*(sum(sum((y_label.*(log(A_3))) .+ ...
                       ((1- y_label).*(log(1- A_3)))))));
    
J = J + Regul_cost;

%% "Backward Pass" 

% while apply backpropagation first remove first column of All theta(Such as Theta1, Theta2) in network

% "Delta Measures how much that node was responsible for any errors in our output"
% Note: Intuitively ,(Delta^l)_j is the error for (A^l)_j(unit j in layer l). ...
% more formally, The Delta values are actually the derivative of the cost function.

% Chain rule = global_gradient * local_gradient

%Partial Derivative of Cost Function w.r.t Activation Function such as (Z_3,Z_2)
Delta_3 = A_3 .- y_label; % Z_3
Delta_2 = Delta_3 * Theta2(:, 2:end) .* sigmoidGradient(Z_2); % Z_2

%Partial Derivative of Cost Function  w.r.t Theta Parameter(weights)susc as (Theta1, Theta2)
Delta_cap2 = Delta_3' * A_02; % Theta2
Delta_cap1 = Delta_2' * A_01; % Theta1

% Regularised Theta
Regul_Theta1 = (lambda/m)*Theta1(:,2:end);
Regul_Theta2 = (lambda/m)*Theta2(:,2:end);


Theta1_grad = (1/m)*Delta_cap1 ;
Theta2_grad = (1/m)*Delta_cap2 ;

% Merge all theta
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + Regul_Theta1;
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + Regul_Theta2;








% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
