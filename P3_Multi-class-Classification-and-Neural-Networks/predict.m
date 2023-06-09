function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== IMPORTANT CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


A_1 = X;                    % Activation function of layer_1
A_1 = [ones(size(X,1), 1), A_1]; % add bias unit in layer_1(unit_zero);
Z_2 = A_1 * Theta1';
A_2 = sigmoid(Z_2);         % Activation function of layer_2
A_2 = [ones(size(A_2, 1), 1), A_2]; % add bias unit in layer_2(unit_zero);
Z_3 = A_2 * Theta2';
A_3 = sigmoid(Z_3);         % Activation function of layer_3
[Value, label] = max(A_3, [], 2);
p = label;







% =========================================================================


end
