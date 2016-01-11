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

% ====================== YOUR CODE HERE ======================
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
%size(Theta1_grad)
%size(Theta2_grad)
%size(y)
        X = [ones(m, 1) X];
%size(X)
        z1 = ( X * Theta1');
        H1 = sigmoid(z1);
%size(H1)
        H1 = [ones(m, 1) H1];
        z2 = ( H1 * Theta2');
        H2 = sigmoid(z2);

	Jk = zeros(size(y));
	yk = zeros(size(y));
	for k = 1:num_labels
	yk = (y==k);
	Jk = Jk +  (-1/m) * ((yk .* log(H2(:,k))) + ((1-yk) .* log(1-H2(:,k))));
	end
	temp1 = Theta1;
	temp2 = Theta2;
	temp1(:,1) = 0;
	temp2(:,1) = 0;
	J = sum(Jk) + (lambda/(2*m)) * (sum(sum(temp1.^2)) + sum(sum(temp2.^2)));


% -------------------------------------------------------------
	D1 = zeros(size(Theta1));
	D2 = zeros(size(Theta2));
	for t= 1:m
	a1 = X(t,:)';

	z2 = Theta1  * a1 ;
	a2 = sigmoid(z2);

        a2 = [1; a2];
	z3 =  Theta2 * a2;
	a3 = sigmoid(z3);

	del3 = zeros(num_labels,1);
	for k = 1:num_labels
	yk = (y==k);
	del3(k) = a3(k) - yk(t);
	end

	del2 = Theta2' * del3;
	del2 = del2(2:end) .* sigmoidGradient(z2);  

        D2 = D2 + del3 * a2';
        D1 = D1 + del2 * a1';
	end 
%	
% =========================================================================
	Theta1_grad = (1/m) * D1 + (lambda/m) * temp1;
	Theta2_grad = (1/m) * D2 + (lambda/m) * temp2;


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
