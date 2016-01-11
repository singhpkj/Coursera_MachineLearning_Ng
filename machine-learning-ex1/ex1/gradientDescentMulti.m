function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    H = (theta' * X')';
    for iter1 = 1:size(X,2)
	theta(iter1) = theta(iter1) - (alpha/m) * sum((H-y) .* X(:,iter1));
    end
%    H = (theta' * X')';
%    Sum1 = sum((H-y) .* X(:,1));
%    Sum2 = sum((H-y) .* X(:,2));
%    Sum3 = sum((H-y) .* X(:,3));
%    J1 = Sum1/(2*m);
%    J2 = Sum2/(2*m);
%    J3 = Sum3/(2*m);
%    theta(1) = theta(1) - alpha * 2 * J1;
%    theta(2) = theta(2) - alpha * 2 * J2;
%    theta(3) = theta(3) - alpha * 2 * J3;








    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
