function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
%    iterations = 500;
    alpha = 0.001;
%    for iter = 1:iterations
       z = (X * theta);
       H = sigmoid(z);
%       for iter1 = 1:size(theta)
%	    grad(iter1) = (1/m) * sum((H -y) .* X(:,iter1));
%	    theta(iter1) = theta(iter1) - alpha *  grad(iter1);
%        end

%    z = (X * theta);
%    H = sigmoid(z);
%    J = (1/m) * sum((-y .* log(H)) - ((1-y) .* log(1.0-H))); 

%    end

    grad = (1/m) * (X' * (H-y))';
    theta = theta - (alpha * grad);    
    J = (1/m) * sum((-y .* log(H)) - ((1-y) .* log(1.0-H))); 


% =============================================================

end
