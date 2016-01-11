function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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
    alpha = 0.001;
%    for iter = 1:iterations
        z = (X * theta);
        H = sigmoid(z);
%        for iter1 = 1:size(theta)
%	    if(iter1 == 1)
%                grad(iter1) = (1/m) * sum((H -y) .* X(:,iter1))  ;
%                theta(iter1) = theta(iter1) - alpha *  grad(iter1);
%	    else
%                grad(iter1) = (1/m) * sum((H -y) .* X(:,iter1)) + (lambda/m) * theta(iter1)  ;
%                theta(iter1) = theta(iter1) - alpha *  grad(iter1);
%        end
%    z = (X * theta);
%    H = sigmoid(z);

%  matrix implimentation
    grad = (1/m) * (X' * (H-y));
    temp = theta;
    temp(1) = 0;
    grad = grad +  (lambda/m) * temp;
    theta = theta - (alpha * grad);
    theta = [0;theta(2:end)];

    z = (X * theta);
    H = sigmoid(z);
    
    J = (1/m) * sum((-y .* log(H)) - ((1-y) .* log(1-H)));
    J = J + lambda/(2*m) * sum(theta(2:end).^2);

%end matrix implementatiin
%    J = (1/m) * sum((-y .* log(H)) - ((1-y) .* log(1.0-H))) + lambda/(2*m) * sum(theta.^2) ;






% =============================================================

end
