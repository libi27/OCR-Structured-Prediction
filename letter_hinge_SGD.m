function W = letter_hinge_SGD(X, Y, lambda, T)
% Inputs
% • X: m-by-d real matrix where each row holds the feature measurements of a different
% handwritten letter.
% • Y: m-element vector of integers between 1 and k, where each entry holds the label of
% the respective row in X.
% • lambda: Non-negative scalar. Regularization parameter.
% • T: Positive integer. Number of SGD iterations to carry out.
% Outputs
% • W: k-by-d real matrix representing the letter-classifier (rows correspond to classes,
% columns correspond to feature measurements) learned via SGD-minimization of
% regularized hinge-loss.
% Description
% The function implements the algorithm you have specified in Part 2. Namely, it learns a letterclassifier
% via SGD-minimization of the regularized hinge-loss.

k = max(Y);
[m, d] = size(X);
W = zeros(k, d);

for t = 1 : T
    i = ceil(rand() * m);
    Z = ones(k, 1);Z(Y(i)) = 0;
    
    [~, z] = max(Z + W * X(i,:)'); 
    
    phi_z = zeros(k, d);
    phi_z(z, :) = X(i, :);
    
    phi_y = zeros(k, d);
    phi_y(Y(i), :) = X(i, :);

    W = (1 - (1/t))*W - (1/(lambda * t)) * (phi_z - phi_y);
end

end