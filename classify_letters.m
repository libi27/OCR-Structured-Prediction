function [ Y ] = classify_letters(X, W)
% Inputs
% • X: m-by-d real matrix where each row holds the feature measurements of a different
% handwritten letter to classify.
% • W: k-by-d real matrix representing a letter-classifier (rows correspond to classes,
% columns correspond to feature measurements).
% Outputs
% • Y: m-element column vector of integers between 1 and k, where each entry holds the
% predicted label (according to W) of the corresponding row in X.
% Description
% The function uses the letter-classifier represented by W to predict the labels of the rows in X.
% This is done with the following linear classification rule
[~,Y] = max(W * X');
Y=Y';
end