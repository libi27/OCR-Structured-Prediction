function Y = classify_word(X, W, A)
% Inputs
% • X: T-by-d real matrix where row t holds the feature measurements of letter t in a Tletter
% handwritten word.
% • W: k-by-d real matrix representing a word-classifier (rows correspond to classes,
% columns correspond to feature measurements).
% • A: k-by-k real (stochastic) matrix holding letter transition probabilities - entry (r,l) holds
% the probability that a label of a letter is l given that the label of the preceding letter is r.
% Outputs
% • Y: T-element column vector of integers between 1 and k, where each entry holds the
% predicted label (according to W) of the corresponding row in X.
% Description
% The function uses the word-classifier represented by W to predict the labels of the rows in X.
% The rows of X are classified jointly (the interaction is defined by the transition matrix A), with
% the following prediction rule
%calculate sizes

T = size(X, 1);
k = size(W, 1);



%initialize values
dynTbl = zeros(k,T);
indices = zeros(k,T);

WX = X*W';
dynTbl(:,1) = WX(1,:)';

for t = 2 : T  
    [val, idx] = max( (dynTbl(:,t-1) * ones(1,k)) + A);
    
    dynTbl(:,t) = (W*X(t,:)') + val';
    indices(:,t) = idx';
    
end


[~, optIdx] = max(dynTbl(:,T));

Y = [indices(optIdx, 2:T) optIdx]';
end


