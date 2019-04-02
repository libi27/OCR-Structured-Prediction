load('datasets/letters.mat');

lambdaRange = -5 : 0.05 : 1;
index = 1;
trainError = zeros(size(lambdaRange));
testError = zeros(size(lambdaRange));
T = 50000;

for i = lambdaRange
    lambda = 10^i;
    W = letter_hinge_SGD(letters.train.X, letters.train.Y, lambda, T);
    
    labels = classify_letters(letters.train.X, W);
    trainError(index) = 1 - sum((labels == letters.train.Y)) / numel(letters.train.Y);
    
    labels = classify_letters(letters.test.X, W);
    testError(index) = 1 - sum((labels == letters.test.Y)) / numel(letters.test.Y);
    
    index = index + 1;
end


hold on;
plot(lambdaRange, trainError, 'b', lambdaRange, testError, 'r');
xlabel('exponent for lambda (10^i)') 
ylabel('error in range [0,1]')
legend('error on training set','error on testing set')


[~,idx] = min(testError);
lambdaOpt = 10^(-5 + 0.05*idx);
W = letter_hinge_SGD(letters.train.X, letters.train.Y, lambdaOpt, T);
C = classify_letters(letters.test.X, W);
figure;conf_mat(letters.test.Y, C);


