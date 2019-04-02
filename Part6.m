load('datasets/letters.mat');
load('W_wrd.mat');
load('datasets/words.mat');
load('A.mat');


W_ltr = letter_hinge_SGD(letters.train.X, letters.train.Y, 10^(-2), 50000);


iteration =  size(words.test, 2);
LetterErr_wrd = zeros(1,iteration);
WordErr_wrd = zeros(1,iteration);
LetterErr_ltr = zeros(1,iteration);
WordErr_ltr = zeros(1,iteration);

for i = 1 : iteration
    %calculate labels and errors for letter and word clasification on W_wrd
    Y_ltr = classify_letters(words.test(i).X, W_wrd);
    LetterErr_wrd(i) = nnz((Y_ltr - words.test(i).Y)) / numel(words.test(i).Y);
    
    Y_wrd = classify_word(words.test(i).X, W_wrd, A);
    WordErr_wrd(i) = nnz((Y_wrd - words.test(i).Y)) / numel(words.test(i).Y);
    
    %calculate labels and errors for letter and word clasification on W_ltr
    Y_ltr = classify_letters(words.test(i).X, W_ltr);
    LetterErr_ltr(i) = nnz((Y_ltr - words.test(i).Y)) / numel(words.test(i).Y);
    
    Y_wrd = classify_word(words.test(i).X, W_ltr, A);
    WordErr_ltr(i) = nnz((Y_wrd - words.test(i).Y)) / numel(words.test(i).Y);

end

nnz(LetterErr_ltr)
nnz(LetterErr_wrd)
nnz(WordErr_ltr)
nnz(WordErr_wrd)
