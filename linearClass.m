% train a very simple, un-regularized linear classifier on the digits
% dataset

load mfeat-pix.txt -ascii;
traindata = zeros(1000,240);
testdata = zeros(1000,240);
% split all data into training and testing sets
for i=1:10
    traindata(100 * (i-1) + 1:100*i,:) = mfeat_pix(200 * (i-1) + 1:200*i - 100,:);
    testdata(100 * (i-1) + 1:100*i,:) = mfeat_pix(200 * (i-1) + 101:200*i ,:);
end

% create the binary target vectors
targets = zeros(1000,10);
for n = 1:10
    targets((n-1)*100+1:n*100,n) = ones(100,1);
end

% compute the linear regression directly on the image vectors
X = pinv(traindata) * targets;

% classify the test patterns
testvotes = testdata * X;
testvotes = testvotes';
[xx, testresults] = max(testvotes);
testresults = testresults - 1;

% compare against the known correct classifications
corrects = zeros(1,1000);
for n = 1:10
    corrects(1,(n-1)*100+1:n*100) = (n-1) * ones(1,100);
end
mismatches = sum(abs(sign(testresults - corrects)));
percentError = 100 * mismatches / 1000