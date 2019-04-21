%%% an elementary script to classify zero vs one images using a single
%%% feature. Here a feature made from a prototype pattern is used

%%% set number of bins
binNumber = 20;

load mfeat-pix.txt -ascii;

%%% compute the feature values for all zeros and ones.

% as a prototype pattern, simply use the first "zero" from the training
% data
prototypeZero = mfeat_pix(1,:);
featureValues = prototypeZero * mfeat_pix(1:400, :)';
featureValuesZerosTrain = featureValues(1,1:100)';
featureValuesZerosTest = featureValues(1,101:200)';
featureValuesOnesTrain = featureValues(1,201:300)';
featureValuesOnesTest = featureValues(1,301:400)';

%%% do the bin counting for the training samples. 

% compute the max and min feature value in the training set
minFVal = min([featureValuesZerosTrain; featureValuesOnesTrain]);
maxFVal = max([featureValuesZerosTrain; featureValuesOnesTrain]);

% create a vector with equally spaced bin borders
binEdges = minFVal:(maxFVal - minFVal)/binNumber:maxFVal;

% open first and last bin to infinity
binEdges(1) = -inf;
binEdges(end) = inf; 

% do the bin counting
zeroCountsTrain = histc(featureValuesZerosTrain, binEdges);
oneCountsTrain = histc(featureValuesOnesTrain, binEdges);

% delete the last entry in count vectors because it is always =0 (check
% Matlab Help for histc to find out why!)
zeroCountsTrain = zeroCountsTrain(1:end-1);
oneCountsTrain = oneCountsTrain(1:end-1);

% determine which of the bins should be considered as representing zeros
% and which ones should be representing ones
bindiffs = zeroCountsTrain - oneCountsTrain;
ZeroBinIndices = bindiffs > 0;
OneBinIndices = bindiffs <= 0;

% compute misclassification rate on testing data
zeroCountsTest = histc(featureValuesZerosTest, binEdges);
oneCountsTest = histc(featureValuesOnesTest, binEdges);
zeroCountsTest = zeroCountsTest(1:end-1);
oneCountsTest = oneCountsTest(1:end-1);
FalseZerosN = sum(zeroCountsTest .* OneBinIndices);
FalseOnesN = sum(oneCountsTest .* ZeroBinIndices);
TotalFalseN = FalseZerosN + FalseOnesN;

disp(sprintf('missclassifications on test data in per cent: %0.3g',...
    100 * TotalFalseN / 200));
    
% plot training data histogram. Could be improved to have the bin borders
% displayed as x axis legend but I am too lazy. 

figure(2);
bar([zeroCountsTrain oneCountsTrain]);


