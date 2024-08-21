% Load the pre-trained feature extraction network and SVM classifier
l = load('featureNet.mat');
featureNet = l.featureNet;

l = load('svmModel.mat');
svmModel = l.svmModel;

% Load the test image datastore
imdsTest = imageDatastore('N:\MATLAB\CS229-master\CK+', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Define the preprocess function for a single image
function img = preprocessSingleImage(filename)
    img = imread(filename);
    img = imresize(img, [224 224]);
    if size(img, 3) == 1
        img = repmat(img, [1 1 3]);
    end
    img = double(img) / 255; % Normalize the image to the range [0, 1]
end

% Preprocess and extract features from the test images
numTestImages = numel(imdsTest.Files);
testFeatures = zeros(numTestImages, 2048); % Assuming the 'avg_pool' layer outputs 2048 features
testLabels = imdsTest.Labels;

for i = 1:numTestImages
    img = preprocessSingleImage(imdsTest.Files{i});
    features = activations(featureNet, img, 'avg_pool', 'OutputAs', 'rows');
    testFeatures(i, :) = features;
end

% Predict labels using the SVM model
[predictedLabels, scores] = predict(svmModel, testFeatures);

% Calculate accuracy
accuracy = mean(predictedLabels == testLabels);
disp(['Test accuracy: ', num2str(accuracy)]);

% Randomly select 9 images to display with their predictions
index = randperm(numTestImages, 9);

figure;
for i = 1:9
    subplot(3,3,i);
    I = readimage(imdsTest, index(i));
    imshow(I);
    label = predictedLabels(index(i));
    probability = max(scores(index(i), :));
    title(string(label) + ", " + num2str(100 * probability, 3) + "%");
end

% Display Confusion Matrix for Test Data
figure;
confusionchart(testLabels, predictedLabels);
title('Confusion Matrix for Test Data');
