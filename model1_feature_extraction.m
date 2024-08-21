net = resnet50;

% Convert the network to a layer graph
lgraph = layerGraph(net);

% Remove the old classification layers
layersToRemove = {'fc1000', 'fc1000_softmax', 'ClassificationLayer_fc1000'};
lgraph = removeLayers(lgraph, layersToRemove);

% Load and prepare the data
emotionData = imageDatastore('N:\MATLAB\CS229-master\CK+', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[emotionTrain, emotionValidation] = splitEachLabel(emotionData, 0.8, 'randomize');

augmenter = imageDataAugmenter('RandRotation', [-10, 10], 'RandScale', [0.8 1.2]);

% Use 'ColorPreprocessing' option to ensure consistent image channels
augmentedTrain = augmentedImageDatastore([224 224], emotionTrain, 'DataAugmentation', augmenter, 'ColorPreprocessing', 'gray2rgb');
augmentedValidation = augmentedImageDatastore([224 224], emotionValidation, 'ColorPreprocessing', 'gray2rgb');

numClasses = numel(categories(emotionTrain.Labels));

% Define new layers
newLayers = [
    flattenLayer('Name', 'flatten')
    fullyConnectedLayer(numClasses, 'Name', 'fc8', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classification')];

% Add new layers to the layer graph
lgraph = addLayers(lgraph, newLayers);

% Connect new layers
% Ensure any previous connection to 'fc8' is removed
if any(strcmp('fc8', lgraph.Connections.Destination))
    lgraph = disconnectLayers(lgraph, 'avg_pool', 'fc8');
end

% Connect 'avg_pool' to 'flatten' layer (if not already connected)
if ~any(strcmp('flatten', lgraph.Connections.Destination))
    lgraph = connectLayers(lgraph, 'avg_pool', 'flatten');
end

% Connect remaining layers
if ~any(strcmp('fc8', lgraph.Connections.Destination))
    lgraph = connectLayers(lgraph, 'flatten', 'fc8');
end

if ~any(strcmp('softmax', lgraph.Connections.Destination))
    lgraph = connectLayers(lgraph, 'fc8', 'softmax');
end

if ~any(strcmp('classification', lgraph.Connections.Destination))
    lgraph = connectLayers(lgraph, 'softmax', 'classification');
end

% Set training options
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-4, ...
    'MaxEpochs', 50, ... 
    'MiniBatchSize', 16, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augmentedValidation, ...
    'ValidationFrequency', floor(numel(augmentedTrain.Files) / 32), ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto', ...
    'L2Regularization', 1e-4, ...
    'GradientThresholdMethod', 'l2norm', ...
    'GradientThreshold', 1.0, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 10);

% Train the network
featureNet = trainNetwork(augmentedTrain, lgraph, options);

% Extract features
featuresTrain = activations(featureNet, augmentedTrain, 'avg_pool', 'OutputAs', 'rows');
featuresValidation = activations(featureNet, augmentedValidation, 'avg_pool', 'OutputAs', 'rows');

% Save features and labels
trainLabels = emotionTrain.Labels;
validationLabels = emotionValidation.Labels;
save('featuresTrain.mat', 'featuresTrain', 'trainLabels');
save('featuresValidation.mat', 'featuresValidation', 'validationLabels');

% Save the trained network
save('featureNet.mat', 'featureNet');
