% Load Pre-trained Features and Labels
load('featuresTrain.mat');
load('featuresValidation.mat');

% Convert labels to categorical if not already
trainLabels = categorical(trainLabels);
validationLabels = categorical(validationLabels);

% Normalize features (Z-score normalization)
meanFeatures = mean(featuresTrain);
stdFeatures = std(featuresTrain);
featuresTrainNorm = (featuresTrain - meanFeatures) ./ stdFeatures;
featuresValidationNorm = (featuresValidation - meanFeatures) ./ stdFeatures;

% Train SVM Classifier with hyperparameter optimization
svmModel = fitcecoc(featuresTrainNorm, trainLabels, 'OptimizeHyperparameters', 'all', ...
    'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName', ...
    'expected-improvement-plus', 'MaxObjectiveEvaluations', 10));

% Save the trained SVM classifier
save('svmModel.mat', 'svmModel');

% Validate SVM Classifier
[predictedLabels, scores] = predict(svmModel, featuresValidationNorm);

% Calculate Accuracy
accuracy = mean(predictedLabels == validationLabels);
disp(['Validation accuracy: ', num2str(accuracy)]);

% Display Confusion Matrix for Validation Data
figure;
confusionchart(validationLabels, predictedLabels);
title('Confusion Matrix for Validation Data');

% Calculate Precision, Recall, and F1 Score
[confMat, order] = confusionmat(validationLabels, predictedLabels);
precision = diag(confMat) ./ sum(confMat, 2);
recall = diag(confMat) ./ sum(confMat, 1)';
f1Score = 2 * (precision .* recall) ./ (precision + recall);

% Display Precision, Recall, and F1 Score
disp(['Precision: ', num2str(mean(precision, 'omitnan'))]);
disp(['Recall: ', num2str(mean(recall, 'omitnan'))]);
disp(['F1 Score: ', num2str(mean(f1Score, 'omitnan'))]);

% Plot ROC Curve for each class
numClasses = numel(order);
figure;
hold on;
legendInfo = cell(numClasses, 1);
for i = 1:numClasses
    [X, Y, ~, AUC] = perfcurve(validationLabels, scores(:, i), order(i));
    plot(X, Y);
    legendInfo{i} = [char(order(i)), ' (AUC: ', num2str(AUC), ')'];
end
xlabel('False positive rate'); 
ylabel('True positive rate');
title('ROC Curves for Each Class');
legend(legendInfo, 'Location', 'Best');
hold off;
