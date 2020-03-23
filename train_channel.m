function [hiddenWeights, outputWeights, error, sample_error,error_tot] = train(activationFunction, dActivationFunction, numberOfHiddenUnits, hiddenWeights, outputWeights, inputValues, targetValues,  epochs, batchSize, learningRate,error_tot)
% trainStochasticSquaredErrorTwoLayerPerceptron Creates a two-layer perceptron
% and trains it on the MNIST dataset.
%
% INPUT:
% activationFunction             : Activation function used in both layers.
% dActivationFunction            : Derivative of the activation
% function used in both layers.
% numberOfHiddenUnits            : Number of hidden units.
% inputValues                    : Input values for training (784 x 60000)
% targetValues                   : Target values for training (1 x 60000)
% epochs                         : Number of epochs to train.
% batchSize                      : Plot error after batchSize images.
% learningRate                   : Learning rate to apply.
%
% OUTPUT:
% hiddenWeights                  : Weights of the hidden layer.
% outputWeights                  : Weights of the output layer.
% 

    % The number of training vectors.
    trainingSetSize = size(inputValues, 2);
    
    % Input vector has 784 dimensions.
    inputDimensions = size(inputValues, 1);
    % We have to distinguish 10 digits.
    outputDimensions = size(targetValues, 1);
    
    % Initialize the weights for the hidden layer and the output layer.
    % hiddenWeights = rand(numberOfHiddenUnits, inputDimensions);
    % outputWeights = rand(outputDimensions, numberOfHiddenUnits);
    
    % hiddenWeights = hiddenWeights./size(hiddenWeights, 2);
    % outputWeights = outputWeights./size(outputWeights, 2);
    

    
    n = zeros(batchSize);
   
    %figure; hold on;
    
    accuracy_curve=zeros(1,epochs);
    error_curve=zeros(1,epochs);
    
    for t = 1: epochs
        for k = 1: batchSize
            % Select which input vector to train on.
            n(k) = floor(rand(1)*trainingSetSize + 1);
            if (t<=50)&&(k==1)
                n(k) = trainingSetSize;
            end
            
            % Propagate the input vector through the network.
            inputVector = inputValues(:, n(k));
            hiddenActualInput = hiddenWeights*inputVector;
            hiddenOutputVector = activationFunction(hiddenActualInput);
            outputActualInput = outputWeights*hiddenOutputVector;
            outputVector = activationFunction(outputActualInput);
            
            targetVector = targetValues(:, n(k));
            
            % Backpropagate the errors.
            outputDelta = dActivationFunction(outputActualInput).*(outputVector - targetVector);
            hiddenDelta = dActivationFunction(hiddenActualInput).*(outputWeights'*outputDelta);
            
            outputWeights = outputWeights - learningRate.*outputDelta*hiddenOutputVector';
            hiddenWeights = hiddenWeights - learningRate.*hiddenDelta*inputVector';

            if (t==1)&&(k==1)
                sample_error = norm(outputVector - targetVector);
            end

            
        end
        
        % Calculate the error for plotting.
        error = 0;
        for k = 1: batchSize
            inputVector = inputValues(:, n(k));
            targetVector = targetValues(:, n(k));
            
            error = error + norm(activationFunction(outputWeights*activationFunction(hiddenWeights*inputVector)) - targetVector, 2);
        end
        error = error/batchSize;
        error_tot = (error_tot*0.9 + error*0.1);
        %plot(t, error,'*');
        error_curve(t)=error;
    end
end