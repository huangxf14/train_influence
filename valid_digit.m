function [Hv1,Hv2,correctlyClassified, classificationErrors, error] = valid_digit(activationFunction, dActivationFunction, hiddenWeights, outputWeights, inputValues, labels, traininput, trainlabel)
% validateTwoLayerPerceptron Validate the twolayer perceptron using the
% validation set.
%
% INPUT:
% activationFunction             : Activation function used in both layers.
% hiddenWeights                  : Weights of the hidden layer.
% outputWeights                  : Weights of the output layer.
% inputValues                    : Input values for training (784 x 10000).
% labels                         : Labels for validation (1 x 10000).
%
% OUTPUT:
% correctlyClassified            : Number of correctly classified values.
% classificationErrors           : Number of classification errors.
% 
    trainsize = size(traininput, 2);
    testSetSize = size(inputValues, 2);
    classificationErrors = 0;
    correctlyClassified = 0;
    error = 0;
    Lod = zeros(size(outputWeights,1),size(outputWeights,2));
    Loh = zeros(size(hiddenWeights,1),size(hiddenWeights,2));
    for t = 1: testSetSize
        inputVector = inputValues(:, t);
        hiddenActualInput = hiddenWeights*inputVector;
        hiddenOutputVector = activationFunction(hiddenActualInput);
        outputActualInput = outputWeights*hiddenOutputVector;
        outputVector = activationFunction(outputActualInput);
        targetVector = zeros(10,1);
        targetVector(labels(t)+1) = 1;

        error = error + norm(outputVector-targetVector,2);
        class = decisionRule(outputVector);
        if class == labels(t) + 1
            correctlyClassified = correctlyClassified + 1;
        else
            classificationErrors = classificationErrors + 1;
        end;
        % Backpropagate the errors.
        outputDelta = (dActivationFunction(outputActualInput).*(outputVector - targetVector));
        hiddenDelta = (dActivationFunction(hiddenActualInput).*(outputWeights'*outputDelta))*inputVector';
        outputDelta = outputDelta*hiddenOutputVector';
        
        Lod = Lod + outputDelta;
        Loh = Loh + hiddenDelta;
        %outputWeights = outputWeights - learningRate.*outputDelta*hiddenOutputVector';
        %hiddenWeights = hiddenWeights - learningRate.*hiddenDelta*inputVector';
    end
    error = error / testSetSize;
    Lod = Lod/testSetSize;
    Loh = Loh/testSetSize;
    %Ltest = [reshape(Lod/testSetSize,1,size(Lod,1)*size(Lod,2)) reshape(Loh/testSetSize,1,size(Loh,1)*size(Loh,2))]';
    Ltest1 = Lod;
    Ltest2 = Loh;
    Hv1 = Ltest1;
    Hv2 = Ltest2;
    trainslelectepoch = 1000;
    trainbatchsize = 10;
    %deltay = zeros(size(Hv));
    for n = 1: trainslelectepoch
        tempHv1 = zeros(size(Hv1));
        tempHv2 = zeros(size(Hv2));
        for m = 1: trainbatchsize
            t= ceil(rand(1,1)*trainsize);
            inputVector = traininput(:, t);
            hiddenActualInput = hiddenWeights*inputVector;
            hiddenOutputVector = activationFunction(hiddenActualInput);
            outputActualInput = outputWeights*hiddenOutputVector;
            %outputVector = activationFunction(outputActualInput);
            %targetVector = trainlabel(:, t);

            % Backpropagate the errors.
            outputDelta = (dActivationFunction(outputActualInput));
            hiddenDelta = (dActivationFunction(hiddenActualInput).*(outputWeights'*outputDelta))*inputVector';
            outputDelta = outputDelta*hiddenOutputVector';
            %deltay = [reshape(outputDelta,size(outputDelta,1)*size(outputDelta,2),1); reshape(hiddenDelta,size(hiddenDelta,1)*size(hiddenDelta,2),1)];
            deltay1 = outputDelta;
            deltay2 = hiddenDelta;
            deltaycoeff = sum(sum(deltay1.*Hv1)) + sum(sum(deltay2.*Hv2));
            tempHv1 = tempHv1 + deltay1 * deltaycoeff;
            tempHv2 = tempHv2 + deltay2 * deltaycoeff;
        end
        tempHv1 = tempHv1 / trainbatchsize;
        tempHv2 = tempHv2 / trainbatchsize;
        Hv1 = Ltest1 + Hv1 - tempHv1;
        Hv2 = Ltest2 + Hv2 - tempHv2;
    end;
    
end

function class = decisionRule(outputVector)
% decisionRule Model based decision rule.
%
% INPUT:
% outputVector      : Output vector of the network.
%
% OUTPUT:
% class             : Class the vector is assigned to.
%

    max = 0;
    class = 1;
    for i = 1: size(outputVector, 1)
        if outputVector(i) > max
            max = outputVector(i);
            class = i;
        end;
    end;
end

function outputVector = evaluateTwoLayerPerceptron(activationFunction, hiddenWeights, outputWeights, inputVector)
% evaluateTwoLayerPerceptron Evaluate two-layer perceptron given by the
% weights using the given activation function.
%
% INPUT:
% activationFunction             : Activation function used in both layers.
% hiddenWeights                  : Weights of hidden layer.
% outputWeights                  : Weights for output layer.
% inputVector                    : Input vector to evaluate.
%
% OUTPUT:
% outputVector                   : Output of the perceptron.
% 

    outputVector = activationFunction(outputWeights*activationFunction(hiddenWeights*inputVector));
end