function [] = applyStochasticSquaredErrorTwoLayerPerceptronMNIST()
%applyStochasticSquaredErrorTwoLayerPerceptronMNIST Train the two-layer
%perceptron using the MNIST dataset and evaluate its performance.

    % Load MNIST.
    inputValues = loadMNISTImages('train-images.idx3-ubyte');
    labels = loadMNISTLabels('train-labels.idx1-ubyte');
    
       
    % Transform the labels to correct target values.
    targetValues = 0.*ones(10, size(labels, 1));
    
    
    for n = 1: size(labels, 1)
        targetValues(labels(n) + 1, n) = 1;
    end;
    
    % Choose form of MLP:
    numberOfHiddenUnits = 700;
    
    % Choose appropriate parameters.
    learningRate = 0.1;
    
    % Choose activation function.
    activationFunction = @logisticSigmoid;
    dActivationFunction = @dLogisticSigmoid;
    
    % Choose batch size and epochs. Remember there are 60k input values.
    batchSize = 4;
    epochs = 4000;
     fprintf('%d\n',size(inputValues));
    fprintf('Train twolayer perceptron with %d hidden units.\n', numberOfHiddenUnits);
    fprintf('Learning rate: %d.\n', learningRate);
    
    inputValues_test = loadMNISTImages('t10k-images.idx3-ubyte');
    labels_test = loadMNISTLabels('t10k-labels.idx1-ubyte');
    
    inputValues_test = inputValues_test(:,1:10000);
    labels_test = labels_test(1:10000);
   % targetValues =  targetValues(:,1:20000);
   % inputValues = inputValues(:,1:20000);
    size(inputValues)
    
    % Input vector has 784 dimensions.
    inputDimensions = size(inputValues, 1);
    % We have to distinguish 10 digits.
    outputDimensions = size(targetValues, 1);
    % Initialize the weights for the hidden layer and the output layer.
    hiddenWeights = rand(numberOfHiddenUnits, inputDimensions);
    outputWeights = rand(outputDimensions, numberOfHiddenUnits);
    
    hiddenWeights = hiddenWeights./size(hiddenWeights, 2);
    outputWeights = outputWeights./size(outputWeights, 2);

    %Distribute sample
    randindex = randperm(size(inputValues,2));
    load('weights');
    device_num = 10;
    initial_size = 50;
    device_size = floor((size(inputValues,2)-initial_size)/device_num);
    initial_size = size(inputValues,2) - device_size * device_num;
    initial_data = inputValues(:,randindex(1:initial_size));
    initial_label = targetValues(:,randindex(1:initial_size));
    device_data =  zeros(device_num,size(inputValues,1),device_size);
    device_label = zeros(device_num,size(targetValues,1),device_size);
    for cnt=1:device_num
        device_data(cnt,:,:) = inputValues(:,randindex(initial_size + (cnt-1) * device_size + 1:initial_size + cnt * device_size));
        device_label(cnt,:,:) = targetValues(:,randindex(initial_size + (cnt-1) * device_size + 1:initial_size + cnt * device_size));
    end
    device_index = zeros(device_num,1);
    
    
    %[hiddenWeights, outputWeights, error] = train_digit(activationFunction, dActivationFunction, numberOfHiddenUnits, hiddenWeights, outputWeights, initial_data, initial_label, epochs, batchSize, learningRate);
%     save('weights','randindex','hiddenWeights','outputWeights')
    % Load validation set.
    
    [x, y, error_tot] = train_digit(activationFunction, dActivationFunction, numberOfHiddenUnits, hiddenWeights, outputWeights, initial_data, initial_label, 25, batchSize, learningRate,0);
    
    % Choose decision rule.
    fprintf('Validation:\n');
    
    [correctlyClassified, classificationErrors] = validateTwoLayerPerceptron(activationFunction, hiddenWeights, outputWeights, inputValues_test, labels_test);
    
    accuracy_array = [correctlyClassified/(correctlyClassified+classificationErrors)]

    Communication_time = 28*28*1000*(4*4*2)/(28*28);
    last_device = 0;
    epochs = 500;
    policy = 1;
    theta = 6;
    cnt = 0;
    last_cnt = 0;
    policy_array=[];

    TW = 10;
    rho = 1.0;
    sigma = 0.5;
    log(1+ sigma*(-2)*log(1-rand(device_num,1)) * 10^0.7)/log(1+10^0.7);
    policy_record = [];
    [Hv1,Hv2] = valid_digit(activationFunction, dActivationFunction, hiddenWeights, outputWeights, inputValues_test, labels_test, initial_data, initial_label);
    while cnt<Communication_time
       
        last_device = floor(rand(1)*10)+1;
        raw_data = device_data(last_device,:,device_index(last_device)+1);
        
        influ_per_pixel = 0;
        best_policy = 0;
        influ_a = [];
        for policy = 1:4
            data = reshape(imresize(imresize(reshape(raw_data,[28,28]),[7*policy,7*policy]),[28,28]),[28*28,1]);
            influ = influence_f(Hv1,Hv2,activationFunction, dActivationFunction, hiddenWeights, outputWeights,data,device_label(last_device,:,device_index(last_device)+1));
            influ_a(end+1) = influ / (policy*policy);
            if influ_per_pixel > influ / (policy*policy)
                influ_per_pixel = influ / (policy*policy);
                best_policy = policy;
            end
        end
       
%         policy = 4;
        policy_record(end+1) = best_policy;
        policy = best_policy;
        device_index(last_device) = device_index(last_device) + 1;
        if policy==0
            continue
        end
        data = reshape(imresize(imresize(reshape(raw_data,[28,28]),[7*policy,7*policy]),[28,28]),[28*28,1]);
        cum_snr = 0;
        while cum_snr < 7*7*policy*policy
            cnt = cnt + 1;
            cum_snr = cum_snr + TW * log(1+ sigma*(-2)*log(1-rand(1,1)) * 10^rho);
        end
        %cnt = cnt + 7*7*policy*policy - 7*7*(policy-1)*(policy-1);
        
            initial_data(:,end+1) = data;
            initial_label(:,end+1) = device_label(last_device,:,device_index(last_device));
 
        
        newepochs = ceil(epochs * 7 * 7 * policy * policy / (28*28)); 
        [hiddenWeights, outputWeights, error, sample_error,error_tot] = train_digit(activationFunction, dActivationFunction, numberOfHiddenUnits, hiddenWeights, outputWeights, initial_data, initial_label, newepochs, batchSize, learningRate,error_tot);
        
        %theta = 100000000;

        if cnt-last_cnt > (4*4*2)*10
            [Hv1,Hv2,correctlyClassified, classificationErrors,test_error] = valid_digit(activationFunction, dActivationFunction, hiddenWeights, outputWeights, inputValues_test, labels_test, initial_data, initial_label);
            %[correctlyClassified, classificationErrors] = validateTwoLayerPerceptron(activationFunction, hiddenWeights, outputWeights, inputValues_test, labels_test);
            last_cnt = last_cnt + (4*4*2)*10;
            cnt/(4*4*2)
            correctlyClassified/(correctlyClassified+classificationErrors)
                    
            accuracy_array(end+1) = correctlyClassified/(correctlyClassified+classificationErrors);
        end
         
    end
    plot(accuracy_array);
    save('digit_influ_new_augH','accuracy_array','policy_record')
    fprintf('Classification errors: %d\n', classificationErrors);
    fprintf('Correctly classified: %d\n', correctlyClassified);
end