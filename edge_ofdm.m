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
    initial_size = 3000;
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
    device_index_flag = zeros(device_num,device_size);
    
    
%     epochs=10000;
%     [hiddenWeights, outputWeights, error] = train_channel(activationFunction, dActivationFunction, numberOfHiddenUnits, hiddenWeights, outputWeights, initial_data, initial_label, epochs, batchSize, learningRate,0);
%     save('weights_3000','randindex','hiddenWeights','outputWeights')
%     Load validation set.
    
    %[x, y, error_tot] = train_channel(activationFunction, dActivationFunction, numberOfHiddenUnits, hiddenWeights, outputWeights, initial_data, initial_label, 25, batchSize, learningRate,0);
    
    % Choose decision rule.
    fprintf('Validation:\n');
    
    [correctlyClassified, classificationErrors, test_error] = validateTwoLayerPerceptron(activationFunction, hiddenWeights, outputWeights, inputValues_test, labels_test);
    %max_error = norm(ones(size(inputValues,1),1),2);
    
    accuracy_array = [correctlyClassified/(correctlyClassified+classificationErrors)]

%    Communication_time = 28*28*1000;%*2;
%     Communication_time = 28*28*1000;
    slot_tot = 100;
    last_device = 0;
    epochs = 2000;
    policy = 2;

    cnt = 0;
    last_cnt = 0;
%     policy_array=[];
    device_error = ones(device_num,1);
    alpha = 2;

    last_slot_v = cnt;
    sigma = 0.5;
    device_v = log(1+ sigma*(-2)*log(1-rand(device_num,1)) * 10^0.7)/log(1+10^0.7);

    best_sample_num = 100;
    best_sample_tot = 200;
    device_best_sample = zeros(device_num,best_sample_num);
    device_sample_importance = zeros(device_num,best_sample_num);
    device_sample_importance_policy = zeros(device_num,best_sample_num,4);
    index_best_sample_num = 0:(best_sample_num-1);

    % ave SNR can transmit an image with a subcarrier in a time slot 
    % 5 subcarrier 
    subcarrier_num = 5;
    TW = 10000;
    ave_snr = 1;

    [Hv1,Hv2] = valid_digit(activationFunction, dActivationFunction, hiddenWeights, outputWeights, inputValues_test, labels_test, initial_data, initial_label);
    for cnt = 1:slot_tot

        % update v
        device_v = log(1 - log(1-rand(subcarrier_num,device_num)) * 10^ave_snr);
        
        % update importance
        for device_cnt=1:device_num
            for device_sample_cnt=1:best_sample_num
                if device_best_sample(device_cnt,device_sample_cnt) == 0 
                    break;
                end;
                %%importance calculation

                raw_data = device_data(device_cnt,:,device_best_sample(device_cnt,device_sample_cnt));
                targetVector = device_label(device_cnt,:, device_best_sample(device_cnt,device_sample_cnt));
                influ_per_pixel = -1000000;
                for policy = 1:4
                    data = reshape(imresize(imresize(reshape(raw_data,[28,28]),[7*policy,7*policy]),[28,28]),[28*28,1]);
                    influ = -influence_f(Hv1,Hv2,activationFunction,dActivationFunction,hiddenWeights,outputWeights,data,targetVector);
                    device_sample_importance_policy(device_cnt,device_sample_cnt,policy) = influ;
                    if influ_per_pixel < influ / (policy*policy)
                        influ_per_pixel = influ / (policy*policy);
                    end
                end
                device_sample_importance(device_cnt,device_sample_cnt) = influ_per_pixel;

                compare_sample = device_sample_cnt - 1;
                temp_index = device_best_sample(device_cnt,device_sample_cnt);
                temp_importance = device_sample_importance(device_cnt,device_sample_cnt);
                temp_importance_policy = device_sample_importance_policy(device_cnt,device_sample_cnt,:);
                while compare_sample > 0
                    if temp_importance > device_sample_importance(device_cnt,compare_sample)
                        device_sample_importance(device_cnt,compare_sample+1) = device_sample_importance(device_cnt,compare_sample);
                        device_best_sample(device_cnt,compare_sample+1) = device_best_sample(device_cnt,compare_sample);
                        device_sample_importance_policy(device_cnt,compare_sample+1,:) = device_sample_importance_policy(device_cnt,compare_sample,:);
                        compare_sample = compare_sample - 1;
                    else
                        break;
                    end
                end
                device_sample_importance(device_cnt,compare_sample+1) = temp_importance;
                device_best_sample(device_cnt,compare_sample+1) = temp_index;
                device_sample_importance_policy(device_cnt,compare_sample+1,:) = temp_importance_policy;
            end

            temp_num = device_size - device_index(device_cnt) - sum(device_best_sample(device_cnt,:)>0);
            explore_num = best_sample_tot-best_sample_num+sum(device_best_sample(device_cnt,:)==0);
            if explore_num > temp_num
                explore_num = temp_num
            end
            temp_index_array = sort(ceil(rand(1,explore_num) * (temp_num-explore_num+1))) + (0:explore_num-1);
            index_pointer = 0;
            index_zero_pointer = 0;
            sample_num = 1;
            flag_one_array = [];
            while sample_num <= explore_num
                while index_zero_pointer < temp_index_array(sample_num)
                    index_pointer = index_pointer + 1;
                    if device_index_flag(device_cnt,index_pointer) == 0 
                        index_zero_pointer = index_zero_pointer + 1;
                    end
                end
                %%importance calculation
                temp_index = index_pointer;
                
                raw_data = device_data(device_cnt,:,index_pointer);
                influ_per_pixel = -100000000;
                temp_importance_policy = zeros(1,4);
                for policy = 1:4
                    data = reshape(imresize(imresize(reshape(raw_data,[28,28]),[7*policy,7*policy]),[28,28]),[28*28,1]);
                    influ = -influence_f(Hv1,Hv2,activationFunction, dActivationFunction, hiddenWeights, outputWeights,data,device_label(device_cnt,:, index_pointer));
                    temp_importance_policy(policy) = influ;
                    if influ_per_pixel < influ / (policy*policy)
                        influ_per_pixel = influ / (policy*policy);
                    end
                end
                temp_importance = influ_per_pixel;

                if temp_importance > device_sample_importance(device_cnt,best_sample_num)
                    if device_best_sample(device_cnt,best_sample_num)>0
                        flag_one_array(end+1) = device_best_sample(device_cnt,best_sample_num);
                    end
                    compare_sample = best_sample_num - 1;
                    while compare_sample > 0
                        if temp_importance > device_sample_importance(device_cnt,compare_sample)
                            device_sample_importance(device_cnt,compare_sample+1) = device_sample_importance(device_cnt,compare_sample);
                            device_best_sample(device_cnt,compare_sample+1) = device_best_sample(device_cnt,compare_sample);
                            device_sample_importance_policy(device_cnt,compare_sample+1,:) = device_sample_importance_policy(device_cnt,compare_sample,:);
                            compare_sample = compare_sample - 1;
                        else
                            break;
                        end
                    end
                    device_sample_importance(device_cnt,compare_sample+1) = temp_importance;
                    device_best_sample(device_cnt,compare_sample+1) = temp_index;
                    device_sample_importance_policy(device_cnt,compare_sample+1,:) = temp_importance_policy;
                    device_index_flag(device_cnt,index_pointer) = 1;
                end
                sample_num = sample_num + 1;
            end
            device_index_flag(device_cnt,flag_one_array) = 0;
        end

        % calculate f(n,s)
        max_sample_size = 4*4*best_sample_num;
        ft = zeros(device_num,max_sample_size,best_sample_num);
        gt = zeros(device_num,max_sample_size,best_sample_num);
        for device_cnt = 1:device_num
            for sample_cnt = 1:best_sample_num
                if sample_cnt > 1
                    ft(device_cnt,:,sample_cnt) = ft(device_cnt,:,sample_cnt-1);
                end
                for sample_size = 1:max_sample_size
                    for policy = 1:4
                        if policy * policy > sample_size
                            break
                        end
                        temp_influ = device_sample_importance_policy(device_cnt,sample_cnt,policy);
                        if policy * policy < sample_size
                            temp_influ = temp_influ + ft(device_cnt,sample_size-policy*policy,sample_cnt-1);
                        end
                        if temp_influ > ft(device_cnt,sample_size,sample_cnt)
                            ft(device_cnt,sample_size,sample_cnt) = temp_influ;
                            gt(device_cnt,sample_size,sample_cnt) = policy;
                        end
                    end
                end
            end
        end

        %allocate subcarrier and get rate for every device
        device_rate = zeros(1,device_num);
        %with rate

        for subcarrier_cnt = 1:subcarrier_num
            [max_rate,max_device] = max(device_v(subcarrier_cnt,:));
            device_rate(max_device) = device_rate(max_device) + max_rate;
        end


        device_rate = floor(TW*device_rate/(7*7*8))
        %select data for transmission according to rate
        for device_cnt = 1:device_num
            temp_size = device_rate(device_cnt);
            for sample_cnt = best_sample_num:-1:1
                policy = gt(device_cnt,temp_size,sample_cnt);
                if policy == 0
                    continue;
                end
                raw_data = device_data(last_device,:,device_best_sample(device_cnt,sample_cnt));
                initial_data(:,end) = reshape(imresize(imresize(reshape(raw_data,[28,28]),[7*policy,7*policy]),[28,28]),[28*28,1]);;
                initial_label(:,end) = device_label(last_device,:,device_best_sample(device_cnt,sample_cnt));
                device_index(device_cnt) = device_index(device_cnt) + 1;
                device_best_sample(device_cnt,sample_cnt) = 0;
                temp_size = temp_size - policy*policy;
            end
            nonempty_pointer = 1;
            for sample_cnt = 1:best_sample_num
                if device_best_sample(device_cnt,sample_cnt) > 0 
                    nonempty_pointer = nonempty_pointer + 1;
                    continue;
                end
                while nonempty_pointer <= best_sample_num
                    nonempty_pointer = nonempty_pointer + 1;
                    if device_best_sample(device_cnt,sample_cnt) > 0
                        break;
                    end;
                end;
                if nonempty_pointer > best_sample_num
                    break;
                end
                device_sample_importance(device_cnt,sample_cnt) = device_sample_importance(device_cnt,nonempty_pointer);
                device_best_sample(device_cnt,sample_cnt) = device_best_sample(device_cnt,nonempty_pointer);
                device_sample_importance_policy(device_cnt,sample_cnt,:) = device_sample_importance_policy(device_cnt,nonempty_pointer,:);
                device_best_sample(device_cnt,nonempty_pointer) = 0;
            end
        end
        
        [hiddenWeights, outputWeights, error, sample_error,error_tot] = train_channel(activationFunction, dActivationFunction, numberOfHiddenUnits, hiddenWeights, outputWeights, initial_data, initial_label, newepochs, batchSize, learningRate,error_tot);
                   
        
            [Hv1,Hv2,correctlyClassified, classificationErrors,test_error] = valid_digit(activationFunction, dActivationFunction, hiddenWeights, outputWeights, inputValues_test, labels_test, initial_data, initial_label);
            %[correctlyClassified, classificationErrors,test_error] = validateTwoLayerPerceptron(activationFunction, hiddenWeights, outputWeights, inputValues_test, labels_test);
            cnt
            correctlyClassified/(correctlyClassified+classificationErrors)
            test_error
            accuracy_array(end+1) = correctlyClassified/(correctlyClassified+classificationErrors);
        end
         
    end
    plot(accuracy_array);
    save('device_filter_new','accuracy_array','device_index');
    fprintf('Classification errors: %d\n', classificationErrors);
    fprintf('Correctly classified: %d\n', correctlyClassified);
    device_index
end