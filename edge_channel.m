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
    
    [x, y, error_tot] = train_channel(activationFunction, dActivationFunction, numberOfHiddenUnits, hiddenWeights, outputWeights, initial_data, initial_label, 25, batchSize, learningRate,0);
    
    % Choose decision rule.
    fprintf('Validation:\n');
    
    [correctlyClassified, classificationErrors, test_error] = validateTwoLayerPerceptron(activationFunction, hiddenWeights, outputWeights, inputValues_test, labels_test);
    max_error = norm(ones(size(inputValues,1),1),2);
    
    accuracy_array = [correctlyClassified/(correctlyClassified+classificationErrors)]

    Communication_time = 28*28*1000;%*2;
%     Communication_time = 28*28*1000;
    slot_tot = Communication_time * 10;
    last_device = 0;
    epochs = 200;
    policy = 2;
    theta = 6;
    theta_change = 0.5;
    cnt = 0;
    last_cnt = 0;
%     policy_array=[];
    device_error = ones(device_num,1);
    alpha = 2;

    last_slot_v = cnt;
    sigma = 0.5;
    device_v = log(1+ sigma*(-2)*log(1-rand(device_num,1)) * 10^0.7)/log(1+10^0.7);

    best_sample_num = 10;
    best_sample_tot = 20;
    device_best_sample = zeros(device_num,best_sample_num);
    device_sample_importance = zeros(device_num,best_sample_num);
    index_best_sample_num = 0:(best_sample_num-1);
    theta_error = 0.7;


    % ave SNR need 8 slot to send the whole message
    % while cnt<Communication_time
    [Hv1,Hv2] = valid_digit(activationFunction, dActivationFunction, hiddenWeights, outputWeights, inputValues_test, labels_test, initial_data, initial_label);
    while cnt<slot_tot

        % update v
        if cnt - last_slot_v > 100
            device_v = log(1+ sigma*(-2)*log(1-rand(device_num,1)) * 10^0.7)/log(1+10^0.7);
            last_slot_v = last_slot_v + 100;
        end
        
        % update importance
        for device_cnt=1:device_num
            for device_sample_cnt=1:best_sample_num
                if device_best_sample(device_cnt,device_sample_cnt) == 0 
                    break;
                end;
                %%importance calculation

                raw_data = device_data(device_cnt,:,device_best_sample(device_cnt,device_sample_cnt));
                %targetVector = device_label(device_cnt,:, device_best_sample(device_cnt,device_sample_cnt))';
                %device_sample_importance(device_cnt,device_sample_cnt) = norm(activationFunction(outputWeights*activationFunction(hiddenWeights*inputVector')) - targetVector, 2);
                influ_per_pixel = 10000;
                for policy = 1:4
                    data = reshape(imresize(imresize(reshape(raw_data,[28,28]),[7*policy,7*policy]),[28,28]),[28*28,1]);
                    influ = influence_f(Hv1,Hv2,activationFunction, dActivationFunction, hiddenWeights, outputWeights,data,device_label(device_cnt,:, device_best_sample(device_cnt,device_sample_cnt)));
                    
                    if influ_per_pixel > influ / (policy*policy)
                        influ_per_pixel = influ / (policy*policy);
                    end
                end
                device_sample_importance(device_cnt,device_sample_cnt) = -influ_per_pixel;

                compare_sample = device_sample_cnt - 1;
                temp_index = device_best_sample(device_cnt,device_sample_cnt);
                temp_importance = device_sample_importance(device_cnt,device_sample_cnt);
                while compare_sample > 0
                    if temp_importance > device_sample_importance(device_cnt,compare_sample)
                        device_sample_importance(device_cnt,compare_sample+1) = device_sample_importance(device_cnt,compare_sample);
                        device_best_sample(device_cnt,compare_sample+1) = device_best_sample(device_cnt,compare_sample);
                        compare_sample = compare_sample - 1;
                    else
                        break;
                    end
                end
                device_sample_importance(device_cnt,compare_sample+1) = temp_importance;
                device_best_sample(device_cnt,compare_sample+1) = temp_index;
            end

            temp_num = device_size - device_index(device_cnt) - best_sample_num + sum(device_best_sample(device_cnt,:)==0);
            if device_index(device_cnt)+best_sample_num-sum(device_best_sample(device_cnt,:)==0)~=sum(device_index_flag(device_cnt,:))
                device_index(device_cnt)
                sum(device_index_flag(device_cnt,:))
            end
            explore_num = best_sample_tot-best_sample_num+sum(device_best_sample(device_cnt,:)==0);
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
                %inputVector = device_data(device_cnt,:,index_pointer);
                %targetVector = device_label(device_cnt,:, index_pointer)';
                %temp_importance = norm(activationFunction(outputWeights*activationFunction(hiddenWeights*inputVector')) - targetVector, 2);
                
                raw_data = device_data(device_cnt,:,index_pointer);
                influ_per_pixel = 10000;
                for policy = 1:4
                    data = reshape(imresize(imresize(reshape(raw_data,[28,28]),[7*policy,7*policy]),[28,28]),[28*28,1]);
                    influ = influence_f(Hv1,Hv2,activationFunction, dActivationFunction, hiddenWeights, outputWeights,data,device_label(device_cnt,:, index_pointer));
                    if influ_per_pixel > influ / (policy*policy)
                        influ_per_pixel = influ / (policy*policy);
                    end
                end
                temp_importance = -influ_per_pixel;

                if temp_importance > device_sample_importance(device_cnt,best_sample_num)
                    if device_best_sample(device_cnt,best_sample_num)>0
                        flag_one_array(end+1) = device_best_sample(device_cnt,best_sample_num);
                    end
                    compare_sample = best_sample_num - 1;
                    while compare_sample > 0
                        if temp_importance > device_sample_importance(device_cnt,compare_sample)
                            device_sample_importance(device_cnt,compare_sample+1) = device_sample_importance(device_cnt,compare_sample);
                            device_best_sample(device_cnt,compare_sample+1) = device_best_sample(device_cnt,compare_sample);
                            compare_sample = compare_sample - 1;
                        else
                            break;
                        end
                    end
                    device_sample_importance(device_cnt,compare_sample+1) = temp_importance;
                    device_best_sample(device_cnt,compare_sample+1) = temp_index;
                    device_index_flag(device_cnt,index_pointer) = 1;
                end
                sample_num = sample_num + 1;
            end
            device_index_flag(device_cnt,flag_one_array) = 0;
        end

        if true
            device_error_exp = device_sample_importance(:,1)./ceil(10./device_v);
%             device_error_exp = 1./ceil(10./device_v);
            randerror = rand(1) * sum(device_error_exp);
            last_device = 1;
            sumerror = device_error_exp(1);
            while sumerror < randerror
                last_device = last_device + 1;
                sumerror = sumerror + device_error_exp(last_device);
            end
%             if sum(device_index==0) == 0
%                 max_reward = device_error_exp(1) + sqrt((alpha * log(sum(device_index)) / device_index(1))/2);
%                 for last_device_cnt=2:device_num
%                     temp_reward = device_error_exp(last_device_cnt) + sqrt((alpha * log(sum(device_index)) / device_index(last_device_cnt))/2);
%                     if temp_reward > max_reward
%                         last_device = last_device_cnt;
%                         max_reward = temp_reward;
%                     end                        
%                 end
%             else
%                 last_device = floor(rand(1)*10)+1;  
%             end
        end

%        last_device = floor(rand(1)*10)+1;  
%         policy_array(end+1) = policy;
%       data = device_data(last_device,:,device_index(last_device)+1);
        
    
        targetVector = device_label(last_device,:,device_best_sample(last_device,1));

        raw_data = device_data(last_device,:,device_best_sample(last_device,1));
        influ_per_pixel = 10000;
        best_policy = 1;
        for policy = 1:4
            data = reshape(imresize(imresize(reshape(raw_data,[28,28]),[7*policy,7*policy]),[28,28]),[28*28,1]);
            influ = influence_f(Hv1,Hv2,activationFunction, dActivationFunction, hiddenWeights, outputWeights,data,targetVector);
            
            if policy == 1
                influ_per_pixel = influ;
            end
            if influ_per_pixel > influ / (policy*policy)
                influ_per_pixel = influ / (policy*policy);
                best_policy = policy;
            end
        end
        policy = best_policy;


        data = reshape(imresize(imresize(reshape(raw_data,[28,28]),[7*policy,7*policy]),[28,28]),[28*28,1]);
        cnt = cnt + 7*7*policy*policy*ceil(10/device_v(last_device));
        
        if true
            initial_data(:,end+1) = data;
%             initial_label(:,end+1) = device_label(last_device,:,device_index(last_device)+1);
            initial_label(:,end+1) = device_label(last_device,:,device_best_sample(last_device,1));
        else
            initial_data(:,end) = data;
            initial_label(:,end) = device_label(last_device,:,device_index(last_device)+1);
        end
        
        newepochs = ceil(epochs * 7 * 7 * policy * policy / (28*28)); 
        [hiddenWeights, outputWeights, error, sample_error,error_tot] = train_channel(activationFunction, dActivationFunction, numberOfHiddenUnits, hiddenWeights, outputWeights, initial_data, initial_label, newepochs, batchSize, learningRate,error_tot);
        
        %theta = 100000000;
%         if (sample_error > theta * error_tot)&&(policy<4)
%             policy = policy + 1;
%         else
%             policy = 1;
%             device_index(last_device) = device_index(last_device) + 1;
%         end
%        device_error(last_device) = device_error(last_device) * 0.5 + 0.5 * sample_error/error_tot;
                
        device_index(last_device) = device_index(last_device) + 1;

        for device_sample_cnt=1:best_sample_num-1
            device_best_sample(last_device,device_sample_cnt) = device_best_sample(last_device,device_sample_cnt+1);
            device_sample_importance(last_device,device_sample_cnt) = device_sample_importance(last_device,device_sample_cnt+1);
        end
        device_best_sample(last_device,best_sample_num) = 0;
        device_sample_importance(last_device,best_sample_num) = 0;

        if cnt-last_cnt > 28*28*10*10
            [Hv1,Hv2,correctlyClassified, classificationErrors,test_error] = valid_digit(activationFunction, dActivationFunction, hiddenWeights, outputWeights, inputValues_test, labels_test, initial_data, initial_label);
            %[correctlyClassified, classificationErrors,test_error] = validateTwoLayerPerceptron(activationFunction, hiddenWeights, outputWeights, inputValues_test, labels_test);
            last_cnt = last_cnt + 28*28*10*10;
            cnt/(28*28*10)
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