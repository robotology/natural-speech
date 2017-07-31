function [ net, trainErr, testErr, trainErr2, testErr2] = nnBackprop_mtk3( net, X, Y, cY2, parnet, options)
% NNBACKPROP_MTK3 Backpropagation training of the Neural Network
% IN
%   net: initial network
%   X: input matrix
%   Y: output matrix
%   cY2: cell of outputs of secondary tasks
%   parnet: net parameters
%   options.valX
%   options.valY
%   options.valY2
%   options.task2ignore

% OUT
%   net: final network
%   trainErr: first task train error over the epochs
%   testErr: first task test error over the epochs
%   trainErr2: second tasks train errors over the epochs
%   testErr2: cell of second tasks test errors over the epochs

nTasks = length(parnet.multitask.ccost2); 
if(nTasks ~= length(parnet.multitask.cactivations2) || nTasks ~= length(cY2))
    error('An error occured with the number of secondary tasks considered\n');
end
%add the primary task in the count
nTasks = nTasks + 1;

% max number of samples for evaluation
maxN4Eval = 100000;

try % check gpu availability
    gd = gpuDevice();
    useGpu = true;
    fprintf(1, 'GPU available\n');
catch
    useGpu = false;
    fprintf(1, 'GPU not available\n');
end
bVerbose = 1;
N = size(X,1);
batchsize = parnet.batchsize;
maxepoch = parnet.maxepoch;
opt = parnet.optimization;
if strcmp(opt,'sgd')
    eta = parnet.learningRate; % initial learning rate
    etaDecay = parnet.learningRateDecay;    % learning rate decay parameter
    if(isfield(parnet,'learningRateMin'))
        etamin = parnet.learningRateMin;
    else
        etamin = -Inf;
    end
    if(isfield(parnet,'learningRateSchedule') && isempty(parnet.learningRateSchedule)==0)
        
        etaSchedule = parnet.learningRateSchedule;
        if(length(etaSchedule) ~= maxepoch)
            error('learningRateSchedule must have as many items as the number of epochs (i.e., maxepoch)\n');
        end
        eta = etaSchedule(1);
    end
    momentum = parnet.momentum;
    
end
trainErr = cell(1,maxepoch);
trainErr2 = cell(1,maxepoch);

testX = options.valX;
testN = size(testX,1);
testY = options.valY;
testY2 = options.valY2{1};
task2w0 = options.task2ignore;

testErr = cell(1,maxepoch);
testErr2 = cell(1,maxepoch);


numWeights = (parnet.units(1:end-2)+1)*parnet.units(2:end-1)';
VV = zeros(numWeights,1,'single');
nl = length(parnet.activations);  % number of layers (hidden + output)
start=1;
for il=1:nl-1 % unroll weights
    VV(start:start+(size(net.w{il},1)+1)*size(net.w{il},2)-1) = reshape([net.w{il}; net.bias{il}],[],1);
    start = start + (size(net.w{il},1)+1)*size(net.w{il},2);
    laststart = start;
end
step = zeros(laststart-1,1);    % last step taken

cY2 = [Y cY2];
clear Y;
lstep = cell(1,nTasks);
%lstep{1} = zeros((parnet.units(end-1)*parnet.units(end))+parnet.units(end),1);
for i=1:nTasks    
    lstep{i} = zeros((parnet.units(end-1)*size(cY2{i},2))+size(cY2{i},2),1);
end
if useGpu
    VV = gpuArray(VV);
    step = gpuArray(step);
    for i=1:nTasks
        lstep{i} = gpuArray(lstep{i});
    end
end

numbatches = ceil(N/batchsize);

fprintf('%6s','Epoch')
switch parnet.cost
    case 'mse'
        fprintf('%15s','Train RMSE');
        if ~isempty(testErr)
            fprintf('%15s','Test RMSE');
        end
        
    case 'ce_softmax'
        fprintf('%25s','Train #misclassified');
        if ~isempty(testErr)
            fprintf('%25s','Test #misclassified');
        end
        
    otherwise
        error('invalid cost: %s',parnet.cost);
end
if(bVerbose)
    fprintf('%16s','Train RMSE2');
    if ~isempty(testErr2)
        fprintf('%16s','Test RMSE2');
    end
end

fprintf('%11s\n','Time');

T0=tic;        
if(N > maxN4Eval)
    irps = randperm(N,maxN4Eval);
end
if (~isempty(testErr) && testN > maxN4Eval)
    tstirps = randperm(testN,maxN4Eval);
end
VVtop = cell(1,nTasks);
VVtop{1} = reshape([net.w{nl}; net.bias{nl}],[],1);
for itask=2:nTasks
    VVtop{itask} = reshape([net.auxw{itask-1}; net.auxbias{itask-1}],[],1);
end
if(useGpu)
    for itask=1:nTasks
        VVtop{itask} = gpuArray(VVtop{itask});
    end
end

mtkparnet = parnet;
mtkparnet.ccost{1} = parnet.cost;
mtkparnet.cactivations2{1} = parnet.activations{end};
for itask=2:nTasks     
    %mtkparnet.units(end) = size(cY2{itask},2);
    mtkparnet.ccost{itask} = parnet.multitask.ccost2{itask-1};
    mtkparnet.cactivations2{itask} = parnet.multitask.cactivations2{itask-1};    
end
parnet2 = parnet;
parnet2.units(end) = size(cY2{2},2);
parnet2.cost = parnet.multitask.ccost2{1};
parnet2.activations{end} = parnet.multitask.cactivations2{1};
for epoch=1:maxepoch
    if strcmp(opt,'sgd')
        if(isfield(parnet,'learningRateSchedule') && isempty(etaSchedule)==0)
            eta = etaSchedule(epoch);
        else
            %if(mod(epoch,nTasks)==0)
                eta = max([eta*etaDecay etamin]);
            %end
        end
    end
    mtkparnet.t2w = parnet.multitask.t2w;
    %parnet.multitask.t2w = max(parnet.multitask.t2w-0.1*(epoch>1),0);
    %fprintf(1,'Secondary task weigth %d\n',parnet.multitask.t2w);
    batchoutput = cell(1,nTasks);
    rp = randperm(N);   % randomize data
    for batch=1:numbatches
    %for batch=1:10    
        %fprintf(1,'Batch number %d\n',batch);        
        i0 = (batch-1)*batchsize+1;
        i1 = min(batch*batchsize,N);
        batchdata = X([rp(i0:i1) rp(1:max(0,batch*batchsize-N))],:);    % fill the last batch with the first randomized examples
        
        for itask=1:nTasks 
            batchoutput{itask} = cY2{itask}([rp(i0:i1) rp(1:max(0,batch*batchsize-N))],:);           
        end
        %VV = [VV(1:laststart-1);VVlast{itask+1}];
        
        if(isempty(task2w0)==0)
            task2w0batch = task2w0([rp(i0:i1) rp(1:max(0,batch*batchsize-N))]);
        end
        
        if useGpu
            batchdata = gpuArray(batchdata);
            for itask=1:nTasks
                batchoutput{itask} = gpuArray(batchoutput{itask});
            end                
            if(isempty(task2w0)==0)
                task2w0batch = gpuArray(task2w0batch);
            end
        end
        switch(opt)
            case 'sgd'
                if(isempty(task2w0))
                    [~,df,df2] = nnCost_mtk3(VV,VVtop,batchdata,batchoutput,mtkparnet);                                                                
                else
                    [~,df,df2] = nnCost_mtk3(VV,VVtop,batchdata,batchoutput,mtkparnet,task2w0batch);
                end
                step = momentum*step - eta*df;
                VV = VV + step;                 
                for itask=1:nTasks
                    lstep{itask} = momentum*lstep{itask} - eta*df2{itask};
                    VVtop{itask} = VVtop{itask} + lstep{itask};
                end                                                
             case 'cgd'
                error('cgd not working yet in multitask learning\n');
                VV = minimize_pct(VV,'nnCost',3,batchdata,batchoutput,mtkparnet);
            case 'lbfgs'
                %%Not working yet
                error('lbfgs not working yet\n');
                options.Method = 'lbfgs';
                options.maxIter = 400;
                options.display = 'on';
                VV = minFunc( @(VV) nnCost(VV,batchdata,batchoutput,mtkparnet), VV, options);
        end        
    end
        
    start=1;    
    for il=1:nl-1   % reshape the unrolled weights
        wAndBias = gather(reshape(VV(start:start+(parnet.units(il)+1)*parnet.units(il+1)-1), parnet.units(il)+1, parnet.units(il+1)));
        if parnet.dropout==0
            net.w{il} = wAndBias(1:end-1,:);
        else    % rescale the weights wrt dropout
            if il>1
                net.w{il} = parnet.dropout * wAndBias(1:end-1,:);
            else
                net.w{il} = 0.8 * wAndBias(1:end-1,:);
            end
        end
        net.bias{il} = wAndBias(end,:);
        start = start + (parnet.units(il)+1)*parnet.units(il+1);
    end
    wAndBias = gather(reshape(VVtop{1}, parnet.units(nl)+1, size(cY2{1},2)));
    if parnet.dropout==0
        net.w{nl} = wAndBias(1:end-1,:);
    else    % rescale the weights wrt dropout               
        net.w{nl} = parnet.dropout * wAndBias(1:end-1,:);        
    end
    net.bias{nl} = wAndBias(end,:);
    %%Compute error
    if(N > maxN4Eval)
        irps = randperm(N,maxN4Eval);
        sX = X(irps,:);
        sY = cY2{1}(irps,:);
        trainOutput = nnFwd(net, sX, parnet);
    else
        trainOutput = nnFwd(net, X, parnet);
    end
    if ~isempty(testErr)
        if(testN > maxN4Eval)
            irps = randperm(testN,maxN4Eval);
            stestX = testX(irps,:);
            stestY = testY(irps,:);
            testOutput = nnFwd(net, stestX, parnet);
        else
            testOutput = nnFwd(net, testX, parnet);
        end
    end
    fprintf('%6i',epoch);
    switch parnet.cost
        case 'mse'
            % using RMSE
            if(N > maxN4Eval)
                trainErr{epoch} = sqrt(mean((trainOutput(:)-sY(:)).^2));
            else
                trainErr{epoch} = sqrt(mean((trainOutput(:)-Y(:)).^2));                
            end
            fprintf('%15.5e',trainErr{epoch});
            if ~isempty(testErr)
                if(testN > maxN4Eval)
                    testErr{epoch} = sqrt(mean((testOutput(:)-stestY(:)).^2));
                else
                    testErr{epoch} = sqrt(mean((testOutput(:)-testY(:)).^2));                    
                end
                fprintf('%15.5e',testErr{epoch});
            end
        case 'ce_softmax'
            % misclassification error
             % misclassification error
            [~, predClass] = max(trainOutput,[],2);
            clear trainOutput;
            if(N > maxN4Eval)
                [~, class] = max(sY,[],2);
                trainErr{epoch} = maxN4Eval-sum(predClass==class);
                fprintf('%25s',[int2str(trainErr{epoch}) '/' int2str(maxN4Eval) ' (' num2str(100*trainErr{epoch}/maxN4Eval,'%1.2f') '%)']);
            else
                [~, class] = max(Y,[],2);
                trainErr{epoch} = N-sum(predClass==class);
                fprintf('%25s',[int2str(trainErr{epoch}) '/' int2str(N) ' (' num2str(100*trainErr{epoch}/N,'%1.2f') '%)']);
            end
            if ~isempty(testErr)
                [~, predClass] = max(testOutput,[],2);
                clear testOutput;
                if(testN > maxN4Eval)
                    [~, class] = max(stestY,[],2);
                    testErr{epoch} = maxN4Eval-sum(predClass==class);
                    fprintf('%25s',[int2str(testErr{epoch}) '/' int2str(maxN4Eval) ' (' num2str(100*testErr{epoch}/maxN4Eval,'%1.2f') '%)']);
                else
                    [~, class] = max(testY,[],2);
                    testErr{epoch} = testN-sum(predClass==class);
                    fprintf('%25s',[int2str(testErr{epoch}) '/' int2str(testN) ' (' num2str(100*testErr{epoch}/testN,'%1.2f') '%)']);
                end
            end
        otherwise
            error('Invalid cost: %s',parnet.cost);
    end
    %% Compute second task error
     if(bVerbose && mtkparnet.t2w > 0)
        wtop1 = net.w{nl};
        btop1 = net.bias{nl};
        wAndBias = gather(reshape(VVtop{2}, parnet2.units(nl)+1, parnet2.units(nl+1)));
        if parnet2.dropout==0
            net.w{nl} = wAndBias(1:end-1,:);
        else    % rescale the weights wrt dropout               
            net.w{nl} = parnet2.dropout * wAndBias(1:end-1,:);        
        end
        net.bias{nl} = wAndBias(end,:);
        Ne = min([N maxN4Eval]);

        trainOutput2 = nnFwd(net, X(1:Ne,:), parnet2);

        if ~isempty(testErr2)
            testNe = min([maxN4Eval size(testX,1)]);
            testOutput2 = nnFwd(net, testX(1:testNe,:), parnet2);
        end
        sY2 = cY2{2}(1:Ne,:);
        % using RMSE
        if(strcmp(parnet.multitask.ccost2{1},'mse'))            
            trainErr2{epoch} = sqrt(mean((trainOutput2(:)-sY2(:)).^2));
            fprintf('%15.5e',trainErr2{epoch});
            if ~isempty(testErr2)
                stestY2 = testY2(1:testNe,:);
                testErr2{epoch} = sqrt(mean((testOutput2(:)-stestY2(:)).^2));
                fprintf('%15.5e',testErr2{epoch});
            end
%         else
%             error ('At present only monitoring mse secondary cost\n');
        elseif(strcmp(parnet.multitask.ccost2{1},'ce_softmax'))            
            [~, predClass] = max(trainOutput2,[],2);
            clear trainOutput2;
            [~, class] = max(sY2,[],2);
            trainErr2{epoch} = Ne-sum(predClass==class);
            fprintf('%25s',[int2str(trainErr2{epoch}) '/' int2str(Ne) ' (' num2str(100*trainErr2{epoch}/Ne,'%1.2f') '%)']);            
            if ~isempty(testErr2)
                stestY2 = testY2(1:testNe,:);
                [~, predClass] = max(testOutput2,[],2);
                clear testOutput2;                
                [~, class] = max(stestY2,[],2);
                testErr2{epoch} = maxN4Eval-sum(predClass==class);
                fprintf('%25s',[int2str(testErr2{epoch}) '/' int2str(Ne) ' (' num2str(100*testErr2{epoch}/Ne,'%1.2f') '%)']);                
            end
        end
        net.w{nl} = wtop1;
        net.bias{nl} = btop1;
    end
    if(isfield(parnet,'earlystop') && parnet.earlystop)
        if(epoch > 2 && testErr{epoch} > testErr{epoch-2} && testErr{epoch-1} > testErr{epoch-2})
            net = pnet_2;
            fprintf(1,'\nEarly stopped\n');
            break;
        end
        if(epoch > 2)
           pnet_2 = pnet_1;            
        end
        pnet_1 = net;
    end
    T=toc(T0);
    fprintf('%11.3f\n',T);
end
    
end


