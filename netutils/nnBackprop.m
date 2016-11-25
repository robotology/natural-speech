function [ net, trainErr, testErr] = nnBackprop( net, X, Y, parnet, varargin )
%NNBACKPROP Backpropagation training of the Neural Network
% IN
%   net: initial network
%   X: input matrix
%   Y: output matrix
%   parnet: net parameters
%   varargin{1}: input test matrix
%   varargin{2}: output test matrix

% OUT
%   net: final network
%   trainErr: first task train error over the epochs
%   testErr: first task test error over the epochs

maxN4Eval = 100000;

try % check gpu availability
    gd = gpuDevice();
    useGpu = true;
    fprintf(1, 'GPU available\n');
catch
    useGpu = false;
    fprintf(1, 'GPU not available\n');
end

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
    momentum = parnet.momentum;
    step = 0;    % last step taken
end
trainErr = cell(1,maxepoch);
if ~isempty(varargin)
    testX = varargin{1};
    testN = size(testX,1);
    testY = varargin{2};
    testErr = cell(1,maxepoch);
    testN = size(testX,1);
else
    testErr = {};
end

numWeights = (parnet.units(1:end-1)+1)*parnet.units(2:end)';
VV = zeros(numWeights,1,'single');
nl = length(parnet.activations);  % number of layers (hidden + output)
if(isfield(parnet,'constGradient') && parnet.constGradient)
    constVV = zeros(numWeights,1,'single');
end
start=1;
for il=1:nl % unroll weights
    VV(start:start+(size(net.w{il},1)+1)*size(net.w{il},2)-1) = reshape([net.w{il}; net.bias{il}],[],1);
    if(isfield(parnet,'constGradient') && parnet.constGradient)
        constVV(start:start+(size(net.w{il},1)+1)*size(net.w{il},2)-1) = reshape([net.wconst{il}; ones(size(net.bias{il}))],[],1);
    end
    start = start + (size(net.w{il},1)+1)*size(net.w{il},2);
end
if useGpu
    VV = gpuArray(VV);
    if(isfield(parnet,'constGradient') && parnet.constGradient)
        constVV = gpuArray(constVV);
    end
end
if(isfield(parnet,'constGradient') && parnet.constGradient)
    options.constVV = constVV;
    clear constVV;
end

numbatches = ceil(N/batchsize);

fprintf('%6s','Epoch')
switch parnet.cost
    case 'mse'
        fprintf('%15s','Train RMSE');
        if ~isempty(testErr)
            fprintf('%15s','Test RMSE');
        end
    case 'ce_logistic'
        fprintf('%15s','Train CE');
        if ~isempty(testErr)
            fprintf('%15s','Test CE');
        end            
    case 'ce_softmax'
        fprintf('%25s','Train #misclassified');
        if ~isempty(testErr)
            fprintf('%25s','Test #misclassified');
        end
        
    otherwise
        error('invalid cost: %s',parnet.cost);
end
fprintf('%11s\n','Time');

T0=tic;        
if(N > maxN4Eval)
    irps = randperm(N,maxN4Eval);
end
if (~isempty(testErr) && testN > maxN4Eval)
    tstirps = randperm(testN,maxN4Eval);
end

for epoch=1:maxepoch
    if strcmp(opt,'sgd')        
        eta = max([eta*etaDecay etamin]);
    end    
    rp = randperm(N);   % randomize data
    for batch=1:numbatches
        %fprintf(1,'Batch number %d\n',batch);
        i0 = (batch-1)*batchsize+1;
        i1 = min(batch*batchsize,N);
        batchdata = X([rp(i0:i1) rp(1:max(0,batch*batchsize-N))],:);    % fill the last batch with the first randomized examples
        batchoutput = Y([rp(i0:i1) rp(1:max(0,batch*batchsize-N))],:);
        if useGpu
            batchdata = gpuArray(batchdata);
            batchoutput = gpuArray(batchoutput);
        end
        switch(opt)
            case 'sgd'
                if(isfield(parnet,'constGradient') && parnet.constGradient)
                    [~,df] = nnCost(VV,batchdata,batchoutput,parnet,options);
                else
                    [~,df] = nnCost(VV,batchdata,batchoutput,parnet);
                end                                
                step = momentum*step - eta*df;
                VV = VV + step;
            case 'cgd'
                VV = minimize_pct(VV,'nnCost',3,batchdata,batchoutput,parnet);
            case 'lbfgs'
                %%Not working yet
                options.Method = 'lbfgs';
                options.maxIter = 400;
                options.display = 'on';
                VV = minFunc( @(VV) nnCost(VV,batchdata,batchoutput,parnet), VV, options);
        end
    end
    
    start=1;
    for il=1:nl   % reshape the unrolled weights
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
    
    %%Compute error
    if(N > maxN4Eval)        
        sX = X(irps,:);
        sY = Y(irps,:);
        trainOutput = nnFwd(net, sX, parnet);
    else
        trainOutput = nnFwd(net, X, parnet);
    end
    if ~isempty(testErr)
        if(testN > maxN4Eval)
            stestX = testX(tstirps,:);
            stestY = testY(tstirps,:);
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
        case 'ce_logistic'
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
%             if(N > maxN4Eval)                 
%                 trainErr{epoch} = - 1./maxN4Eval .* sum(sum(sY.*log(trainOutput) + (1-sY).*log(1-trainOutput)));
%             else
%                 trainErr{epoch} = - 1./N .* sum(sum(Y.*log(trainOutput) + (1-Y).*log(1-trainOutput)));                
%             end
%             fprintf('%15.5e',trainErr{epoch});
%             if ~isempty(testErr)
%                 if(testN > maxN4Eval)
%                     testErr{epoch} = - 1./maxN4Eval .* sum(sum(stestY.*log(testOutput) + (1-stestY).*log(1-testOutput)));
%                 else
%                     testErr{epoch} = - 1./testN .* sum(sum(testY.*log(testOutput) + (1-testY).*log(1-testOutput)));                    
%                 end
%                 fprintf('%15.5e',testErr{epoch});
%             end
        case 'ce_softmax'
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
    if(isfield(parnet,'earlystop') && parnet.earlystop)
        if(epoch > 7 && testErr{epoch} > testErr{epoch-2} && testErr{epoch-1} > testErr{epoch-2})
            net = pnet_2;
            fprintf(1,'Early stopped\n');
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


