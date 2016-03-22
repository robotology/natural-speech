function [ net, trainErr, testErr ] = nnBackprop( net, X, Y, parnet, varargin )
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
%   trainErr: train error over the epochs
%   testErr: test error over the epochs

try % check gpu availability
    gd = gpuDevice();
    useGpu = true;
catch
    useGpu = false;
end

N = size(X,1);
batchsize = parnet.batchsize;
maxepoch = parnet.maxepoch;
opt = parnet.optimization;
if strcmp(opt,'sgd')
    eta = parnet.learningRate; % initial learning rate
    etaDecay = parnet.learningRateDecay;    % learning rate decay parameter
    momentum = parnet.momentum;
    step = 0;    % last step taken
end
trainErr = cell(1,maxepoch);
if ~isempty(varargin)
    testX = varargin{1};
    testN = size(testX,1);
    testY = varargin{2};
    testErr = cell(1,maxepoch);
else
    testErr = {};
end

numWeights = (parnet.units(1:end-1)+1)*parnet.units(2:end)';
VV = zeros(numWeights,1,'single');
nl = length(parnet.activations);  % number of layers (hidden + output)
start=1;
for il=1:nl % unroll weights
    VV(start:start+(size(net.w{il},1)+1)*size(net.w{il},2)-1) = reshape([net.w{il}; net.bias{il}],[],1);
    start = start + (size(net.w{il},1)+1)*size(net.w{il},2);
end
if useGpu
    VV = gpuArray(VV);
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
fprintf('%11s\n','Time');

T0=tic;        

for epoch=1:maxepoch
    if strcmp(opt,'sgd')
        eta = eta * etaDecay;
    end
    
    rp = randperm(N);   % randomize data
    for batch=1:numbatches
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
                [~,df] = nnCost(VV,batchdata,batchoutput,parnet);
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
    trainOutput = nnFwd(net, X, parnet);
    if ~isempty(testErr)
        testOutput = nnFwd(net, testX, parnet);
    end
    fprintf('%6i',epoch);
    switch parnet.cost
        case 'mse'
            % using RMSE
            trainErr{epoch} = sqrt(mean((trainOutput(:)-Y(:)).^2));
            fprintf('%15.5e',trainErr{epoch});
            if ~isempty(testErr)
                testErr{epoch} = sqrt(mean((testOutput(:)-testY(:)).^2));
                fprintf('%15.5e',testErr{epoch});
            end
        case 'ce_softmax'
            % misclassification error
            [~, predClass] = max(trainOutput,[],2);
            [~, class] = max(Y,[],2);
            trainErr{epoch} = N-sum(predClass==class);
            fprintf('%25s',[int2str(trainErr{epoch}) '/' int2str(N) ' (' num2str(100*trainErr{epoch}/N,'%1.2f') '%)']);
            if ~isempty(testErr)
                [~, predClass] = max(testOutput,[],2);
                [~, class] = max(testY,[],2);
                testErr{epoch} = testN-sum(predClass==class);
                fprintf('%25s',[int2str(testErr{epoch}) '/' int2str(testN) ' (' num2str(100*testErr{epoch}/testN,'%1.2f') '%)']);
            end
        otherwise
            error('Invalid cost: %s',parnet.cost);
    end
    T=toc(T0);
    fprintf('%11.3f\n',T);
end
    
end


