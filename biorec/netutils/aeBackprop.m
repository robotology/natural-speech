function [ net, trainErr, testErr ] = aeBackprop( net, X, parnet, varargin )
%AEBACKPROP Backpropagation training of the AutoEncoder
% IN
%   net: initial network
%   X: input train matrix
%   parnet: net parameters
%   varargin{1}: input test matrix
%   varargin{2}: train labels
%   varargin{3}: 
%   varargin{4}: 
% OUT
%   net: final network
%   trainErr: train error over the epochs
%   testErr: test error over the epochs

maxepoch = parnet.maxepoch;
N = size(X,1);
trainErr = cell(1,maxepoch);
if length(varargin)>=1 && ~isempty(varargin{1})
    testX = varargin{1};
    testErr = cell(1,maxepoch);
else
    testErr = {};
end
if parnet.shid || parnet.pairsim
    if length(varargin)>=2 && ~isempty(varargin{2}) % supervised case
        trainLabels = varargin{2};
        % for each frame, compute the first and last consecutive frame with the same label
        firstIdx = zeros(N,1);
        lastIdx = zeros(N,1);
        first = 1;
        last = 1;
        while last<N
            firstIdx(last) = first;
            if trainLabels(last+1) ~= trainLabels(last)
                lastIdx(first:last) = last;
                first = last + 1;
            end
            last = last + 1;
        end
        firstIdx(N) = first;
        lastIdx(first:N) = N;
    else    % unsupervised case
        % (trick so that each frame is replaced by the following one, and the last by the previous one)
        firstIdx = [2:N N-1]';
        lastIdx = [2:N N-1]';
    end
end
aehmm = 0;
if isfield(parnet,'aehmm')
    if parnet.aehmm
        aehmm = 1;
        if length(varargin)<4 || isempty(varargin{3}) || isempty(varargin{4})
            error('Missing arguments for aehmm');
        end
        nhlErr = cell(1,maxepoch);
    end
end

nhl = length(parnet.activations)/2;
parEncNet.units = parnet.units(1:nhl+1);
parEncNet.activations = parnet.activations(1:nhl);
parDecNet.units = parnet.units(nhl+1:end);
parDecNet.activations = parnet.activations(nhl+1:end);

try % check gpu availability
    gd = gpuDevice();
    useGpu = true;
catch
    useGpu = false;
end

batchsize = parnet.batchsize;
if strcmp(parnet.optimization,'sgd')
    eta = parnet.learningRate; % initial learning rate
    etaDecay = parnet.learningRateDecay;    % learning rate decay parameter
    momentum = parnet.momentum;
    step = 0;    % last step taken
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

if parnet.forceBin==1
    deterministicNoise = sqrt(parnet.noiseVariance)*randn(N,parnet.units(nl/2));
end

numbatches = ceil(N/batchsize);

if ~strcmp(parnet.cost,'mse')
    error('invalid cost: %s',parnet.cost);
end
fprintf('%6s%15s','Epoch','Train RMSE');
if ~isempty(testErr)
    fprintf('%15s','Test RMSE');
end
if aehmm && strcmp(varargin{4},'enc')
    fprintf('%15s','X-entropy');
end
fprintf('%11s\n','Time');

Y = X;
if ~isempty(testErr)
    testY = testX;
end

varargcost = {};

T0=tic;

for epoch=1:maxepoch
    if strcmp(parnet.optimization,'sgd')
        eta = eta * etaDecay;
    end
    
    rp = randperm(N);   % randomize data
    
    for batch=1:numbatches
        i0 = (batch-1)*batchsize+1;
        i1 = min(batch*batchsize,N);
        batchIdxs = [rp(i0:i1) rp(1:max(0,batch*batchsize-N))]; % fill the last batch with the first randomized examples
        batchdata = X(batchIdxs,:);
        batchoutput = Y(batchIdxs,:);
        if parnet.pairsim
            type = 'long';  % choose among all the frames with the same label
            %type = 'short'; % choose among the adjacent frames with the same label
            if strcmp(type,'short')||length(varargin)<2||isempty(varargin{2})   % if 'short' or unsupervised case
                for i=1:batchsize
                    if rand < 0.33
                        ori = batchIdxs(i);
                        newi = firstIdx(ori) + randi(lastIdx(ori)-firstIdx(ori)+1) - 1;
                        batchdata(i,:) = X(newi,:);
                    end
                end
            elseif strcmp(type,'long')
                for i=1:batchsize
                    if rand < 0.33
                        ori = batchIdxs(i);
                        sameLabelIdxs = find(trainLabels==trainLabels(ori));
                        batchdata(i,:) = X(sameLabelIdxs(randi(length(sameLabelIdxs))),:);
                    end
                end
            end
        end
        if parnet.noisy
            batchdata = batchdata + 0.4 * randn(size(batchdata));
        end
        if useGpu
            batchdata = gpuArray(batchdata);
            batchoutput = gpuArray(batchoutput);
        end
        if parnet.shid
            batchdata2 = zeros(size(batchdata),'single');
            for i=1:batchsize
                ori = batchIdxs(i);
                if lastIdx(ori)>ori
                    newi = ori + 1;
                elseif firstIdx(ori)<ori
                    newi = ori - 1;
                else
                    newi = ori;
                end
                batchdata2(i,:) = X(newi,:);
            end
            if useGpu
                batchdata2 = gpuArray(batchdata2);
            end
            varargcost{1} = batchdata2;
        end
        if parnet.forceBin==1
            batchnoise = deterministicNoise(batchIdxs,:);
            if useGpu
                batchnoise = gpuArray(batchnoise);
            end
            varargcost{2} = batchnoise;
        end
        
        switch(parnet.optimization)
            case 'sgd'
                [~,df] = aeCost(VV,batchdata,batchoutput,parnet,varargcost{:});
                step = momentum*step - eta*df;
                VV = VV + step;
            case 'cgd'
                VV = minimize_pct(VV,'aeCost',3,batchdata,batchoutput,parnet,varargcost{:});
            case 'lbfgs'
                %%Not working yet
                options.Method = 'lbfgs';
                options.maxIter = 400;
                options.display = 'on';
                VV = minFunc( @(VV) aeCost(VV,batchdata,batchoutput,parnet,varargcost{:}), VV, options);
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
    
    % compute error
    encNet = net;
    encNet.w = net.w(1:nhl);
    encNet.bias = net.bias(1:nhl);
    decNet = net;
    decNet.w = net.w(nhl+1:end);
    decNet.bias = net.bias(nhl+1:end);
    
    trainEnc = nnFwd(encNet, X, parEncNet);
    trainOutput = nnFwd(decNet, trainEnc, parDecNet);
    
    if ~isempty(testErr)
        testOutput = nnFwd(net, testX, parnet);
    end
    
    fprintf('%6i',epoch);
    trainErr{epoch} = sqrt(mean((trainOutput(:)-Y(:)).^2));
    fprintf('%15.5e',trainErr{epoch});
    if ~isempty(testErr)
        testErr{epoch} = sqrt(mean((testOutput(:)-testY(:)).^2));
        fprintf('%15.5e',testErr{epoch});
    end
    if aehmm && strcmp(varargin{4},'enc')
        pq = varargin{3};
        nhlErr{epoch} = -1./(size(pq,1))*sum(sum(pq.*log(trainEnc)+(1-pq).*log(1-trainEnc)));
        fprintf('%15.5e',nhlErr{epoch});
    end
    T=toc(T0);
    fprintf('%11.3f\n',T);
end
    
end