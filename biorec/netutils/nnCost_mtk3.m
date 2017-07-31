function [ f, df, df2 ] = nnCost_mtk3( VV, cVV2, X, cY, parnet, varargin )
% NNCOST_MTK3 Computes the cost function and its gradient using backpropagation
% IN
%   VV: unrolled vector of weights
%   cVV2: urolled vector of weights used specifically for the secondary
%           target
%   X: input matrix
%   Y: output matrix for primary target
%   cY: outputmatrix for secondary target(s)    
%   parnet: net parameters
%   varargin{1}: array indicating examples where task2 must not be
%   performed
% OUT
%   f: cost function
%   df: gradient
%   df2: part of gradient due to the secondary target(s)

nl = length(parnet.activations);  % number of layers (hidden + output)
N = size(X,1);   % number of cases
lambda = parnet.weightDecay;   % weight decay parameter
dropout = parnet.dropout;
if(isfield(parnet,'vsparse'))
    vsparse = parnet.vsparse;
else
    vsparse = [];
end
    
nTasks = length(cY);

if(length(varargin)>0)
    t2w0 = varargin{1};
    rN = N - sum(t2w0);
    ignoreTargets = 1;
else
    %t2w0 = zeros(N,1);
    ignoreTargets = 0;
end

w=cell(1,nl-1);
start=1;
for il=1:nl-1   % reshape the unrolled weights
    w{il} = reshape(VV(start:start+(parnet.units(il)+1)*parnet.units(il+1)-1), parnet.units(il)+1, parnet.units(il+1));
    start = start + (parnet.units(il)+1)*parnet.units(il+1);
end

w2=cell(1,nTasks);
for itask=1:nTasks
    w2{itask} = reshape(cVV2{itask}, parnet.units(nl)+1, size(cY{itask},2));
end

probs = cell(1,nl);
probs{1} = X;
%% Forward pass
for il=1:nl-1
    if dropout>0
        if il>1
            p = dropout;
        else
            p = 0.8;
        end
        if isa(X,'gpuArray')    % if using GPU
            probs{il} = probs{il} .* (p > gpuArray.rand(size(probs{il})));
        else
            probs{il} = probs{il} .* (p > rand(size(probs{il})));
        end
    end
    probs{il+1} = [probs{il} ones(N,1)] * w{il};
    switch parnet.activations{il}
        case 'linear'
            %nop
        case 'sigm'
            probs{il+1} = 1 ./ (1 + exp(-probs{il+1}));
        case 'tanh'
            probs{il+1} = tanh(probs{il+1});
        case 'relu'
            probs{il+1} = 0.5 * (probs{il+1} + abs(probs{il+1}));
%         case 'softmax'
%             probs{il+1} = bsxfun(@minus, probs{il+1}, max(probs{il+1}, [], 2));  % wlog
%             probs{il+1} = exp(probs{il+1});
%             probs{il+1} = bsxfun(@rdivide, probs{il+1}, sum(probs{il+1},2));
        otherwise
            error('Invalid activation for layer %i: %s', il, parnet.activations{il});
    end    
end
probs2 = cell(1,nTasks);
for itask=1:nTasks            
    probs2{itask} = [probs{end} ones(N,1)] * w2{itask};
    switch parnet.cactivations2{itask}
        case 'linear'
            %nop
        case 'sigm'
            probs2{itask} = 1 ./ (1 + exp(-probs2{itask}));
        case 'tanh'
            probs2{itask} = tanh(probs2{itask});
        case 'relu'
            probs2{itask} = 0.5 * (probs2{itask} + abs(probs2{itask}));
        case 'softmax'
            probs2{itask} = bsxfun(@minus, probs2{itask}, max(probs2{itask}, [], 2));  % wlog
            probs2{itask} = exp(probs2{itask});
            probs2{itask} = bsxfun(@rdivide, probs2{itask}, sum(probs2{itask},2));
        otherwise
            error('Invalid activation for task %i: %s', itask, parnet.cactivations2{itask});
    end    
end
%% Backpropagation
delta2 = cell(1,nTasks);
dw2 = cell(1,nTasks);
delta = cell(1,nl-1);
dw = cell(1,nl-1);
%cf = zeros(1,nTasks);
for itask=1:nTasks
    if(itask == 1)
        t2w = 1;
    else
        t2w = parnet.t2w;
    end
    eN = N;
    res = probs2{itask}-cY{itask};
    if(ignoreTargets && itask > 1)
        res(t2w0,:) = res(t2w0,:) .* 0;
        eN = rN;
    end
    switch parnet.ccost{itask}        
        case 'mse'            
            cf(itask) = 1/eN * 1/2 * t2w *sum(sum(res.^2)); 
            delta2{itask} = 1/eN * t2w * res;
            switch parnet.cactivations2{itask}
                case 'linear'
                    %nop
                case 'sigm'
                    delta2{itask} = delta2{itask} .* probs2{itask} .* (1-probs2{itask});
                case 'tanh'
                    delta2{itask} = delta2{itask} .* (1-probs2{itask}.^2);
                case 'relu'
                    delta2{itask} = delta2{itask} .* (probs2{itask}>0);
                otherwise
                    error('Invalid activation for the output layer with mse cost: %s',parnet.cactivations2{itask});
            end
        case 'ce_logistic'
            if strcmp(parnet.cactivations2{itask},'sigm')
                probs2{itask}(probs2{itask}==0) = eps;    % anti-log(0)
                res1 = cY{itask}.*log(probs2{itask}) + (1-cY{itask}).*log(1-probs2{itask});
                if(ignoreTargets && itask > 1)
                    res1(t2w0,:) = res1(t2w0,:) .* 0;
                end
                cf(itask) = - 1/eN * t2w * sum(sum(res1));            
                delta2{itask} = 1/eN * t2w * res;
               
            else
                error('ce_logistic cost requires a sigmoid output layer');
            end
        case 'ce_softmax'
            if strcmp(parnet.cactivations2{itask},'softmax')
                probs2{itask}(probs2{itask}==0) = eps;    % anti-log(0)
                cf(itask) = - 1/N * t2w * sum(sum(cY{itask}.*log(probs2{itask})));            
                delta2{itask} = 1/N * t2w * (probs2{itask}-cY{itask});
            else
                error('ce_softmax cost requires a softmax output layer');
            end
        otherwise
            error('Invalid cost function: %s',parnet.multitask.ccost2{itask});
    end
end

f = lambda/2 * (sum(VV.^2));
for itask = 1:nTasks
    f = f + cf(itask) + lambda/2 * sum(cVV2{itask}.^2);  % weight decay
    dw2{itask} = [probs{nl} ones(N,1)]' * delta2{itask};
end


for il=nl-1:-1:1
    if(il==nl-1)
        delta{il} = zeros(N,parnet.units(end-1),class(VV));
        for itask=1:nTasks
            delta{il} = delta{il} + delta2{itask} * w2{itask}(1:end-1,:)';
        end
    else
        delta{il} = delta{il+1} * w{il+1}(1:end-1,:)';    
    end
    if(isempty(vsparse)==0 && vsparse(il)~=0)
        if(strcmp(parnet.activations{il},'relu'))
            res = probs{il+1} - zeros(size(probs{il+1}),class(VV));
            f = f + 1/N * 1/2 * vsparse(il) *sum(sum(res.^2)); 
            delta{il} = delta{il} + 1/N * vsparse(il) * res;
        else
            error('Sparsity for activations different from relu not implemented yet\n');
        end
    end
        
    switch parnet.activations{il}
        case 'linear'
            %nop
        case 'sigm'
            delta{il} = delta{il} .* probs{il+1} .* (1-probs{il+1});
        case 'tanh'
            delta{il} = delta{il} .* (1-probs{il+1}.^2);
        case 'relu'
            delta{il} = delta{il} .* (probs{il+1}>0);
        otherwise
            error('Invalid activation for the %i-th hidden layer: %s',il,parnet.activations{il});
    end
    dw{il} = [probs{il} ones(N,1)]' * delta{il};
end

df = zeros(size(VV),class(VV));    % same size and type of VV

start=1;
for il=1:nl-1 % unroll the derivatives
    df(start:start+length(dw{il}(:))-1) = dw{il}(:);
    start = start + length(dw{il}(:));
end
df = df + lambda * VV; % weight decay derivative

df2 = cell(1,nTasks);
for itask=1:nTasks
    df2{itask} = dw2{itask}(:);
    df2{itask} = df2{itask} + lambda * cVV2{itask}; 
end
end