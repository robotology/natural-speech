function [ f, df ] = nnCost( VV, X, Y, parnet )
%NNCOST Computes the cost function and its gradient using backpropagation
% IN
%   VV: unrolled vector of weights
%   X: input matrix
%   Y: output matrix
%   parnet: net parameters
% OUT
%   f: cost function
%   df: gradient

nl = length(parnet.activations);  % number of layers (hidden + output)
N = size(X,1);   % number of cases
lambda = parnet.weightDecay;   % weight decay parameter
dropout = parnet.dropout;

w=cell(1,nl);
start=1;
for il=1:nl   % reshape the unrolled weights
    w{il} = reshape(VV(start:start+(parnet.units(il)+1)*parnet.units(il+1)-1), parnet.units(il)+1, parnet.units(il+1));
    start = start + (parnet.units(il)+1)*parnet.units(il+1);
end

probs = cell(1,nl+1);
probs{1} = X;

%% Forward pass
for il=1:nl
    if dropout>0
        if(strcmp(parnet.activations,'relu')==0)
            error('At present dropout only works with relu units\n');
        end
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
        case 'softmax'
            probs{il+1} = bsxfun(@minus, probs{il+1}, max(probs{il+1}, [], 2));  % wlog
            probs{il+1} = exp(probs{il+1});
            probs{il+1} = bsxfun(@rdivide, probs{il+1}, sum(probs{il+1},2));
        otherwise
            error('Invalid activation for layer %i: %s', il, parnet.activations{il});
    end
end

%% Backpropagation
delta = cell(1,nl);
dw = cell(1,nl);
switch parnet.cost
    case 'mse'
        f = 1/N * 1/2 * sum(sum((probs{nl+1}-Y).^2));
        delta{nl} = 1/N * (probs{nl+1}-Y);
        switch parnet.activations{end}
            case 'linear'
                %nop
            case 'sigm'
                delta{nl} = delta{nl} .* probs{nl+1} .* (1-probs{nl+1});
            case 'tanh'
                delta{nl} = delta{nl} .* (1-probs{nl+1}.^2);
            case 'relu'
                delta{nl} = delta{nl} .* (probs{nl+1}>0);
            otherwise
                error('Invalid activation for the output layer with mse cost: %s',parnet.activations{end});
        end
    case 'ce_logistic'
        if strcmp(parnet.activations{end},'sigm')
            f = - 1/N * sum(sum(Y.*log(probs{nl+1}) + (1-Y).*log(1-probs{nl+1})));
            delta{nl} = 1/N * (probs{nl+1}-Y);
        else
            error('ce_logistic cost requires a sigmoid output layer');
        end
    case 'ce_softmax'
        if strcmp(parnet.activations{end},'softmax')
            probs{nl+1}(probs{nl+1}==0) = eps;    % anti-log(0)
            f = - 1/N * sum(sum(Y.*log(probs{nl+1})));
            delta{nl} = 1/N * (probs{nl+1}-Y);
        else
            error('ce_softmax cost requires a softmax output layer');
        end
    otherwise
        error('Invalid cost function: %s',parnet.cost);
end

f = f + lambda/2 * sum(VV.^2);  % weight decay

dw{nl} = [probs{nl} ones(N,1)]' * delta{nl};

for il=nl-1:-1:1
    delta{il} = delta{il+1} * w{il+1}(1:end-1,:)';
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
for il=1:nl % unroll the derivatives
    df(start:start+length(dw{il}(:))-1) = dw{il}(:);
    start = start + length(dw{il}(:));
end

df = df + lambda * VV; % weight decay derivative

end