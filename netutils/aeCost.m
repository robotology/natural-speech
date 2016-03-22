function [ f, df ] = aeCost( VV, X, Y, parnet, varargin )
%AECOST Computes the AE cost function and its gradient using backpropagation
% IN
%   VV: unrolled vector of weights
%   X: input matrix
%   Y: output matrix
%   parnet: net parameters
% OUT
%   f: cost function
%   df: gradient

nl = length(parnet.activations);  % number of layers (hidden + output)
iel = nl/2;  % index of the encoding layer
N = size(X,1);   % number of cases
lambdaWD = parnet.weightDecay;   % weight decay parameter
dropout = parnet.dropout;
if parnet.shid
    if length(varargin)>=1 && ~isempty(varargin{1})
        X2 = varargin{1};
    else
        error('X2 required for Segmental AE');
    end
    lambdaSCAE = 0.1;
end
sparse = parnet.sparsepen > 0;
if sparse
    beta = parnet.sparsepen;
    rho = parnet.sparsepar;
end
binarize = parnet.binarize;
forceBin = parnet.forceBin;
if forceBin==1
    if length(varargin)>=2 && ~isempty(varargin{2})
        batchnoise = varargin{2};
    else
        error('Noise required for forceBin=1');
    end
elseif forceBin==2
    gamma = parnet.entropyWeight;
end

if isa(X,'gpuArray')
    useGpu = true;
else
    useGpu = false;
end

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
        if il>1
            p = dropout;
        else
            p = 0.8;
        end
        if useGpu
            probs{il} = probs{il} .* (p > gpuArray.rand(size(probs{il})));
        else
            probs{il} = probs{il} .* (p > rand(size(probs{il})));
        end
    end
    probs{il+1} = [probs{il} ones(N,1)] * w{il};
    if il==iel  % encoding layer
        if forceBin==1
            probs{il+1} = probs{il+1} + batchnoise;
        end
    end
    switch parnet.activations{il}
        case 'linear'
            %nop
        case 'sigm'
            probs{il+1} = 1 ./ (1 + exp(-probs{il+1}));
        otherwise
            error('Invalid activation for layer %i: %s', il, parnet.activations{il});
    end
    if il==iel  % encoding layer
        if sparse
            rhos = mean(probs{il+1},1);
        end
        if forceBin==2 || binarize>0
            encActivations = probs{il+1};
            if binarize==1
                probs{il+1} = probs{il+1} > 0.5;
            elseif binarize==2
                if useGpu
                    probs{il+1} = probs{il+1} > gpuArray.rand(size(probs{il+1}));
                else
                    probs{il+1} = probs{il+1} > rand(size(probs{il+1}));
                end
            end
        end
    end
end
if parnet.shid
    probs2 = cell(1,iel+1);
    probs2{1} = X2;
    for il=1:iel
        probs2{il+1} = [probs2{il} ones(N,1)] * w{il};
        switch parnet.activations{il}
            case 'sigm'
                probs2{il+1} = 1 ./ (1 + exp(-probs2{il+1}));
            otherwise
                error('Invalid activation for layer %i: %s', il, parnet.activations{il});
        end
    end
end

if binarize>0
    probs{iel+1} = encActivations;
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
            otherwise
                error('Invalid activation for the output layer with mse cost: %s',parnet.activations{end});
        end
    otherwise
        error('Invalid cost function: %s',parnet.cost);
end

f = f + lambdaWD/2 * sum(VV.^2);  % weight decay
if sparse   % sparsity
    rhos(rhos<eps) = eps;
    rhos(rhos>1-eps) = 1-eps;
    f = f + beta*sum(rho*log(rho./rhos)+(1-rho)*log((1-rho)./(1-rhos)));
end
if forceBin==2  % cost term due to activations far from 0 or 1
    encActivations(encActivations<eps) = eps;
    encActivations(encActivations>1-eps) = 1-eps;
    f = f - gamma/N*sum(sum(encActivations.*log(encActivations)+(1-encActivations).*log(1-encActivations)));
end

dw{nl} = [probs{nl} ones(N,1)]' * delta{nl};

for il=nl-1:-1:1
    delta{il} = delta{il+1} * w{il+1}(1:end-1,:)';
    if il==iel
        if sparse
            delta{il} = bsxfun(@plus, delta{il}, beta/N*(-(rho./rhos)+(1-rho)./(1-rhos)));
        end
        if forceBin==2
            delta{il} = delta{il} + gamma/N*(-log(encActivations)+log(1-encActivations));
        end
    end
    switch parnet.activations{il}
        case 'linear'
            %nop
        case 'sigm'
            delta{il} = delta{il} .* probs{il+1} .* (1-probs{il+1});
        otherwise
            error('Invalid activation for the %i-th hidden layer: %s',il,parnet.activations{il});
    end
    dw{il} = [probs{il} ones(N,1)]' * delta{il};
end
if parnet.shid
    hprobs = probs{iel+1};
    hprobs2 = probs2{iel+1};
    hprobs(hprobs<eps) = eps;
    hprobs(hprobs>1-eps) = 1-eps;
    hprobs2(hprobs2<eps) = eps;
    hprobs2(hprobs2>1-eps) = 1-eps;
    
    f = f - lambdaSCAE/N*sum(sum(hprobs.*log(hprobs2) + (1-hprobs).*log((1-hprobs2))));
    
    delta1 = cell(1,iel);
    dw1 = cell(1,iel);
    delta2 = cell(1,iel);
    dw2 = cell(1,iel);
    
    delta1{iel} = lambdaSCAE/N*((hprobs.*(1-hprobs)).*log((1-hprobs2)./hprobs2));
    delta2{iel} = lambdaSCAE/N*(hprobs2 - hprobs);  %%(segno opposto rispetto al pdf)
    
    dw1{iel} = [probs{iel} ones(N,1)]' * delta1{iel};
    for il=iel-1:-1:1
        delta1{il} = delta1{il+1} * w{il+1}(1:end-1,:)';
        switch parnet.activations{il}
            case 'sigm'
                delta1{il} = delta1{il} .* probs{il+1} .* (1-probs{il+1});
            otherwise
                error('Invalid activation for the %i-th hidden layer: %s',il,parnet.activations{il});
        end
        dw1{il} = [probs{il} ones(N,1)]' * delta1{il};
    end
    
    dw2{iel} = [probs{iel} ones(N,1)]' * delta2{iel};
    for il=iel-1:-1:1
        delta2{il} = delta2{il+1} * w{il+1}(1:end-1,:)';
        switch parnet.activations{il}
            case 'sigm'
                delta2{il} = delta2{il} .* probs2{il+1} .* (1-probs2{il+1});
            otherwise
                error('Invalid activation for the %i-th hidden layer: %s',il,parnet.activations{il});
        end
        dw2{il} = [probs2{il} ones(N,1)]' * delta2{il};
    end
    
    for ihl=1:iel            
        dw{ihl} = dw{ihl} + dw1{ihl} + dw2{ihl};
    end
end

df = zeros(size(VV),class(VV));    % same size and type of VV
start=1;
for il=1:nl % unroll the derivatives
    df(start:start+length(dw{il}(:))-1) = dw{il}(:);
    start = start + length(dw{il}(:));
end

df = df + lambdaWD * VV; % weight decay derivative

end