function [ output ] = nnFwd( net, X, parnet )
%NNFWD Performs a forward pass on the net, computing its output for the
%given data
% IN
%   net: the net
%   X: input matrix
%   parnet: net parameters
% OUT
%   output: output of the net

nl = length(parnet.activations);  %number of layers (hidden+output)
N = size(X,1);   %number of cases

probs = X;

for il=1:nl
    probs = [probs ones(N,1)] * [net.w{il}; net.bias{il}];
    switch parnet.activations{il}
        case 'linear'
            %nop
        case 'sigm'
            probs = 1 ./ (1 + exp(-probs));
        case 'tanh'
            probs = tanh(probs);
        case 'relu'
            probs = 0.5 * (probs + abs(probs));
        case 'softmax'
            probs = bsxfun(@minus, probs, max(probs, [], 2));  % wlog
            probs = exp(probs);
            probs = bsxfun(@rdivide, probs, sum(probs,2));
        otherwise
            error('Invalid activation for layer %i: %s', il, parnet.activations{il});
    end
end

output=probs;

end