function [ output ] = nnFwd( net, X, parnet, varargin )
%NNFWD Performs a forward pass on the net, computing its output for the
%given data
% IN
%   net: the net
%   X: input matrix
%   parnet: net parameters
%   varargin{1}: net layer whose avctivations are returned by nnFwd
%                default is the last layer
% OUT
%   output: output of the net or of the layer indicated by varargin{1}

if(length(varargin) > 0)
    nl = varargin{1};
else
    nl = length(parnet.activations);  %number of layers (hidden+output)
end
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