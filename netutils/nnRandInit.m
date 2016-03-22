function [ net ] = nnRandInit( parnet )
%NNINIT Randomly initializes the net. Heuristics taken from the "deepnet"
%library by Miguel A. Carreira-Perpinan and Weiran Wang.
% IN
%   parnet: net parameters
% OUT
%   net: initialized network

nl = length(parnet.activations);  %number of layers (hidden+output)
fprintf('Random initialization of the weights\n');

for il = 1:nl
    nIn = parnet.units(il);
    nOut = parnet.units(il+1);
    net.w{il} = zeros(nIn,nOut,'single');
    net.bias{il} = zeros(1,nOut,'single');
    
    switch parnet.activations{il}
        case {'linear','softmax'}
            net.w{il} = 2*(rand(nIn,nOut,'single')-0.5)/sqrt(nIn);
        case 'sigm'
            net.w{il} = 8*(rand(nIn,nOut,'single')-0.5)*sqrt(6)/sqrt(nIn+nOut);
        case 'tanh'
            net.w{il} = 2*(rand(nIn,nOut,'single')-0.5)*sqrt(6)/sqrt(nIn+nOut);
        case 'relu'
            net.w{il} = 2*(rand(nIn,nOut,'single')-0.5)*0.01;
            net.bias{il} = rand(1,nOut,'single')*0.1;            
        otherwise
            error('Invalid activation for layer %i: %s', il, parnet.activations{il});
    end
    %%%
    %net.w{il} = 0.1*randn(nIn,nOut,'single');
    %net.bias{il} = 0.1*randn(1,nOut,'single');
    %%%
end

end