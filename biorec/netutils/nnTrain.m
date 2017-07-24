function [ net, prenet] = nnTrain( X, Y, parnet, varargin )
%NNTRAIN Training of the Neural Network
% IN
%   X: input train matrix
%   Y: output train matrix
%   parnet: net parameters
%   varargin{1}: input test matrix
%   varargin{2}: output test matrix
%   varargin{3}: net from Acoustic to Articulatory Mapping
% OUT
%   net: final net
%   prenet: initialization net (i.e., net before back-prop)

switch parnet.pretrain
    case 'rand'
        prenet = nnRandInit(parnet);
    case 'loadnet'
        prenet = parnet.initnet;
    case 'rbm'
        if ~prod(strcmp(parnet.activations(1:end-1),'sigm'))
            error('RBM pretraining supported only if all the hidden layers are sigmoid');
        end
        fprintf('RBM pre-training\n');
        dbnparam.units = parnet.units(1:end-1);
        dbnparam.vgaus = parnet.vgaus;
        dbnparam.hgaus = parnet.hgaus;
        dbnparam.rbmparam = parnet.rbmparam;
        dbnparam.rbmvgausparam = parnet.rbmvgausparam;
        dbnparam.rbmhgausparam = parnet.rbmhgausparam;
        if strcmp(parnet.activations(end),'sigm')
            dbnparam.units = [dbnparam.units parnet.units(end)];
        end
        prenet = dbnTrain(X,dbnparam);
        if ~strcmp(parnet.activations(end),'sigm')  % initialize output layer weights
            fprintf('Initializing output layer weights\n');
            parnet1 = parnet;   % first N-1 layers
            parnet1.units = parnet1.units(1:end-1);
            parnet1.activations = parnet1.activations(1:end-1);
            lastLayerInput = nnFwd(prenet,X,parnet1);
            parnet2 = parnet;   % last layer
            parnet2.units = parnet.units(end-1:end);
            parnet2.activations = parnet.activations(end);
            parnet2.maxepoch = parnet.outputInit_maxepoch;
            lastLayerPreNet = nnRandInit(parnet2);
            lastLayerNet = nnBackprop(lastLayerPreNet,lastLayerInput,Y,parnet2);
            prenet.w{end+1} = lastLayerNet.w{1};
            prenet.bias{end+1} = lastLayerNet.bias{1};
        end
    case 'aam'
        if length(varargin)>=3 && ~isempty(varargin{3})
            prenet = varargin{3};
        else
            error('Net required for AAM-based pretraining');
        end
        fprintf('Initializing output layer weights\n');
        parnet1 = parnet;   % first N-1 layers
        parnet1.units = parnet1.units(1:end-1);
        parnet1.activations = parnet1.activations(1:end-1);
        lastLayerInput = nnFwd(prenet,X,parnet1);
        parnet2 = parnet;   % last layer
        parnet2.units = parnet.units(end-1:end);
        parnet2.activations = parnet.activations(end);
        parnet2.maxepoch = parnet.outputInit_maxepoch;
        lastLayerPreNet = nnRandInit(parnet2);
        lastLayerNet = nnBackprop(lastLayerPreNet,lastLayerInput,Y,parnet2);
        prenet.w{end+1} = lastLayerNet.w{1};
        prenet.bias{end+1} = lastLayerNet.bias{1};
    otherwise
        error('Invalid pretraining type: %s',parnet.pretrain);
end
if(isfield(parnet,'constGradient') && parnet.constGradient)
    if(strcmp(parnet.optimization,'sgd')==0)
        error('At present, constGradinet only works with stochastic gradient descent\n');
    end
    prenet = buildConstEleNet(prenet,parnet);
end
fprintf('Fine-tuning\n');
[net, trainErr, testErr] = nnBackprop(prenet,X,Y,parnet,varargin{:});

end