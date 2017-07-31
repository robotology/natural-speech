function [ encNet, decNet, parDecNet, fullNet, parFullNet ] = aeTrain( X, parnet, varargin )
%AETRAIN Training of the AutoEncoder
% IN
%   X: input train matrix
%   parnet: AE parameters
%   varargin{1}: input test matrix
%   varargin{2}: train targets
% OUT
%   encNet: encoding part of the AE
%   decNet: decoding part of the AE
%   parDecNet: decoder parameter
%   fullNet: full AE
%   parFullAE: full AE parameters



if ~prod(strcmp(parnet.activations(1:end),'sigm'))
    error('AE training supported only if all the layers are sigmoid');
end
if length(varargin)>=1 && ~isempty(varargin{1})
    varargbackprop{1} = varargin{1};    % testX
end
if parnet.shid || parnet.pairsim
    if length(varargin)>=2 && ~isempty(varargin{2})
        trainTargets = varargin{2};
        [trainLabels, ~] = find(trainTargets');
        varargbackprop{2} = trainLabels;
    end
end

nhl = length(parnet.activations);   % number of hidden layers (up to the encoding layer)
%% Pre-training
fprintf('Pre-training the AE\n');
switch parnet.pretrain
    case 'rand'
        prenet = nnRandInit(parnet);
        for ihl=1:nhl
            prenet.gbias{ihl} = zeros(1,parnet.units(ihl),'single');
        end
    case 'rbm'  % stack of RBMs
        dbnparam.units = parnet.units;
        dbnparam.vgaus = parnet.vgaus;
        dbnparam.hgaus = parnet.hgaus;
        dbnparam.rbmparam = parnet.rbmparam;
        dbnparam.rbmvgausparam = parnet.rbmvgausparam;
        dbnparam.rbmhgausparam = parnet.rbmhgausparam;
        
        prenet = dbnTrain(X,dbnparam);
    case 'ae'   % stack of 1-layer AEs
        aeX = X;
        for ihl=1:nhl
            fprintf('Pretraining hidden layer %i with autoencoder\n',ihl);
            preaeparam = parnet;
            preaeparam.units = [parnet.units(ihl) parnet.units(ihl+1) parnet.units(ihl)];
            if (ihl==1 && parnet.vgaus) || (ihl==nhl && parnet.hgaus)
                preaeparam.activations = {parnet.activations{ihl}, 'linear'};
            else
                preaeparam.activations = {parnet.activations{ihl}, 'sigm'};
            end
            preaeparam.maxepoch = parnet.preae_maxepoch;
            preae = nnRandInit(preaeparam);
            preae = nnBackprop(preae,aeX,aeX,preaeparam);
            
            prenet.w{ihl} = preae.w{1};
            prenet.bias{ihl} = preae.bias{1};
            prenet.gbias{ihl} = preae.bias{2};
            
            if ihl<nhl
                preencparam.units = [parnet.units(ihl) parnet.units(ihl+1)];
                preencparam.activations = parnet.activations(ihl);
                preenc.w = preae.w(1);
                preenc.bias = preae.bias(1);
                aeX = nnFwd(preenc,aeX,preencparam);
            end
        end
    otherwise
        error('Invalid AE pretraining type: %s',parnet.pretrain);
end

for ihl = 1:nhl % 'mirror' the net
    prenet.w{nhl+ihl} = prenet.w{nhl-ihl+1}';
    prenet.bias{nhl+ihl} = prenet.gbias{nhl-ihl+1};
    parnet.units(nhl+ihl+1) = parnet.units(nhl-ihl+1);
    if ihl<nhl
        parnet.activations{nhl+ihl} = parnet.activations{nhl-ihl};
    else
        if parnet.vgaus
            parnet.activations{nhl+ihl} = 'linear';
        else
            parnet.activations{nhl+ihl} = 'sigm';
        end
    end
end
parFullNet = parnet;
%% Fine-tuning
fprintf('Fine-tuning\n');
fullNet = aeBackprop(prenet, X, parnet, varargbackprop{:});

encNet.w = fullNet.w(1:nhl); % the encoder is the first half
encNet.bias = fullNet.bias(1:nhl);

decNet.w = fullNet.w(nhl+1:end);  % the decoder is the second half
decNet.bias = fullNet.bias(nhl+1:end);
parDecNet = parFullNet;
parDecNet.units = parFullNet.units(nhl+1:end);
parDecNet.activations = parFullNet.activations(nhl+1:end);

end