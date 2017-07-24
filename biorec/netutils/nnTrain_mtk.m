function [ net, prenet ] = nnTrain_mtk( X, Y, Y2, parnet, initnet, options )
%NNTRAIN Training of the Neural Network
% IN
%   X: input train matrix
%   Y: output train matrix
%   Y2: output of secondary task/ cell of outputs of secondary tasks
%   parnet: net parameters
%   options.valX
%   options.valY
%   options.valY2
%   options.task2ignore


% OUT
%   net: final net

switch parnet.pretrain
    case 'rand'
        prenet = nnRandInit(parnet);
    case 'loadnet'
        prenet = initnet;
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
    otherwise
        error('Invalid pretraining type for multitask net: %s',parnet.pretrain);
end

if((strcmp(parnet.multitask.mtktype,'alternate') || strcmp(parnet.multitask.mtktype,'joint_top')) && parnet.multitask.t2w ~= 0)
    nTasks = length(parnet.multitask.ccost2);
    prenet.auxw = cell(1,nTasks);
    for i=1:nTasks
        auxparnet = parnet;   % last layer
        auxparnet.units = [parnet.units(end-1) size(Y2{i},2)];
        auxparnet.activations = parnet.multitask.cactivations2(i);
        auxparnet.maxepoch = parnet.outputInit_maxepoch;
        lastLnet = nnRandInit(auxparnet);
        prenet.auxw{i} = lastLnet.w{1};
        prenet.auxbias{i} = lastLnet.bias{1};
    end    
end

fprintf('Fine-tuning\n');
%     if(strcmp(parnet.multitask.mtktype,'joint') || parnet.multitask.t2w==0)
%         [net, trainErr, testErr, trainErr2, testErr2] = nnBackprop_mtk(prenet,X,Y,Y2,parnet,options);
%     elseif(strcmp(parnet.multitask.mtktype,'alternate'))
%         [net, trainErr, testErr, trainErr2, testErr2] = nnBackprop_mtk2(prenet,X,Y,Y2,parnet,options);
%     else
        [net, trainErr, testErr, trainErr2, testErr2] = nnBackprop_mtk3(prenet,X,Y,Y2,parnet,options);
%    end
% end