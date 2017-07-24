function [ net ] = dbnTrain( X, dbnparam )
%DBNTRAIN Training of the Deep Belief Network
% IN
%   X: input matrix
%   dbnparam: DBN parameters
% OUT
%   net: final DBN

nl = length(dbnparam.units) - 1;    % number of layers (excluding the input)
vgaus = dbnparam.vgaus;
hgaus = dbnparam.hgaus;

if nl==1 && vgaus && hgaus
    error('1-layer DBN with both guassian visible and hidden units ignores non-linear correlations');
end

poshidprobs = X;

for il=1:nl
    fprintf(1,'Training DBN layer %d: %d-%d \n',il,dbnparam.units(il),dbnparam.units(il+1));
    if il==1 && vgaus
        [net.w{il}, net.bias{il}, poshidprobs, net.gbias{il}] = rbmTrain(poshidprobs,dbnparam.units(il+1),dbnparam.rbmvgausparam);
    elseif il==nl && hgaus
        [net.w{il}, net.bias{il}, poshidprobs, net.gbias{il}] = rbmTrain(poshidprobs,dbnparam.units(il+1),dbnparam.rbmhgausparam);
    else
        [net.w{il}, net.bias{il}, poshidprobs, net.gbias{il}] = rbmTrain(poshidprobs,dbnparam.units(il+1),dbnparam.rbmparam);
    end
end

end