function [ w, hb, allposhidprobs, vb ] = rbmTrain( X, numhid, rbmparam )
%RBMTRAIN Training of the Restricted Boltzmann Machine
% IN
%   X: input matrix
%   numhid: number of hidden units
%   rbmparam: RBM parameters
% OUT
%   w: weights
%   hb: hidden bias weights
%   allposhidprobs: hidden layer activations for all data
%   vb: visible bias weights

try % check gpu availability
    gd = gpuDevice();
    useGpu = true;
    fprintf(1, 'GPU available\n');
catch
    useGpu = false;
end

maxepoch = rbmparam.maxepoch;
epsilonw = rbmparam.epsilonw;   % Learning rate for weights
epsilonvb = rbmparam.epsilonvb; % Learning rate for biases of visible units
epsilonhb = rbmparam.epsilonhb; % Learning rate for biases of hidden units
weightcost = rbmparam.weightcost;
momentum = rbmparam.initialmomentum;
finalmomentum = rbmparam.finalmomentum;
hgaus = rbmparam.hgaus;
vgaus = rbmparam.vgaus;

N = size(X,1);
numdims = size(X,2);
batchsize=rbmparam.batchsize;
numbatches = ceil(N/batchsize);

vishid = 0.1*randn(numdims,numhid,'single');
if useGpu
    vishid = gpuArray(vishid);
end
hidbiases  = zeros(1,numhid,class(vishid));
visbiases  = zeros(1,numdims,class(vishid));
vishidinc  = zeros(numdims,numhid,class(vishid));
hidbiasinc = zeros(1,numhid,class(vishid));
visbiasinc = zeros(1,numdims,class(vishid));

allposhidprobs = zeros(N,numhid,'single');

fprintf('%6s %15s %11s\n','Epoch','Error','Time');

T0=tic;        

for epoch=1:maxepoch
    errsum = 0;
    rp = randperm(N);   % randomize data
    for batch=1:numbatches
        i0 = (batch-1)*batchsize+1;
        i1 = min(batch*batchsize,N);
        batchdata = X([rp(i0:i1) rp(1:max(0,batch*batchsize-N))],:);    % fill the last batch with the first randomized cases
        if useGpu
            batchdata = gpuArray(batchdata);
        end
        
        arg1 = bsxfun(@plus, batchdata*vishid, hidbiases);
        if ~hgaus
            arg1 = 1 ./ (1 + exp(-arg1));
        end
        poshidprobs = arg1;
        posprods = batchdata' * poshidprobs;
        poshidact = sum(poshidprobs);
        posvisact = sum(batchdata);
        %%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%
        if ~hgaus
            if useGpu
                poshidstates = (poshidprobs > gpuArray.rand(batchsize,numhid)) + 0.0;
            else
                poshidstates = (poshidprobs > rand(batchsize,numhid)) + 0.0;
            end
        else
            if useGpu
                poshidstates = poshidprobs + gpuArray.randn(batchsize,numhid);
            else
                poshidstates = poshidprobs + randn(batchsize,numhid);
            end
        end
        %%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%
        arg1 = bsxfun(@plus, poshidstates*vishid', visbiases);
        if ~vgaus
            arg1 = 1 ./ (1 + exp(-arg1));
        end
        negdata = arg1;
        if(epoch < maxepoch)
            arg1 = bsxfun(@plus, negdata*vishid, hidbiases);
            if ~hgaus
                arg1 = 1 ./ (1 + exp(-arg1));
            end
            neghidprobs = arg1;
            negprods = negdata' * neghidprobs;
            neghidact = sum(neghidprobs);
            negvisact = sum(negdata);
            %%%%%%%%% END OF NEGATIVE PHASE%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if epoch>5,
                momentum=finalmomentum;
            end;
            %%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%
            vishidinc = momentum*vishidinc + epsilonw*((posprods-negprods)/batchsize - weightcost*vishid);
            visbiasinc = momentum*visbiasinc + (epsilonvb/batchsize)*(posvisact-negvisact);
            hidbiasinc = momentum*hidbiasinc + (epsilonhb/batchsize)*(poshidact-neghidact);
            vishid = vishid + vishidinc;
            visbiases = visbiases + visbiasinc;
            hidbiases = hidbiases + hidbiasinc;
            %%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%  
        else
            allposhidprobs([rp(i0:i1) rp(1:max(0,batch*batchsize-N))],:) = gather(poshidprobs);
        end
        err = sum(sum((batchdata-negdata).^2));               
        errsum = err + errsum;      
    end        
    T = toc(T0);
    fprintf('%6i %15.0f %11.3f\n',epoch,errsum,T);
end
w = gather(vishid);
hb = gather(hidbiases);
vb = gather(visbiases);

end