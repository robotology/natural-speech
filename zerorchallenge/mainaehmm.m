clear all;
close all;
 
parnet = struct(...
    'units',[60 100 50 6],...   % input + hidden (last is the encoding layer)
    'activations',{{'sigm','sigm','sigm'}},... % hidden
    'pretrain','rbm',...    % rbm/ae/rand
    'preae_maxepoch',50,... % for 'ae' pretraining, number of epochs for each layer
    'vgaus',1,...   % for 'rbm' pretraining, gaussian visible units
    'hgaus',0,...   % for 'rbm' pretraining, gaussian hidden units
    'pairsim',0,... % SAE using similarity heuristic
    'shid',0,...    % SCAE using similarity heuristic
    'noisy',0,...   % DAE
    'binarize',0,...    % binarize encodings in the fwd pass: 1) threshold 0.5; 2) threshold stochastically
    'forceBin',0,...    % force binarization: 1) add noise to encoding layer input; 2) additional entropy cost term
    'noiseVariance',16,...  % for forceBin=1, variance of the noise
    'entropyWeight',1,...   % for forceBin=2, weight of the entropy term
    'sparsepen',0,...   % sparsity penalty (beta)
    'sparsepar',0.05,...    % sparsity parameter (rho)
    'batchsize',1000,...
    'maxepoch',10,...
    'optimization','cgd',...    % cgd/sgd
    'cost','mse',...
    'dropout',0,...
    'learningRate',0.1,...          % SGD parameters
    'learningRateDecay',0.99,...    %%
    'momentum',0.9,...              %%
    'weightDecay',0.0001,...
    'rbmparam',struct(...           
        'vgaus',0,...
        'hgaus',0,...
        'batchsize',50,...
        'maxepoch',10,...
        'epsilonw',0.1,...
        'epsilonvb',0.1,...
        'epsilonhb',0.1,...
        'weightcost',0.0002,...
        'initialmomentum',0.5,...
        'finalmomentum',0.9...
        ),...
    'rbmhgausparam',struct(...
        'vgaus',0,...
        'hgaus',1,...
        'batchsize',50,...
        'maxepoch',20,...
        'epsilonw',0.001,...
        'epsilonvb',0.001,...
        'epsilonhb',0.001,...
        'weightcost',0.0002,...
        'initialmomentum',0.5,...
        'finalmomentum',0.9...
        ),...
    'rbmvgausparam',struct(...
        'vgaus',1,...
        'hgaus',0,...
        'batchsize',50,...
        'maxepoch',20,...
        'epsilonw',0.001,...
        'epsilonvb',0.001,...
        'epsilonhb',0.001,...
        'weightcost',0.0002,...
        'initialmomentum',0.5,...
        'finalmomentum',0.9...
        )...
);

mode = 'enc';
maxiter = 2;

%%%filedata = '/home/local/HUMANOIDS/badino/MyProjects/speechdata/zeroR/sampleData';
filedata = 'sampleDataSINGLE';
%%%filenet = 'aehmmnet_6h_force00_maxiter5_enc_sc0_v2_02-Mar-2015';
filenet = [];%'Nets trainate/aehmmnet_6h_force00_maxiter5_enc_sc0_v2_02-Mar-2015';
%Xitsonga

%aehfile = ['aehmmnet_6h_force00_enc-rec-beta05_noscaled_tsonga_maxiter_10' date];
aehfile = ['pippo_' date];
%English

outdir1 = ['outdir1_' date];
outdir2 = ['outdir2_' date];

Q = 2.^parnet.units(end);

mkdir(outdir1);
mkdir(outdir2);
load(filedata);
if ~isempty(filenet)
    load(filenet);
    parnet = prevhae.parnet;
%     parnet = parnet_unclassifier;
%     prevhae.net = net;
%     prevhae.parnet = parnet;
    [hae, prevhae, loglik, storehae] = hmmauto_learn(traindata,parnet,wordTrainStart,wordTrainEnd,testdata,mode,'maxiter',maxiter,'net',prevhae.net);
else
    [hae, prevhae, loglik, storehae] = hmmauto_learn(traindata,parnet,wordTrainStart,wordTrainEnd,testdata,mode,'maxiter',maxiter);
end

save(aehfile,'hae','parnet','prevhae','loglik','storehae');
ntrutts = length(wordTrainStart);
ntsutts = length(wordTestStart);
Nu = ntrutts + ntsutts;
utt.data = cell(1,Nu);
utt.name = cell(1,Nu);

alldata = [traindata;testdata];
allStart = zeros(1,Nu);
allEnd = zeros(1,Nu);
for i=1:ntrutts        
    utt.data{i} = traindata(wordTrainStart(i):wordTrainEnd(i),:);
    utt.name{i} = wordTrainName{i};
end

for i=1:ntsutts
    utt.data{i+ntrutts} = testdata(wordTestStart(i):wordTestEnd(i),:);  
    utt.name{i+ntrutts} = wordTestName{i};
end

allStart(1:ntrutts) = wordTrainStart;
allEnd(1:ntrutts) = wordTrainEnd;
allStart(ntrutts+1:end) = wordTestStart + size(traindata,1);
allEnd(ntrutts+1:end) = wordTestEnd + size(traindata,1);

utt.encoded1oK = hmmauto_decode(hae,alldata,allStart,allEnd,mode);

len = 0;
for i=1:Nu
    len = len+length(rmRepInarow(utt.encoded1oK{i}));
end
alen = len./Nu;

%printoutaeh_zr(outdir1,utt.name,utt.encoded1oK,Q);

nohmmutts = hmmauto_decode(prevhae,alldata,allStart,allEnd,mode);

len = 0;
for i=1:Nu
    len = len+length(rmRepInarow(nohmmutts{i}));
end
alen_nohmm = len./Nu;

%printoutaeh_zr(outdir2,utt.name,nohmmutts,Q);
