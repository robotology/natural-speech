function [hae, prevhae, loglik, storehae] = hmmauto_learn(traindata,parnet,trStart,trEnd,testdata,mode,varargin)
% HMMAUTO_LEARN Trains a HMM-Encoder
% IN
%   traindata: matrix of training data, one row per acoustic feature frame
%   parnet: struct of hyper-parameters of the autoencoder
%   trStart: vector whose elements indicate the starting frame (within
%           training data) of each utterance
%   trEnd: vector whose elements indicate the starting frame (within
%          training data) of each utterance
%   testdata: matrix of testing data
%   mode: % defines the way subword posterior probabilities are extracted. 
%           Keep mode = enc
%   varargin: contains Expectation-Maximization hyper-parameters
% OUT
%   hae: HMM-Encoder
%   prevhae: inizial HMM_Encoder where actually no HMM as been trained
%   loglik: overall log-likelihood of the trained HMM-Encoder on the training
%          data
%   storehae: array of HMM-Encoders, one for each EM iteration

parnet.aehmm = 0;
H = parnet.units(end);

if H > 8
    error('The final number of states is too large');
end
nutts = length(trStart);
Q = 2^H;

[net, maxiter, thresh, adj_prior, adj_trans adj_ae, decNet, parDecNet, fullNet, parFullAE] = ...
    process_options(varargin,'net','','maxiter', 3,'thresh', 1e-4,'adj_prior', 1,'adj_trans', 1,...
    'adj_ae',1,'decNet','','parDecNet','','fullNet','','parFullAE','');

prior = ones(Q,1)*(1/Q);
allpriors = prior';
transmat = repmat(prior',Q,1);

if isempty(net)
    % autoencoder training
    [net, decNet, parDecNet, fullNet, parFullAE] = aeTrain(traindata,parnet,testdata);
    endata = nnFwd(net,traindata,parnet);
end
prevhae.net = net;
prevhae.parnet = parnet;
prevhae.prior = prior;
prevhae.transmat = transmat;
prevhae.allpriors = allpriors;

if sum(trStart(2:end)-trEnd(1:end-1)-1)~=0
    fprintf(1, 'WARNING: Some frames will be ignored during the HMM-AE training\n');
    newdata = [];
    for i=1:nutts
        newdata = [newdata;traindata(trStart(i):trEnd(i),:)];
    end
    traindata = newdata;
    clear newdata;
end

norder = randperm(size(traindata,1));
loglik = zeros(1,maxiter);

for iter=1:maxiter
    % Expectation  step
    B = ae_prob(traindata, allpriors,net, parnet,mode);
         
    [loglik(iter), exp_num_trans, exp_num_visits1, exp_num_visitsall, allgammas] = ess_aehmm(prior,transmat,B,trStart,trEnd);
    fprintf(1,'Loglikelihood after Expectation step  interation %d : %f\n',iter, loglik(iter));
    
    % Maximization step
    if adj_prior
        prior = normalise(exp_num_visits1);
    end
    if adj_trans 
        transmat = mk_stochastic(exp_num_trans);
    end
    if adj_ae                
       
        [explogs, allpriors] = getexplogs(traindata,allgammas,exp_num_visitsall,net,parnet,mode);
        net = aeBackprop(fullNet,traindata(norder,:),parFullAE,testdata,[],explogs(norder,:),mode);
    end
    storehae(iter).net = net;
    storehae(iter).parnet = parnet;
    storehae(iter).prior = prior;
    storehae(iter).transmat = transmat;
    storehae(iter).allpriors = allpriors;
end

hae.net = net;
hae.parnet = parnet;
hae.prior = prior;
hae.transmat = transmat;
hae.allpriors = allpriors;

%   
%   if verbose, fprintf(1, 'iteration %d, loglik = %f\n', num_iter, loglik); end
%   num_iter =  num_iter + 1;
%   converged = em_converged(loglik, previous_loglik, thresh);
%   previous_loglik = loglik;
%   LL = [LL loglik];
% end











    