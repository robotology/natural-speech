function [loglik, exp_num_trans, exp_num_visits1,  exp_num_visitsall, allgammas] = ...
    ess_aehmm(prior, transmat, post,starts,ends)

sxb = 10;
iblock = 1;

numex = length(starts);
Q = length(prior);

exp_num_trans = zeros(Q);
exp_num_visits1 = zeros(Q,1);
exp_num_visitsall = zeros(Q,1);
%allgammas = cell(1,numex);
allgammas = zeros(Q,size(post,1));
nblock = ceil(numex/sxb);
loglik = 0;

for ex=1:numex
    B = post(starts(ex):ends(ex),:)';
    T = size(B,2);
    if(T > 2)

        [alpha, beta, gamma,  current_loglik, xi] = fwdback(prior, transmat, B);
        
        loglik = loglik +  current_loglik;
        %  if verbose, fprintf(1, 'll at ex %d = %f\n', ex, loglik); end
        
        exp_num_trans = exp_num_trans + sum(xi,3);
        exp_num_visits1 = exp_num_visits1 + gamma(:,1);
        exp_num_visitsall = exp_num_visitsall + sum(gamma,2); 
%        allgammas{ex} = gamma;
        allgammas(:,starts(ex):ends(ex)) = gamma;
        
        if(mod(ex,sxb)==1)
            fprintf(1,'Processing block %d of %d\n',iblock,nblock);
            iblock = iblock + 1;
        end
    end
end
allgammas = allgammas';

end