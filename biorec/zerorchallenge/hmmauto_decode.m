function decutts = hmmauto_decode(hae,data,starts,ends,mode)

sxb = 100;
iblock = 1;


nutts = length(starts);
nblock = ceil(nutts/sxb);
decutts = cell(1,nutts);
logprior = log(hae.prior);
logtransmat = log(hae.transmat);
B = ae_prob(data,hae.allpriors,hae.net,hae.parnet,mode);
Butts = cell(1,nutts); 
for i=1:nutts
    Butts{i} = B(starts(i):ends(i),:)';
end
for i=1:nutts

    if size(Butts{i},2)>1
        decutts{i} = viterbi_path_log(logprior,logtransmat,log(Butts{i}));
    else
        [tmp decutts{i}] = max(Butts{i});
    end
    if mod(i,sxb)==1        
        fprintf(1,'Processing block %d of %d\n',iblock,nblock);
        iblock = iblock + 1;
    end
    
end