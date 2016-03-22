function [per predseq totdels totins totsubs sops substitution testseq] = computePhoneRecognitionError(cseq,testtargets,vSP,vEP,state)

nutts = length(vSP);
testtmpseq = cell(1,nutts);
testseq = cell(1,nutts);
predseq = cell(1,nutts);
sops = cell(1,nutts);
totdels = 0;
totins = 0;
totsubs = 0;

nph = size(testtargets,2)/state;
substitution = zeros(nph,1);
nphonetype = zeros(nph,1);

[tmp testsinglet] = max(testtargets,[],2);
for i=1:nutts
    testtmpseq{i} = ceil(testsinglet(vSP(i):vEP(i),:)./state)';
    cseq{i} = ceil(cseq{i}./state); 
end
errc = 0;
phonc = 0;
for i=1:nutts
    predseq{i} = rmRepInarow(cseq{i});
    testseq{i} = rmRepInarow(testtmpseq{i});
    [d dels ins subs sops{i}] = edit_distance_levenshtein(testseq{i},predseq{i},'');
    errc = errc + d;
    totdels = totdels + dels;
    totins = totins + ins;
    totsubs = totsubs + subs;
    phonc = phonc + length(testseq{i});
    for h = 1:length(testseq{i})
        nphonetype(testseq{i}(h)) = nphonetype(testseq{i}(h))+1;
    end
    
    tmpops = sops{i};
    tmpops(tmpops=='i') = [];
    sub = find(tmpops == 's');
    for h = 1:length(sub)
        substitution(testseq{i}(sub(h))) = substitution(testseq{i}(sub(h)))+1;
    end
    
    clear sub tmpops
end
per = errc/phonc;
substitution = substitution./nphonetype;
totdels = totdels/phonc;
totins = totins./phonc;
totsubs = totsubs./phonc;



