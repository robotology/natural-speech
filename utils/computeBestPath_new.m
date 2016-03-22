function [cseq scseq] = computeBestPath_new(p_unigm,p_bigm,s_unigm,s_bigm,post,vSP,vEP)

nsent = length(vSP);
cseq = cell(1,nsent);
scseq = cell(1,nsent);
nstates = length(s_unigm)/length(p_unigm);
wlm = 1;
for i=1:nsent
    spost = post(:,vSP(i):vEP(i));
    cseq{i} = viterbi_path_wlm(p_unigm,p_bigm,s_bigm,spost,nstates,wlm);
    s_priors = repmat(s_unigm,1,size(spost,2));
    spost = spost ./ s_priors;
    scseq{i} = viterbi_path_wlm(p_unigm,p_bigm,s_bigm,spost,nstates,wlm);
end