function [p_unigm,p_bigm,s_unigm,s_bigm] = recomputeLM(slabs,nstates,sentenceStart,tuniques)

N = size(slabs,1);
NS = size(slabs,2);
NP = NS/nstates;
csu = sum(slabs,1);
%if(tuniques)
%zerov = 1;
%     zerov = min([round(csu/2) 1]);
%     csu(csu==0) = zerov;
%     nz = sum((csu==0))*zerov;
% else
%     nz = 0;
% end
nz = 0;
s_unigm = (csu./(N+nz))';
s_unigm(s_unigm==0) = 1;
cs_bigm = zeros(NS);
s_bigm = zeros(NS);
for i=1:N-1
    ps = find(slabs(i,:) == 1);
    ns = find(slabs(i+1,:) == 1);
    cs_bigm(ps,ns) = cs_bigm(ps,ns) + 1; 
end

for i=1:NS
    s_bigm(i,:) = cs_bigm(i,:) ./ csu(i);    
end
    
plabs = state2phone(slabs,nstates,0);

fplabs = plabs(sentenceStart,:);
cpsu = sum(fplabs,1);
if(tuniques)
    nz = sum(cpsu==0);
    cpsu(cpsu==0) = 1;
else
    nz = 0;
end

p_unigm = (cpsu./(length(sentenceStart)+nz))';

p_bigm = zeros(NP);
for i=1:N-1
    pp = find(plabs(i,:) == 1);
    np = find(plabs(i+1,:) == 1);
    p_bigm(pp,np) = p_bigm(pp,np) + 1; 
end

for i=1:NS
    if(mod(i,nstates)==0)
        j = ceil(i/nstates);
        if(tuniques==0)
            p_bigm(j,:) = s_bigm(i,1:nstates:end);
            p_bigm(j,:) = p_bigm(j,:) ./ sum(p_bigm(j,:));
        else
            tmp = cs_bigm(i,1:nstates:end);
            tmp(tmp==0) = 1;
            p_bigm(j,:) = tmp./sum(tmp);            
        end
        
        a = s_bigm(i,i);
        s_bigm(i,:) = 0;
        s_bigm(i,i) = a;
    end
    
end




