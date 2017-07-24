function plabels = state2phone(traintargets,nstates,removeSil)

Ns = size(traintargets,2) - removeSil;
if(mod(Ns,nstates) ~= 0)
    error('Number of states is not a multiple of nstates\n');
end

plabels = zeros(size(traintargets,1),ceil((Ns+removeSil)/nstates));

[~,ids] = max(traintargets,[],2);
ids = ceil(ids/nstates);

 for i=1:size(plabels,1)
     plabels(i,ids(i)) = 1;
 end