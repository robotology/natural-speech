function stateids = getStateId(targets,nstates,removeSil)

Ns = size(targets,2) - removeSil;
if(mod(Ns,nstates) ~= 0)
    error('Number of states is not a multiple of nstates\n');
end

[~,ids] = max(targets,[],2);
dids = mod(ids,nstates)+1;

stateids = zeros(size(targets,1),nstates);
for i=1:length(ids)
    stateids(i,dids(i)) = 1;
end