function [err] = computeReconstructionErrorEachPhoneme(data,recdata,targets)
%function [err] = computeReconstructionErrorEachPhoneme(data,recdata,targets,stats)

for ph = 1:size(targets,2)
index=find(targets(:,ph)==1);
ori = data(index,:);
recori = recdata(index,:);
err(ph) = computeReconstructionError(ori,recori);
%err(ph) = computeReconstructionError(ori,recori,stats);
clear ori recori
end

