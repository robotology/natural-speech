function newdata = groupFeatures(data,nwnds)

% data = NxOF matrix where N is the number of examples and OF is the overall
% number of feature vectors
% nwds = no. of windows, i.e. number of feature vectors 
nf = size(data,2)/nwnds;
newdata = zeros(size(data));
for i=1:nf
    newdata(:,nwnds*(i-1)+1:nwnds*i) = data(:,i:nf:end);
end
