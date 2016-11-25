function nframes = getNeighborPhones(frames,ss,se)

nframes{1} = zeros(size(frames));
nframes{2} = zeros(size(frames));
[~, imax] = max(frames,[],2);
plab = frames(1,:);
for i=2:size(frames,1)    
    if(imax(i) ~= imax(i-1))
        plab = frames(i-1,:);
    end
    nframes{1}(i,:) = plab;
end
nframes{1}(ss,:) = frames(ss,:);

nlab = frames(end,:);
for i=size(frames,1)-1:-1:1    
    if(imax(i)~=imax(i+1))
        nlab = frames(i+1,:);
    end
    nframes{2}(i,:) = nlab; 
end
nframes{2}(se,:) = frames(se,:);