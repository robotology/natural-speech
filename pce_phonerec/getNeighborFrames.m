function nframes = getNeighborFrames(frames,ss,se)

nframes{1} = [frames(1,:);frames(1:end-1,:)];
nframes{1}(ss,:) = frames(ss,:);

nframes{2} = [frames(2:end,:);frames(end,:)];
nframes{2}(se,:) = frames(se,:);