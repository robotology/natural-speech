function nframes = getNeighborFrames(frames,ss,se)
%GETNEIGHBORFRAMES Function used by mtkpr_baseline to select the the next
% and preceeding label vectors (i.e., neighbor frames of each i-th
% frame)
% IN
%   frame: matrix of label vectors
%   ss: vector of elements indicating the first frame of each utterance
%   se: vector of elements indicating the last frame of each utterance
%
% OUT
%    nframes: matrix where each i-th row contains the neighbor frames of the
%             i-th frame


nframes{1} = [frames(1,:);frames(1:end-1,:)];
nframes{1}(ss,:) = frames(ss,:);

nframes{2} = [frames(2:end,:);frames(end,:)];
nframes{2}(se,:) = frames(se,:);