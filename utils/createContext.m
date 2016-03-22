function [ frameData, ftargets, newSentenceStart, newSentenceEnd ] = createContext( data,targets,sentenceStart,sentenceEnd,Nframes,varargin )
%CREATECONTEXT Creates context for the data frames
removeSil=0;
if length(varargin)>=1 && ~isempty(varargin{1})
    removeSil = varargin{1};
end

if removeSil
    %removing added labels from target labels
    ftargets = false(size(targets,1)-2*length(sentenceStart),size(targets,2));
    start=1;
    for ii = 1:length(sentenceStart)
        ftargets(start:start+sentenceEnd(ii)-sentenceStart(ii)-2,:) = targets(sentenceStart(ii)+1:sentenceEnd(ii)-1,:);
        start=start+sentenceEnd(ii)-sentenceStart(ii)-1;
    end
    ftargets=ftargets(:,1:end-1);
else
    ftargets=targets;
end

[frameData, newSentenceStart, newSentenceEnd] = windowData(data,Nframes,sentenceStart,sentenceEnd,removeSil);

end
