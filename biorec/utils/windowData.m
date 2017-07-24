function [newData, newWordStart, newWordEnd] = windowData(data,windowSize,wordStart,wordEnd,removeSil)
%WINDOWDATA Groups data into windows
%   windowData(data,windowSize,wordStart,wordEnd,removeSil) groups words 
%   data into windows of windowSize frames, removing the first and last if
%   removeSil=1, and updating newWordStart and newWordEnd accordingly

newWordStart=wordStart;
newWordEnd=wordEnd;
if(windowSize==1 && ~removeSil)
    newData=data;
    return
end
wordSize=wordEnd-wordStart+1;    % #frames in each word
if(removeSil && min(wordSize)<=2)
    error('Some words too short for removeSil');
end
nFeatures=size(data,2);
nWords=length(wordStart);
nDummyFramesL=floor((windowSize-1)/2)-removeSil;  % number of dummy frames at the beginning
nDummyFramesR=ceil((windowSize-1)/2)-removeSil;   % number of dummy frames at the end

newData=zeros(size(data,1)-nWords*2*removeSil,windowSize*nFeatures,'single');    % preallocate
data=reshape(data',1,[]); % unroll data
currRow=1;
for i=1:nWords
    if(nDummyFramesL >= 0)
        dummyVectorL=repmat(data((wordStart(i)-1)*nFeatures+1:wordStart(i)*nFeatures),1,nDummyFramesL);
    else
        dummyVectorL=[];
        wordStart(i)=wordStart(i)+1;
    end
    if(nDummyFramesR >= 0)
        dummyVectorR=repmat(data((wordEnd(i)-1)*nFeatures+1:wordEnd(i)*nFeatures),1,nDummyFramesR);
    else
        dummyVectorR=[];
        wordEnd(i)=wordEnd(i)-1;
    end
    newWordVector=[dummyVectorL data((wordStart(i)-1)*nFeatures+1:wordEnd(i)*nFeatures) dummyVectorR];
    newWordStart(i)=currRow;
    for j=1:wordSize(i)-2*removeSil
        newData(currRow,:)=newWordVector((j-1)*nFeatures+1:(j-1)*nFeatures+windowSize*nFeatures);
        currRow=currRow+1;
    end
    newWordEnd(i)=currRow-1;
end

end