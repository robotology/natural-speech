function [data, labels, varargout] = loadAndNormData_short(datafile,splitfile,datatype,normtype,removeSil,varargin)

timitsils = [64 65 66 82 83 84 133 134 135 139 140 141];
%timitsils = [82 83  84];
load (datafile);
load (splitfile);

if strcmp(datatype,'train')
    gIdx = trainIdx;
elseif(strcmp(datatype,'validation'))
    gIdx = valIdx;
elseif(strcmp(datatype,'test'))
    gIdx = testIdx;
end
if ~isempty(varargin)
    maxSentences = min(varargin{1},length(gIdx));
    gIdx = gIdx(1:maxSentences);
end

csStart= sentenceStart(gIdx);
csEnd= sentenceEnd(gIdx);
csLength=csEnd-csStart+1;
tmpsentenceStart = zeros(1,length(csStart));
tmpsentenceEnd = zeros(1,length(csEnd));
data = zeros(sum(csLength),size(audio,2));
labels = zeros(sum(csLength),size(label,2));
start = 1;
for ii=1:length(gIdx)
    nrmv = 0;
    tmpsentenceStart(ii)=start; 
    if(removeSil)
        tmpdata = audio(csStart(ii):csEnd(ii),:);   
        tmplabs = label(csStart(ii):csEnd(ii),:);
        [~, ids] = max(tmplabs,[],2);
        isils = ismember(ids,timitsils);
        nrmv = sum(isils);
        tmpdata(isils,:) = [];
        tmplabs(isils,:) = [];
        data(start:start+csLength(ii)-1-nrmv,:) =  tmpdata;
        labels(start:start+csLength(ii)-1-nrmv,:) =  tmplabs;
    else
        labels(start:start+csLength(ii)-1,:) = label(csStart(ii):csEnd(ii),:);
        data(start:start+csLength(ii)-1,:) = audio(csStart(ii):csEnd(ii),:);        
    end
    start=start+csLength(ii)-nrmv;
    tmpsentenceEnd(ii)=start-1;
end

if(removeSil)
    data(tmpsentenceEnd(end)+1:end,:) = [];
    labels(tmpsentenceEnd(end)+1:end,:) = [];
 %   labels(:,timitsils) = [];
end
data = fnormData(data,size(data,2),normtype);
varargout{1} = tmpsentenceStart;
varargout{2} = tmpsentenceEnd;


