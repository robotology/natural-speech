function [alldata, label, audioend, varargout] = loadAndNormData(datafile,splitfile,datatype,split,stream1,stream2,NoAuF,NoMF,normtype,varargin)

load (datafile);
load (splitfile);
if (strcmp(notes.splitype,'cv1-4audio-1motor-1sbj') || strcmp(notes.splitype,'cv_imitation'))
    error ('At present this splittype for recognition is not possible\n');
end

sentenceLength=sentenceEnd-sentenceStart+1;
Audio = zeros(size(audio));
RealMotor = zeros(size(realMotor,1),NoMF);
RecMotor = zeros(size(recMotor,1),NoMF);
labels = false(size(label));

start=1;

if strcmp(datatype,'train')
    tmpsentenceStart=zeros(1,length(trainIdx{split}));
    tmpsentenceEnd=zeros(1,length(trainIdx{split}));
    for ii=1:length(trainIdx{split})
        tmpsentenceStart(ii)=start;
        idx=trainIdx{split}(ii);
        Audio(start:start+sentenceLength(idx)-1,:)=audio(sentenceStart(idx):sentenceEnd(idx),:);
        RealMotor(start:start+sentenceLength(idx)-1,:)=realMotor(sentenceStart(idx):sentenceEnd(idx),1:NoMF);
        RecMotor(start:start+sentenceLength(idx)-1,:)=recMotor(sentenceStart(idx):sentenceEnd(idx),1:NoMF);
        labels(start:start+sentenceLength(idx)-1,:)=label(sentenceStart(idx):sentenceEnd(idx),:);
        start=start+sentenceLength(idx);
        tmpsentenceEnd(ii)=start-1;
    end
    varargout{1} = tmpsentenceStart;
    varargout{2} = tmpsentenceEnd;
    audio = Audio(1:tmpsentenceEnd(end),:);
    realMotor = RealMotor(1:tmpsentenceEnd(end),:);
    recMotor = RecMotor(1:tmpsentenceEnd(end),:);
    label = labels(1:tmpsentenceEnd(end),:);
elseif strcmp(datatype,'test')
    tmpsentenceStart=zeros(1,length(testIdx{split}));
    tmpsentenceEnd=zeros(1,length(testIdx{split}));
    for ii=1:length(testIdx{split})
        tmpsentenceStart(ii)=start;
        idx=testIdx{split}(ii);
        Audio(start:start+sentenceLength(idx)-1,:)=audio(sentenceStart(idx):sentenceEnd(idx),:);
        RealMotor(start:start+sentenceLength(idx)-1,:)=realMotor(sentenceStart(idx):sentenceEnd(idx),1:NoMF);
        RecMotor(start:start+sentenceLength(idx)-1,:)=recMotor(sentenceStart(idx):sentenceEnd(idx),1:NoMF);
        labels(start:start+sentenceLength(idx)-1,:)=label(sentenceStart(idx):sentenceEnd(idx),:);
        start=start+sentenceLength(idx);
        tmpsentenceEnd(ii)=start-1;
    end
    varargout{1} = tmpsentenceStart;
    varargout{2} = tmpsentenceEnd;
    audio = Audio(1:tmpsentenceEnd(end),:);
    realMotor = RealMotor(1:tmpsentenceEnd(end),:);
    recMotor = RecMotor(1:tmpsentenceEnd(end),:);
    label = labels(1:tmpsentenceEnd(end),:);
elseif strcmp(datatype,'all')
    display('Normalizing the whole data set (no distinction between training and testing data)');
    varargout{1} = sentenceStart;
    varargout{2} = sentenceEnd;
    audio = Audio;
    RealMotor = realMotor(:,1:NoMF);
    RecMotor = recMotor(:,1:NoMF);
    clear realMotor recMotor
    realMotor = RealMotor;
    recMotor = RecMotor;
else
    error('Undefined datatype in function loadandnormdata\n');
end
if (strcmp(stream1,'audio') || strcmp(stream1,'joint'))
    stream = 'audio';
    if(normtype == 3)
        normData3;
    elseif(normtype==4)
        normData4;
    end
    alldata = single(ndata);
    audioend = size(ndata,2);
    clear ndata
    if (strcmp(stream2,'realMotor') || strcmp(stream2,'recMotor') || strcmp(stream1,'joint'))
        if (strcmp(stream1,'joint'))
            stream = 'realMotor';
        else
            stream = stream2;
        end;
        if(normtype == 3)
            normData3;
        elseif(normtype == 4)
            normData4;
        end
        alldata = [alldata ndata];
        clear ndata;
    else
        Nfeat2 = 0;
    end
elseif (strcmp(stream1,'realMotor') || strcmp(stream1,'recMotor'))
    stream = stream1;
    if(normtype == 3)
        normData3;
    elseif(normtype == 4)
        normData4;
    end
    alldata = single(ndata);
    audioend = 0;
    clear ndata;
end
if strcmp(stream2,'audio')
    error ('ERROR: audio in stream 2\n');
end
