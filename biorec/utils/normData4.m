switch stream
    case 'audio'
        undata = audio;
        Nfeat = NoAuF;
    case 'realMotor'
        undata = realMotor;
        Nfeat = NoMF;
    case 'recMotor'
        undata = recMotor;
        Nfeat = NoMF;
    case 'audio4mapping'
        undata = audio4map;
        Nfeat = NoAuF_map;
    case 'mappedaudio1'
        undata = audioListener;
        Nfeat = NoAuF;
    case 'mappedaudio2'
        undata = audioSpeaker;
        Nfeat = NoAuF;
    case 'mappedmotor'
        undata = realMotorListener;
        Nfeat = NoMF;
end



% first rescale data in [0 1] range
for ifeat=1:Nfeat
    fundata = undata(:,ifeat:Nfeat:end);
    ox = size(fundata,1);
    oy = size(fundata,2);
    fundata = reshape(fundata,1,ox*oy);
    m = min(fundata);
    M = max(fundata);
    fundata = ((fundata - m) / (M-m)) - 0.5;   
    fundata = reshape(fundata,ox,oy);
    undata(:,ifeat:Nfeat:end) = fundata;
end

%then transform data again in order to have 0 mean and std 1 
% for ifeat=1:Nfeat
%     fundata = undata(:,ifeat:Nfeat:end);
%     ox = size(fundata,1);
%     oy = size(fundata,2);
%     fundata = reshape(fundata,1,ox*oy);
%     m = mean(fundata);
%     sd = std(fundata);
%     fundata = (fundata - m) ./ (sd);   
%     fundata = reshape(fundata,ox,oy);
%     undata(:,ifeat:Nfeat:end) = fundata;
% end
            
ndata = undata;

