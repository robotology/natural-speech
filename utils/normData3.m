switch stream
    case 'audio'
        undata = audio;
        Nfeat = NoAuF;
        sindx = 1;
    case 'realMotor'
        undata = realMotor;
        Nfeat = NoMF;
        sindx = 2;
    case 'recMotor'
        undata = recMotor;
        Nfeat = NoMF;
        sindx = 4;
    case 'audio4mapping'
        undata = audio4map;
        Nfeat = NoAuF_map;
        sindx = 3;
    case 'mappedaudio1'
        undata = audioListener;
        Nfeat = NoAuF;
        sindx = 5;
    case 'mappedaudio2'
        undata = audioSpeaker;
        Nfeat = NoAuF;
        sindx = 6;
    case 'mappedmotor'
        undata = realMotorListener;
        Nfeat = NoMF;
        sindx = 7;
end

%sbj = unique(subjects);


if(~isempty(varargin) &&  strcmp(datatype,'test'))
    stats = varargin{1};
    for ifeat=1:Nfeat
        fundata = undata(:,ifeat);
        ox = size(fundata,1);
        oy = size(fundata,2);
        if(stats{sindx}(ifeat).sd > 0)
            fundata = (fundata - stats{sindx}(ifeat).m) ./ (stats{sindx}(ifeat).sd);
        else
            fundata = fundata - stats{sindx}(ifeat).m;
        end
        fundata = reshape(fundata,ox,oy);
        undata(:,ifeat:Nfeat:end) = fundata;
    end   
    
% elseif length(sbj)>0
%     for sb = 1:length(sbj)
%         index = find(subjects==sbj(sb));
%         undata1 = undata(index,:);
%         for ifeat=1:Nfeat
%         fundata = undata1(:,ifeat);
%         ox = size(fundata,1);
%         oy = size(fundata,2);
%         fundata = reshape(fundata,1,ox*oy);
%         stats{sindx}(ifeat).m(sb) = mean(fundata);
%         stats{sindx}(ifeat).sd(sb) = std(fundata);
%         if(stats{sindx}(ifeat).sd(sb) > 0)
%             fundata = (fundata - stats{sindx}(ifeat).m(sb)) ./ (stats{sindx}(ifeat).sd(sb));
%         else
%             fundata = fundata - stats{sindx}(ifeat).m(sb);
%         end
%         fundata = reshape(fundata,ox,oy);
%         undata(index,ifeat:Nfeat:end) = fundata;
%         end
%         clear undata1 index
%     end
else
    for ifeat=1:Nfeat
        fundata = undata(:,ifeat);
        ox = size(fundata,1);
        oy = size(fundata,2);
        fundata = reshape(fundata,1,ox*oy);
        stats{sindx}(ifeat).m = mean(fundata);
        stats{sindx}(ifeat).sd = std(fundata);
        if(stats{sindx}(ifeat).sd > 0)
            fundata = (fundata - stats{sindx}(ifeat).m) ./ (stats{sindx}(ifeat).sd);
        else
            fundata = fundata - stats{sindx}(ifeat).m;
        end
        fundata = reshape(fundata,ox,oy);
        undata(:,ifeat:Nfeat:end) = fundata;
    end
end


ndata = undata;

