function err = computeReconstructionError(ori,rec,varargin)
%function err = computeReconstructionError(ori,rec,varargin)

maxrand = max([50000 size(ori,1)]);

% possible modes
if(length(varargin) > 0)
    if(~(strcmp(varargin{1},'fast') || strcmp(varargin{1},'savemem') || strcmp(varargin{1},'sample')))
        mode = 'fast';
    else
        mode = varargin{1};
    end
else
    mode = 'fast';
end
if(length(varargin) > 1)
    compdeltas = varargin{2};
else
    compdeltas = 1;
end

D1 = size(ori,1);
D2 = size(ori,2);

if(strcmp(mode,'sample'))
    rindx = ceil(maxrand*rand(1,maxrand));
    ori = ori(rindx,:);
    rec = rec(rindx,:);
end

if(strcmp(mode,'savemem'))
    
    nori = fnormData(ori, D2,'minmax01');
    nrec = fnormData(rec, D2,'minmax01');
else
    min_ori = repmat(min(ori,[],1),D1,1);
    min_rec = repmat(min(rec,[],1),D1,1);

    max_ori = repmat(max(ori,[],1),D1,1);
    max_rec = repmat(max(rec,[],1),D1,1);

    if(sum(sum(max_ori-min_ori))~=0)
        nori = (ori - min_ori)./(max_ori - min_ori);
    else
        nori = ori;
    end

    if(sum(sum(max_rec-min_rec))~=0)
        nrec = (rec - min_rec)./(max_rec - min_rec);
    else
        nrec = rec;
    end
end

if(compdeltas && mod(D2,3)==0)
    d = D2/3;
end

% if isempty(varargin) ==0
%     stats = varargin;
%     mm = 1;
% 
%     pori = zeros(size(ori));
%     prec = zeros(size(rec));
%     for feat = 1:size(ori,2)
%         pori(:,feat) = nori(:,feat).*stats{2}(feat).sd+ones(size(ori,1),1)*stats{2}(feat).m;
%         prec(:,feat) = nrec(:,feat).*stats{2}(feat).sd+ones(size(ori,1),1)*stats{2}(feat).m;
%     end
%     
% else
%     mm = 0;
% end


%vavg average on samples
%eavg average on samples and features
%each for each articulatory features

%;err = (1/(size(nori,2)*size(ori,1)))*sum(sum((nori-nrec).^2));  %errore di ricostruzione complessivo
%nrmse normalized root mean squared error
if(strcmp(mode,'savemem'))
    err.nrmse.vavg = 0;
    err.nrmse.eavg = zeros(1,D2);
    err.rmse.vavg  = 0;
    err.rmse.eavg = zeros(1,D2);
    err.nrmse.each = zeros(1,D2);
    err.rmse.each = zeros(1,D2);
    
    err.corr.avg = 0;
    err.corr.each = zeros(1,D2);
    err.corr.posavg = 0;
    err.corr.velavg = 0;
    err.corr.accavg = 0;
            
    for i=1:D2
         err.nrmse.each(i) = err.nrmse.each(i) + sum((nori(:,i)-nrec(:,i)).^2);
         err.rmse.each(i) = err.rmse.each(i) + sum((ori(:,i)-rec(:,i)).^2);
                  
         err.corr.each(i) = corr(nori(:,i),nrec(:,i));
         err.corr.avg = err.corr.avg + err.corr.each(i); 
         if(compdeltas && mod(D2,3)==0)
            if(i<=d)
                err.corr.posavg = err.corr.posavg + err.corr.each(i);
            elseif(i > d && i <=2*d)
                err.corr.velavg = err.corr.velavg + err.corr.each(i);
            elseif(i > 2*d)
                err.corr.accavg = err.corr.accavg + err.corr.each(i);
            end
        end
    end                                        


     err.nrmse.vavg = sqrt((1/D1)*sum(err.nrmse.each));
     err.nrmse.eavg = (1/D2)*sum(sqrt((1/D1)*err.nrmse.each));
     err.rmse.vavg = sqrt((1/D1)*sum(err.rmse.each));
     err.rmse.eavg = (1/D2)*sum(sqrt((1/D1)*err.rmse.each));
     err.nrmse.each = sqrt(1/D1*err.nrmse.each);
     err.rmse.each = sqrt(1/D1*err.rmse.each);
     
     err.corr.avg = err.corr.avg / D2;

     if(compdeltas && mod(D2,3)==0)
        err.corr.posavg = err.corr.posavg./(D2/3);
        err.corr.velavg = err.corr.velavg./(D2/3);
        err.corr.accavg = err.corr.accavg./(D2/3);
     end
     
      if(compdeltas && mod(D2,3)==0)
        err.nrmse.vpos = sqrt((1/D1)*sum(err.nrmse.each(1:d)));
        err.nrmse.vvel = sqrt((1/D1)*sum(err.nrmse.each(d+1:2*d)));
        err.nrmse.vacc = sqrt((1/D1)*sum(err.nrmse.each((2*d)+1:3*d)));
        err.nrmse.epos = (1/(D2/3))*sum(sqrt((1/D1)*err.nrmse.each(1:d)));
        err.nrmse.evel = (1/(D2/3))*sum(sqrt((1/D1)*err.nrmse.each(d+1:2*d)));
        err.nrmse.eacc = (1/(D2/3))*sum(sqrt((1/D1)*err.nrmse.each((2*d)+1:3*d)));
        
        err.rmse.vpos = sqrt((1/D1)*sum(err.rmse.each(1:d)));
        err.rmse.vvel = sqrt((1/D1)*sum(err.rmse.each(d+1:2*d)));
        err.rmse.vacc = sqrt((1/D1)*sum(err.rmse.each((2*d)+1:3*d)));
        err.rmse.epos = (1/(D2/3))*sum(sqrt((1/D1)*err.rmse.each(1:d)));
        err.rmse.evel = (1/(D2/3))*sum(sqrt((1/D1)*err.rmse.each(d+1:2*d)));
        err.rmse.eacc = (1/(D2/3))*sum(sqrt((1/D1)*err.rmse.each((2*d)+1:3*d)));
     end
    
else
    nDf = nori - nrec;
    Df = ori-rec;
    err.nrmse.vavg = sqrt((1/D1)*sum(sum((nDf.^2),2)));
    err.nrmse.eavg = (1/D2)*sum(sqrt((1/D1)*(sum(((nDf).^2),1))));

    err.nrmse.each = zeros(1,D2);

    %rmse root mean squared error
    err.rmse.vavg = sqrt((1/D1)*sum(sum((Df.^2),2)));
    err.rmse.eavg = (1/D2)*sum(sqrt((1/D1)*(sum((Df.^2),1))));

    err.rmse.each = zeros(1,D2);

    % if mm
    %     %rmse root mean squared error in mm
    %     err.prmse.vavg = sqrt((1/size(pori,1))*sum(sum(((pori-prec).^2),2)));
    %     err.prmse.eavg = (1/size(pori,2))*sum(sqrt((1/size(pori,1))*(sum(((pori-prec).^2),1))));
    %     
    %     err.prmse.each = zeros(1,size(ori,2));
    % end


    err.corr.avg = 0;
    err.corr.each = zeros(1,D2);
    if(compdeltas && mod(D2,3)==0)
        err.corr.posavg = 0;
        err.corr.velavg = 0;
        err.corr.accavg = 0;
    end
    for i=1:length(err.rmse.each)
        err.nrmse.each(i) = sqrt((1/D1)*sum((Df(:,i)).^2));
        err.rmse.each(i) = sqrt((1/D1)*sum((Df(:,i)).^2));

    %     if mm
    %         err.prmse.each(i) = sqrt((1/size(pori,1))*sum((pori(:,i)-prec(:,i)).^2));
    %     end

        err.corr.each(i) = corr(nori(:,i),nrec(:,i));
        err.corr.avg = err.corr.avg + err.corr.each(i); 
        if(compdeltas && mod(D2,3)==0)
            if(i<=d)
                err.corr.posavg = err.corr.posavg + err.corr.each(i);
            elseif(i > d && i <=2*d)
                err.corr.velavg = err.corr.velavg + err.corr.each(i);
            elseif(i > 2*d)
                err.corr.accavg = err.corr.accavg + err.corr.each(i);
            end
        end
    end
    err.corr.avg = err.corr.avg / D2;

    if(compdeltas && mod(D2,3)==0)
        err.corr.posavg = err.corr.posavg./(D2/3);
        err.corr.velavg = err.corr.velavg./(D2/3);
        err.corr.accavg = err.corr.accavg./(D2/3);
    end

    if(compdeltas && mod(D2,3)==0)

        err.nrmse.vpos = sqrt((1/D1)*sum(sum(((nori(:,1:d)-nrec(:,1:d)).^2),2)));
        err.nrmse.vvel = sqrt((1/D1)*sum(sum(((nori(:,d+1:2*d)-nrec(:,d+1:2*d)).^2),2)));
        err.nrmse.vacc = sqrt((1/D1)*sum(sum(((nori(:,(2*d)+1:end)-nrec(:,(2*d)+1:end)).^2),2)));    
        err.nrmse.epos = (1/(D2/3))*sum(sqrt((1/D1)*(sum(((nori(:,1:d)-nrec(:,1:d)).^2),1))));
        err.nrmse.evel = (1/(D2/3))*sum(sqrt((1/D1)*(sum(((nori(:,d+1:2*d)-nrec(:,d+1:2*d)).^2),1))));
        err.nrmse.eacc = (1/(D2/3))*sum(sqrt((1/D1)*(sum(((nori(:,(2*d)+1:end)-nrec(:,(2*d)+1:end)).^2),1))));

        err.rmse.vpos = sqrt((1/D1)*sum(sum(((ori(:,1:d)-rec(:,1:d)).^2),2)));
        err.rmse.vvel = sqrt((1/D1)*sum(sum(((ori(:,d+1:2*d)-rec(:,d+1:2*d)).^2),2)));
        err.rmse.vacc = sqrt((1/D1)*sum(sum(((ori(:,(2*d)+1:end)-rec(:,(2*d)+1:end)).^2),2)));
        err.rmse.epos = (1/(D2/3))*sum(sqrt((1/D1)*(sum(((ori(:,1:d)-rec(:,1:d)).^2),1))));
        err.rmse.evel = (1/(D2/3))*sum(sqrt((1/D1)*(sum(((ori(:,d+1:2*d)-rec(:,d+1:2*d)).^2),1))));
        err.rmse.eacc = (1/(D2/3))*sum(sqrt((1/D1)*(sum(((ori(:,(2*d)+1:end)-rec(:,(2*d)+1:end)).^2),1))));

    end
end
    

