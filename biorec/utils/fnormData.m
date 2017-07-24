function [ndata p1 p2] = fnormData(undata,Nfeat,type)

%Nfeat is always an integer
%Nfeat = size(undata,2)/Nframes;

p1 = [];
p2 = [];
if(strcmp(type,'m0s1'))
    for ifeat=1:Nfeat
        fundata = undata(:,ifeat:Nfeat:end);
        ox = size(fundata,1);
        oy = size(fundata,2);
        fundata = reshape(fundata,1,ox*oy);
        m = mean(fundata);
        sd = std(fundata);
        p1 = [p1 m];
        p2 = [p2 sd];
        if(sd==0)
            fundata = fundata - m;
        else
            fundata = (fundata - m) ./ (sd);
        end
        fundata = reshape(fundata,ox,oy);
        undata(:,ifeat:Nfeat:end) = fundata;
    end
elseif(strcmp(type,'minmax01'))
    for ifeat=1:Nfeat
        fundata = undata(:,ifeat:Nfeat:end);
        ox = size(fundata,1);
        oy = size(fundata,2);
        fundata = reshape(fundata,1,ox*oy);
        m = min(fundata);
        M = max(fundata);
        p1 = [p1 m];
        p2 = [p2 M];        
        fundata = (fundata - m) / (M-m);
        fundata = fundata * 0.8;
        fundata = fundata + 0.1;
        fundata = reshape(fundata,ox,oy);
        undata(:,ifeat:Nfeat:end) = fundata;
    end
end            
% rescale data in the range [0 1]
ndata = undata;

