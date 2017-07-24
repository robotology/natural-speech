function dsdata = discretizeData(ae,parae,dec,pardec,data)

%rmse or cossim
simmeasure = 'rmse';
endata = nnFwd(ae,data,parae);
decdata = nnFwd(dec,endata,pardec);

nEncUnits = parae.units(end);
nsym = (2^nEncUnits)-1;
numdata = dec2bin(1:nsym,nEncUnits);

decnumdata = nnFwd(dec,numdata,pardec);

if(strcmp(simmeasure,'rmse'))
    rmsem = zeros(size(decdata,1),nsym);
    for i = 1:nsym
        rtmp = repmat(decnumdata(i,:),size(decdata,1),1);
        rmsem(:,i) = sqrt((sum((decdata - rtmp).^2,2))/size(decdata,2));
    end
    [~, p] = min(rmsem,[],2);
else
    nd1 = sqrt(sum(decdata.^2,2));
    nd2 = sqrt(sum(decnumdata.^2,2));
    csim = (decdata * decnumdata')./(nd1*nd2');
    [~, p] = max(csim,[],2);    
end

dsdata = dec2bin(p,nEncUnits) - '0';
