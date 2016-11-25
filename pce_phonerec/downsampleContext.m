function ndata = downsampleContext(data,nf,ds)

nw = size(data,2)/nf;
cw = ceil(nw/2);

vw = [fliplr(cw-ds-1:-ds-1:1) cw:ds+1:nw];

vc = [];

for i=1:length(vw)    
    start = (vw(i)-1)*nf+1;
    vc = [vc start:start+nf-1];
end

ndata = data(:,vc);