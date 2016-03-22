function [explogs cpriors] = getexplogs(obs,gammas,num_visitsall,net,parnet,mode,varargin)

if strcmp(mode,'dec'==0) && strcmp(mode,'enc')
    error('Unknown mode in ae_prob\n');
end

scaled = 1;
alpha = 0.001;

nh = parnet.units(end);
Q = 2.^nh;
T = size(obs,1);
if strcmp(mode,'dec')
    if length(varargin)<2 || isempty(varargin{1}) || isempty(varargin{2})
        error('Mode "dec" requires that the decoder and its parameters are passed as arguments');
    end
    decNet = varargin{1};
    parDecNet = varargin{2};
    explogs = zeros(size(obs));
    cpriors = [];
elseif strcmp(mode,'enc')
    explogs = zeros(T,nh);
    cpriors = num_visitsall./sum(num_visitsall);    
    cpriors(cpriors<0.0001) = 0.0001;
    %cpriors = alpha + (1-alpha) .* num_visitsall./sum(num_visitsall);
    cpriors = cpriors';
    if scaled
%        gammas = gammas./repmat(cpriors,T,1);
        logpriors = -log(cpriors);
        gammas = gammas.*repmat(logpriors,T,1);
        sumgamma = sum(gammas,2);
        sumgamma(sumgamma<0.0001) = 0.0001;
        gammas = gammas ./repmat(sumgamma,1,Q);
    end
end

for q=0:Q-1   
    zq = dec2bin(q,nh)-'0';
    zq = repmat(zq,T,1);
    if strcmp(mode,'dec')
        deczq = nnFwd(decNet,zq,parDecNet);
        gg = repmat(gammas(:,q+1),1,size(deczq,2));
        explogs = explogs+(deczq .* gg);
    end
    if strcmp(mode,'enc')
%         nzq = 1-zq;
%         if(q==0)
%             hq = undbnencode_hgpu(net,length(parnet.hlayers),obs,parnet.ogaus,parnet.sigm);
%         end
%        hpost(:,q+1) = prod(((hq.^zq).*((1-hq).^nzq)),2)./priors(q+1);
        explogs = explogs + zq .*repmat(gammas(:,q+1),1,nh);
    end            
end
if strcmp(mode,'enc')
%    explogs = explogs ./Q;
    explogs(explogs<0.0001) = 0.0001;
    explogs(explogs>0.9999) = 0.9999;
end

