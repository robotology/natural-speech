function pr = ae_prob(obs,priors,net,parnet,mode,varargin)

scaled = 1;
alpha = 0.001;

if ~strcmp(mode,'rec') && ~strcmp(mode,'dec'==0) && ~strcmp(mode,'enc')
    error('Unknown mode in ae_prob\n');
end
obs = obs';
nh = parnet.units(end);
Q = 2^nh;
T = size(obs,2);
nF = size(obs,1);
pr = zeros(T,Q);
pr2 = zeros(T,Q);
for q=0:Q-1
    zq = dec2bin(q,nh)-'0';
    zq = repmat(zq,T,1);
    if strcmp(mode,'rec') || strcmp(mode,'dec')
        if length(varargin)<2 || isempty(varargin{1}) || isempty(varargin{2})
            error('Mode %s requires that the decoder and its parameters are passed as arguments',mode);
        end
        decNet = varargin{1};
        parDecNet = varargin{2};
        deczq = nnFwd(decNet,zq,parDecNet);
%        pr(:,q+1) = mvnpdf(obs'- deczq,mu,sigma);
        pr(:,q+1) = exp(-sum((obs'- deczq).^2,2));
    end
    if strcmp(mode,'rec') || strcmp(mode,'enc')
        nzq = 1-zq;
        if q==0
            hq = nnFwd(net,obs',parnet);
        end
        pr2(:,q+1) = prod(((hq.^zq).*((1-hq).^nzq)),2);
    end            
end

if strcmp(mode,'rec')
    pr= pr.*pr2;
elseif strcmp(mode,'enc')
    if scaled
        priors(priors < eps) = eps;
        pr2 =  pr2./repmat(priors,T,1);
%        pr2 =  pr2./(alpha + ((1-alpha) .* repmat(priors,T,1)));
        sumpr2 = repmat(sum(pr2,2),1,Q);
        pr = pr2 ./sumpr2;
    else
        pr  = pr2;
    end
end

end