function [recnet ptrainy ptesty] = recLinearRegress(trainx,trainy,testx,varargin)

if(~isempty(varargin) && varargin{1}==0)
    outfunc{1} = 'logsig';
else
    outfunc{1} = 'purelin';
end

recnet = newff(minmax(double(trainx')),[size(trainy,2)],outfunc,'trainscg','learngdm','msereg');

recnet.trainParam.epochs = 500;
recnet.performParam.ratio = 1;
recnet.trainParam.show = NaN;
recnet.trainParam.goal = 1e-5;
recnet = train(recnet,double(trainx'),double(trainy'));
ptrainy = sim(recnet,double(trainx'));
if(~isempty(testx))
    ptesty = sim(recnet,double(testx'));
else
    ptesty = [];
end

ptrainy = ptrainy';
ptesty  = ptesty';