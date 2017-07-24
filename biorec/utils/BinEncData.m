function [thrs, bintrain, bintest] = BinEncData(net,parnet,traindata,testdata)

nrands = 30000;
randdata = rand(nrands,size(traindata,2));

enranddata = nnFwd(net,randdata,parnet);

if(parnet.sparse==0)
    thrs = mean(enranddata,1);
% else
%     error(''
%      fp = which('kmeans');
%      fp = strrep(fp,'/kmeans.m','');
%      if(strfind(fp,'voicebox'))   
%        rmpath(fp);
%      end
%      rmpath(fp);
%      for i=1:size(enranddata,2)
%         indx = kmeans(enranddata(:,i),2);
%         a = enranddata(find(indx==1),i);
%         b = enranddata(find(indx==2),i);
%         if(min(a) > max(b))
%             thrs(i) = (min(a) + max(b))/2;
%         elseif(min(b) > max(a))
%             thrs(i) = (min(b) + max(a))/2;
%         else
%             error('Clustering error at node %d', i);
%         end
%      end
%      if(strfind(fp,'voicebox'))   
%         addpath(fp);
%      end
end

bintrain = nnFwd(net,traindata,parnet);
bintest = nnFwd(net,testdata,parnet);

bintrain = bsxfun(@gt,bintrain,thrs);
if ~isempty(testdata)
    bintest = bsxfun(@gt,bintest,thrs);
end