function [cm cmr accuracy raccuracy] = createConfusionMatrix(target,pred,nclass,state,rframes,vSP,vEP)

cm = zeros(nclass/state,nclass/state);
cmr = zeros(nclass/state,nclass/state);

target = ceil(target./state);
pred = ceil(pred./state);

for i=1:length(target)
    cm(target(i),pred(i)) = cm(target(i),pred(i)) + 1;
end

for l = 1:length(vSP)
    for j = vSP(l):vEP(l)
        r = 0;
        if pred(j) ~= target(j)
            for fr = 1:rframes
                if ((j>vSP(l)+fr && j<vEP(l)-fr+1) && (pred(j) == target(j-fr) || pred(j) == target(j+fr)))
                    r = 1;
                elseif j>vSP(l)+fr && (pred(j) == target(j-fr))
                    r = 1;
                elseif j<vEP(l)-fr+1 && pred(j) == target(j+fr)
                    r = 1;
                end
            end

            if r==0 
                cmr(target(j),pred(j)) = cmr(target(j),pred(j))+1;
            end  
        else
            cmr(target(j),pred(j)) = cmr(target(j),pred(j))+1;
        end
    end   
end

Ntot = sum(sum(cm));
Nrtot = sum(sum(cmr));
Nok = sum(diag(cm));
NoR = sum(diag(cmr));
accuracy = Nok/Ntot;
raccuracy = NoR/Nrtot;