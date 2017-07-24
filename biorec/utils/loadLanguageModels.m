function [p_uni p_bi s_uni s_bi] = loadLanguageModels(p_unifile,p_bifile,s_unifile,s_bifile,nstates,blog)

p_uni = dlmread(p_unifile,'\t');
mp = dlmread(p_bifile,'\t');
s_uni = dlmread(s_unifile,'\t');
ms = dlmread(s_bifile,'\t');

if(length(p_uni) ~= length(s_uni)/nstates)
    error('Inconsistency in the number of phones\n');
end

if(rem(length(mp),length(p_uni))==0)
    p_bi = reshape(mp,length(p_uni),length(p_uni));
    p_bi = p_bi';
else
    error('the number of phone bigrams must be the lenght(unigram).^2\n');
end


if(rem(length(ms),length(s_uni)))==0
     s_bi = reshape(ms,length(s_uni),length(s_uni));
     s_bi = s_bi';

else
    tmp = -1*ones(length(s_uni),nstates)/0;
    k=1;
    for z=0:length(p_uni)-1
        for i=1:nstates
            for j=1:nstates
                if(i==j || (i+1==j))
                    tmp(i+(z*nstates),j) = ms(k);
                    k = k+1;
                end
            end
        end
    end

    sst = (nstates-1)*2 + 1;
    if(length(ms)/sst == length(p_uni))
        s_bi = -1*ones(size(tmp,1),size(tmp,1))/0;    
        for i=1:nstates:size(tmp,1)-nstates+1
            s_bi(i:i+nstates-1,i:i+nstates-1) = tmp(i:i+nstates-1,:);
        end
    else
        error('the number of state bigrams is different\n');
    end
end


if(blog == 0)
    p_uni = 10.^(p_uni);
    s_uni = 10.^(s_uni);
    p_bi = 10.^(p_bi);
    s_bi = 10.^(s_bi);
end