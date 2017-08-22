function [d dels ins subs sops]=edit_distance_levenshtein(t,s,itemtype)
m=length(t);
n=length(s);

d=zeros(m+1,n+1);
dels=zeros(m+1,n+1);
ins=zeros(m+1,n+1);
subs=zeros(m+1,n+1);

%bktr=zeros(m+1,n+1);
% initialize distance matrix
%deletion
for i=0:m
    d(i+1,1)=i;
    ops(i+1,1)='d';
    dels(i+1,1)=i;
end
%insertion
for j=0:n
    d(1,j+1)=j;
    ops(1,j+1)='i';
    ins(1,j+1)=j;
end


for j=2:n+1
    for i=2:m+1
%         dels(i,j)=dels(i-1,j-1);
%         ins(i,j)=ins(i-1,j-1);
%         subs(i,j)=subs(i-1,j-1);
        if ((strcmp(itemtype,'string') && strcmp(t(i-1),s(j-1))) ||  t(i-1) == s(j-1))
            d(i,j)=d(i-1,j-1);
            dels(i,j)=dels(i-1,j-1);
            ins(i,j)=ins(i-1,j-1);
            subs(i,j)=subs(i-1,j-1);
            ops(i,j) = 'n';
            bktr{1}(i,j) = i-1;
            bktr{2}(i,j) = j-1;
        else
            %         d(i,j)=min([ ...
            %           d(i-1,j) + 1, ...  % deletion
            %           d(i,j-1) + 1, ...  % insertion
            %           d(i-1,j-1) + 1 ... % substitution
            %           ]);
            [d(i,j) pos] = min([ ...
                d(i-1,j) + 1, ...  % deletion
                d(i,j-1) + 1, ...  % insertion
                d(i-1,j-1) + 1 ... % substitution
                ]);
            if(pos==1)
                dels(i,j) = dels(i-1,j) + 1;                
                ins(i,j)=ins(i-1,j);
                subs(i,j)=subs(i-1,j);
                ops(i,j) = 'd';
                bktr{1}(i,j) = i-1;
                bktr{2}(i,j) = j;
            elseif(pos==2)
                ins(i,j) = ins(i,j-1) + 1;
                dels(i,j)=dels(i,j-1);
                subs(i,j)=subs(i,j-1);    
                ops(i,j) = 'i';
                bktr{1}(i,j) = i;
                bktr{2}(i,j) = j-1;
            elseif(pos==3)
                subs(i,j) = subs(i-1,j-1) + 1;
                dels(i,j)=dels(i-1,j-1);
                ins(i,j)=ins(i-1,j-1);        
                ops(i,j) = 's';
                bktr{1}(i,j) = i-1;
                bktr{2}(i,j) = j-1;
            end
            
        end
    end
end
d=d(m+1,n+1);
dels=dels(m+1,n+1);
ins=ins(m+1,n+1);
subs=subs(m+1,n+1);
i = m+1;
j = n+1;
k = 1;
while(i > 1 || j > 1) 
    sops(k) = ops(i,j);
    if(j > 1 && i > 1)
        newi = bktr{1}(i,j);
        j = bktr{2}(i,j);
        i = newi;
    elseif(i == 1)
        j = j - 1;
    elseif(j == 1)
        i = i - 1;
    end
    
   k = k+1;
end
sops = sops(end:-1:1);
pippo = 1;