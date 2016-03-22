function nseq  = rmRepInarow(seq)

nseq(1) = seq(1);
j = 1;
for i=2:length(seq)
    if(seq(i)~=nseq(j))
        j = j+1;
        nseq(j) = seq(i);
    end
end