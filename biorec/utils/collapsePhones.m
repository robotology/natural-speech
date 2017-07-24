function newseq = collapsePhones(seq,phMap,id2pMap,p2idMap)

newseq = cell(1,length(seq));
for i=1:length(seq)
    a = seq{i};
    newseq{i} = seq{i};
    for j=1:length(a)
        ph = id2pMap(a(j));
        if(isKey(phMap,ph))
            b = phMap(ph);
            newseq{i}(j) = p2idMap(b);            
        end
    end
end
pippo = 1;