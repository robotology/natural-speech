function [per, totdels, totins, totsubs, sops, substitution ] = computePhoneRecognitionError_shortPhS(testseq,predseq,nph,corpus)

timitPH = {'aa','ae','ah','ao','aw','ax','axh','axr','ay','b','bcl','ch','d','dcl',...
         'dh','dx','eh','el','em','en','eng','epi','er','ey','f','g','gcl','hC','hh',...
         'hv','ih','ix','iy','jh','k','kcl','l','m','n','ng','nx','ow','oy','p','pau',...
         'pcl','q','r','s','sh','t','tcl','th','uh','uw','ux','v','w','y','z','zh'};

timitAlloph = {'ao','axh','ax','axr','hv','ix','el','em','en','nx','eng','zh','ux',...
          'pcl','tcl','kcl','bcl','dcl','pau','epi','q'};
timitMainph = {'aa','ah','ah','er','hh','ih','l','m','n','n','ng','sh','uw','hC','hC',...
          'hC','hC','hC','hC','hC','hC'};      
      
      

% ao aa
% axh ah
% ax ah
% axr er
% hv hh
% ix ih
% el l
% em m
% en n
% nx n
% eng ng
% zh sh
% ux uw
% pcl sil
% tcl sil
% kcl sil
% bcl sil
% dcl sil
% hC sil
% pau sil
% epi sil
% q sil


if(strcmp(corpus,'timit'))    
    phSet = timitPH;
    phMap = containers.Map(timitAlloph,timitMainph);        
else
    fprintf(1,'Corpus %s unknown\n',corpus);
end

ids = 1:length(phSet);
p2idMap = containers.Map(phSet,ids);
id2pMap = containers.Map(ids,phSet);

testseq = collapsePhones(testseq,phMap,id2pMap,p2idMap);
predseq = collapsePhones(predseq,phMap,id2pMap,p2idMap);

nutts = length(testseq);


sops = cell(1,nutts);
totdels = 0;
totins = 0;
totsubs = 0;

substitution = zeros(nph,1);
nphonetype = zeros(nph,1);

errc = 0;
phonc = 0;
for i=1:nutts
    [d, dels, ins, subs, sops{i}] = edit_distance_levenshtein(testseq{i},predseq{i},'');
    errc = errc + d;
    totdels = totdels + dels;
    totins = totins + ins;
    totsubs = totsubs + subs;
    phonc = phonc + length(testseq{i});
    for h = 1:length(testseq{i})
        nphonetype(testseq{i}(h)) = nphonetype(testseq{i}(h))+1;
    end
    
    tmpops = sops{i};
    tmpops(tmpops=='i') = [];
    sub = find(tmpops == 's');
    for h = 1:length(sub)
        substitution(testseq{i}(sub(h))) = substitution(testseq{i}(sub(h)))+1;
    end    
    clear sub tmpops
end
per = errc/phonc;
substitution = substitution./nphonetype;
totdels = totdels/phonc;
totins = totins./phonc;
totsubs = totsubs./phonc;



