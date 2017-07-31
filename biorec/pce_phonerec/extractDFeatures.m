function dfM = extractDFeatures(labels,corpus)

% EXTRACTDFEATURES Extracts the vector of distinctive/articulatory 
% features from each phonetic label
% IN
%  labels: N x K matrix. N = No. of frames/observations, K = No. of phone labels
%          a phone label is represented as a 1-of-K vector
%  corpus: name of the dataset
%
% OUT
%  dfM : N x F matrix. F = number of distinctive features

%distincive features
%old (pre-timit)
% dFkeys = {'vowel','diphtong','close','mid','open','front','central','back',...
%     'long','short','close2','mid2','open2','front2','central2','back2',...
%     'long2','short2','nasalized','consonant','voiced','unvoiced','fricative',...
%     'nasal','stop','approximant','affricate','labial','dental','alveolar',...
%     'lateral','post-alveolar','palatal','velar','glottal','syllabic'};
dFkeys = {'vowel','diphtong','close','close-mid','mid','open-mid','open','front',...
    'central','back','long','short','close2','close-mid2','mid2','open-mid2','open2',...
    'front2','central2','back2','long2','short2','nasalized','rmerged','consonant','voiced',...
    'unvoiced','fricative','nasal','stop','approximant','affricate','labial','dental',...
    'alveolar','lateral','post-alveolar','palatal','velar','glottal','syllabic','flapping',...
    'silence'};

nF = length(dFkeys);
dFids = 1:nF;
dFmap = containers.Map(dFkeys,dFids);


%mngu0 phone set
mnguPH = {'@','@@','@U','A','D','E','E@','I','I@','N','O','OI','Q','S','T','U',...
        'U@','V','Z','a','aI','aU','b','d','dZ','eI','f','g','h','i','j','k',...
        'l','l!','lw','m','m!','n','n!','o^','p','r','s','t','tS','u','v',...
        'w','z'};
timitPH = {'aa','ae','ah','ao','aw','ax','axh','axr','ay','b','bcl','ch','d','dcl',...
         'dh','dx','eh','el','em','en','eng','epi','er','ey','f','g','gcl','hC','hh',...
         'hv','ih','ix','iy','jh','k','kcl','l','m','n','ng','nx','ow','oy','p','pau',...
         'pcl','q','r','s','sh','t','tcl','th','uh','uw','ux','v','w','y','z','zh'};
         
if(strcmp(corpus,'mngu0'))
    lenpH = length(mnguPH);
    values = 1:lenpH;
    pHmap = containers.Map(values,mnguPH);
elseif(strcmp(corpus,'timit'))
    lenpH = length(timitPH);
    values = 1:lenpH;
    pHmap = containers.Map(values,timitPH);
    
else
    error('Corpus unknown\n');
end

nP= size(labels,2);
if(nP ~= lenpH)
    error('Number of phonemes in the phone set is different from number of input columns\n');
end

N = size(labels,1);
dfM = zeros(N,nF);

if(strcmp(corpus,'mngu0'))
    for i=1:N
       ip = find(labels(i,:)==1);
        if(strcmp(pHmap(ip),'@'))
            dfM(i,dFmap('vowel')) = 1;dfM(i,dFmap('central')) = 1;dfM(i,dFmap('mid')) = 1;
            dfM(i,dFmap('short')) = 1;
        elseif(strcmp(pHmap(ip),'@@'))
            dfM(i,dFmap('vowel')) = 1;dfM(i,dFmap('central')) = 1;dfM(i,dFmap('mid')) = 1;
            dfM(i,dFmap('long')) = 1;
        elseif(strcmp(pHmap(ip),'@U'))
            dfM(i,dFmap('vowel')) = 1;dfM(i,dFmap('diphtong')) = 1;
            dfM(i,dFmap('central')) = 1;dfM(i,dFmap('mid')) = 1;dfM(i,dFmap('short')) = 1;
            dfM(i,dFmap('back2')) = 1;dfM(i,dFmap('close2')) = 1;dfM(i,dFmap('short2')) = 1;    
        elseif(strcmp(pHmap(ip),'A'))
            dfM(i,dFmap('vowel')) = 1;dfM(i,dFmap('back')) = 1;dfM(i,dFmap('open')) = 1;
            dfM(i,dFmap('long')) = 1;
        elseif(strcmp(pHmap(ip),'D'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('voiced')) = 1;dfM(i,dFmap('fricative')) = 1;
            dfM(i,dFmap('dental')) = 1;    
        elseif(strcmp(pHmap(ip),'E'))
            dfM(i,dFmap('vowel')) = 1;dfM(i,dFmap('front')) = 1;dfM(i,dFmap('mid')) = 1;
            dfM(i,dFmap('short')) = 1;
        elseif(strcmp(pHmap(ip),'E@'))
            dfM(i,dFmap('vowel')) = 1;dfM(i,dFmap('diphtong')) = 1;
            dfM(i,dFmap('front')) = 1;dfM(i,dFmap('mid')) = 1;dfM(i,dFmap('short')) = 1;
            dfM(i,dFmap('central2')) = 1;dfM(i,dFmap('mid2')) = 1;dfM(i,dFmap('short2')) = 1;  
        elseif(strcmp(pHmap(ip),'I'))
            dfM(i,dFmap('vowel')) = 1;dfM(i,dFmap('front')) = 1;dfM(i,dFmap('close')) = 1;
            dfM(i,dFmap('short')) = 1;
        elseif(strcmp(pHmap(ip),'I@'))
            dfM(i,dFmap('vowel')) = 1;dfM(i,dFmap('diphtong')) = 1;
            dfM(i,dFmap('front')) = 1;dfM(i,dFmap('close')) = 1;dfM(i,dFmap('short')) = 1;
            dfM(i,dFmap('central2')) = 1;dfM(i,dFmap('mid2')) = 1;dfM(i,dFmap('short2')) = 1; 
        elseif(strcmp(pHmap(ip),'N'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('unvoiced')) = 1;dfM(i,dFmap('nasal')) = 1;
            dfM(i,dFmap('velar')) = 1;
        elseif(strcmp(pHmap(ip),'O'))
            dfM(i,dFmap('vowel')) = 1;dfM(i,dFmap('back')) = 1;dfM(i,dFmap('mid')) = 1;
            dfM(i,dFmap('long')) = 1;
        elseif(strcmp(pHmap(ip),'OI'))
            dfM(i,dFmap('vowel')) = 1;dfM(i,dFmap('diphtong')) = 1;
            dfM(i,dFmap('back')) = 1;dfM(i,dFmap('mid')) = 1;dfM(i,dFmap('short')) = 1;
            dfM(i,dFmap('front2')) = 1;dfM(i,dFmap('close2')) = 1;dfM(i,dFmap('short2')) = 1;
        elseif(strcmp(pHmap(ip),'Q'))
            dfM(i,dFmap('vowel')) = 1;dfM(i,dFmap('back')) = 1;dfM(i,dFmap('open')) = 1;
            dfM(i,dFmap('short')) = 1;    
        elseif(strcmp(pHmap(ip),'S'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('unvoiced')) = 1;dfM(i,dFmap('fricative')) = 1;
            dfM(i,dFmap('post-alveolar')) = 1;    
        elseif(strcmp(pHmap(ip),'T'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('unvoiced')) = 1;dfM(i,dFmap('fricative')) = 1;
            dfM(i,dFmap('dental')) = 1;
        elseif(strcmp(pHmap(ip),'U'))
            dfM(i,dFmap('vowel')) = 1;dfM(i,dFmap('back')) = 1;dfM(i,dFmap('close')) = 1;
            dfM(i,dFmap('short')) = 1;
        elseif(strcmp(pHmap(ip),'U@'))
            dfM(i,dFmap('vowel')) = 1;dfM(i,dFmap('diphtong')) = 1;
            dfM(i,dFmap('back')) = 1;dfM(i,dFmap('close')) = 1;dfM(i,dFmap('short')) = 1;
            dfM(i,dFmap('central2')) = 1;dfM(i,dFmap('mid2')) = 1;dfM(i,dFmap('short2')) = 1; 
        elseif(strcmp(pHmap(ip),'V'))
            dfM(i,dFmap('vowel')) = 1;dfM(i,dFmap('central')) = 1;dfM(i,dFmap('open')) = 1;
            dfM(i,dFmap('short')) = 1;
        elseif(strcmp(pHmap(ip),'Z'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('voiced')) = 1;dfM(i,dFmap('fricative')) = 1;
            dfM(i,dFmap('alveolar')) = 1;
        elseif(strcmp(pHmap(ip),'a'))
            dfM(i,dFmap('vowel')) = 1;dfM(i,dFmap('front')) = 1;dfM(i,dFmap('open')) = 1;
            dfM(i,dFmap('short')) = 1;
        elseif(strcmp(pHmap(ip),'aI'))
            dfM(i,dFmap('vowel')) = 1;dfM(i,dFmap('diphtong')) = 1;
            dfM(i,dFmap('front')) = 1;dfM(i,dFmap('open')) = 1;dfM(i,dFmap('short')) = 1;
            dfM(i,dFmap('front2')) = 1;dfM(i,dFmap('close2')) = 1;dfM(i,dFmap('short2')) = 1;
        elseif(strcmp(pHmap(ip),'aU'))
            dfM(i,dFmap('vowel')) = 1;dfM(i,dFmap('diphtong')) = 1;
            dfM(i,dFmap('front')) = 1;dfM(i,dFmap('open')) = 1;dfM(i,dFmap('short')) = 1;
            dfM(i,dFmap('back2')) = 1;dfM(i,dFmap('close2')) = 1;dfM(i,dFmap('short2')) = 1; 
        elseif(strcmp(pHmap(ip),'b'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('voiced')) = 1;dfM(i,dFmap('stop')) = 1;
            dfM(i,dFmap('labial')) = 1; 
        elseif(strcmp(pHmap(ip),'d'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('voiced')) = 1;dfM(i,dFmap('stop')) = 1;
            dfM(i,dFmap('alveolar')) = 1;  
        elseif(strcmp(pHmap(ip),'dZ'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('voiced')) = 1;dfM(i,dFmap('affricate')) = 1;
            dfM(i,dFmap('post-alveolar')) = 1;
        elseif(strcmp(pHmap(ip),'eI'))
            dfM(i,dFmap('vowel')) = 1;dfM(i,dFmap('diphtong')) = 1;
            dfM(i,dFmap('front')) = 1;dfM(i,dFmap('mid')) = 1;dfM(i,dFmap('short')) = 1;
            dfM(i,dFmap('front2')) = 1;dfM(i,dFmap('close2')) = 1;dfM(i,dFmap('short2')) = 1; 
        elseif(strcmp(pHmap(ip),'f'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('unvoiced')) = 1;dfM(i,dFmap('fricative')) = 1;
            dfM(i,dFmap('labial')) = 1; 
        elseif(strcmp(pHmap(ip),'g'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('voiced')) = 1;dfM(i,dFmap('stop')) = 1;
            dfM(i,dFmap('velar')) = 1;
        elseif(strcmp(pHmap(ip),'h'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('unvoiced')) = 1;dfM(i,dFmap('fricative')) = 1;
            dfM(i,dFmap('glottal')) = 1; 
        elseif(strcmp(pHmap(ip),'i'))
            dfM(i,dFmap('vowel')) = 1;dfM(i,dFmap('front')) = 1;dfM(i,dFmap('close')) = 1;
            dfM(i,dFmap('long')) = 1;
        elseif(strcmp(pHmap(ip),'j'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('voiced')) = 1;dfM(i,dFmap('approximant')) = 1;
            dfM(i,dFmap('palatal')) = 1;  
        elseif(strcmp(pHmap(ip),'k'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('unvoiced')) = 1;dfM(i,dFmap('stop')) = 1;
            dfM(i,dFmap('velar')) = 1;
        elseif(strcmp(pHmap(ip),'l'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('voiced')) = 1;dfM(i,dFmap('lateral')) = 1;
            dfM(i,dFmap('alveolar')) = 1; 
        elseif(strcmp(pHmap(ip),'l!'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('voiced')) = 1;dfM(i,dFmap('lateral')) = 1;
            dfM(i,dFmap('alveolar')) = 1;dfM(i,dFmap('syllabic')) = 1;    
        elseif(strcmp(pHmap(ip),'lw'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('voiced')) = 1;dfM(i,dFmap('approximant')) = 1;
            dfM(i,dFmap('velar')) = 1;dfM(i,dFmap('syllabic')) = 1;
        elseif(strcmp(pHmap(ip),'m'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('voiced')) = 1;dfM(i,dFmap('nasal')) = 1;
            dfM(i,dFmap('labial')) = 1;
        elseif(strcmp(pHmap(ip),'m!'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('voiced')) = 1;dfM(i,dFmap('nasal')) = 1;
            dfM(i,dFmap('labial')) = 1;dfM(i,dFmap('syllabic')) = 1;
        elseif(strcmp(pHmap(ip),'n'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('voiced')) = 1;dfM(i,dFmap('nasal')) = 1;
            dfM(i,dFmap('alveolar')) = 1; 
        elseif(strcmp(pHmap(ip),'n!'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('voiced')) = 1;dfM(i,dFmap('nasal')) = 1;
            dfM(i,dFmap('alveolar')) = 1;dfM(i,dFmap('syllabic')) = 1;
        elseif(strcmp(pHmap(ip),'o^'))
            dfM(i,dFmap('vowel')) = 1;dfM(i,dFmap('back')) = 1;dfM(i,dFmap('mid')) = 1;    
            dfM(i,dFmap('nasalized')) = 1;dfM(i,dFmap('short')) = 1;  
        elseif(strcmp(pHmap(ip),'p'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('unvoiced')) = 1;dfM(i,dFmap('stop')) = 1;
            dfM(i,dFmap('labial')) = 1;
        elseif(strcmp(pHmap(ip),'r'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('voiced')) = 1;dfM(i,dFmap('approximant')) = 1;
            dfM(i,dFmap('post-alveolar')) = 1;
        elseif(strcmp(pHmap(ip),'s'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('unvoiced')) = 1;dfM(i,dFmap('fricative')) = 1;
            dfM(i,dFmap('alveolar')) = 1;
        elseif(strcmp(pHmap(ip),'t'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('unvoiced')) = 1;dfM(i,dFmap('stop')) = 1;
            dfM(i,dFmap('alveolar')) = 1;
        elseif(strcmp(pHmap(ip),'u'))
            dfM(i,dFmap('vowel')) = 1;dfM(i,dFmap('back')) = 1;dfM(i,dFmap('close')) = 1;    
            dfM(i,dFmap('long')) = 1;
        elseif(strcmp(pHmap(ip),'v'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('voiced')) = 1;dfM(i,dFmap('fricative')) = 1;
            dfM(i,dFmap('labial')) = 1;
        elseif(strcmp(pHmap(ip),'w'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('voiced')) = 1;dfM(i,dFmap('approximant')) = 1;
            dfM(i,dFmap('velar')) = 1;    
         elseif(strcmp(pHmap(ip),'z'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('voiced')) = 1;dfM(i,dFmap('fricative')) = 1;
            dfM(i,dFmap('alveolar')) = 1;   
        end
       if(mod(i,ceil(N/100))==1)
           fprintf(1,'Row %d of %d\n',i,N);
       end
    end

elseif(strcmp(corpus,'timit'))
    for i=1:N
       ip = find(labels(i,:)==1);
       if(strcmp(pHmap(ip),'aa'))
            dfM(i,dFmap('vowel')) = 1;dfM(i,dFmap('back')) = 1;dfM(i,dFmap('open')) = 1;
            dfM(i,dFmap('short')) = 1;
       elseif(strcmp(pHmap(ip),'ae'))
            dfM(i,dFmap('vowel')) = 1;dfM(i,dFmap('front')) = 1;dfM(i,dFmap('open')) = 1;
            dfM(i,dFmap('short')) = 1;
       elseif(strcmp(pHmap(ip),'ah'))
            dfM(i,dFmap('vowel')) = 1;dfM(i,dFmap('central')) = 1;dfM(i,dFmap('open-mid')) = 1;
            dfM(i,dFmap('short')) = 1;
       elseif(strcmp(pHmap(ip),'ao'))
            dfM(i,dFmap('vowel')) = 1;dfM(i,dFmap('back')) = 1;dfM(i,dFmap('open-mid')) = 1;
            dfM(i,dFmap('long')) = 1;
       elseif(strcmp(pHmap(ip),'aw'))
            dfM(i,dFmap('vowel')) = 1;dfM(i,dFmap('diphtong')) = 1;
            dfM(i,dFmap('central')) = 1;dfM(i,dFmap('open')) = 1;dfM(i,dFmap('short')) = 1;
            dfM(i,dFmap('back2')) = 1;dfM(i,dFmap('close-mid2')) = 1;dfM(i,dFmap('short2')) = 1;     
       elseif(strcmp(pHmap(ip),'ax')||strcmp(pHmap(ip),'axh'))
            dfM(i,dFmap('vowel')) = 1;dfM(i,dFmap('central')) = 1;dfM(i,dFmap('mid')) = 1;
            dfM(i,dFmap('short')) = 1;
       elseif(strcmp(pHmap(ip),'axr'))
            dfM(i,dFmap('vowel')) = 1;dfM(i,dFmap('central')) = 1;dfM(i,dFmap('mid')) = 1;
            dfM(i,dFmap('rmerged')) = 1;dfM(i,dFmap('short')) = 1;
       elseif(strcmp(pHmap(ip),'ay'))
            dfM(i,dFmap('vowel')) = 1;dfM(i,dFmap('diphtong')) = 1;
            dfM(i,dFmap('central')) = 1;dfM(i,dFmap('open')) = 1;dfM(i,dFmap('short')) = 1;
            dfM(i,dFmap('front2')) = 1;dfM(i,dFmap('close-mid2')) = 1;dfM(i,dFmap('short2')) = 1;
       elseif(strcmp(pHmap(ip),'b') || strcmp(pHmap(ip),'bcl'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('voiced')) = 1;dfM(i,dFmap('stop')) = 1;
            dfM(i,dFmap('labial')) = 1;       
       elseif(strcmp(pHmap(ip),'ch'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('unvoiced')) = 1;dfM(i,dFmap('affricate')) = 1;
            dfM(i,dFmap('post-alveolar')) = 1;     
       elseif(strcmp(pHmap(ip),'d') || strcmp(pHmap(ip),'dcl'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('voiced')) = 1;dfM(i,dFmap('stop')) = 1;
            dfM(i,dFmap('alveolar')) = 1;       
       elseif(strcmp(pHmap(ip),'dh'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('voiced')) = 1;dfM(i,dFmap('fricative')) = 1;
            dfM(i,dFmap('dental')) = 1;
       elseif(strcmp(pHmap(ip),'dx'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('voiced')) = 1;dfM(i,dFmap('stop')) = 1;
            dfM(i,dFmap('alveolar')) = 1;dfM(i,dFmap('flapping')) = 1;
       elseif(strcmp(pHmap(ip),'eh'))
            dfM(i,dFmap('vowel')) = 1;dfM(i,dFmap('front')) = 1;dfM(i,dFmap('open-mid')) = 1;
            dfM(i,dFmap('short')) = 1;
       elseif(strcmp(pHmap(ip),'el'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('voiced')) = 1;dfM(i,dFmap('lateral')) = 1;
            dfM(i,dFmap('alveolar')) = 1;dfM(i,dFmap('syllabic')) = 1;
       elseif(strcmp(pHmap(ip),'em'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('voiced')) = 1;dfM(i,dFmap('nasal')) = 1;
            dfM(i,dFmap('labial')) = 1;dfM(i,dFmap('syllabic')) = 1;        
       elseif(strcmp(pHmap(ip),'en'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('voiced')) = 1;dfM(i,dFmap('nasal')) = 1;
            dfM(i,dFmap('alveolar')) = 1;dfM(i,dFmap('syllabic')) = 1;
       elseif(strcmp(pHmap(ip),'eng') || strcmp(pHmap(ip),'ng'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('unvoiced')) = 1;dfM(i,dFmap('nasal')) = 1;
            dfM(i,dFmap('velar')) = 1;dfM(i,dFmap('syllabic')) = 1;
       elseif(strcmp(pHmap(ip),'epi'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('fricative')) = 1;dfM(i,dFmap('silence')) = 1;
       elseif(strcmp(pHmap(ip),'er'))
            dfM(i,dFmap('vowel')) = 1;dfM(i,dFmap('central')) = 1;dfM(i,dFmap('open-mid')) = 1;
            dfM(i,dFmap('rmerged')) = 1;dfM(i,dFmap('short')) = 1;  
       elseif(strcmp(pHmap(ip),'ey'))
            dfM(i,dFmap('vowel')) = 1;dfM(i,dFmap('diphtong')) = 1;
            dfM(i,dFmap('front')) = 1;dfM(i,dFmap('mid')) = 1;dfM(i,dFmap('short')) = 1;
            dfM(i,dFmap('front2')) = 1;dfM(i,dFmap('close-mid2')) = 1;dfM(i,dFmap('short2')) = 1;
       elseif(strcmp(pHmap(ip),'f'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('unvoiced')) = 1;dfM(i,dFmap('fricative')) = 1;
            dfM(i,dFmap('labial')) = 1; 
       elseif(strcmp(pHmap(ip),'g') || strcmp(pHmap(ip),'gcl'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('voiced')) = 1;dfM(i,dFmap('stop')) = 1;
            dfM(i,dFmap('velar')) = 1;      
       elseif(strcmp(pHmap(ip),'hC') || strcmp(pHmap(ip),'pau') || strcmp(pHmap(ip),'q'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('silence')) = 1;
       elseif(strcmp(pHmap(ip),'hh') || strcmp(pHmap(ip),'hv'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('unvoiced')) = 1;dfM(i,dFmap('fricative')) = 1;
            dfM(i,dFmap('glottal')) = 1;         
       elseif(strcmp(pHmap(ip),'ih') || strcmp(pHmap(ip),'ix'))
            dfM(i,dFmap('vowel')) = 1;dfM(i,dFmap('front')) = 1;dfM(i,dFmap('close-mid')) = 1;
            dfM(i,dFmap('short')) = 1;    
       elseif(strcmp(pHmap(ip),'iy'))
            dfM(i,dFmap('vowel')) = 1;dfM(i,dFmap('front')) = 1;dfM(i,dFmap('close')) = 1;
            dfM(i,dFmap('long')) = 1;
       elseif(strcmp(pHmap(ip),'jh'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('post-alveolar')) = 1;dfM(i,dFmap('voiced')) = 1;
            dfM(i,dFmap('affricate')) = 1;
       elseif(strcmp(pHmap(ip),'k') || strcmp(pHmap(ip),'kcl'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('unvoiced')) = 1;dfM(i,dFmap('stop')) = 1;
            dfM(i,dFmap('velar')) = 1;
       elseif(strcmp(pHmap(ip),'l'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('voiced')) = 1;dfM(i,dFmap('lateral')) = 1;
            dfM(i,dFmap('alveolar')) = 1;
       elseif(strcmp(pHmap(ip),'m'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('voiced')) = 1;dfM(i,dFmap('nasal')) = 1;
            dfM(i,dFmap('labial')) = 1;       
       elseif(strcmp(pHmap(ip),'n') || strcmp(pHmap(ip),'nx'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('voiced')) = 1;dfM(i,dFmap('nasal')) = 1;
            dfM(i,dFmap('alveolar')) = 1;
        elseif(strcmp(pHmap(ip),'ng'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('unvoiced')) = 1;dfM(i,dFmap('nasal')) = 1;
            dfM(i,dFmap('velar')) = 1;     
       elseif(strcmp(pHmap(ip),'ow'))
            dfM(i,dFmap('vowel')) = 1;dfM(i,dFmap('diphtong')) = 1;
            dfM(i,dFmap('back')) = 1;dfM(i,dFmap('mid')) = 1;dfM(i,dFmap('short')) = 1;
            dfM(i,dFmap('back2')) = 1;dfM(i,dFmap('close-mid2')) = 1;dfM(i,dFmap('short2')) = 1; 
       elseif(strcmp(pHmap(ip),'oy'))
            dfM(i,dFmap('vowel')) = 1;dfM(i,dFmap('diphtong')) = 1;
            dfM(i,dFmap('back')) = 1;dfM(i,dFmap('mid')) = 1;dfM(i,dFmap('short')) = 1;
            dfM(i,dFmap('front2')) = 1;dfM(i,dFmap('close-mid2')) = 1;dfM(i,dFmap('short2')) = 1;
       elseif(strcmp(pHmap(ip),'p') || strcmp(pHmap(ip),'pcl'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('unvoiced')) = 1;dfM(i,dFmap('stop')) = 1;
            dfM(i,dFmap('labial')) = 1;
       elseif(strcmp(pHmap(ip),'r'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('voiced')) = 1;dfM(i,dFmap('approximant')) = 1;
            dfM(i,dFmap('post-alveolar')) = 1;
       elseif(strcmp(pHmap(ip),'s'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('unvoiced')) = 1;dfM(i,dFmap('fricative')) = 1;
            dfM(i,dFmap('alveolar')) = 1;
       elseif(strcmp(pHmap(ip),'sh'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('unvoiced')) = 1;dfM(i,dFmap('fricative')) = 1;
            dfM(i,dFmap('post-alveolar')) = 1;
       elseif(strcmp(pHmap(ip),'t') || strcmp(pHmap(ip),'tcl'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('unvoiced')) = 1;dfM(i,dFmap('stop')) = 1;
            dfM(i,dFmap('alveolar')) = 1;  
       elseif(strcmp(pHmap(ip),'th'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('unvoiced')) = 1;dfM(i,dFmap('fricative')) = 1;
            dfM(i,dFmap('dental')) = 1;
       elseif(strcmp(pHmap(ip),'uh'))
            dfM(i,dFmap('vowel')) = 1;dfM(i,dFmap('back')) = 1;dfM(i,dFmap('close-mid')) = 1;
            dfM(i,dFmap('short')) = 1;
       elseif(strcmp(pHmap(ip),'uw') || strcmp(pHmap(ip),'ux'))
            dfM(i,dFmap('vowel')) = 1;dfM(i,dFmap('back')) = 1;dfM(i,dFmap('close')) = 1;
            dfM(i,dFmap('long')) = 1;
       elseif(strcmp(pHmap(ip),'v'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('voiced')) = 1;dfM(i,dFmap('fricative')) = 1;
            dfM(i,dFmap('labial')) = 1;
       elseif(strcmp(pHmap(ip),'w'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('voiced')) = 1;dfM(i,dFmap('approximant')) = 1;
            dfM(i,dFmap('velar')) = 1;
       elseif(strcmp(pHmap(ip),'y'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('voiced')) = 1;dfM(i,dFmap('approximant')) = 1;
            dfM(i,dFmap('palatal')) = 1;
       elseif(strcmp(pHmap(ip),'z'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('voiced')) = 1;dfM(i,dFmap('fricative')) = 1;
            dfM(i,dFmap('alveolar')) = 1;
        elseif(strcmp(pHmap(ip),'zh'))
            dfM(i,dFmap('consonant')) = 1;dfM(i,dFmap('voiced')) = 1;dfM(i,dFmap('fricative')) = 1;
            dfM(i,dFmap('post-alveolar')) = 1;     
       end
    end
end

zeroc = zeros(1,size(dfM,2));
for i=1:size(dfM,2)
    if(sum(dfM(:,i))==0)
        zeroc(i) = 1;
    end
end
dfM(:,zeroc==1) = [];


