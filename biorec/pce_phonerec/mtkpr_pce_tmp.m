% this is the main of pce_phonerec

clear all;
close all;


corpus = 'timit';

dirin = '/DATA/badino/prdata/inputdata/timit/';
datafile = [dirin 'timitData18-Aug-2015.mat'];
splitfile = [dirin 'timitSplit18-Aug-2015.mat'];

% type of input feature normalization
normtype = 'm0s1';
%distinctive feature file. it should be generated only once (genereting it
%takes several mins)
dfMapfile = 'dfMap_timit_09-Feb-2016'; %with silences

%language model file
%lmfile = ([dirin 'timitlanguageModels_nosils']);
    
% fileinitnet = 'bestDNN-sil-9Feb2016';

%no. of states in the HMMs
nstates = 3;
removeSil = 0;
ignoreSil = 1;
Nframes4class = 1;
Nframes4Reg = 27;
downsample = 0;

% hidden layer from which phonetic context embeddings are extracted
nle = 1;

loadDFMap = 1;
rframes = 2;
bValidation = 1;
bloadlm = 0;
insertStates = 0;
bReconstructionError = 0;

% hyper-parameters of the DNN that computes phone state posteriors
parnet_classifier=struct(...
     'units',[NaN 2000 2000 2000 2000 NaN],...
    'activations',{{'relu','relu','relu','relu','softmax'}},...
    'pretrain','rand',...   % other options 'rbm' or 'aam', or 'loadnet'. 'aam' for Acoustic to Articulatory Mapping - based pretraining
    'outputInit_maxepoch',5,... %  if 'rbm'/'aam' pretraining, number of epochs to initialize last layer weights. default is 5
    'vgaus',1,...
    'hgaus',0,...
    'batchsize',1000,...
    'maxepoch',20,... %for mngu0 50 - timit 20-25
    'optimization','sgd',...
    'cost','ce_softmax',...
    'dropout',0,...
    'learningRate',0.1,...  %original timit 0.1
    'learningRateDecay',.75,... %for mngu0 .99 - timit .8   
    'learningRateMin',.0005,...
    'momentum',0.9,...  %momentum 0.9
    'weightDecay',0.0001,...  %for mngu0 0.0001 - timit 0
    'earlystop',1,...
    'multitask',struct(...
        'mtktype','joint_top',... % do not change
        'firstnode',1,...
        'lastnode',NaN,...
        'cost2','mse',... %alternatives are 'ce_softmax' and  'ce_logistic'
        'ccost2',{{'mse'}},...
        'cactivations2',{{'relu'}},...
        't2w',0.1...
        ),...        
    'rbmparam',struct(...
        'vgaus',0,...
        'hgaus',0,...
        'batchsize',50,...
        'maxepoch',75,...
        'epsilonw',0.1,...
        'epsilonvb',0.1,...
        'epsilonhb',0.1,...
        'weightcost',0.0002,...
        'initialmomentum',0.5,...
        'finalmomentum',0.9...
        ),...
    'rbmhgausparam',struct(...
        'vgaus',0,...
        'hgaus',1,...
        'batchsize',50,...
        'maxepoch',225,...
        'epsilonw',0.001,...
        'epsilonvb',0.001,...
        'epsilonhb',0.001,...
        'weightcost',0.0002,...
        'initialmomentum',0.5,...
        'finalmomentum',0.9...
        ),...
    'rbmvgausparam',struct(...
        'vgaus',1,...
        'hgaus',0,...
        'batchsize',50,...
        'maxepoch',225,...
        'epsilonw',0.001,...
        'epsilonvb',0.001,...
        'epsilonhb',0.001,...
        'weightcost',0.0002,...
        'initialmomentum',0.5,...
        'finalmomentum',0.9...
        )...    
    );

% hyper-parameters of the DNN from which phonetic context embeddings are
% extracted

parnet_regress1 = struct(...
    'units',[NaN 300 300 300 NaN],... % input + hidden + output {555 370} 
    'activations',{{'relu','relu','relu','sigm'}},... % hidden + output
    'vsparse',[0 0 0 0],...
    'pretrain','rand',...   % rbm/rand
    'outputInit_maxepoch',10,...    % if 'rbm' pretraining, number of epochs to initialize last layer weights before running backprop. default iis 10
    'vgaus',0,...
    'hgaus',0,...
    'batchsize',1000,...
    'maxepoch',1,...  %40
    'optimization','sgd',...
    'cost','mse',...
    'dropout',0,...
    'learningRate',0.1,...
    'learningRateDecay',0.95,... %mngu0 .99 - timit .95
    'learningRateMin',0.001,...    
    'momentum',0.9,...
    'weightDecay',0.0001,... %mngu0 .0001 - timit 0
    'constGradient',0,...
    'constMtx',struct(...
        'createconstMtx',1,...
        'hlayers',[1:2],...
        'wnds',{{9,15}},... %9,15
        'slides',{{9,15}},... %9,15
        'hxw',{{15,10}}... %15, 10
        ),...          
    'rbmparam',struct(...
        'vgaus',0,...
        'hgaus',0,...
        'batchsize',50,...
        'maxepoch',75,...
        'epsilonw',0.1,...
        'epsilonvb',0.1,...
        'epsilonhb',0.1,...
        'weightcost',0.0002,...
        'initialmomentum',0.5,...
        'finalmomentum',0.9...
        ),...
    'rbmhgausparam',struct(...
        'vgaus',0,...
        'hgaus',1,...
        'batchsize',50,...
        'maxepoch',225,...
        'epsilonw',0.001,...
        'epsilonvb',0.001,...
        'epsilonhb',0.001,...
        'weightcost',0.0002,...
        'initialmomentum',0.5,...
        'finalmomentum',0.9...
        ),...
    'rbmvgausparam',struct(...
        'vgaus',1,...
        'hgaus',0,...
        'batchsize',50,...
        'maxepoch',225,...
        'epsilonw',0.001,...
        'epsilonvb',0.001,...
        'epsilonhb',0.001,...
        'weightcost',0.0002,...
        'initialmomentum',0.5,...
        'finalmomentum',0.9...
        )...
    );


c = clock;
wdir = datestr(c);
wdir(wdir==' ') = [];
wdir = ['nets_mtk3_27w-relu-3H300-300-300-_1Hpe_mse_normt2wis2' corpus '_' wdir];
mkdir(wdir);

load(splitfile);

if(strcmp(parnet_regress1.activations{end},'sigm'))
    normtype = 'minmax01';
    fprintf(1,'Changing norm type to minmax01\n');
end
[traindata, traintargets, trainSentenceStart, trainSentenceEnd] = loadAndNormData_short(datafile,splitfile,'train',normtype,removeSil);
[valdata, valtargets, valSentenceStart, valSentenceEnd] = loadAndNormData_short(datafile,splitfile,'validation',normtype,removeSil);
[testdata, testtargets, testSentenceStart, testSentenceEnd] = loadAndNormData_short(datafile,splitfile,'test',normtype,removeSil);

NoAuF = size(traindata,2);

%%%%% Compute Phonetic Context embeddings %%%%%%%

% 1. extract phone labels 
if(loadDFMap==0)
    trainplabels = state2phone(traintargets,nstates,0);
    valplabels = state2phone(valtargets,nstates,0);
    testplabels = state2phone(testtargets,nstates,0);
end

% 2. extract distinctive features from phone labels
if(insertStates)
    trainSids = getStateId(traintargets,nstates,0);
    valSids = getStateId(valtargets,nstates,0);
    testSids = getStateId(testtargets,nstates,0);

end
%clear traintargets testtargets valtargets;

if(loadDFMap)
    [traindfM, valdfM, testdfM] = getDFs(dfMapfile,insertStates);
else
    [traindfM, valdfM, testdfM] = getDFs('',insertStates,trainplabels, valplables, testplables, corpus);
    dfMapfile = ['dfMap_' corpus '_' date]; 
    save(dfMapfile,'traindfM''valdfM','testdfM');
end

NoDF = size(traindfM,2);

if(ignoreSil)
    trainsil = find(traindfM(:,end)==1);
    valsil = find(valdfM(:,end)==1);
    testsil = find(testdfM(:,end)==1);
end

% 3. create contex
if(downsample)
    nW = (Nframes4Reg-1)*(downsample+1)+1;
else
    nW = Nframes4Reg;
end

[traindfM, traindata] = createContext(traindfM,traindata,trainSentenceStart,trainSentenceEnd,nW,0);
[valdfM, valdata] = createContext(valdfM,valdata,valSentenceStart,valSentenceEnd,nW,0);
[testdfM, testdata] = createContext(testdfM,testdata,testSentenceStart,testSentenceEnd,nW,0);


if(ignoreSil)
    traindfM(trainsil,:) = [];
    traindata(trainsil,:) = [];     
    valdfM(valsil,:) = [];
    valdata(valsil,:) = [];
    testdfM(testsil,:) = [];
    testdata(testsil,:) = [];   
end

if(downsample)
    traindfM = downsampleContext(traindfM,NoDF,downsample);
    valdfM = downsampleContext(valdfM,NoDF,downsample);
    testdfM = downsampleContext(testdfM,NoDF,downsample);
end

% 4. do regression on audio 
%specify classification and regression nn's input and output size
parnet_regress1.units(1) = Nframes4Reg*NoDF;
parnet_regress1.units(end) = NoAuF;

if(parnet_regress1.constGradient)
    traindfM = groupFeatures(traindfM,Nframes4Reg);
    valdfM = groupFeatures(valdfM,Nframes4Reg);
    testdfM = groupFeatures(testdfM,Nframes4Reg);

end
regnet1 = nnTrain(traindfM,traindata,parnet_regress1,valdfM,valdata);
runfolder = cd(wdir);
save('regnet1','regnet1','parnet_regress1');
cd(runfolder);    

if(bReconstructionError)
    rectraindata = nnFwd(regnet1,traindfM,parnet_regress1);    
    recvaldata = nnFwd(regnet1,valdfM,parnet_regress1);
    rectestdata = nnFwd(regnet1,testdfM,parnet_regress1);
    rectrainaudioerr = computeReconstructionError(traindata,rectraindata,'fast',0);
    recvalaudioerr = computeReconstructionError(valdata,recvaldata,'fast',0);   
    rectestaudioerr = computeReconstructionError(testdata,rectestdata,'fast',0);       
end
clear traindata testdata valdata;

%%%%%%% PC-embedding learning ends here %%%%%%%%%%%%%%%%%%%%


%%%%%%%% Prepare data for pone recognition %%%%%%%%%%%
normtype = 'm0s1';
[traindata, traintargets, trainSentenceStart, trainSentenceEnd] = loadAndNormData_short(datafile,splitfile,'train',normtype,removeSil);
[testdata, testtargets, testSentenceStart, testSentenceEnd] = loadAndNormData_short(datafile,splitfile,'test',normtype,removeSil);
[valdata, valtargets, valSentenceStart, valSentenceEnd] = loadAndNormData_short(datafile,splitfile,'validation',normtype,removeSil);

if(bloadlm == 0)
    [p_unigm,p_bigm,s_unigm,s_bigm] = recomputeLM(traintargets,nstates,trainSentenceStart,1);
else
    load(lmfile);
end

[traindata,traintargets,trainSentenceStart,trainSentenceEnd] = createContext(traindata,traintargets,trainSentenceStart,trainSentenceEnd,Nframes4class,0);
[valdata, valtargets,valSentenceStart,valSentenceEnd]  = createContext(valdata,valtargets,valSentenceStart,valSentenceEnd,Nframes4class,0);
[testdata, testtargets,testSentenceStart,testSentenceEnd]  = createContext(testdata,testtargets,testSentenceStart,testSentenceEnd,Nframes4class,0);


%%%% generate phonetic context embeddings as secondary targets for the DNN
%%%% that computes phone state posteriors


[traindfM, valdfM, testdfM] = getDFs(dfMapfile,insertStates);
if(ignoreSil)
    trainsil = traindfM(:,end)==1;
else
    trainsil = [];
end

[traindfM, traindata] = createContext(traindfM,traindata,trainSentenceStart,trainSentenceEnd,nW,0);
[valdfM, valdata] = createContext(valdfM,valdata,valSentenceStart,valSentenceEnd,nW,0);
[testdfM, testdata] = createContext(testdfM,testdata,testSentenceStart,testSentenceEnd,nW,0);

if(downsample)
    traindfM = downsampleContext(traindfM,NoDF,downsample);
    valdfM = downsampleContext(valdfM,NoDF,downsample);
    testdfM = downsampleContext(testdfM,NoDF,downsample);
end

embtrainplabels = nnFwd(regnet1,traindfM,parnet_regress1,nle);
clear trainplabels;
embvalplabels = nnFwd(regnet1,valdfM,parnet_regress1,nle);
embtestplabels = nnFwd(regnet1,testdfM,parnet_regress1,nle);

clear testplabels valplabels;
%clear regnet1;

csum = sum(embtrainplabels,1);
czero = (csum==0);
zratio = sum(czero)/size(embtrainplabels,2);
fprintf(1,'Ratio of zero columns is %f\n',zratio);
embtrainplabels(:,czero) = [];
embtestplabels(:,czero) = [];
embvalplabels(:,czero) = [];

zetrain = (embtrainplabels ==0);
spars = mean(sum(zetrain,2))/size(embtrainplabels,2);
fprintf(1,'Average sparsity of the encoding is %f\n',spars);
efnum = round(size(embtrainplabels,2)*(1-spars));

clear spars zetrain;

%%%%% Phone recognition training and evaluation %%%%% 

parnet_classifier.units(1) = Nframes4class * NoAuF;
parnet_classifier.units(end) = size(traintargets,2);

initnet = [];
mtkoptions.valX = valdata ;
mtkoptions.valY = valtargets;
mtkoptions.valY2 = {embvalplabels};    
mtkoptions.task2ignore = trainsil;
parnet_classifier.multitask.lastnode = size(embtrainplabels,2);
orilearningRate = parnet_classifier.learningRate;
%t2norm = size(traintargets,2)/efnum;
t2norm = 2;
iter = 1;
for t2w = 0:.1:0       
    parnet_classifier.multitask.t2w = t2w*t2norm;

    if(t2w==0 && strcmp(parnet_classifier.pretrain,'loadnet')==0)        
%         [classnet, initnet] = nnTrain_mtk(traindata,traintargets,embtrainplabels,parnet_classifier,initnet,mtkoptions);
        [classnet, initnet] = nnTrain(traindata,traintargets,parnet_classifier,valdata,valtargets);
        parnet_classifier.pretrain = 'loadnet';
    elseif(strcmp(parnet_classifier.pretrain,'loadnet'))
        if(isempty(initnet))
            load(fileinitnet);
        end
        classnet = nnTrain_mtk(traindata,traintargets,{embtrainplabels},parnet_classifier,initnet,mtkoptions);
    elseif(t2w>0)
        classnet = nnTrain_mtk(traindata,traintargets,{embtrainplabels},parnet_classifier,initnet,mtkoptions);
    else
        error('something is wrong with net inizialization \n'); 
    end

    post = nnFwd(classnet,testdata,parnet_classifier)';
    [~, cpred] = max(post);        
    [rseq, srseq]= computeBestPath_new(p_unigm,p_bigm,s_unigm,s_bigm,post,testSentenceStart,testSentenceEnd);
    clear post;
    [tmp, target] = max(testtargets,[],2);
    [confmatrix, rconfmatrix, cacc(iter), rcacc] = createConfusionMatrix(target,cpred,size(testtargets,2),nstates, rframes, testSentenceStart, testSentenceEnd);
    [rerrb(iter), predseq, totdels, totins, totsubs, moperations, rsubstitution, testseq] = computePhoneRecognitionError(rseq,testtargets,testSentenceStart,testSentenceEnd,nstates);
    [serrb(iter), spredseq, stotdels, stotins, stotsubs, smoperations, ssubstitution] = computePhoneRecognitionError(srseq,testtargets,testSentenceStart,testSentenceEnd,nstates);

    if(strcmp(corpus,'timit'))
        [rerr(iter), totdels, totins, totsubs, moperations, substitution ] = computePhoneRecognitionError_shortPhS(testseq,predseq,round(size(testtargets,2)/nstates),'timit');
        [serr(iter), stotdels, stotins, stotsubs, smoperations, ssubstitution ] = computePhoneRecognitionError_shortPhS(testseq,spredseq,round(size(testtargets,2)/nstates),'timit');
    else
        rerr = rerrb;
        serr = serrb;
    end

    post = nnFwd(classnet,valdata,parnet_classifier)';
    [~, cpred] = max(post);
    [rseq, srseq]= computeBestPath_new(p_unigm,p_bigm,s_unigm,s_bigm,post,valSentenceStart,valSentenceEnd);
    clear post;
    [tmp, target] = max(valtargets,[],2);
    [confmatrix, rconfmatrix, valcacc(iter), rcacc(iter)] = createConfusionMatrix(target,cpred,size(valtargets,2),nstates, rframes, valSentenceStart, valSentenceEnd);
    [valrerrb(iter), predseq, totdels, totins, totsubs, moperations, rsubstitution, valseq] = computePhoneRecognitionError(rseq,valtargets,valSentenceStart,valSentenceEnd,nstates);
    [valserrb(iter), spredseq, stotdels, stotins, stotsubs, smoperations, ssubstitution] = computePhoneRecognitionError(srseq,valtargets,valSentenceStart,valSentenceEnd,nstates);

    if(strcmp(corpus,'timit'))
        [valrerr(iter), totdels, totins, totsubs, moperations, substitution ] = computePhoneRecognitionError_shortPhS(valseq,predseq,round(size(valtargets,2)/nstates),'timit');
        [valserr(iter), stotdels, stotins, stotsubs, smoperations, ssubstitution ] = computePhoneRecognitionError_shortPhS(valseq,spredseq,round(size(valtargets,2)/nstates),'timit');
    else
        valrerr = valrerrb;
        valserr = valserrb;
    end
    fprintf(1,'Value of rerr and valrerr for t2w %f are %f and %f\n', t2w, rerr(iter),valrerr(iter)); 
    iter = iter+1;

end
runfolder = cd(wdir);

save(['results_mtk'],'rerr','serr','cacc','valrerr','valserr','valcacc','classnet','initnet','parnet_regress1','parnet_classifier','rerrb','serrb');
cd(runfolder);    
