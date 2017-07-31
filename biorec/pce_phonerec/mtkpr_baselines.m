% MTKPR_BASELINE Trains and tests THE multi-task learning-based phone recognition system
% of Seltzer and Droppo (see README for references)

clear all;
close all;

corpus = 'timit';

dirin = '/DATA/badino/prdata/inputdata/timit/';
datafile = [dirin 'timitData_40fbankse_02-Feb-2016.mat'];
splitfile = [dirin 'timitSplit18-Aug-2015.mat'];
lmfile = '';

normtype = 'm0s1';
fileinitnet = '';
    
split = 1;
nstates = 3;
removeSil = 0;
ignoreSil = 0;
Nframes4class = 11;

rframes = 2;
bValidation = 1;
bloadlm = 0;
bphones = 1;



% hyper-parameters of the DNN that computes phone state posteriors 
parnet_classifier=struct(...
     'units',[NaN 2000 2000 2000 2000 NaN],...
    'activations',{{'relu','relu','relu','relu','softmax'}},...
    'pretrain','rand',...   % other options 'rbm' or 'aam', or 'loadnet'. 'aam' for Acoustic to Articulatory Mapping - based pretraining
    'outputInit_maxepoch',5,... %  if 'rbm'/'aam' pretraining, number of epochs to initialize last layer weights. default is 5
    'vgaus',1,...
    'hgaus',0,...
    'batchsize',1000,...
    'maxepoch',20,... % timit 20-25
    'optimization','sgd',...
    'cost','ce_softmax',...
    'dropout',0,...
    'learningRate',0.1,...  %original timit 0.1
    'learningRateDecay',.75,... %for mngu0 .99 - timit .75   
    'learningRateMin',.0005,...
    'momentum',0.9,...  %momentum 0.9
    'weightDecay',0.000,...  %for mngu0 0.0001 - timit 0
    'earlystop',1,...
    'multitask',struct(...
        'mtktype','joint_top',... % possible values: alternate, joint, joint_top
        'firstnode',1,...
        'lastnode',NaN,...
        'cost2','mse',... %alternative is 'ce_logistic'
        'ccost2',{{'ce_softmax','ce_softmax'}},...
        'cactivations2',{{'softmax','softmax'}},...
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

c = clock;
wdir = datestr(c);
wdir(wdir==' ') = [];
wdir = ['nets_mtk3_11w-baseline' date '_' corpus '_' wdir];
mkdir(wdir);

[traindata, traintargets, trainSentenceStart, trainSentenceEnd] = loadAndNormData_short(datafile,splitfile,'train',normtype,removeSil);
[testdata, testtargets, testSentenceStart, testSentenceEnd] = loadAndNormData_short(datafile,splitfile,'test',normtype,removeSil);
[valdata, valtargets, valSentenceStart, valSentenceEnd] = loadAndNormData_short(datafile,splitfile,'validation',normtype,removeSil);
NoAuF = size(traindata,2);

if(bloadlm == 0)
    [p_unigm,p_bigm,s_unigm,s_bigm] = recomputeLM(traintargets,nstates,trainSentenceStart,1);
else
    load(lmfile);
end

[traindata,traintargets,trainSentenceStart,trainSentenceEnd] = createContext(traindata,traintargets,trainSentenceStart,trainSentenceEnd,Nframes4class,0);
[testdata, testtargets,testSentenceStart,testSentenceEnd]  = createContext(testdata,testtargets,testSentenceStart,testSentenceEnd,Nframes4class,0);
[valdata, valtargets,valSentenceStart,valSentenceEnd]  = createContext(valdata,valtargets,valSentenceStart,valSentenceEnd,Nframes4class,0);

trainplabels = state2phone(traintargets,nstates,0);
%testplabels = state2phone(testtargets,nstates,0);
valplabels = state2phone(valtargets,nstates,0);

if(bphones)
    traintargets2 = getNeighborPhones(trainplabels,trainSentenceStart,trainSentenceEnd);
else
    traintargets2 = getNeighborFrames(trainplabels,trainSentenceStart,trainSentenceEnd);
end

parnet_classifier.units(1) = Nframes4class * NoAuF;
parnet_classifier.units(end) = size(traintargets,2);

initnet = [];
mtkoptions.valX = valdata ;
mtkoptions.valY = valtargets;
if(bphones)
    mtkoptions.valY2 = getNeighborPhones(valplabels,valSentenceStart,valSentenceEnd); 
else
    mtkoptions.valY2 = getNeighborFrames(valplabels,valSentenceStart,valSentenceEnd); 
end
mtkoptions.task2ignore = [];
iter = 1;
for t2w = 0:.1:1       
    parnet_classifier.multitask.t2w = t2w;

    if(t2w==0 && strcmp(parnet_classifier.pretrain,'loadnet')==0)       
        [classnet, initnet] = nnTrain(traindata,traintargets,parnet_classifier,valdata,valtargets);
        parnet_classifier.pretrain = 'loadnet';
    elseif(strcmp(parnet_classifier.pretrain,'loadnet'))
        if(isempty(initnet))
            load(fileinitnet);
        end
        classnet = nnTrain_mtk(traindata,traintargets,traintargets2,parnet_classifier,initnet,mtkoptions);
    elseif(t2w>0)
        classnet = nnTrain_mtk(traindata,traintargets,traintargets2,parnet_classifier,initnet,mtkoptions);
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

    
        [rerr(iter), totdels, totins, totsubs, moperations, substitution ] = computePhoneRecognitionError_shortPhS(testseq,predseq,round(size(testtargets,2)/nstates),'timit');
        [serr(iter), stotdels, stotins, stotsubs, smoperations, ssubstitution ] = computePhoneRecognitionError_shortPhS(testseq,spredseq,round(size(testtargets,2)/nstates),'timit');
    

    post = nnFwd(classnet,valdata,parnet_classifier)';
    [~, cpred] = max(post);
    [rseq, srseq]= computeBestPath_new(p_unigm,p_bigm,s_unigm,s_bigm,post,valSentenceStart,valSentenceEnd);
    clear post;
    [tmp, target] = max(valtargets,[],2);
    [confmatrix, rconfmatrix, valcacc(iter), rcacc(iter)] = createConfusionMatrix(target,cpred,size(valtargets,2),nstates, rframes, valSentenceStart, valSentenceEnd);
    [valrerrb(iter), predseq, totdels, totins, totsubs, moperations, rsubstitution, valseq] = computePhoneRecognitionError(rseq,valtargets,valSentenceStart,valSentenceEnd,nstates);
    [valserrb(iter), spredseq, stotdels, stotins, stotsubs, smoperations, ssubstitution] = computePhoneRecognitionError(srseq,valtargets,valSentenceStart,valSentenceEnd,nstates);

    
        [valrerr(iter), totdels, totins, totsubs, moperations, substitution ] = computePhoneRecognitionError_shortPhS(valseq,predseq,round(size(valtargets,2)/nstates),'timit');
        [valserr(iter), stotdels, stotins, stotsubs, smoperations, ssubstitution ] = computePhoneRecognitionError_shortPhS(valseq,spredseq,round(size(valtargets,2)/nstates),'timit');
    
    fprintf(1,'Value of rerr and valrerr for t2w %f are %f and %f\n', t2w, rerr(iter),valrerr(iter)); 
    iter = iter+1;

end
runfolder = cd(wdir);

save(['results_mtk'],'rerr','serr','cacc','valrerr','valserr','valcacc','classnet','initnet','parnet_classifier','rerrb','serrb');
cd(runfolder);    


