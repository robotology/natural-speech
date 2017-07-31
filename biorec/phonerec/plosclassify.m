function plosclassify(varargin)
% PLOSCLASSIFY Trains and tests an articulatory phone recognition system
% Requires datastes of articulatory data
% Needs inivar to be configured
% varargin{1}: folder where inivar is located

if ~isempty(varargin)
    cd(varargin{1});
    inivar;
    cd('..');
else
    inivar;
end

audioonly = 0;
motoronly = 0;
if caudioType==0 && cmotorType==0
    error('At least one kind of information must be used!\n');
elseif cmotorType == 0 && caudioType~=3
    audioonly = 1;
    stream1 = 'audio';
    stream2 = '';
elseif caudioType==0
    motoronly = 1;
    stream1 = 'realMotor';
    stream2 = '';
else
    stream1 = 'audio';
    stream2 = 'realMotor';
end
if caudioType == 2 || bReconstruct==2 || bReconstruct==4 || caudioType == 3 || caudioType == 4 || caudioType == 5
    encodeaudio = 1;
else
    encodeaudio = 0;
end
if cmotorType == 2 || cmotorType == 4 || (cmotorType ~=5 && cmotorType ~=6 && bReconstruct >= 2 && bReconstruct <=5)
    encodemotor = 1;
elseif cmotorType == 5 || cmotorType == 6
    encodemotor = 2;
else
    encodemotor = 0;
end
if cmotorType < 3 && bReconstruct > 0
    error('Motor reconstruction required but not used\n');
end
if strcmp(parnet_classifier.pretrain,'aam') && bReconstruct==0
    error('Reconstruction required for aam pretraining');
end

for split=1:numSplits
    fprintf('split:%g\n',split);
    [traindata, traintargets, ~, trainSentenceStart, trainSentenceEnd] = loadAndNormData(datafile,splitfile,'train',split,stream1,stream2,NoAuF,NoMF,normtype); 
    [testdata, testtargets, audioend, testSentenceStart, testSentenceEnd] = loadAndNormData(datafile,splitfile,'test',split,stream1,stream2,NoAuF,NoMF,normtype);
    ntrain = length(trainSentenceStart);
    ntest = length(testSentenceEnd);
    [p_unigm,p_bigm,s_unigm,s_bigm] = loadLanguageModels(p_unifile{split},p_bifile{split},s_unifile{split},s_bifile{split},state,0);
    oritrainaudio = traindata(:,1:audioend);
    oritestaudio = testdata(:,1:audioend);
    oritrainmotor = traindata(:,audioend+1:end);
    oritestmotor = testdata(:,audioend+1:end);
    if encodeaudio
        fprintf('Audio encoding\n');
        trainaeaudio = createContext(oritrainaudio,traintargets,trainSentenceStart,trainSentenceEnd,nfae_audio);
        testaeaudio = createContext(oritestaudio,testtargets,testSentenceStart,testSentenceEnd,nfae_audio);
        varargae={};
        if parae_audio.pairsim || parae_audio.shid
            varargae{2} = traintargets;
        end
        ae_audio = aeTrain(trainaeaudio,parae_audio,testaeaudio,varargae{:});
        entrainaudio = nnFwd(ae_audio,trainaeaudio,parae_audio);
        entestaudio = nnFwd(ae_audio,testaeaudio,parae_audio);
    end
    if encodemotor
        fprintf('Motor encoding\n');
        trainaemotor = createContext(oritrainmotor,traintargets,trainSentenceStart,trainSentenceEnd,nfae_motor); 
        testaemotor = createContext(oritestmotor,testtargets,testSentenceStart,testSentenceEnd,nfae_motor);
        if encodemotor==1
            varargae={};
            if parae_motor.pairsim || parae_motor.shid
                varargae{2} = traintargets;
            end
            ae_motor = aeTrain(trainaemotor,parae_motor,testaemotor,varargae{:});
            entrainmotor = nnFwd(ae_motor,trainaemotor,parae_motor);
            entestmotor = nnFwd(ae_motor,testaemotor,parae_motor);
        elseif encodemotor==2
            ae_motor = nnTrain(trainaemotor,traintargets,parae_motor,testaemotor,testtargets);
            entrainmotor = nnFwd(ae_motor,trainaemotor,parae_motor);
            entestmotor = nnFwd(ae_motor,testaemotor,parae_motor);
        end
        if parae_motor.vgaus
            entrainmotor = fnormData(entrainmotor,parae_motor.units(end),'m0s1');
            entestmotor = fnormData(entestmotor,parae_motor.units(end),'m0s1');
        end
    end
    if ~audioonly && ~motoronly && bReconstruct>0
        fprintf('Motor reconstruction\n');
        if bReconstruct == 1     
            trainaudio = createContext(oritrainaudio,traintargets,trainSentenceStart,trainSentenceEnd,nf_audio);
            testaudio = createContext(oritestaudio,testtargets,testSentenceStart,testSentenceEnd,nf_audio);
            recnet = nnTrain(trainaudio,oritrainmotor,parnet_regress,testaudio,oritestmotor);
            trainmotor = nnFwd(recnet,trainaudio,parnet_regress);
            testmotor = nnFwd(recnet,testaudio,parnet_regress);
            rectrainerr{split} = computeReconstructionError(oritrainmotor,trainmotor);    
            rectesterr{split} = computeReconstructionError(oritestmotor,testmotor);
        elseif bReconstruct == 2 || bReconstruct == 4
            entrainaudio = createContext(entrainaudio,traintargets,trainSentenceStart,trainSentenceEnd,nf_audio);
            entestaudio = createContext(entestaudio,testtargets,testSentenceStart,testSentenceEnd,nf_audio);
            if blinregr
                [recnet,entrainmotor,entestmotor] = recLinearRegress(entrainaudio,entrainmotor,entestaudio,parae_motor.hgaus);
            else
                recnet = nnTrain(entrainaudio,entrainmotor,parnet_regress,entestaudio,entestmotor);
                entrainmotor = nnFwd(recnet,entrainaudio,parnet_regress);
                entestmotor = nnFwd(recnet,entestaudio,parnet_regress);
            end
        elseif bReconstruct == 3 || bReconstruct == 5
            trainaudio = createContext(oritrainaudio,traintargets,trainSentenceStart,trainSentenceEnd,nf_audio);
            testaudio = createContext(oritestaudio,testtargets,testSentenceStart,testSentenceEnd,nf_audio);
            if blinregr
                [recnet, entrainmotor, entestmotor] = recLinearRegress(trainaudio,entrainmotor,testaudio,parae_motor.hgaus);
            else
                recnet = nnTrain(trainaudio,entrainmotor,parnet_regress,testaudio,entestmotor);
                entrainmotor = nnFwd(recnet,trainaudio,parnet_regress);
                entestmotor = nnFwd(recnet,testaudio,parnet_regress);
            end
%         elseif bReconstruct == 6
%             trainaudio = createContext(oritrainaudio,traintargets,trainSentenceStart,trainSentenceEnd,nf_audio);
%             testaudio = createContext(oritestaudio,testtargets,testSentenceStart,testSentenceEnd,nf_audio);
%             trainmotor = createContext(oritrainmotor,traintargets,trainSentenceStart,trainSentenceEnd,nf_motor);
%             testmotor = createContext(oritestmotor,testtargets,testSentenceStart,testSentenceEnd,nf_motor);
%             parae_audio.hlayers=parae_audio.units(2:end);
%             parae_audio.ogaus=parae_audio.hgaus;
%             parae_audio.finetune=0;
%             parae_audio.rbmbatchsize=50;
%             parae_audio.labrbm=0;
%             parae_motor.hlayers=parae_motor.units(2:end);
%             parae_motor.ogaus=parae_motor.hgaus;
%             parae_motor.finetune=0;
%             parae_motor.rbmbatchsize=50;
%             parae_motor.labrbm=0;
%             parsrbm.rbmbatchsize=50;
%             parsrbm.rbmparam=struct('vgaus',0,'hgaus',0,'batchsize',50,'maxepoch',1,'epsilonw',0.1,'epsilonvb',0.1,'epsilonhb',0.1,'weightcost',0.0002,'initialmomentum',0.5,'finalmomentum',0.9);
%             parsrbm.rbmhgausparam=struct('vgaus',0,'hgaus',1,'batchsize',50,'maxepoch',1,'epsilonw',0.001,'epsilonvb',0.001,'epsilonhb',0.001,'weightcost',0.0002,'initialmomentum',0.5,'finalmomentum',0.9);
%             parsrbm.rbmvgausparam=struct('vgaus',1,'hgaus',0,'batchsize',50,'maxepoch',1,'epsilonw',0.001,'epsilonvb',0.001,'epsilonhb',0.001,'weightcost',0.0002,'initialmomentum',0.5,'finalmomentum',0.9);
%             [recnet, parjnet, trainmotor] = dbndeepautomix_pct(trainaudio,trainmotor,parae_audio,parae_motor,parsrbm,parjnet,mixtype,testaudio,testmotor);
%             if mixtype==3
%                 test_avgmotor = zeros(size(testmotor));
%                 testmotor = idbnfwd_hgpu(recnet,length(parjnet.hlayers),[testaudio test_avgmotor],parjnet.agaus);   %%MANCA FILE!!!!
%                 testmotor = testmotor(:,size(testaudio,2)+1:end);
%             end
%             if mixtype==2 || mixtype==1
%                 testmotor = dbnfwd_hgpu(recnet,length(parjnet.hlayers)+1,testaudio,parjnet.agaus);
%             end
%             if mixtype==2
%                 testmotor = testmotor(:,size(testaudio,2)+1:end);
%             end
%             rectrainerr{split} = computeReconstructionError(oritrainmotor,trainmotor);
%             rectesterr{split} = computeReconstructionError(oritestmotor,testmotor);
%             rectrainerr_each_phoneme{split} = computeReconstructionErrorEachPhoneme(oritrainmotor,trainmotor,traintargets);
%             rectesterr_each_phoneme{split}=computeReconstructionErrorEachPhoneme(oritestmotor,testmotor,testtargets);
%             filename  = ['reconstructedMotorData_mixtype',num2str(mixtype),'_split',num2str(split)];
%             parnet_regress = parjnet;
        end
    end
    
    classtraindata = [];
    classtestdata = [];
    if caudioType == 1
        classtraindata = oritrainaudio;
        classtestdata = oritestaudio;
    elseif caudioType == 2
        classtraindata = entrainaudio;
        classtestdata = entestaudio;
    elseif caudioType == 3
        NoEAuF = parae_audio.units(end);
        tmpentrainaudio = fnormData(entrainaudio,NoEAuF,'m0s1');
        classtraindata = [oritrainaudio tmpentrainaudio];
        tmpentestaudio = fnormData(entestaudio,NoEAuF,'m0s1');
        classtestdata = [oritestaudio tmpentestaudio];
        clear tmpentrainaudio tmpentestaudio
    end
    
    if cmotorType == 1 || cmotorType == 3
        if cmotorType==1
            trainmotor = oritrainmotor;
            testmotor = oritestmotor;
        end
        if cmotorType == 3 && parae_motor.vgaus
            trainmotor = fnormData(trainmotor,NoMF,'m0s1');
            testmotor = fnormData(testmotor,NoMF,'m0s1');
        end
        classtraindata = [classtraindata trainmotor];
        classtestdata =  [classtestdata testmotor];
        clear trainmotor testmotor
    elseif(cmotorType == 2 || cmotorType >= 4)
        NoEMF = parae_motor.units(end);
        if(parae_motor.hgaus)
            entrainmotor = fnormData(entrainmotor,NoEMF,'m0s1');
            entestmotor = fnormData(entestmotor,NoEMF,'m0s1');
        end
        classtraindata = [classtraindata entrainmotor];
        classtestdata =  [classtestdata entestmotor];
        clear entrainmotor entestmotor
    end
    
    fprintf('Classification\n');
    [classtraindata, classtraintargets, tmptrainSentenceStart, tmptrainSentenceEnd] = createContext(classtraindata,traintargets,trainSentenceStart,trainSentenceEnd,Nframes4class,removeSil);
    [classtestdata, classtesttargets, tmptestSentenceStart, tmptestSentenceEnd] = createContext(classtestdata,testtargets,testSentenceStart,testSentenceEnd,Nframes4class,removeSil);
    
    if strcmp(parnet_classifier.pretrain,'aam')
        classnet = nnTrain(classtraindata,classtraintargets,parnet_classifier,classtestdata,classtesttargets,recnet);
    else
        classnet = nnTrain(classtraindata,classtraintargets,parnet_classifier,classtestdata,classtesttargets);
    end
    
    post = nnFwd(classnet,classtestdata,parnet_classifier)';
    [~, cpred] = max(post);
    [rseq, srseq]= computeBestPath_new(p_unigm,p_bigm,s_unigm,s_bigm,post,tmptestSentenceStart,tmptestSentenceEnd);
    [tmp, classtarget] = max(classtesttargets,[],2);
    [confmatrix, rconfmatrix, cacc, rcacc] = createConfusionMatrix(classtarget,cpred,size(classtesttargets,2),state, rframes, tmptestSentenceStart, tmptestSentenceEnd);
    [rerr, predseq, totdels, totins, totsubs, moperations, rsubstitution, testseq] = computePhoneRecognitionError(rseq,classtesttargets,tmptestSentenceStart,tmptestSentenceEnd,state);
    [serr, spredseq, stotdels, stotins, stotsubs, smoperations, ssubstitution] = computePhoneRecognitionError(srseq,classtesttargets,tmptestSentenceStart,tmptestSentenceEnd,state);
end

% Save params, nets and results
varstosave = {'parnet_classifier','classnet','rerr','serr','cacc','cpred','confmatrix','classtarget'};
if encodeaudio
    varstosave = [varstosave, {'parae_audio','ae_audio'}];
end
if encodemotor
    varstosave = [varstosave, {'parae_motor','ae_motor'}];
end
if bReconstruct>0
    varstosave = [varstosave, {'parnet_regress','recnet'}];
end
mkdir(folderName);
cd(folderName);
save('output',varstosave{:});
cd('..');

end