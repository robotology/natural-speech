% Main ZeroSpeech with Binnarized Autoencoders

clear all;

load('sampleDataSINGLE.mat');

NoAuF = size(traindata,2);
windowSize = 1; % size of the frame context

% struct of hyperparameters of the autoencoder

parae = struct(...
    'units',[NoAuF*windowSize 500 100 16],...   % input + hidden (last is the encoding layer)
    'activations',{{'sigm','sigm','sigm'}},... % hidden
    'pretrain','rbm',...    % rbm/ae/rand
    'preae_maxepoch',50,... % for 'ae' pretraining, number of epochs for each layer
    'vgaus',1,...   % for 'rbm' pretraining, gaussian visible units
    'hgaus',0,...   % for 'rbm' pretraining, gaussian hidden units
    'pairsim',0,... % SAE using similarity heuristic
    'shid',0,...    % SCAE using similarity heuristic
    'noisy',0,...   % DAE
    'binarize',0,...    % binarize encodings in the fwd pass: 1) threshold 0.5; 2) threshold stochastically
    'forceBin',0,...    % force binarization: 1) add noise to encoding layer input; 2) additional entropy cost term
    'noiseVariance',16,...  % for forceBin=1, variance of the noise
    'entropyWeight',1,...   % for forceBin=2, weight of the entropy term
    'sparsepen',0,...   % sparsity penalty (beta)
    'sparsepar',0.05,...    % sparsity parameter (rho)
    'batchsize',1000,...
    'maxepoch',50,...
    'optimization','cgd',...    % cgd/sgd
    'cost','mse',...
    'dropout',0,...
    'learningRate',0.1,...          % SGD parameters
    'learningRateDecay',0.99,...    %%
    'momentum',0.9,...              %%
    'weightDecay',0.0001,...
    'rbmparam',struct(...           
        'vgaus',0,...
        'hgaus',0,...
        'batchsize',50,...
        'maxepoch',225,...
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
        'maxepoch',75,...
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
        'maxepoch',75,...
        'epsilonw',0.001,...
        'epsilonvb',0.001,...
        'epsilonhb',0.001,...
        'weightcost',0.0002,...
        'initialmomentum',0.5,...
        'finalmomentum',0.9...
        )...
);

discretization = 1; % threshold=0.5
% discretization = 2; % binary stochastic units
% discretization = 3; % discretizeData
% discretization = 4; % threshold=mean activations random data
% discretization = 5; % threshold=median activations train/test data

% create directory
clk=clock;
dir=[num2str(clk(3)) '-' num2str(clk(2)) '_'];
dir=[dir 'binarize' num2str(parae.binarize) '_'];
dir=[dir 'forceBin' num2str(parae.forceBin) '_'];
dir=[dir '['];
for i=2:length(parae.units)-1
    dir=[dir num2str(parae.units(i)) '-'];
end
dir=[dir num2str(parae.units(end)) ']_'];
dir=[dir 'opt-' parae.optimization '_'];
dir=[dir 'sparse' num2str(parae.sparsepen) '_'];
dir=[dir '_N' num2str(windowSize)];
mkdir(dir);

[traindata, wordTrainStart, wordTrainEnd] = windowData(traindata,windowSize,wordTrainStart,wordTrainEnd,0);
[testdata, wordTestStart, wordTestEnd] = windowData(testdata,windowSize,wordTestStart,wordTestEnd,0);

ntest = size(testdata,1);
ntrain = size(traindata,1);

% Autoencoder training
[ae, aeDec, parDec] = aeTrain(traindata,parae,testdata);
entraindata = nnFwd(ae,traindata,parae);
entestdata = nnFwd(ae,testdata,parae);

cd(dir);
save(['entraindata_' strrep(strrep(datestr(clock, 0),':',';'),' ','_')], 'entraindata');
save(['entestdata_' strrep(strrep(datestr(clock, 0),':',';'),' ','_')], 'entestdata');
save(['ae_'  dir], 'ae', 'parae');
cd('..');

wordTrain = struct('name',[],'label',[],'spc',[],'dec',[],'raw', [], 'encoded', [],'decoded',[], 'encodedscr',[]);
wordTest= struct('name',[],'label',[],'spc',[],'dec',[],'raw', [], 'encoded', [],'decoded',[], 'encodedscr',[]);
%{
raw: original data
encoded: encoded data
encodedscr: binarized encoded data
dec: binarized encoded data converted to decimal
%}

% Group training data into (training) words
for k = 1:length(wordTrainName)
    wordTrain(k).name=wordTrainName{k};
    wordTrain(k).raw=traindata(wordTrainStart(k):wordTrainEnd(k),:);
    wordTrain(k).encoded=entraindata(wordTrainStart(k):wordTrainEnd(k),:);
end

% Group testing data into (testing) words
for k=1:length(wordTestName)
     wordTest(k).name=wordTestName{k};
     wordTest(k).raw=testdata(wordTestStart(k):wordTestEnd(k),:);
     wordTest(k).encoded=entestdata(wordTestStart(k):wordTestEnd(k),:);
end

switch discretization        
    case 1  %%thrs=0.5
        entraindata = entraindata>=0.5;
        entestdata = entestdata>=0.5;
    case 2  %% binary stochastic units
        entraindata = entraindata > rand(size(entraindata));
        entestdata = entestdata > rand(size(entestdata));
    case 3
        entraindata = discretizeData(ae,parae,aeDec,parDec,traindata);
        entestdata = discretizeData(ae,parae,aeDec,parDec,testdata);
    case 4  %%thrs=mean activations random data
        [thrs entraindata entestdata] = BinEncData(net,parae,traindata,testdata);
        cd(dir);
        save(['thrs_' strrep(strrep(datestr(clock, 0),':',';'),' ','_')], 'thrs');
        save(['binentraindata_' strrep(strrep(datestr(clock, 0),':',';'),' ','_')], 'entraindata');
        save(['binentestdata_' strrep(strrep(datestr(clock, 0),':',';'),' ','_')], 'entestdata');
        cd('..');
    case 5  %%thrs=median activations train/test data
        entraindata = bsxfun(@ge,entraindata,median(entraindata,1));
        entestdata = bsxfun(@ge,entestdata,median(entestdata,1));
        
        cd(dir);
        save(['thrs_train' strrep(strrep(datestr(clock, 0),':',';'),' ','_')], 'train_median');
        save(['thrs_test' strrep(strrep(datestr(clock, 0),':',';'),' ','_')], 'test_median');
        save(['binentraindata_' strrep(strrep(datestr(clock, 0),':',';'),' ','_')], 'entraindata');
        save(['binentestdata_' strrep(strrep(datestr(clock, 0),':',';'),' ','_')], 'entestdata');
        cd('..');
end

mean_nbits = mean(sum(entestdata,2));
max_nbits = max(sum(entestdata,2));  

decentraindata = zeros(ntrain,1);
decentestdata = zeros(ntest,1);

%decimal conversion
if parae.units(end)<32  %% if < 32 bit convert with bin2dec
    for i = 1:ntrain
        decentraindata(i,1) = bin2dec(strrep(num2str(entraindata(i,:)),' ',''))+1;
        if i<=ntest
            decentestdata(i,1) = bin2dec(strrep(num2str(entestdata(i,:)),' ',''))+1;
        end
    end
else    %% if > 32 bit convert "sparsely" (e.g. 000,001,010,100,011,101,110,111)
    decentraindata = SparseBin2Dec(entraindata);
    decentestdata = SparseBin2Dec(entestdata);
end


for k = 1:length(wordTrainName) %% to each train/test word, assign vector of encoded frames both decimal and binary
    wordTrain(k).dec = decentraindata(wordTrainStart(k):wordTrainEnd(k),1);
    wordTrain(k).encodedscr = entraindata(wordTrainStart(k):wordTrainEnd(k),:);
    if k<=length(wordTestName)
        wordTest(k).dec = decentestdata(wordTestStart(k):wordTestEnd(k),1);
        wordTest(k).encodedscr = entestdata(wordTestStart(k):wordTestEnd(k),:);
    end
end

cd(dir);
save('words.mat','wordTrain','wordTest');
cd('..');

%% Output

cd(dir);

% real-numbered outputs
mkdir('output');
cd('output');
for i=1:length(wordTrain)
    fd=fopen([wordTrain(i).name '.fea'],'w');
    time=0.0125;
    for j=1:size(wordTrain(i).encoded,1)
        fprintf(fd,'%f',time);
        for k=1:length(wordTrain(i).encoded(j,:))
            fprintf(fd,' %f',wordTrain(i).encoded(j,k));
        end
        fprintf(fd,'\n');
        time=time+0.01;
    end
    fclose(fd);
end
for i=1:length(wordTest)
    fd=fopen([wordTest(i).name '.fea'],'w');
    time=0.0125;
    for j=1:size(wordTest(i).encoded,1)
        fprintf(fd,'%f',time);
        for k=1:length(wordTest(i).encoded(j,:))
            fprintf(fd,' %f',wordTest(i).encoded(j,k));
        end
        fprintf(fd,'\n');
        time=time+0.01;
    end
    fclose(fd);
end

cd('..');

% Binary outputs
mkdir('outputBin');
cd('outputBin');
for i=1:length(wordTrain)
    fd=fopen([wordTrain(i).name '.fea'],'w');
    time=0.0125;
    for j=1:size(wordTrain(i).encodedscr,1)
        fprintf(fd,'%f',time);
        for k=1:length(wordTrain(i).encodedscr(j,:))
            fprintf(fd,' %f',wordTrain(i).encodedscr(j,k));
        end
        fprintf(fd,'\n');
        time=time+0.01;
    end
    fclose(fd);
end
for i=1:length(wordTest)
    fd=fopen([wordTest(i).name '.fea'],'w');
    time=0.0125;
    for j=1:size(wordTest(i).encodedscr,1)
        fprintf(fd,'%f',time);
        for k=1:length(wordTest(i).encodedscr(j,:))
            fprintf(fd,' %f',wordTest(i).encodedscr(j,k));
        end
        fprintf(fd,'\n');
        time=time+0.01;
    end
    fclose(fd);
end

% 1-of-K outputs
numSymbols=2^size(wordTrain(1).encodedscr,2);
for i=1:length(wordTrain)
    fd=fopen([wordTrain(i).name '.fea'],'w');
    time=0.0125;
    for j=1:size(wordTrain(i).encodedscr,1)
        fprintf(fd,'%f',time);
        symbol=bin2dec(char('0'+wordTrain(i).encodedscr(j,:)));
        for k=1:symbol
            fprintf(fd,' %f',0);
        end
        fprintf(fd,' %f',1);
        for k=1:numSymbols-symbol-1
            fprintf(fd,' %f',0);
        end
        fprintf(fd,'\n');
        time=time+0.01;
    end
    fclose(fd);
end
for i=1:length(wordTest)
    fd=fopen([wordTest(i).name '.fea'],'w');
    time=0.0125;
    for j=1:size(wordTest(i).encodedscr,1)
        fprintf(fd,'%f',time);
        symbol=bin2dec(char('0'+wordTest(i).encodedscr(j,:)));
        for k=1:symbol
            fprintf(fd,' %f',0);
        end
        fprintf(fd,' %f',1);
        for k=1:numSymbols-symbol-1
            fprintf(fd,' %f',0);
        end
        fprintf(fd,'\n');
        time=time+0.01;
    end
    fclose(fd);
end

cd('../..');