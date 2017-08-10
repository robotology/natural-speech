function [traindfM, valdfM, testdfM] = getDFs(dfMapfile,varargin)
%GETDFS Loads or computes a vector of distinctive/articulatory features for
% each frame of acoustic features
% IN
%   dfMapfile: file name where articulatory features are saved
%   varargin{1}: label matrix of the training set
%   varargin{2}: label matrix of the validation set
%   varargin{1}: label matrix of the testing set
%   varargin{1}: name of the dataset (currenlty: timit or mngu0)
% OUT
%   traindDFM: articulatory feature matrix of the training set
%   valdDFM: articulatory feature matrix of the validation set
%   testdDFM: articulatory feature matrix of the testing set


if(isempty(dfMapfile)==0)
    load(dfMapfile)
elseif(length(varargin) >=4)
    traindfM = extractDFeatures(varargin{1},varargin{4});
    valdfM = extractDFeatures(varargin{2},varargin{1});
    testdfM = extractDFeatures(varargin{3},varargin{1});      
else
    error('Wrong nukber of input arguments<n');
end

% if(insertStates)    
%     traindfM = [traindfM trainSids];
%     valdfM = [valdfM valSids];
%     testdfM = [testdfM testSids];
%     clear trainSids testSids valSids;
% end