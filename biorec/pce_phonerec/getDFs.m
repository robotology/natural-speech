function [traindfM, valdfM, testdfM] = getDFs(dfMapfile,insertStates,varargin)

if(isempty(dfMapfile)==0)
    load(dfMapfile)
elseif(length(varargin) >=4)
    
    traindfM = extractDFeatures(varargin{1},varargin{4});
    valdfM = extractDFeatures(varargin{2},varargin{1});
    testdfM = extractDFeatures(varargin{3},varargin{1});    

    
else
    error('Wrong nukber of input arguments<n');
end


if(insertStates)    
    traindfM = [traindfM trainSids];
    valdfM = [valdfM valSids];
    testdfM = [testdfM testSids];
    clear trainSids testSids valSids;
end