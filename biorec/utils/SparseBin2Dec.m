%SPARSEBIN2DEC Converts binary data to decimal not according to a positional
% numbering, but enumerating increasingly wrt the number of '1's and then,
% within the strings with the same number of '1's, in a positional manner
% e.g. the numbers with 4 digits are ordered: 0000, 0001, 0010, 0100, 1000,
% 0011, 0101, 0110, 1001, 1010, 1100, 0111, 1011, 1101, 1110, 1111

function dec = SparseBin2Dec(data)
n=sum(data,2);  %% #bits '1' for each row
dec=ones(size(data,1),1);   %% start from 1
for i=1:size(data,1)
    for j=0:n(i)-1  %% calculate as a starting base the number of strings having a #bits '1' lesser than n(i)
        dec(i)=dec(i)+nchoosek(size(data,2),double(j));
    end
    num1=n(i);  %% #bits '1' remaining
    for j=size(data,2):-1:1;    %% every bit '1' means we have to sum the # of possible strings that can be formed "on the right" of it containing the same number of remaining '1's
        if(data(i,size(data,2)-j+1)==1 && num1<j)
            dec(i)=dec(i)+nchoosek(j-1,double(num1));
            num1=num1-1;
        end
    end
end