function numgrad = computeNumericalGradient(J, theta)
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the function value at theta. 
  
numgrad = zeros(size(theta));
EPSILON=10^-4;

E=eye(length(theta));   % identity matrix from which we take the needed column each time

for i=1:length(theta)
    numgrad(i)=gather(J(theta+EPSILON*E(:,i))-J(theta-EPSILON*E(:,i)))/(2*EPSILON);
end

end