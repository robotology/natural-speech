function [] = nnCheckNumericalGradient()
% Gradient checking

options.units=[3; 3; 2; 5; 4];
options.activations{1}='sigm';
options.activations{2}='relu';
options.activations{3}='tanh';
options.activations{4}='softmax';
options.cost='ce_softmax';
options.weightDecay=0;
options.dropout=0;

% Evaluate the function and gradient at w
w=gpuArray.rand((options.units(1:end-1)+1)'*options.units(2:end),1);  

X=gpuArray.rand(5,3);

%Y=gpuArray.rand(5,4);  % for cost 'mse'
Y=gpuArray([1 0 0 0;1 0 0 0;0 0 1 0;0 0 0 1;0 1 0 0]);  % for cost 'ce_softmax'

[value, grad] = nnCost(w,X,Y,options);
grad=single(grad);

% Numerically compute the gradient
numgrad = computeNumericalGradient(@(w) nnCost(w,X,Y,options),w);

% Visually examine the two gradient computations.  The two columns
% you get should be very similar. 
disp([numgrad grad]);

fprintf('The above two columns you get should be very similar.\n(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n');

% Evaluate the norm of the difference between two solutions.  
% If you have a correct implementation, then diff below should be < ~10^-9 
diff = norm(numgrad-grad)/norm(numgrad+grad);
disp(diff); 
fprintf('Norm of the difference between numerical and analytical gradient (should be < 1e-9)\n\n');

end