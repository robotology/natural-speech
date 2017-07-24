function [] = aeCheckNumericalGradient()
% Gradient checking

w=gpuArray.rand(53,1);

options.units=[3; 3; 2; 5; 3];
options.activations{1}='sigm';
options.activations{2}='sigm';
options.activations{3}='sigm';
options.activations{4}='sigm';
options.cost='mse';
options.weightDecay=0.2;
options.dropout=0;
options.noisy=0;
varargcost={};

options.shid=1;
varargcost{1}=gpuArray.rand(5,3);   %%X2

options.pairsim=0;
options.sparsepen=0;
options.sparsepar=0.05;
options.binarize=0;
options.forceBin=0;
options.entropyWeight=0.5;

X=gpuArray.rand(5,3);

[value, grad] = aeCost(w,X,X,options,varargcost{:});
grad=single(grad);

% Numerically compute the gradient
numgrad = computeNumericalGradient(@(w) aeCost(w,X,X,options,varargcost{:}),w);

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

