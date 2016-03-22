function [gpu_X, fX, i,bbug] = minimize_pct(gpu_X, f, length, varargin)

% Minimize a differentiable multivariate function.
%
% Usage: [gpu_X, fX, i] = minimize(gpu_X, f, length, P1, P2, P3, ... )
%
% where the starting point is given by "X" (D by 1), and the function named in
% the string "f", must return a function value and a vector of partial
% derivatives of f wrt X, the "length" gives the length of the run: if it is
% positive, it gives the maximum number of line searches, if negative its
% absolute gives the maximum allowed number of function evaluations. You can
% (optionally) give "length" a second component, which will indicate the
% reduction in function value to be expected in the first line-search (defaults
% to 1.0). The parameters P1, P2, P3, ... are passed on to the function f.
%
% The function returns when either its length is up, or if no further progress
% can be made (ie, we are at a (local) minimum, or so close that due to
% numerical problems, we cannot get any closer). NOTE: If the function
% terminates within a few iterations, it could be an indication that the
% function values and derivatives are not consistent (ie, there may be a bug in
% the implementation of your "f" function). The function returns the found
% solution "X", a vector of function values "fX" indicating the progress made
% and "i" the number of iterations (line searches or function evaluations,
% depending on the sign of "length") used.
%
% The Polack-Ribiere flavour of conjugate gradients is used to compute search
% directions, and a line search using quadratic and cubic polynomial
% approximations and the Wolfe-Powell stopping criteria is used together with
% the slope ratio method for guessing initial step sizes. Additionally a bunch
% of checks are made to make sure that exploration is taking place and that
% extrapolation will not be unboundedly large.
%
% See also: checkgrad
%
% Copyright (C) 2001 - 2006 by Carl Edward Rasmussen (2006-09-08).

INT = 0.1;    % don't reevaluate within 0.1 of the limit of the current bracket
EXT = 3.0;                  % extrapolate maximum 3 times the current step-size
MAX = 20;                         % max 20 function evaluations per line search
RATIO = 10;                                       % maximum allowed slope ratio
SIG = 0.1; RHO = SIG/2; % SIG and RHO are the constants controlling the Wolfe-
% Powell conditions. SIG is the maximum allowed absolute ratio between
% previous and new slopes (derivatives in the search direction), thus setting
% SIG to low (positive) values forces higher precision in the line-searches.
% RHO is the minimum allowed fraction of the expected (from the slope at the
% initial point in the linesearch). Constants must satisfy 0 < RHO < SIG < 1.
% Tuning of SIG (depending on the nature of the function to be optimized) may
% speed up the minimization; it is probably not worth playing much with RHO.

% The code falls naturally into 3 parts, after the initial line search is
% started in the direction of steepest descent. 1) we first enter a while loop
% which uses point 1 (p1) and (p2) to compute an extrapolation (p3), until we
% have extrapolated far enough (Wolfe-Powell conditions). 2) if necessary, we
% enter the second loop which takes p2, p3 and p4 chooses the subinterval
% containing a (local) minimum, and interpolates it, unil an acceptable point
% is found (Wolfe-Powell conditions). Note, that points are always maintained
% in order p0 <= p1 <= p2 < p3 < p4. 3) compute a new search direction using
% conjugate gradients (Polack-Ribiere flavour), or revert to steepest if there
% was a problem in the previous line-search. Return the best value so far, if
% two consecutive line-searches fail, or whenever we run out of function
% evaluations or line-searches. During extrapolation, the "f" function may fail
% either with an error or returning Nan or Inf, and minimize should handle this
% gracefully.

bbug = 0;


if max(size(length)) == 2, red=length(2); length=length(1); else red=1; end
if length>0, S='Linesearch'; else S='Function evaluation'; end

i = 0;                                            % zero the run length counter
ls_failed = 0;                             % no previous line search has failed
tic;
[f0 gpu_df0] = feval(f, gpu_X, varargin{:});          % get function value and gradient
fX = single(f0);
i = i + (length<0);  % count epochs?!

gpu_s = -gpu_df0;
gpu_d0 = sum(-gpu_s.*gpu_s);           % initial search direction (steepest) and slope
d0 = gather(gpu_d0);
x3 = red/(1-d0);                                  % initial step is red/(|s|+1)

while i < abs(length)                                      % while not finished
    i = i + (length>0);                                      
    
    gpu_X0 = gpu_X; F0 = f0; gpu_dF0 = gpu_df0;                   % make a copy of current values
    if length>0, M = MAX; else M = min(MAX, -length-i); end
    
    while 1                             % keep extrapolating as long as necessary
        x2 = 0; f2 = f0; d2 = d0; f3 = f0; f4 = f0; gpu_df3 = gpu_df0;
        success = 0;
        while ~success && M > 0
            try
                M = M - 1; i = i + (length<0);                         
                gpu_x3 = gpuArray(x3);
                gpu_arg1 = gpu_X+gpu_x3*gpu_s;
                [f3 gpu_df3] = feval(f, gpu_arg1, varargin{:});
                clear gpu_arg1;
               
                df3 = gather(gpu_df3);
                if isnan(f3) || isinf(f3) || any(isnan(df3)+isinf(df3)), error(''), end
                success = 1;
            catch                                % catch any error which occured in f
                x3 = (x2+x3)/2;                                  % bisect and try again
            end
        end
        if f3 < F0
            gpu_x3 = gpuArray(x3);
            gpu_X0 = gpu_X+gpu_x3*gpu_s; 
            F0 = f3; 
            gpu_dF0 = gpu_df3; 
        end         % keep best values
        gpu_d3 = sum(gpu_df3.*gpu_s);
        d3 = gather(gpu_d3);

        if d3 > SIG*d0 || f3 > f0+x3*RHO*d0 || M == 0  % are we done extrapolating?
            break
        end
        x1 = x2; f1 = f2; d1 = d2;                        % move point 2 to point 1
        x2 = x3; f2 = f3; d2 = d3;                        % move point 3 to point 2
        A = 6*(f1-f2)+3*(d2+d1)*(x2-x1);                 % make cubic extrapolation
        B = 3*(f2-f1)-(2*d1+d2)*(x2-x1);
        %%%%%x3 = x1-d1*(x2-x1)^2/(B+sqrt(B*B-A*d1*(x2-x1))); % num. error possible, ok!
        x3 = x1-d1*(x2-x1)^2/(B+sqrt(complex(B*B-A*d1*(x2-x1))));
        if ~isreal(x3) || isnan(x3) || isinf(x3) || x3 < 0 % num prob | wrong sign?
            x3 = x2*EXT;                                 % extrapolate maximum amount
        elseif x3 > x2*EXT                  % new point beyond extrapolation limit?
            x3 = x2*EXT;                                 % extrapolate maximum amount
        elseif x3 < x2+INT*(x2-x1)         % new point too close to previous point?
            x3 = x2+INT*(x2-x1);
        end
    end                                                       % end extrapolation
    
    while ((abs(d3) > -SIG*d0) || (f3 > f0+x3*RHO*d0)) && M > 0  % keep interpolating
        if d3 > 0 || f3 > f0+x3*RHO*d0                         % choose subinterval
            x4 = x3; f4 = f3; d4 = d3;                      % move point 3 to point 4
         else
             x2 = x3; f2 = f3; d2 = d3;                      % move point 3 to point 2
             
        end
        if f4 > f0
            x3 = x2-(0.5*d2*(x4-x2)^2)/(f4-f2-d2*(x4-x2));  % quadratic interpolation
            
        else
            bbug = 1;
            x4 = x3; %BUG FIXING. NEEDS TO BE CHECKED!!!   
            d4 = d3;
            A = 6*(f2-f4)/(x4-x2)+3*(d4+d2);                    % cubic interpolation
            B = 3*(f4-f2)-(2*d2+d4)*(x4-x2);
            x3 = x2+(sqrt(B*B-A*d2*(x4-x2)^2)-B)/A;        % num. error possible, ok!
            
        end
        if isnan(x3) || isinf(x3)
            x3 = (x2+x4)/2;               % if we had a numerical problem then bisect
        end
        x3 = max(min(x3, x4-INT*(x4-x2)),x2+INT*(x4-x2));  % don't accept too close
        gpu_x3 = gpuArray(x3);
        gpu_arg1 = gpu_X+gpu_x3*gpu_s;
        [f3 gpu_df3] = feval(f, gpu_arg1, varargin{:});
        clear gpu_arg1;
        if f3 < F0, gpu_X0 = gpu_X+gpu_x3*gpu_s; F0 = f3; gpu_dF0 = gpu_df3; end         % keep best values
        M = M - 1; i = i + (length<0);                             % count epochs?!

        gpu_d3 = sum(gpu_df3.*gpu_s);% new slope
        d3 = gather(gpu_d3);

    end                                                       % end interpolation
    
    if abs(d3) < -SIG*d0 && f3 < f0+x3*RHO*d0          % if line search succeeded
        gpu_x3 = gpuArray(x3);
        gpu_X = gpu_X+gpu_x3*gpu_s; 
        f0 = f3; 
        fX = [fX' f0]';
%        fprintf('%s %6i;  Value %4.6e\r', S, i, f0);
        gpu_s = (sum(gpu_df3.*gpu_df3)-sum(gpu_df0.*gpu_df3))/(sum(gpu_df0.*gpu_df0))*gpu_s - gpu_df3;   % Polack-Ribiere CG direction

        gpu_df0 = gpu_df3;                                               % swap derivatives
        d3 = d0; gpu_d0 = sum(gpu_df0.*gpu_s); d0 = gather(gpu_d0);
        if d0 > 0                                      % new slope must be negative
            gpu_s = -gpu_df0; gpu_d0 = sum(-gpu_s.*gpu_s); d0 = gather(gpu_d0);                   % otherwise use steepest direction
        end
        x3 = x3 * min(RATIO, d3/(d0-realmin));          % slope ratio but max RATIO
        ls_failed = 0;                              % this line search did not fail
    else
        gpu_X = gpu_X0; f0 = F0; gpu_df0 = gpu_dF0;                     % restore best point so far
        if ls_failed || i > abs(length)         % line search failed twice in a row
            break;                             % or we ran out of time, so we give up
        end
        gpu_s = -gpu_df0; gpu_d0 = sum(-gpu_s.*gpu_s); d0 = gather(gpu_d0);                                        % try steepest
        x3 = 1/(1-d0);
        ls_failed = 1;                                    % this line search failed
    end
end

end