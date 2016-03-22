function path = viterbi_path_wlm(prior, p_transmat, s_transmat, obslik, nstates,lmw)

% This is a modified version of viterbi_path, by Kevin P. Murphy
% VITERBI Find the most-probable (Viterbi) path through the HMM state trellis.
% path = viterbi(prior, transmat, obslik)
%
% Inputs:
% prior(i) = Pr(Q(1) = i)
% transmat(i,j) = Pr(Q(t+1)=j | Q(t)=i)
% obslik(i,t) = Pr(y(t) | Q(t)=i)
%
% Outputs:
% path(t) = q(t), where q1 ... qT is the argmax of the above expression.


% delta(j,t) = prob. of the best sequence of length t-1 and then going to state j, and O(1:t)
% psi(j,t) = the best predecessor state, given that we ended up in state j at t


transmat = s_transmat;
for j=1:nstates:size(transmat,2)-nstates+1
    for i=nstates:nstates:size(transmat,1)
        transmat(i,j) = p_transmat(ceil(i/nstates),ceil(j/nstates)) * (1-s_transmat(i,i));
    end
end

scaled = 1;

T = size(obslik, 2);
prior = prior(:);
Q = size(s_transmat,1);
nP = length(prior);
delta = zeros(Q,T);
psi = zeros(Q,T);
path = zeros(1,T);
scale = ones(1,T);

t=1;

delta(1:nstates:end,t) = (prior.^lmw) .* obslik(1:nstates:end,t);
delta(1:nstates:end,t) = obslik(1:nstates:end,t);
if scaled
  [delta(:,t), n] = normalise(delta(:,t));
  scale(t) = 1/n;
end
psi(:,t) = 0; % arbitrary value, since there is no predecessor to t=1
for t=2:T
  for j=1:Q
        
        [delta(j,t), psi(j,t)] = max(delta(:,t-1) .* (transmat(:,j).^lmw));
        delta(j,t) = delta(j,t) * obslik(j,t);
  end
  if scaled
    [delta(:,t), n] = normalise(delta(:,t));
    scale(t) = 1/n;
  end
end
[p, path(T)] = max(delta(:,T));
for t=T-1:-1:1
  path(t) = psi(path(t+1),t+1);
end

% If scaled==0, p = prob_path(best_path)
% If scaled==1, p = Pr(replace sum with max and proceed as in the scaled forwards algo)
% Both are different from p(data) as computed using the sum-product (forwards) algorithm

%if 0
if scaled
  loglik = -sum(log(scale));
  %loglik = prob_path(prior, transmat, obslik, path);
else
  loglik = log(p);
end
end
