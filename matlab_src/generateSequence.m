function tokenSeq = generateSequence(net, seedTokens, totalLen, windowLen, temperature)
%GENERATESEQUENCE Autoregressively sample a token sequence from a trained net.
%   net         - trained network from trainMusicLSTM
%   seedTokens  - vector of initial token indices (e.g. from a real piece)
%   totalLen    - desired total sequence length (including seed)
%   windowLen   - context length the network was trained with
%   temperature - >0; 1.0 = neutral, <1 more conservative, >1 more random
%
%   REST token index is assumed to be 1 (as produced by buildTokenDataset).

if nargin < 5, temperature = 1.0; end
REST_TOKEN = 1;

tokenSeq = zeros(totalLen, 1);
seedLen = numel(seedTokens);
tokenSeq(1:seedLen) = seedTokens(:);

for i = seedLen+1:totalLen
    startIdx = max(1, i-windowLen);
    context = tokenSeq(startIdx:i-1);
    if numel(context) < windowLen
        context = [REST_TOKEN * ones(windowLen-numel(context),1); context]; %#ok<AGROW>
    end
    Xin = {context'}; % 1-by-windowLen row, as required by the trained net
    probs = predict(net, Xin);
    tokenSeq(i) = sampleWithTemperature(probs(:), temperature);
end
end

function idx = sampleWithTemperature(probs, temperature)
probs = double(probs);
probs = max(probs, 1e-8);
logp = log(probs) / temperature;
p = exp(logp - max(logp));
p = p / sum(p);
edges = [0; cumsum(p)];
r = rand();
idx = find(r < edges(2:end), 1, 'first');
if isempty(idx)
    [~, idx] = max(p);
end
end
