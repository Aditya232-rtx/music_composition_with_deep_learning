function plotTrainingHistory(info, outFile)
%PLOTTRAININGHISTORY Save training/validation loss + accuracy curves to a PNG.
%   info    - the TrainingInfo returned by trainNetwork (trainMusicLSTM.m)
%   outFile - output image path, e.g. 'training_curves.png'

fig = figure('Visible', 'off', 'Position', [100 100 900 700]);
tiledlayout(fig, 2, 1);

nexttile; hold on;
plotSeries(info, 'TrainingLoss', '-', 'Training Loss');
plotSeries(info, 'ValidationLoss', 'o-', 'Validation Loss');
legend('Location', 'best'); xlabel('Iteration'); ylabel('Loss');
title('Loss'); grid on;

nexttile; hold on;
plotSeries(info, 'TrainingAccuracy', '-', 'Training Accuracy');
plotSeries(info, 'ValidationAccuracy', 'o-', 'Validation Accuracy');
legend('Location', 'best'); xlabel('Iteration'); ylabel('Accuracy (%)');
title('Accuracy'); grid on;

exportgraphics(fig, outFile, 'Resolution', 150);
close(fig);
fprintf('Saved training curves to %s\n', outFile);
end

function plotSeries(info, fieldName, style, label)
if ~(isprop(info, fieldName) || isfield(info, fieldName))
    return;
end
y = info.(fieldName);
idx = find(~isnan(y));
if isempty(idx)
    return;
end
plot(idx, y(idx), style, 'DisplayName', label);
end
