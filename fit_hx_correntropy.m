%% fit_hx_correntropy.m
% 使用多带宽 correntropy 原子拟合 RBSMF 的 Hx 鲁棒损失
%
% RBSMF:
%   Hx(z)  = c * log(1 + z/c), z >= 0
%   psi(z) = dHx/dz = c/(c+z) = 1/(1+z/c)
%
% Correntropy 原子:
%   phi_s(z)  = s * (1 - exp(-z^2/(2s^2)))
%   phi_s'(z) = (z/s) * exp(-z^2/(2s^2))
%
% 说明:
%   所有 correntropy 原子的导数在 z=0 处均为 0，而 Hx'(0)=1。
%   因而仅使用 correntropy 原子时，无法在包含 z=0 的区间上
%   一致逼近 Hx 的影响函数。这里在 [tau,L] 上拟合/评价导数，
%   同时在完整区间 [0,L] 上拟合/评价损失。
%
% MATLAB R2019b+ 可直接运行（使用 tiledlayout 和 sgtitle）。

clear; clc; close all;

%% ---------------- 配置 ----------------
cfg.c = 1.0;
cfg.L = 10.0;

% 导数拟合区间下界；对应论文中 [tau_b, 1] 的有限区间思想
cfg.tau = 0.2;

% 候选字典和稀疏原子数
cfg.Qgrid = 200;
cfg.numAtoms = 15;

cfg.Jfit = 1600;
cfg.Jeval = 5000;

% 联合拟合中损失块的相对权重
% 1.0 表示损失和导数分别归一化后等权
cfg.lossWeight = 1.0;

cfg.verbose = true;

%% ---------------- 目标函数 ----------------
zLossFit = linspace(0, cfg.L, cfg.Jfit).';
zDerFit  = linspace(cfg.tau, cfg.L, cfg.Jfit).';

[HxLossFit, ~] = target_hx(zLossFit, cfg.c);
[~, HxDerFit]  = target_hx(zDerFit, cfg.c);

%% ---------------- 线性相对带宽字典 ----------------
omega = (1:cfg.Qgrid).' / cfg.Qgrid;
bandwidths = cfg.L * omega;

AlossFit = correntropy_loss_dictionary(zLossFit, bandwidths);
AderFit  = correntropy_derivative_dictionary(zDerFit, bandwidths);

%% ---------------- 联合归一化拟合 ----------------
scaleLoss = max(norm(HxLossFit), eps);
scaleDer  = max(norm(HxDerFit), eps);

Afit = [AderFit / scaleDer; ...
        cfg.lossWeight * AlossFit / scaleLoss];

bfit = [HxDerFit / scaleDer; ...
        cfg.lossWeight * HxLossFit / scaleLoss];

[selected, coeffSelected] = sparse_nnls_forward( ...
    Afit, bfit, cfg.numAtoms, cfg.verbose);

coeff = zeros(cfg.Qgrid, 1);
coeff(selected) = coeffSelected;

%% ---------------- 完整区间评价 ----------------
zEval = linspace(0, cfg.L, cfg.Jeval).';
[HxTrue, psiTrue] = target_hx(zEval, cfg.c);

HxHat  = correntropy_loss_dictionary(zEval, bandwidths) * coeff;
psiHat = correntropy_derivative_dictionary(zEval, bandwidths) * coeff;

maskDer = zEval >= cfg.tau;

lossMetrics = approximation_metrics(HxTrue, HxHat);
derMetricsRestricted = approximation_metrics( ...
    psiTrue(maskDer), psiHat(maskDer));
derMetricsFull = approximation_metrics(psiTrue, psiHat);

%% ---------------- 输出 ----------------
fprintf('\n============================================================\n');
fprintf('RBSMF Hx fitting by multi-bandwidth correntropy\n');
fprintf('Hx(z) = %.4g * log(1 + z/%.4g)\n', cfg.c, cfg.c);
fprintf('Loss interval:      [0, %.4g]\n', cfg.L);
fprintf('Derivative interval:[%.4g, %.4g]\n', cfg.tau, cfg.L);
fprintf('Candidate atoms:    %d\n', cfg.Qgrid);
fprintf('Selected atoms:     %d\n', numel(selected));
fprintf('============================================================\n');

fprintf('\nLoss metrics on [0, L]:\n');
print_metrics(lossMetrics);

fprintf('\nDerivative metrics on [tau, L]:\n');
print_metrics(derMetricsRestricted);

fprintf('\nDerivative metrics on [0, L] (reported only for reference):\n');
print_metrics(derMetricsFull);

fprintf('\nSelected bandwidths and coefficients:\n');
fprintf(' index      bandwidth          coefficient\n');
for k = 1:numel(selected)
    fprintf(' %5d    %12.6f      %14.8e\n', ...
        selected(k), bandwidths(selected(k)), coeffSelected(k));
end

%% ---------------- 保存数值结果 ----------------
T = table(zEval, HxTrue, HxHat, psiTrue, psiHat, ...
    'VariableNames', {'z','HxTrue','HxApprox','PsiTrue','PsiApprox'});
writetable(T, 'hx_correntropy_fit_curves.csv');

Ts = table(selected(:), bandwidths(selected), coeffSelected(:), ...
    'VariableNames', {'DictionaryIndex','Bandwidth','Coefficient'});
writetable(Ts, 'hx_correntropy_selected_atoms.csv');

save('hx_correntropy_fit_result.mat', ...
    'cfg','selected','coeffSelected','bandwidths', ...
    'lossMetrics','derMetricsRestricted','derMetricsFull');

fprintf('\n结果已保存：\n');
fprintf('  hx_correntropy_fit_curves.csv\n');
fprintf('  hx_correntropy_selected_atoms.csv\n');
fprintf('  hx_correntropy_fit_result.mat\n');

%% ---------------- 绘图 ----------------
figure('Color','w','Name','RBSMF Hx fitting', ...
       'Position',[100 100 1150 760]);

tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

nexttile;
plot(zEval, HxTrue, 'LineWidth', 2); hold on;
plot(zEval, HxHat, '--', 'LineWidth', 2);
xlabel('Residual norm z');
ylabel('Loss');
legend('RBSMF H_x','Correntropy approximation','Location','best');
grid on;
title(sprintf('Loss relative L_2 error: %.3e', lossMetrics.relL2));

nexttile;
plot(zEval, psiTrue, 'LineWidth', 2); hold on;
plot(zEval, psiHat, '--', 'LineWidth', 2);
xline(cfg.tau, ':', '\tau');
xlabel('Residual norm z');
ylabel('Influence function');
legend('\psi_{H_x}','Correntropy approximation','Location','best');
grid on;
title(sprintf('[\\tau,L] derivative relative L_2 error: %.3e', ...
      derMetricsRestricted.relL2));

nexttile;
semilogy(zEval, abs(HxTrue-HxHat)+eps, 'LineWidth', 1.5);
xlabel('Residual norm z');
ylabel('Absolute loss error');
grid on;
title('Pointwise loss error');

nexttile;
stem(bandwidths(selected), coeffSelected, 'filled', ...
     'LineWidth', 1.2);
xlabel('Bandwidth s');
ylabel('Coefficient');
grid on;
title(sprintf('%d selected correntropy atoms', numel(selected)));

sgtitle(sprintf(['RBSMF H_x fit: L=%.3g, \\tau=%.3g, ' ...
                 'Q=%d, selected=%d'], ...
                 cfg.L, cfg.tau, cfg.Qgrid, numel(selected)));

%% ======================== 局部函数 ========================

function [L, dL] = target_hx(z, c)
    % RBSMF Hx loss for nonnegative residual norm z
    z = z(:);
    L = c .* log1p(z ./ c);
    dL = c ./ (c + z);
end

function A = correntropy_derivative_dictionary(z, bandwidths)
    z = z(:);
    s = bandwidths(:).';
    A = (z ./ s) .* exp(-(z.^2) ./ (2 .* s.^2));
end

function A = correntropy_loss_dictionary(z, bandwidths)
    z = z(:);
    s = bandwidths(:).';
    A = s .* (1 - exp(-(z.^2) ./ (2 .* s.^2)));
end

function m = approximation_metrics(yTrue, yHat)
    yTrue = yTrue(:);
    yHat = yHat(:);
    err = yTrue - yHat;

    m.rmse = sqrt(mean(err.^2));
    m.relL2 = norm(err) / max(norm(yTrue), eps);
    m.peakNormalizedMax = ...
        max(abs(err)) / max(max(abs(yTrue)), eps);
    m.maxAbs = max(abs(err));
end

function print_metrics(m)
    fprintf('  RMSE                         = %.6e\n', m.rmse);
    fprintf('  Relative L2 error            = %.6e\n', m.relL2);
    fprintf('  Peak-normalized max error    = %.6e\n', ...
        m.peakNormalizedMax);
    fprintf('  Maximum absolute error       = %.6e\n', m.maxAbs);
end

function [selected, xSelected] = sparse_nnls_forward( ...
    A, b, maxAtoms, verbose)

    [~, Q] = size(A);
    selected = [];
    xSelected = [];
    currentResidual = norm(b);

    for k = 1:maxAtoms
        remaining = setdiff(1:Q, selected);
        bestResidual = inf;
        bestQ = [];
        bestX = [];

        for q = remaining
            trial = [selected, q];
            xTrial = nnls_solve(A(:,trial), b);
            rTrial = norm(A(:,trial) * xTrial - b);

            if rTrial < bestResidual
                bestResidual = rTrial;
                bestQ = q;
                bestX = xTrial;
            end
        end

        if isempty(bestQ)
            break;
        end

        relativeImprovement = ...
            (currentResidual - bestResidual) / ...
            max(currentResidual, eps);

        if k > 1 && relativeImprovement < 1e-12
            break;
        end

        selected = [selected, bestQ]; %#ok<AGROW>
        xSelected = bestX;
        currentResidual = bestResidual;

        if verbose
            fprintf('step %2d: atom %3d, residual %.6e\n', ...
                k, bestQ, bestResidual);
        end
    end

    if isempty(selected)
        error('未选择到有效原子。');
    end

    xSelected = nnls_solve(A(:,selected), b);

    keep = xSelected > 1e-12 * max(1, max(xSelected));
    selected = selected(keep);
    xSelected = nnls_solve(A(:,selected), b);
end

function x = nnls_solve(A, b)
    if exist('lsqnonneg', 'file') == 2
        x = lsqnonneg(A, b);
        return;
    end

    % 无 Optimization Toolbox 时的投影梯度备用实现
    n = size(A,2);
    x = zeros(n,1);
    step = 1 / (norm(A,2)^2 + eps);

    for iter = 1:30000
        grad = A' * (A*x - b);
        xNew = max(0, x - step*grad);

        if norm(xNew-x) <= 1e-11 * max(1, norm(x))
            x = xNew;
            break;
        end
        x = xNew;
    end
end
