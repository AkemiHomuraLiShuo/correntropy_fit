%% scan_log_fit_range.m
% 扫描：固定原子数时，多带宽 correntropy 对 smooth squared log
% 在多大有限区间 [0,L] 内仍能保持高精度逼近。
%
% 目标：
%   L(z)  = 0.5*c^2*log(1 + (z/c)^2)
%   L'(z) = z/(1 + (z/c)^2)
%
% Correntropy 原子：
%   phi_s(z)  = s*(1-exp(-z^2/(2*s^2)))
%   phi_s'(z) = (z/s)*exp(-z^2/(2*s^2))
%
% 默认扫描固定 10 原子、L=1,...,20。
% 输出绝对 RMSE、相对 L2 误差、峰值归一化最大误差，
% 并自动报告满足阈值的最大区间。
%
% MATLAB R2016b+ 可直接运行。

clear; clc; close all;

%% ---------------- 用户配置 ----------------
cfg.c = 1.0;

% 默认只扫描 10 原子。要比较不同原子数可改为：
% cfg.atomCounts = [2 4 6 8 10 12 15];
cfg.atomCounts = 10;

cfg.rangeMaxList = 1:1:20;

cfg.dictionaryMode = 'paper';
% 'paper':
%   对每个区间 [0,L] 设 sigmaRef=L，
%   s_q = (q/Qgrid)*L，严格保持线性相对带宽字典。
%
% 'expanded':
%   使用固定最小带宽和随 L 扩大的对数带宽字典，
%   更适合测试“函数表示能力极限”，但不再是严格论文线性字典。

cfg.Qgrid = 100;
cfg.Jfit = 1200;
cfg.Jeval = 3000;

cfg.expandedMinBandwidth = 0.02;
cfg.expandedMaxFactor = 1.5;  % expanded 模式下最大带宽 = factor*L

cfg.fitLossToo = false;
cfg.lossWeight = 1.0;
cfg.verbose = false;

% “高精度”的默认判据。可根据论文标准调整。
cfg.threshold.derivativeRelL2 = 1e-3;
cfg.threshold.derivativePeakMax = 5e-3;
cfg.threshold.lossRelL2 = 1e-3;
cfg.threshold.lossPeakMax = 5e-3;

%% ---------------- 扫描 ----------------
numA = numel(cfg.atomCounts);
numL = numel(cfg.rangeMaxList);

result = repmat(struct(), numA, numL);

fprintf('开始扫描：dictionary=%s, Qgrid=%d\n', ...
    cfg.dictionaryMode, cfg.Qgrid);

for ia = 1:numA
    numAtoms = cfg.atomCounts(ia);

    for iL = 1:numL
        Lmax = cfg.rangeMaxList(iL);

        zFit = linspace(0, Lmax, cfg.Jfit).';
        zEval = linspace(0, Lmax, cfg.Jeval).';

        [targetLossFit, targetDerFit] = target_log_loss(zFit, cfg.c);
        [targetLoss, targetDer] = target_log_loss(zEval, cfg.c);

        bandwidths = make_bandwidth_dictionary(Lmax, cfg);

        AderFit = correntropy_derivative_dictionary(zFit, bandwidths);
        AlossFit = correntropy_loss_dictionary(zFit, bandwidths);

        if cfg.fitLossToo
            scaleDer = max(norm(targetDerFit), eps);
            scaleLoss = max(norm(targetLossFit), eps);
            Afit = [AderFit/scaleDer; ...
                    cfg.lossWeight*AlossFit/scaleLoss];
            bfit = [targetDerFit/scaleDer; ...
                    cfg.lossWeight*targetLossFit/scaleLoss];
        else
            Afit = AderFit;
            bfit = targetDerFit;
        end

        [selected, coeffSelected] = sparse_nnls_forward( ...
            Afit, bfit, numAtoms, cfg.verbose);

        coeff = zeros(numel(bandwidths), 1);
        coeff(selected) = coeffSelected;

        targetDerHat = correntropy_derivative_dictionary( ...
            zEval, bandwidths) * coeff;
        targetLossHat = correntropy_loss_dictionary( ...
            zEval, bandwidths) * coeff;

        m = approximation_metrics( ...
            targetDer, targetDerHat, targetLoss, targetLossHat);

        passed = ...
            m.derivativeRelL2 <= cfg.threshold.derivativeRelL2 && ...
            m.derivativePeakMax <= cfg.threshold.derivativePeakMax && ...
            m.lossRelL2 <= cfg.threshold.lossRelL2 && ...
            m.lossPeakMax <= cfg.threshold.lossPeakMax;

        result(ia, iL).numAtoms = numAtoms;
        result(ia, iL).Lmax = Lmax;
        result(ia, iL).bandwidths = bandwidths(selected);
        result(ia, iL).coefficients = coeffSelected;
        result(ia, iL).metrics = m;
        result(ia, iL).passed = passed;

        fprintf(['atoms=%2d, L=%5.1f | dRel=%.3e, dMax=%.3e, ' ...
                 'LRel=%.3e, LMax=%.3e | %s\n'], ...
            numAtoms, Lmax, ...
            m.derivativeRelL2, m.derivativePeakMax, ...
            m.lossRelL2, m.lossPeakMax, ...
            string_pass(passed));
    end
end

%% ---------------- 汇总表 ----------------
rows = [];
for ia = 1:numA
    for iL = 1:numL
        r = result(ia, iL);
        rows = [rows; ...
            r.numAtoms, r.Lmax, ...
            r.metrics.derivativeRMSE, ...
            r.metrics.derivativeRelL2, ...
            r.metrics.derivativePeakMax, ...
            r.metrics.lossRMSE, ...
            r.metrics.lossRelL2, ...
            r.metrics.lossPeakMax, ...
            double(r.passed)]; %#ok<AGROW>
    end
end

T = array2table(rows, 'VariableNames', { ...
    'NumAtoms','RangeMax','DerivativeRMSE','DerivativeRelL2', ...
    'DerivativePeakMax','LossRMSE','LossRelL2','LossPeakMax','Passed'});

disp(T);

writetable(T, 'log_fit_range_scan.csv');
fprintf('\n结果已保存到 log_fit_range_scan.csv\n');

%% ---------------- 最大高精度区间 ----------------
fprintf('\n================ 最大高精度区间 ================\n');
for ia = 1:numA
    passMask = [result(ia, :).passed];
    if any(passMask)
        lastIndex = find(passMask, 1, 'last');
        fprintf('%2d 原子：最大通过区间 [0, %.3g]\n', ...
            cfg.atomCounts(ia), result(ia, lastIndex).Lmax);
    else
        fprintf('%2d 原子：没有区间通过当前阈值\n', cfg.atomCounts(ia));
    end
end
fprintf('==================================================\n');

%% ---------------- 误差趋势图 ----------------
figure('Color','w','Name','Maximum fitting range scan', ...
       'Position',[100 100 1100 760]);

tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

nexttile;
hold on;
for ia = 1:numA
    semilogy(cfg.rangeMaxList, ...
        arrayfun(@(r) r.metrics.derivativeRelL2, result(ia,:)), ...
        '-o','LineWidth',1.5);
end
yline(cfg.threshold.derivativeRelL2,'--','Threshold');
xlabel('Fitting interval upper bound L');
ylabel('Derivative relative L2 error');
legend(atom_legend(cfg.atomCounts),'Location','best');
grid on;
title('Derivative relative error');

nexttile;
hold on;
for ia = 1:numA
    semilogy(cfg.rangeMaxList, ...
        arrayfun(@(r) r.metrics.derivativePeakMax, result(ia,:)), ...
        '-o','LineWidth',1.5);
end
yline(cfg.threshold.derivativePeakMax,'--','Threshold');
xlabel('Fitting interval upper bound L');
ylabel('Derivative peak-normalized max error');
legend(atom_legend(cfg.atomCounts),'Location','best');
grid on;
title('Derivative maximum error');

nexttile;
hold on;
for ia = 1:numA
    semilogy(cfg.rangeMaxList, ...
        arrayfun(@(r) r.metrics.lossRelL2, result(ia,:)), ...
        '-o','LineWidth',1.5);
end
yline(cfg.threshold.lossRelL2,'--','Threshold');
xlabel('Fitting interval upper bound L');
ylabel('Loss relative L2 error');
legend(atom_legend(cfg.atomCounts),'Location','best');
grid on;
title('Loss relative error');

nexttile;
hold on;
for ia = 1:numA
    semilogy(cfg.rangeMaxList, ...
        arrayfun(@(r) r.metrics.lossPeakMax, result(ia,:)), ...
        '-o','LineWidth',1.5);
end
yline(cfg.threshold.lossPeakMax,'--','Threshold');
xlabel('Fitting interval upper bound L');
ylabel('Loss peak-normalized max error');
legend(atom_legend(cfg.atomCounts),'Location','best');
grid on;
title('Loss maximum error');

sgtitle(sprintf('Log fit range scan; dictionary=%s', cfg.dictionaryMode));

%% ---------------- 单独绘制 [0,10]、10 原子结果 ----------------
targetAtoms = 10;
targetRange = 10;

ia = find(cfg.atomCounts == targetAtoms, 1);
iL = find(cfg.rangeMaxList == targetRange, 1);

if ~isempty(ia) && ~isempty(iL)
    r = result(ia, iL);
    z = linspace(0, targetRange, cfg.Jeval).';
    [lossTrue, derTrue] = target_log_loss(z, cfg.c);

    allBandwidths = make_bandwidth_dictionary(targetRange, cfg);
    coeff = zeros(numel(allBandwidths),1);

    % 根据带宽值恢复索引；字典值唯一。
    for k = 1:numel(r.bandwidths)
        [~, idx] = min(abs(allBandwidths-r.bandwidths(k)));
        coeff(idx) = r.coefficients(k);
    end

    derHat = correntropy_derivative_dictionary(z, allBandwidths)*coeff;
    lossHat = correntropy_loss_dictionary(z, allBandwidths)*coeff;

    figure('Color','w','Name','10 atoms on [0,10]', ...
           'Position',[150 150 1100 430]);

    tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

    nexttile;
    plot(z,derTrue,'LineWidth',2); hold on;
    plot(z,derHat,'--','LineWidth',2);
    xlabel('Residual norm z');
    ylabel('Derivative');
    legend('Target log derivative','10-atom approximation', ...
           'Location','best');
    grid on;
    title(sprintf('[0,10], derivative rel. error %.3e', ...
          r.metrics.derivativeRelL2));

    nexttile;
    plot(z,lossTrue,'LineWidth',2); hold on;
    plot(z,lossHat,'--','LineWidth',2);
    xlabel('Residual norm z');
    ylabel('Loss');
    legend('Target log loss','10-atom approximation', ...
           'Location','best');
    grid on;
    title(sprintf('[0,10], loss rel. error %.3e', ...
          r.metrics.lossRelL2));

    fprintf('\n[0,10]、10原子所选带宽和系数：\n');
    fprintf(' bandwidth          coefficient\n');
    for k = 1:numel(r.bandwidths)
        fprintf(' %12.6f       %12.6e\n', ...
            r.bandwidths(k), r.coefficients(k));
    end
end

%% ======================== 局部函数 ========================
function [L, dL] = target_log_loss(z, c)
    L = 0.5*c^2 .* log1p((z./c).^2);
    dL = z ./ (1 + (z./c).^2);
end

function bandwidths = make_bandwidth_dictionary(Lmax, cfg)
    switch lower(cfg.dictionaryMode)
        case 'paper'
            omega = (1:cfg.Qgrid).' / cfg.Qgrid;
            bandwidths = Lmax * omega;

        case 'expanded'
            sMin = cfg.expandedMinBandwidth;
            sMax = cfg.expandedMaxFactor * Lmax;
            bandwidths = logspace(log10(sMin),log10(sMax), ...
                                  cfg.Qgrid).';

        otherwise
            error('未知 dictionaryMode: %s', cfg.dictionaryMode);
    end
end

function A = correntropy_derivative_dictionary(z, bandwidths)
    z = z(:);
    s = bandwidths(:).';
    A = (z./s).*exp(-(z.^2)./(2*s.^2));
end

function A = correntropy_loss_dictionary(z, bandwidths)
    z = z(:);
    s = bandwidths(:).';
    A = s.*(1-exp(-(z.^2)./(2*s.^2)));
end

function m = approximation_metrics(dTrue,dHat,LTrue,LHat)
    dErr = dTrue-dHat;
    LErr = LTrue-LHat;

    m.derivativeRMSE = sqrt(mean(dErr.^2));
    m.derivativeRelL2 = norm(dErr)/max(norm(dTrue),eps);
    m.derivativePeakMax = max(abs(dErr))/max(max(abs(dTrue)),eps);

    m.lossRMSE = sqrt(mean(LErr.^2));
    m.lossRelL2 = norm(LErr)/max(norm(LTrue),eps);
    m.lossPeakMax = max(abs(LErr))/max(max(abs(LTrue)),eps);
end

function [selected,xSelected] = sparse_nnls_forward(A,b,maxAtoms,verbose)
    [~,Q] = size(A);
    selected = [];
    currentResidual = norm(b);
    xSelected = [];

    for k = 1:maxAtoms
        remaining = setdiff(1:Q,selected);
        bestResidual = inf;
        bestQ = [];
        bestX = [];

        for q = remaining
            trial = [selected,q];
            xTrial = nnls_solve(A(:,trial),b);
            rTrial = norm(A(:,trial)*xTrial-b);

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
            (currentResidual-bestResidual)/max(currentResidual,eps);

        if k > 1 && relativeImprovement < 1e-10
            break;
        end

        selected = [selected,bestQ]; %#ok<AGROW>
        xSelected = bestX;
        currentResidual = bestResidual;

        if verbose
            fprintf('step %d: atom %d, residual %.6e\n', ...
                k,bestQ,bestResidual);
        end
    end

    xSelected = nnls_solve(A(:,selected),b);
    keep = xSelected > 1e-12*max(1,max(xSelected));
    selected = selected(keep);
    xSelected = nnls_solve(A(:,selected),b);
end

function x = nnls_solve(A,b)
    if exist('lsqnonneg','file') == 2
        x = lsqnonneg(A,b);
        return;
    end

    n = size(A,2);
    x = zeros(n,1);
    step = 1/(norm(A,2)^2+eps);

    for iter = 1:20000
        grad = A'*(A*x-b);
        xNew = max(0,x-step*grad);

        if norm(xNew-x) <= 1e-10*max(1,norm(x))
            x = xNew;
            break;
        end
        x = xNew;
    end
end

function s = string_pass(flag)
    if flag
        s = 'PASS';
    else
        s = 'FAIL';
    end
end

function labels = atom_legend(atomCounts)
    labels = arrayfun(@(x) sprintf('%d atoms',x), ...
        atomCounts,'UniformOutput',false);
end
