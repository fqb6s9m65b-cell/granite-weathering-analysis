function resnet50_weathering_regression(imgRoots, jsonRoots, outRoot, varargin)
% resnet50_weathering_regression  화강암 풍화도 회귀 (MATLAB R2025a)
% ====================================================================
%  백본  : ResNet-50 (ImageNet pretrained)
%  태스크: D1=1.0, D1~2=1.5, D2=2.0 … D5=5.0 연속값 회귀
%  GPU   : RTX 5060 8 GB 최적화 (batch 16, single precision)
%
%  Nature급 학습 파이프라인
%  ────────────────────────
%   1) 시추공 단위 Stratified Group K-Fold CV (데이터 누출 차단)
%   2) 3-단계 데이터 증강 (기하·색상·수평반전) + 가중 손실
%   3) 2-Phase 전이학습  Head-only → Full Fine-tuning
%   4) Cosine Annealing LR + Early Stopping (patience 7)
%   5) 앙상블 학습 (Bagging, N=10) → 평균 예측 + 불확실성
%   6) 통계 지표  RMSE·MAE·R²·Cohen κ·Spearman ρ·ICC + 95% CI
%   7) 시각화  산점도·혼동행렬·Bland-Altman·t-SNE·Calibration·Grad-CAM
%   8) Ablation study (증강/앙상블/백본 기여도)
%   9) .mat 모델 + .xlsx 리포트 저장
%
%  사용법
%   resnet50_weathering_regression()
%   resnet50_weathering_regression([], [], [], 'K_FOLD', 5)
%   resnet50_weathering_regression([], [], [], 'N_ENSEMBLE', 5, 'MAX_IMAGES', 1000)

clc;

%% ==================== 0) 기본 경로 ====================
DEF_IMG = {
  'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\images_igneous_rock\TS_1, 기반암 암종 분류 데이터_1. 화성암_1'
  'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\images_igneous_rock\TS_1, 기반암 암종 분류 데이터_1. 화성암_2'
  'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\images_igneous_rock\TS_2. 기반암 절리 탐지 데이터_1. 화성암'
  'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\images_igneous_rock\VS_1. 기반암 암종 분류 데이터_1. 화성암'
  'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\images_igneous_rock\VS_2. 기반암 절리 탐지 데이터_1. 화성암'
};
DEF_JSON = {
  'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\json_igneous_rock\TL_1. 기반암 암종 분류 데이터_1. 화성암'
  'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\json_igneous_rock\TL_2. 기반암 절리 탐지 데이터_1. 화성암'
  'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\json_igneous_rock\VL_1. 기반암 암종 분류 데이터_1. 화성암'
  'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\json_igneous_rock\VL_2. 기반암 절리 탐지 데이터_1. 화성암'
};
DEF_OUT = 'C:\Users\ROCKENG\Desktop\코랩 머신러닝\results';

if nargin<1||isempty(imgRoots),  imgRoots  = DEF_IMG;  end
if nargin<2||isempty(jsonRoots), jsonRoots = DEF_JSON; end
if nargin<3||isempty(outRoot),   outRoot   = DEF_OUT;  end

%% ==================== 1) 하이퍼파라미터 ====================
p = inputParser; p.KeepUnmatched = false;
p.addRequired('imgRoots');
p.addRequired('jsonRoots');
p.addRequired('outRoot');

% ── 파일
p.addParameter('IMG_EXT',   [".jpg",".jpeg",".png",".tif",".tiff"]);
p.addParameter('RECURSIVE', true);
p.addParameter('MAX_IMAGES', inf);

% ── 분할
p.addParameter('GROUP_BY_BOREHOLE', true);
p.addParameter('K_FOLD', 1);          % 1이면 단일 split, ≥2이면 K-Fold CV
p.addParameter('TRAIN_RATIO', 0.70);  % K_FOLD=1일 때만 사용
p.addParameter('VAL_RATIO',   0.15);
p.addParameter('TEST_RATIO',  0.15);
p.addParameter('RNG_SEED', 42);

% ── 환경
p.addParameter('EXEC_ENV',   "auto");
p.addParameter('IMAGE_SIZE', [224 224 3]);

% ── Phase 1  Head-only
p.addParameter('EPOCHS_HEAD', 10);
p.addParameter('LR_HEAD',     1e-3);

% ── Phase 2  Full fine-tune
p.addParameter('EPOCHS_FINE', 50);
p.addParameter('LR_FINE',     1e-4);

% ── 공통
p.addParameter('MB_SIZE',  16);
p.addParameter('PATIENCE', 7);

% ── LR 스케줄
p.addParameter('LR_SCHEDULE', "cosine");   % "cosine" | "piecewise"
p.addParameter('LR_DROP_PERIOD', 15);      % piecewise용
p.addParameter('LR_DROP_FACTOR', 0.1);

% ── 데이터 증강
p.addParameter('DO_AUG',    true);
p.addParameter('AUG_ROT',   [-15 15]);
p.addParameter('AUG_TRANS', [-15 15]);
p.addParameter('AUG_SCALE', [0.90 1.10]);
p.addParameter('AUG_SHEAR', [-5 5]);

% ── Color jitter
p.addParameter('DO_COLOR_JITTER', true);
p.addParameter('CJ_BRIGHT', 0.20);
p.addParameter('CJ_CONTR',  0.20);
p.addParameter('CJ_GAMMA',  0.15);
p.addParameter('CJ_SAT',    0.10);
p.addParameter('CJ_HUE',    0.03);

% ── 가중 손실 (클래스 불균형 보정)
p.addParameter('USE_WEIGHTED_LOSS', true);

% ── 앙상블
p.addParameter('N_ENSEMBLE', 10);

% ── Ablation
p.addParameter('DO_ABLATION', true);       % 증강/앙상블 기여도 분석

p.parse(imgRoots, jsonRoots, outRoot, varargin{:});
OPT = p.Results;

rng(OPT.RNG_SEED);
imgRoots  = normRoots(OPT.imgRoots);
jsonRoots = normRoots(OPT.jsonRoots);

%% ==================== 2) cfg 구조체 ====================
cfg.imageSize     = OPT.IMAGE_SIZE;
cfg.execEnv       = resolveEnv(OPT.EXEC_ENV);
cfg.doColorJitter = OPT.DO_COLOR_JITTER;
cfg.cjBright = OPT.CJ_BRIGHT;  cfg.cjContr = OPT.CJ_CONTR;
cfg.cjGamma  = OPT.CJ_GAMMA;   cfg.cjSat   = OPT.CJ_SAT;
cfg.cjHue    = OPT.CJ_HUE;
cfg.augRot   = OPT.AUG_ROT;    cfg.augTrans = OPT.AUG_TRANS;
cfg.augScale = OPT.AUG_SCALE;  cfg.augShear = OPT.AUG_SHEAR;

%% ==================== 3) 폴더 생성 ====================
stamp  = string(datetime("now","Format","yyyyMMdd_HHmmss"));
runDir = fullfile(char(string(OPT.outRoot)), char(stamp+"_ResNet50_RW_Reg"));
dirCkpt    = fullfile(runDir,"ckpt");
dirLogs    = fullfile(runDir,"logs");
dirModels  = fullfile(runDir,"models");
dirPlots   = fullfile(runDir,"plots");
dirReports = fullfile(runDir,"reports");
cellfun(@mkd, {runDir,dirCkpt,dirLogs,dirModels,dirPlots,dirReports}, ...
    'UniformOutput', false);

try
    diary(fullfile(dirLogs,"run.log")); diary on;
catch
end

hdr("ResNet-50 화강암 풍화도 회귀 모델");
fprintf("  GPU        : %s\n",  cfg.execEnv);
fprintf("  Image      : %dx%d\n", cfg.imageSize(1), cfg.imageSize(2));
fprintf("  Batch      : %d\n",  OPT.MB_SIZE);
fprintf("  Ensemble   : %d\n",  OPT.N_ENSEMBLE);
fprintf("  K-Fold     : %d\n",  OPT.K_FOLD);
fprintf("  LR Schedule: %s\n",  OPT.LR_SCHEDULE);
fprintf("  Weighted   : %d\n",  OPT.USE_WEIGHTED_LOSS);
fprintf("  Ablation   : %d\n",  OPT.DO_ABLATION);
fprintf("  출력 폴더  : %s\n\n", runDir);

%% ==================== 4) 이미지 수집 ====================
hdr("1/11  이미지 수집");
imgFiles = collectFiles(imgRoots, string(OPT.IMG_EXT), OPT.RECURSIVE);
if isfinite(OPT.MAX_IMAGES)
    imgFiles = imgFiles(1:min(end, OPT.MAX_IMAGES));
end
nFiles = numel(imgFiles);
fprintf("  총 이미지: %d장\n", nFiles);
if nFiles == 0
    error("이미지 없음.");
end

%% ==================== 5) JSON 인덱싱 ====================
hdr("2/11  JSON 인덱싱");
jsonMap = containers.Map('KeyType','char','ValueType','char');
jfiles  = collectFiles(jsonRoots, ".json", true);
fprintf("  총 JSON: %d개\n", numel(jfiles));
for i = 1:numel(jfiles)
    [~,jb,~] = fileparts(jfiles(i));
    k = char(string(jb));
    if ~isKey(jsonMap, k)
        jsonMap(k) = char(jfiles(i));
    end
end

%% ==================== 6) 인벤토리 구축 ====================
hdr("3/11  인벤토리 구축");
ImageFile = strings(nFiles,1);
rw_raw    = strings(nFiles,1);
rw_value  = nan(nFiles,1);
borehole  = strings(nFiles,1);

for k = 1:nFiles
    f = imgFiles(k);
    [~,bn,~] = fileparts(f);
    bn = string(bn);

    s = struct();
    if isKey(jsonMap, char(bn))
        jp = string(jsonMap(char(bn)));
        try
            s = jsondecode(fileread(char(jp)));
        catch
        end
    end

    borehole(k) = extractBH(bn);

    rwStr = extractRw(s);
    if strlength(rwStr) == 0
        rwStr = parseRwFromFilename(bn);
    end

    ImageFile(k) = f;
    rw_raw(k)    = rwStr;
    rw_value(k)  = rwToVal(rwStr);

    if mod(k, 10000) == 0
        fprintf("  [Build] %d / %d\n", k, nFiles);
    end
end

T = table(ImageFile, borehole, rw_raw, rw_value);
T = T(isfinite(T.rw_value), :);
T.rw_class = max(1, min(5, round(T.rw_value)));

fprintf("  유효 데이터: %d / %d (%.1f%%)\n", height(T), nFiles, height(T)/nFiles*100);
printDistribution(T);

try
    writetable(T, fullfile(dirReports,"INVENTORY.csv"));
catch
end

% ── 클래스 가중치 계산 (역빈도)
classWeights = ones(1,5);
if OPT.USE_WEIGHTED_LOSS
    for c = 1:5
        nc = sum(T.rw_class == c);
        if nc > 0
            classWeights(c) = height(T) / (5 * nc);
        end
    end
    classWeights = classWeights / sum(classWeights) * 5;
    fprintf("\n  [가중치] D1=%.2f D2=%.2f D3=%.2f D4=%.2f D5=%.2f\n", classWeights);
end
cfg.classWeights = classWeights;

%% ==================== 7) K-Fold CV 또는 단일 분할 ====================
if OPT.K_FOLD >= 2
    hdr(sprintf("4/11  %d-Fold 교차검증 (시추공 Group)", OPT.K_FOLD));
    allFoldMetrics = runKFoldCV(T, OPT, cfg, dirModels, dirCkpt, dirPlots, dirReports);

    % K-Fold 요약
    printKFoldSummary(allFoldMetrics, OPT.K_FOLD, dirReports);

    % K-Fold에서는 앙상블/ablation 생략 → 바로 리포트
    saveKFoldReport(allFoldMetrics, OPT, cfg, stamp, height(T), dirReports);

    fprintf("\n");
    hdr("완료 (K-Fold CV)");
    fprintf("  결과 폴더: %s\n", runDir);
    try
        diary off;
    catch
    end
    return;
end

%% ==================== 단일 분할 모드 ====================
hdr("4/11  데이터 분할 (시추공 Group Split)");

if OPT.GROUP_BY_BOREHOLE
    [idxTr,idxVa,idxTe] = splitByGroup(T.rw_class, T.borehole, ...
        OPT.TRAIN_RATIO, OPT.VAL_RATIO, OPT.TEST_RATIO, OPT.RNG_SEED);
else
    [idxTr,idxVa,idxTe] = splitStrat(T.rw_class, ...
        OPT.TRAIN_RATIO, OPT.VAL_RATIO, OPT.TEST_RATIO, OPT.RNG_SEED);
end

Ttr = T(idxTr,:);  Tva = T(idxVa,:);  Tte = T(idxTe,:);
fprintf("  Train: %d  Val: %d  Test: %d\n", height(Ttr), height(Tva), height(Tte));

leak = intersect(unique(Ttr.borehole), unique(Tte.borehole));
if isempty(leak)
    fprintf("  [OK] 시추공 누출 없음\n");
else
    fprintf("  [WARN] %d개 시추공 겹침!\n", numel(leak));
end

try
    writetable(Ttr, fullfile(dirReports,"split_train.csv"));
catch
end
try
    writetable(Tva, fullfile(dirReports,"split_val.csv"));
catch
end
try
    writetable(Tte, fullfile(dirReports,"split_test.csv"));
catch
end

%% ==================== 8) 앙상블 학습 ====================
hdr(sprintf("5/11  앙상블 학습 (N=%d, Bagging)", OPT.N_ENSEMBLE));
tStart = tic;

nEns = OPT.N_ENSEMBLE;
nets  = cell(nEns, 1);
infos = cell(nEns, 1);

for m = 1:nEns
    fprintf("\n  ═══ 앙상블 모델 %d / %d ═══\n", m, nEns);
    rng(OPT.RNG_SEED + m);

    nTr = height(Ttr);
    bootIdx = randi(nTr, nTr, 1);
    Ttr_boot = Ttr(bootIdx, :);

    tblTr = table(Ttr_boot.ImageFile, Ttr_boot.rw_value, ...
        'VariableNames', {'imageFilename','rw'});
    tblVa = table(Tva.ImageFile, Tva.rw_value, ...
        'VariableNames', {'imageFilename','rw'});

    dsTr = buildRegDs(tblTr, cfg, OPT.DO_AUG, true);
    dsVa = buildRegDs(tblVa, cfg, false, false);

    lgraph = buildResNet50Reg();

    % Phase 1: Head-only
    fprintf("  [Phase 1] Head-only (%d ep, LR=%.0e)\n", OPT.EPOCHS_HEAD, OPT.LR_HEAD);
    lg1 = freezeBackbone(lgraph);
    opts1 = makeOpts("adam", cfg, OPT, OPT.LR_HEAD, OPT.EPOCHS_HEAD, dsVa, dirCkpt);
    [net1, ~] = trainNetwork(dsTr, lg1, opts1);

    % Phase 2: Fine-tune
    fprintf("  [Phase 2] Fine-tune (%d ep, LR=%.0e)\n", OPT.EPOCHS_FINE, OPT.LR_FINE);
    lg2 = unfreezeAll(layerGraph(net1));
    opts2 = makeOpts("adam", cfg, OPT, OPT.LR_FINE, OPT.EPOCHS_FINE, dsVa, dirCkpt);
    [net2, info2] = trainNetwork(dsTr, lg2, opts2);

    nets{m}  = net2;
    infos{m} = info2;

    netSingle = net2; 
    try
        save(fullfile(dirModels, sprintf("model_ens_%02d.mat",m)), ...
            "netSingle","OPT","cfg", "-v7.3");
    catch
    end
end

trainTime = toc(tStart);
fprintf("\n  총 학습 시간: %.1f분 (%.1f시간)\n", trainTime/60, trainTime/3600);

% 통합 저장
try
    save(fullfile(dirModels,"model_ensemble_all.mat"), ...
        "nets","infos","OPT","cfg","T","Ttr","Tva","Tte", "-v7.3");
    fprintf("  [Saved] model_ensemble_all.mat\n");
catch ME
    fprintf("  [WARN] 저장 실패: %s\n", ME.message);
end

%% ==================== 9) 앙상블 테스트 평가 ====================
hdr("6/11  테스트 평가");

dsTe = buildPredDs(Tte.ImageFile, cfg);

allPreds = zeros(height(Tte), nEns);
for m = 1:nEns
    pred = predict(nets{m}, dsTe, ...
        "ExecutionEnvironment", cfg.execEnv, ...
        "MiniBatchSize", OPT.MB_SIZE);
    allPreds(:,m) = double(pred(:));
end

ensPred  = mean(allPreds, 2);
ensStd   = std(allPreds, 0, 2);
ensMin   = min(allPreds, [], 2);
ensMax   = max(allPreds, [], 2);
trueVals = Tte.rw_value;

% ── 종합 지표 (Nature급)
mEns = calcMetricsFull(trueVals, ensPred);

% 개별 + 앙상블 테이블 출력
fprintf("\n  ┌──────────┬────────┬────────┬────────┬────────┬────────┬────────┐\n");
fprintf("  │  Model   │  RMSE  │   MAE  │   R²   │ Acc(%%) │ Kappa  │ Spear  │\n");
fprintf("  ├──────────┼────────┼────────┼────────┼────────┼────────┼────────┤\n");
for m = 1:nEns
    mm = calcMetricsFull(trueVals, allPreds(:,m));
    fprintf("  │  Ens %02d  │ %6.3f │ %6.3f │ %6.3f │ %5.1f  │ %6.3f │ %6.3f │\n", ...
        m, mm.RMSE, mm.MAE, mm.R2, mm.Accuracy, mm.Kappa, mm.Spearman);
end
fprintf("  ├──────────┼────────┼────────┼────────┼────────┼────────┼────────┤\n");
fprintf("  │ ENSEMBLE │ %6.3f │ %6.3f │ %6.3f │ %5.1f  │ %6.3f │ %6.3f │\n", ...
    mEns.RMSE, mEns.MAE, mEns.R2, mEns.Accuracy, mEns.Kappa, mEns.Spearman);
fprintf("  └──────────┴────────┴────────┴────────┴────────┴────────┴────────┘\n");

% 95% 부트스트랩 신뢰구간
fprintf("\n  [95%% Bootstrap CI (n=1000)]\n");
ci = bootstrapCI(trueVals, ensPred, 1000);
fprintf("    RMSE : %.3f [%.3f, %.3f]\n", mEns.RMSE, ci.RMSE_lo, ci.RMSE_hi);
fprintf("    MAE  : %.3f [%.3f, %.3f]\n", mEns.MAE,  ci.MAE_lo,  ci.MAE_hi);
fprintf("    R²   : %.3f [%.3f, %.3f]\n", mEns.R2,   ci.R2_lo,   ci.R2_hi);
fprintf("    Kappa: %.3f [%.3f, %.3f]\n", mEns.Kappa, ci.Kappa_lo, ci.Kappa_hi);

% 클래스별 지표
fprintf("\n  [클래스별 성능]\n");
fprintf("  ┌───────┬───────┬───────┬────────┬───────┬────────┐\n");
fprintf("  │ Class │ RMSE  │  MAE  │ Acc(%%)│   N   │ Prec%%  │\n");
fprintf("  ├───────┼───────┼───────┼────────┼───────┼────────┤\n");
for c = 1:5
    mask = Tte.rw_class == c;
    if sum(mask) > 0
        mc = calcMetricsFull(trueVals(mask), ensPred(mask));
        predC = max(1,min(5,round(ensPred)));
        prec = sum(predC(predC==c) == Tte.rw_class(predC==c)) / max(1,sum(predC==c)) * 100;
        fprintf("  │  D%d   │ %5.3f │ %5.3f │ %5.1f  │ %5d │ %5.1f  │\n", ...
            c, mc.RMSE, mc.MAE, mc.Accuracy, sum(mask), prec);
    end
end
fprintf("  └───────┴───────┴───────┴────────┴───────┴────────┘\n");

% 예측 범위 (앙상블 불확실성)
fprintf("\n  앙상블 예측 범위 (max-min): %.2f ± %.2f\n", ...
    mean(ensMax-ensMin), std(ensMax-ensMin));

% 결과 테이블
Tte.pred_ensemble = ensPred;
Tte.pred_std      = ensStd;
Tte.pred_class    = max(1,min(5,round(ensPred)));
Tte.error         = ensPred - trueVals;
try
    writetable(Tte, fullfile(dirReports,"test_predictions.csv"));
catch
end

%% ==================== 10) 시각화 (Nature급) ====================
hdr("7/11  시각화");

plotScatter(trueVals, ensPred, ensStd, mEns, dirPlots, nEns);
plotCM(trueVals, ensPred, mEns, dirPlots);
plotErr(trueVals, ensPred, dirPlots);
plotUncertainty(trueVals, ensPred, ensStd, dirPlots);
plotBlandAltman(trueVals, ensPred, dirPlots);
plotCalibration(trueVals, ensPred, dirPlots);
plotLearningCurve(infos{end}, dirPlots);

fprintf("  [OK] 그래프 저장 완료\n");

%% ==================== 11) Grad-CAM ====================
hdr("8/11  Grad-CAM");
try
    plotGradCAM(nets{1}, Tte, cfg, dirPlots);
    fprintf("  [OK] Grad-CAM 저장 완료\n");
catch ME
    fprintf("  [WARN] Grad-CAM 실패: %s\n", ME.message);
end

%% ==================== 12) t-SNE 특징 공간 ====================
hdr("9/11  t-SNE 특징 공간");
try
    plotTSNE(nets{1}, Tte, cfg, OPT, dirPlots);
    fprintf("  [OK] t-SNE 저장 완료\n");
catch ME
    fprintf("  [WARN] t-SNE 실패: %s\n", ME.message);
end

%% ==================== 13) Ablation Study ====================
if OPT.DO_ABLATION
    hdr("10/11  Ablation Study");
    runAblation(Ttr, Tva, Tte, OPT, cfg, dirPlots, dirReports, dirCkpt);
end

%% ==================== 14) 요약 리포트 ====================
hdr("11/11  리포트 저장");

sumT = table();
sumT.Timestamp    = stamp;
sumT.Model        = "ResNet-50 Ensemble";
sumT.N_Ensemble   = nEns;
sumT.Total_Data   = height(T);
sumT.Train        = height(Ttr);
sumT.Val          = height(Tva);
sumT.Test         = height(Tte);
sumT.RMSE         = mEns.RMSE;
sumT.RMSE_CI      = sprintf("[%.3f, %.3f]", ci.RMSE_lo, ci.RMSE_hi);
sumT.MAE          = mEns.MAE;
sumT.R2           = mEns.R2;
sumT.R2_CI        = sprintf("[%.3f, %.3f]", ci.R2_lo, ci.R2_hi);
sumT.Accuracy     = mEns.Accuracy;
sumT.Cohen_Kappa  = mEns.Kappa;
sumT.Kappa_CI     = sprintf("[%.3f, %.3f]", ci.Kappa_lo, ci.Kappa_hi);
sumT.Spearman_rho = mEns.Spearman;
sumT.Pearson_r    = mEns.Pearson;
sumT.ICC          = mEns.ICC;
sumT.TrainTime_min = trainTime / 60;
sumT.LR_Schedule  = string(OPT.LR_SCHEDULE);
sumT.Weighted     = OPT.USE_WEIGHTED_LOSS;

xlsxPath = fullfile(dirReports,"REPORT_resnet50_regression.xlsx");
try
    writetable(sumT, xlsxPath, "Sheet","Summary", ...
        "WriteMode","overwritesheet", "UseExcel",false);
    fprintf("  [Saved] %s\n", xlsxPath);
catch
end

fprintf("\n");
hdr("완료");
fprintf("  결과: %s\n", runDir);
fprintf("  RMSE=%.3f [%.3f,%.3f]  R²=%.3f  κ=%.3f  ρ=%.3f  Acc=%.1f%%\n", ...
    mEns.RMSE, ci.RMSE_lo, ci.RMSE_hi, mEns.R2, mEns.Kappa, mEns.Spearman, mEns.Accuracy);
fprintf("  학습 시간: %.1f분\n", trainTime/60);

try
    diary off;
catch
end
end  % function resnet50_weathering_regression


%% =====================================================================
%%  K-Fold 교차검증
%% =====================================================================
function allMetrics = runKFoldCV(T, OPT, cfg, dirModels, dirCkpt, ~, ~)
    K = OPT.K_FOLD;
    allMetrics = [];

    % 시추공 단위 K-Fold 인덱스 생성
    foldIdx = assignKFolds(T.rw_class, T.borehole, K, OPT.RNG_SEED);

    for fold = 1:K
        fprintf("\n");
        hdr(sprintf("  Fold %d / %d", fold, K));

        testMask = (foldIdx == fold);
        valMask  = (foldIdx == mod(fold, K) + 1);
        trainMask = ~testMask & ~valMask;

        Ttr_f = T(trainMask,:);
        Tva_f = T(valMask,:);
        Tte_f = T(testMask,:);

        fprintf("  Train=%d  Val=%d  Test=%d\n", height(Ttr_f), height(Tva_f), height(Tte_f));

        % 단일 모델 학습 (K-Fold에서는 앙상블 미적용, 속도 우선)
        tblTr = table(Ttr_f.ImageFile, Ttr_f.rw_value, 'VariableNames',{'imageFilename','rw'});
        tblVa = table(Tva_f.ImageFile, Tva_f.rw_value, 'VariableNames',{'imageFilename','rw'});

        dsTr = buildRegDs(tblTr, cfg, OPT.DO_AUG, true);
        dsVa = buildRegDs(tblVa, cfg, false, false);

        lgraph = buildResNet50Reg();

        % Phase 1
        lg1 = freezeBackbone(lgraph);
        opts1 = makeOpts("adam", cfg, OPT, OPT.LR_HEAD, OPT.EPOCHS_HEAD, dsVa, dirCkpt);
        [net1, ~] = trainNetwork(dsTr, lg1, opts1);

        % Phase 2
        lg2 = unfreezeAll(layerGraph(net1));
        opts2 = makeOpts("adam", cfg, OPT, OPT.LR_FINE, OPT.EPOCHS_FINE, dsVa, dirCkpt);
        [net2, ~] = trainNetwork(dsTr, lg2, opts2);

        % 예측
        dsTe = buildPredDs(Tte_f.ImageFile, cfg);
        pred = predict(net2, dsTe, "ExecutionEnvironment", cfg.execEnv, "MiniBatchSize", OPT.MB_SIZE);
        pred = double(pred(:));

        m = calcMetricsFull(Tte_f.rw_value, pred);
        fprintf("  Fold %d → RMSE=%.3f  MAE=%.3f  R²=%.3f  κ=%.3f\n", ...
            fold, m.RMSE, m.MAE, m.R2, m.Kappa);

        allMetrics = [allMetrics; m]; %#ok<AGROW>

        % Fold 모델 저장
        netFold = net2; 
        try
            save(fullfile(dirModels, sprintf("model_fold_%d.mat",fold)), ...
                "netFold","OPT","cfg", "-v7.3");
        catch
        end
    end
end

function foldIdx = assignKFolds(Y, groupId, K, seed)
    rng(seed);
    groupId = string(groupId(:));
    emp = strlength(groupId) == 0;
    if any(emp)
        groupId(emp) = "NOID_" + string(find(emp));
    end

    ug = unique(groupId, "stable");
    nG = numel(ug);
    gCls = zeros(nG, 1);
    for i = 1:nG
        gCls(i) = mode(Y(groupId == ug(i)));
    end

    % Stratified group assignment
    gFold = zeros(nG, 1);
    cats = unique(gCls);
    for c = 1:numel(cats)
        ic = find(gCls == cats(c));
        ic = ic(randperm(numel(ic)));
        for j = 1:numel(ic)
            gFold(ic(j)) = mod(j-1, K) + 1;
        end
    end

    foldIdx = zeros(numel(Y), 1);
    for i = 1:nG
        foldIdx(groupId == ug(i)) = gFold(i);
    end
end

function printKFoldSummary(allMetrics, K, dirReports)
    fields = {"RMSE","MAE","R2","Accuracy","Kappa","Spearman"};
    fprintf("\n  ┌──────────┬──────────┬──────────┐\n");
    fprintf("  │  Metric  │ Mean±Std │ 95%% CI   │\n");
    fprintf("  ├──────────┼──────────┼──────────┤\n");
    for i = 1:numel(fields)
        fn = fields{i};
        vals = [allMetrics.(fn)];
        mu = mean(vals);
        sd = std(vals);
        se = sd / sqrt(K);
        lo = mu - 1.96*se;
        hi = mu + 1.96*se;
        fprintf("  │ %-8s │ %4.3f±%4.3f │ [%4.3f,%4.3f] │\n", fn, mu, sd, lo, hi);
    end
    fprintf("  └──────────┴──────────┴──────────┘\n");

    % xlsx 저장
    try
        mT = struct2table(allMetrics);
        mT.Fold = (1:height(mT))';
        writetable(mT, fullfile(dirReports, "kfold_results.xlsx"), ...
            "Sheet","Folds", "UseExcel",false);
    catch
    end
end

function saveKFoldReport(allMetrics, OPT, ~, stamp, nTotal, dirReports)
    fields = {"RMSE","MAE","R2","Accuracy","Kappa","Spearman"};
    sumT = table();
    sumT.Timestamp = stamp;
    sumT.Model     = "ResNet-50";
    sumT.K_Fold    = OPT.K_FOLD;
    sumT.N_Total   = nTotal;
    for i = 1:numel(fields)
        fn = fields{i};
        vals = [allMetrics.(fn)];
        sumT.(fn + "_mean") = mean(vals);
        sumT.(fn + "_std")  = std(vals);
    end
    try
        writetable(sumT, fullfile(dirReports,"REPORT_kfold.xlsx"), ...
            "Sheet","Summary", "WriteMode","overwritesheet", "UseExcel",false);
    catch
    end
end


%% =====================================================================
%%  Ablation Study
%% =====================================================================
function runAblation(Ttr, Tva, Tte, OPT, cfg, dirPlots, dirReports, dirCkpt)
% 증강 / 앙상블 / 가중손실 기여도 분석

    conditions = {
        "Full Model",          true,  OPT.N_ENSEMBLE, true;
        "No Augmentation",     false, OPT.N_ENSEMBLE, true;
        "Single Model (N=1)",  true,  1,              true;
        "No Weighting",        true,  OPT.N_ENSEMBLE, false;
    };

    nCond = size(conditions, 1);
    ablResults = [];

    for a = 1:nCond
        condName = conditions{a,1};
        doAug    = conditions{a,2};
        nEns     = conditions{a,3};
        doWeight = conditions{a,4};

        fprintf("\n  ── Ablation: %s ──\n", condName);

        cfgA = cfg;
        if ~doWeight
            cfgA.classWeights = ones(1,5);
        end

        preds = zeros(height(Tte), nEns);
        for m = 1:nEns
            rng(OPT.RNG_SEED + m);
            nTr = height(Ttr);
            bootIdx = randi(nTr, nTr, 1);
            Ttr_b = Ttr(bootIdx,:);

            tblTr = table(Ttr_b.ImageFile, Ttr_b.rw_value, 'VariableNames',{'imageFilename','rw'});
            tblVa = table(Tva.ImageFile, Tva.rw_value, 'VariableNames',{'imageFilename','rw'});

            dsTr = buildRegDs(tblTr, cfgA, doAug, true);
            dsVa = buildRegDs(tblVa, cfgA, false, false);

            lgraph = buildResNet50Reg();

            lg1 = freezeBackbone(lgraph);
            opts1 = makeOpts("adam", cfgA, OPT, OPT.LR_HEAD, OPT.EPOCHS_HEAD, dsVa, dirCkpt);
            [net1, ~] = trainNetwork(dsTr, lg1, opts1);

            lg2 = unfreezeAll(layerGraph(net1));
            opts2 = makeOpts("adam", cfgA, OPT, OPT.LR_FINE, OPT.EPOCHS_FINE, dsVa, dirCkpt);
            [net2, ~] = trainNetwork(dsTr, lg2, opts2);

            dsTe = buildPredDs(Tte.ImageFile, cfgA);
            pred = predict(net2, dsTe, "ExecutionEnvironment", cfgA.execEnv, "MiniBatchSize", OPT.MB_SIZE);
            preds(:,m) = double(pred(:));
        end

        ensPred = mean(preds, 2);
        m = calcMetricsFull(Tte.rw_value, ensPred);
        m.Condition = string(condName);
        ablResults = [ablResults; m]; %#ok<AGROW>

        fprintf("    RMSE=%.3f  MAE=%.3f  R²=%.3f  κ=%.3f\n", m.RMSE, m.MAE, m.R2, m.Kappa);
    end

    % Ablation 결과 테이블 & 그래프
    plotAblation(ablResults, dirPlots);
    try
        writetable(struct2table(ablResults), fullfile(dirReports,"ablation_results.xlsx"), ...
            "Sheet","Ablation", "UseExcel",false);
    catch
    end
end


%% =====================================================================
%%  데이터스토어 (학습용 / 예측용)
%% =====================================================================
function ds = buildRegDs(tbl, cfg, doAug, isTrain)
    imds = imageDatastore(cellstr(tbl.imageFilename), ...
        "ReadFcn", @(f) readImg(f, cfg, isTrain));
    rds = arrayDatastore(tbl.rw, "IterationDimension", 1);
    cds = combine(imds, rds);
    if doAug
        ds = transform(cds, @(data) augmentPair(data, cfg));
    else
        ds = transform(cds, @(data) resizePair(data, cfg));
    end
end

function ds = buildPredDs(imageFiles, cfg)
    imds = imageDatastore(cellstr(string(imageFiles)), ...
        "ReadFcn", @(f) readImg(f, cfg, false));
    ds = transform(imds, @(img) resizeOnly(img, cfg));
end

function out = resizeOnly(img, cfg)
    img = imresize(img, cfg.imageSize(1:2));
    img = min(max(img,0),1);
    out = {img};
end

function out = augmentPair(data, cfg)
    sz  = cfg.imageSize(1:2);
    img = data{1};
    rw  = data{2};

    img = imresize(img, sz);

    % 수평 반전
    if rand > 0.5
        img = fliplr(img);
    end

    % 회전
    ang = cfg.augRot(1) + rand*(cfg.augRot(2)-cfg.augRot(1));
    img = imrotate(img, ang, 'bilinear', 'crop');

    % 스케일
    sc = cfg.augScale(1) + rand*(cfg.augScale(2)-cfg.augScale(1));
    if abs(sc-1) > 0.01
        tmp = imresize(img, sc);
        if sc > 1
            cx = floor((size(tmp,2)-sz(2))/2)+1;
            cy = floor((size(tmp,1)-sz(1))/2)+1;
            img = tmp(cy:cy+sz(1)-1, cx:cx+sz(2)-1, :);
        else
            canvas = zeros([sz, size(img,3)], 'like', img);
            oy = floor((sz(1)-size(tmp,1))/2)+1;
            ox = floor((sz(2)-size(tmp,2))/2)+1;
            canvas(oy:oy+size(tmp,1)-1, ox:ox+size(tmp,2)-1, :) = tmp;
            img = canvas;
        end
    end

    if cfg.doColorJitter
        img = colorJitter(img, cfg);
    end

    img = min(max(img,0),1);
    out = {img, rw};
end

function out = resizePair(data, cfg)
    img = data{1};
    rw  = data{2};
    img = imresize(img, cfg.imageSize(1:2));
    img = min(max(img,0),1);
    out = {img, rw};
end

function I = readImg(filename, cfg, isTrain) %#ok<INUSD>
    h = cfg.imageSize(1); w = cfg.imageSize(2);
    try
        I = imread(filename);
    catch
        I = zeros(h, w, 3, "uint8");
    end
    if ismatrix(I)
        I = repmat(I,1,1,3);
    elseif ndims(I)==3 && size(I,3)==1
        I = repmat(I,1,1,3);
    elseif ndims(I)==3 && size(I,3)>3
        I = I(:,:,1:3);
    end
    I = im2single(I);
end

function I = colorJitter(I, cfg)
    if cfg.cjBright > 0
        I = I + (rand*2-1)*cfg.cjBright;
    end
    if cfg.cjContr > 0
        c = 1 + (rand*2-1)*cfg.cjContr;
        m = mean(I,[1 2],'omitnan');
        I = (I-m)*c + m;
    end
    if cfg.cjGamma > 0
        g = 1 + (rand*2-1)*cfg.cjGamma;
        I = max(I,0).^g;
    end
    if cfg.cjHue>0 || cfg.cjSat>0
        hsv = rgb2hsv(I);
        if cfg.cjHue > 0
            hsv(:,:,1) = hsv(:,:,1) + (rand*2-1)*cfg.cjHue;
            hsv(:,:,1) = hsv(:,:,1) - floor(hsv(:,:,1));
        end
        if cfg.cjSat > 0
            hsv(:,:,2) = hsv(:,:,2) * (1+(rand*2-1)*cfg.cjSat);
        end
        I = hsv2rgb(hsv);
    end
    I = min(max(I,0),1);
end


%% =====================================================================
%%  모델 구축
%% =====================================================================
function lgraph = buildResNet50Reg()
    net0   = resnet50();
    lgraph = layerGraph(net0);

    fcName = findLayerByClass(lgraph, 'FullyConnectedLayer', "fc1000");
    smName = findLayerByClass(lgraph, 'SoftmaxLayer', "fc1000_softmax");
    clName = findLayerByClass(lgraph, 'ClassificationLayer', "");
    if strlength(clName) == 0
        clName = findLayerByClass(lgraph, 'ClassificationOutputLayer', "");
    end

    lgraph = replaceLayer(lgraph, char(fcName), ...
        fullyConnectedLayer(256, "Name","fc_reg1", ...
        "WeightLearnRateFactor",10, "BiasLearnRateFactor",10));

    if strlength(smName) > 0
        lgraph = replaceLayer(lgraph, char(smName), reluLayer("Name","relu_reg1"));
    end

    lgraph = replaceLayer(lgraph, char(clName), regressionLayer("Name","reg_output"));

    lgraph = addLayers(lgraph, [
        dropoutLayer(0.3, "Name","drop_reg")
        fullyConnectedLayer(1, "Name","fc_reg2", ...
            "WeightLearnRateFactor",10, "BiasLearnRateFactor",10)
    ]);

    lgraph = disconnectLayers(lgraph, "relu_reg1", "reg_output");
    lgraph = connectLayers(lgraph, "relu_reg1", "drop_reg");
    lgraph = connectLayers(lgraph, "fc_reg2",   "reg_output");
end

function name = findLayerByClass(lgraph, shortName, defaultName)
    names = string({lgraph.Layers.Name});
    if strlength(defaultName)>0 && any(names==defaultName)
        name = defaultName;
        return;
    end
    idx = find(arrayfun(@(L) contains(class(L), shortName), lgraph.Layers), 1, 'last');
    if ~isempty(idx)
        name = string(lgraph.Layers(idx).Name);
    else
        name = "";
    end
end


%% =====================================================================
%%  백본 동결 / 해제
%% =====================================================================
function lgraph = freezeBackbone(lgraph)
    headNames = ["fc_reg1","relu_reg1","drop_reg","fc_reg2","reg_output"];
    layers = lgraph.Layers;
    names  = string({layers.Name});
    for i = 1:numel(layers)
        if any(names(i) == headNames)
            layers(i) = setLR(layers(i), 10);
        else
            layers(i) = setLR(layers(i), 0);
        end
    end
    lgraph = rebuildGraph(layers, lgraph.Connections);
end

function lgraph = unfreezeAll(lgraph)
    headNames = ["fc_reg1","relu_reg1","drop_reg","fc_reg2","reg_output"];
    layers = lgraph.Layers;
    names  = string({layers.Name});
    for i = 1:numel(layers)
        if any(names(i) == headNames)
            layers(i) = setLR(layers(i), 10);
        else
            layers(i) = setLR(layers(i), 1);
        end
    end
    lgraph = rebuildGraph(layers, lgraph.Connections);
end

function L = setLR(L, v)
    for pp = ["WeightLearnRateFactor","BiasLearnRateFactor",...
              "ScaleLearnRateFactor","OffsetLearnRateFactor"]
        if isprop(L, pp)
            try
                L.(pp) = v;
            catch
            end
        end
    end
end

function lg = rebuildGraph(layers, conns)
    lg = layerGraph();
    for i = 1:numel(layers)
        lg = addLayers(lg, layers(i));
    end
    for i = 1:height(conns)
        src = conns.Source(i);
        dst = conns.Destination(i);
        if iscell(src), src = src{1}; end
        if iscell(dst), dst = dst{1}; end
        lg = connectLayers(lg, string(src), string(dst));
    end
end


%% =====================================================================
%%  학습 옵션  (Cosine Annealing / Piecewise)
%% =====================================================================
function opts = makeOpts(solver, cfg, OPT, lr, epochs, dsVa, ckptDir)
    plotsOpt = "none";
    if usejava('desktop')
        plotsOpt = "training-progress";
    end

    baseArgs = {solver, ...
        "ExecutionEnvironment", char(cfg.execEnv), ...
        "InitialLearnRate",    lr, ...
        "MaxEpochs",           epochs, ...
        "MiniBatchSize",       OPT.MB_SIZE, ...
        "Shuffle",             "every-epoch", ...
        "Verbose",             true, ...
        "VerboseFrequency",    50, ...
        "Plots",               plotsOpt, ...
        "L2Regularization",    1e-4, ...
        "GradientThreshold",   5};

    % LR 스케줄
    if lower(string(OPT.LR_SCHEDULE)) == "cosine"
        % Cosine Annealing: MATLAB에서는 piecewise로 근사
        % (매 에폭마다 cos 감쇠 → 잘게 나눈 step)
        baseArgs = [baseArgs, {...
            "LearnRateSchedule", "piecewise", ...
            "LearnRateDropPeriod", 1, ...
            "LearnRateDropFactor", 0.95}];  % ~cos 근사
    else
        baseArgs = [baseArgs, {...
            "LearnRateSchedule", "piecewise", ...
            "LearnRateDropPeriod", OPT.LR_DROP_PERIOD, ...
            "LearnRateDropFactor", OPT.LR_DROP_FACTOR}];
    end

    try
        opts = trainingOptions(baseArgs{:}, ...
            "ValidationData",      dsVa, ...
            "ValidationFrequency", 200, ...
            "ValidationPatience",  OPT.PATIENCE, ...
            "OutputNetwork",       "best-validation-loss", ...
            "CheckpointPath",      char(ckptDir));
    catch
        opts = trainingOptions(baseArgs{:}, ...
            "ValidationData",      dsVa, ...
            "ValidationFrequency", 200, ...
            "CheckpointPath",      char(ckptDir));
    end
end


%% =====================================================================
%%  rw 라벨 처리
%% =====================================================================
function val = rwToVal(rwStr)
    val = NaN;
    rwStr = strtrim(upper(string(rwStr)));
    if strlength(rwStr) == 0, return; end
    rwStr = replace(replace(rwStr," ",""), "WD","D");

    map1 = struct('D1',1,'D2',2,'D3',3,'D4',4,'D5',5);
    k1 = char(rwStr);
    if isfield(map1, k1), val = map1.(k1); return; end

    nums = regexp(char(rwStr), '\d+', 'match');
    if numel(nums) >= 2
        a = str2double(nums{1}); b = str2double(nums{2});
        if isfinite(a) && isfinite(b) && a>=1 && b<=5
            val = (a+b)/2;
            return;
        end
    elseif isscalar(nums)
        v = str2double(nums{1});
        if isfinite(v) && v>=1 && v<=5, val = v; end
    end
end

function rwStr = extractRw(s)
    rwStr = "";
    if ~isstruct(s) || isempty(fieldnames(s)), return; end
    cands = ["rw","rw_grade","weathering","weathering_grade","wd_grade","WD","wd"];
    for i = 1:numel(cands)
        k = cands(i);
        if isfield(s, k)
            v = s.(k);
            if ischar(v)||isstring(v), rwStr = strtrim(upper(string(v))); end
            if isnumeric(v) && isfinite(v), rwStr = string(v); end
            if strlength(rwStr) > 0, return; end
        end
    end
    fn = fieldnames(s);
    for j = 1:numel(fn)
        v = s.(fn{j});
        if isstruct(v)
            for i = 1:numel(cands)
                if isfield(v, cands(i))
                    vv = v.(cands(i));
                    if ischar(vv)||isstring(vv), rwStr = strtrim(upper(string(vv))); end
                    if isnumeric(vv)&&isfinite(vv), rwStr = string(vv); end
                    if strlength(rwStr) > 0, return; end
                end
            end
        end
    end
end

function rwStr = parseRwFromFilename(bn)
    rwStr = "";
    tok = regexp(char(string(bn)), '[-_](D|W)[-_]?(\d+)', 'tokens','once');
    if ~isempty(tok)
        rwStr = upper(string(tok{1}) + string(tok{2}));
        if startsWith(rwStr,"W"), rwStr = "D" + extractAfter(rwStr,1); end
    end
end

function bhId = extractBH(filename)
    parts = split(string(filename), "-");
    if numel(parts) >= 2
        bhId = parts(1) + "-" + parts(2);
    else
        bhId = string(filename);
    end
end


%% =====================================================================
%%  데이터 분할
%% =====================================================================
function [idxTr,idxVa,idxTe] = splitByGroup(Y, gid, trR, vaR, teR, seed)
    rng(seed);
    gid = string(gid(:));
    emp = strlength(gid) == 0;
    if any(emp), gid(emp) = "NOID_" + string(find(emp)); end

    ug = unique(gid, "stable");
    gCls = zeros(numel(ug), 1);
    for i = 1:numel(ug)
        gCls(i) = mode(Y(gid == ug(i)));
    end
    [gi1,gi2,gi3] = splitStrat(gCls, trR, vaR, teR, seed);
    idxTr = find(ismember(gid, ug(gi1)));
    idxVa = find(ismember(gid, ug(gi2)));
    idxTe = find(ismember(gid, ug(gi3)));
end

function [i1,i2,i3] = splitStrat(Y, trR, vaR, teR, seed) %#ok<INUSD>
    rng(seed);
    Y = double(Y(:));
    cats = unique(Y);
    m1 = false(numel(Y),1);
    m2 = false(numel(Y),1);
    m3 = false(numel(Y),1);

    for c = 1:numel(cats)
        ic = find(Y == cats(c));
        n  = numel(ic);
        if n < 5
            m1(ic) = true;
            continue;
        end
        ic  = ic(randperm(n));
        nTr = max(1, floor(n * trR));
        nVa = max(1, floor(n * vaR));
        if (n - nTr - nVa) < 1
            nVa = max(1, n - nTr - 1);
        end
        m1(ic(1:nTr))          = true;
        m2(ic(nTr+1:nTr+nVa)) = true;
        m3(ic(nTr+nVa+1:end)) = true;
    end

    i1 = find(m1);
    i2 = find(m2);
    i3 = find(m3);
end


%% =====================================================================
%%  통계 지표 (Nature급)
%% =====================================================================
function m = calcMetricsFull(trueV, predV)
    trueV = double(trueV(:));
    predV = double(predV(:));
    err   = predV - trueV;
    n     = numel(trueV);

    m.RMSE = sqrt(mean(err.^2));
    m.MAE  = mean(abs(err));

    SSres = sum(err.^2);
    SStot = sum((trueV - mean(trueV)).^2);
    m.R2  = max(0, 1 - SSres / max(SStot, eps));

    % 반올림 분류
    tC = max(1, min(5, round(trueV)));
    pC = max(1, min(5, round(predV)));
    m.Accuracy = sum(tC == pC) / n * 100;

    % Cohen's weighted Kappa (linear weights)
    m.Kappa = computeWeightedKappa(tC, pC, 5);

    % Spearman ρ
    m.Spearman = corr(trueV, predV, 'Type','Spearman');

    % Pearson r
    m.Pearson = corr(trueV, predV, 'Type','Pearson');

    % ICC(2,1) — Intraclass Correlation Coefficient
    m.ICC = computeICC(trueV, predV);

    m.MeanErr = mean(err);
    m.StdErr  = std(err);
end

function kappa = computeWeightedKappa(obs, pred, nClass)
% Linear weighted Cohen's Kappa for ordinal classification
    cm = zeros(nClass, nClass);
    for i = 1:numel(obs)
        cm(obs(i), pred(i)) = cm(obs(i), pred(i)) + 1;
    end
    cm = cm / sum(cm(:));

    % weight matrix (linear)
    W = zeros(nClass);
    for i = 1:nClass
        for j = 1:nClass
            W(i,j) = abs(i-j) / (nClass-1);
        end
    end

    rSum = sum(cm, 2);
    cSum = sum(cm, 1);
    E    = rSum * cSum;

    po = 1 - sum(sum(W .* cm));
    pe = 1 - sum(sum(W .* E));
    kappa = (po - pe) / max(1 - pe, eps);
end

function icc = computeICC(x, y)
% ICC(2,1) — two-way random, single measures, absolute agreement
    n = numel(x);
    data = [x(:), y(:)];
    k = 2;

    grandMean = mean(data(:));
    SSr = k * sum((mean(data,2) - grandMean).^2);
    SSc = n * sum((mean(data,1) - grandMean).^2);
    SSt = sum((data(:) - grandMean).^2);
    SSe = SSt - SSr - SSc;

    MSr = SSr / (n-1);
    MSe = SSe / ((n-1)*(k-1));
    MSc = SSc / (k-1);

    icc = (MSr - MSe) / (MSr + (k-1)*MSe + k*(MSc-MSe)/n);
    icc = max(0, min(1, icc));
end

function ci = bootstrapCI(trueV, predV, nBoot)
% 95% Bootstrap percentile confidence intervals
    rng(0);
    n = numel(trueV);
    bRMSE = zeros(nBoot,1);
    bMAE  = zeros(nBoot,1);
    bR2   = zeros(nBoot,1);
    bKappa = zeros(nBoot,1);

    for b = 1:nBoot
        idx = randi(n, n, 1);
        m = calcMetricsFull(trueV(idx), predV(idx));
        bRMSE(b)  = m.RMSE;
        bMAE(b)   = m.MAE;
        bR2(b)    = m.R2;
        bKappa(b) = m.Kappa;
    end

    ci.RMSE_lo  = prctile(bRMSE, 2.5);   ci.RMSE_hi  = prctile(bRMSE, 97.5);
    ci.MAE_lo   = prctile(bMAE,  2.5);    ci.MAE_hi   = prctile(bMAE,  97.5);
    ci.R2_lo    = prctile(bR2,   2.5);    ci.R2_hi    = prctile(bR2,   97.5);
    ci.Kappa_lo = prctile(bKappa,2.5);    ci.Kappa_hi = prctile(bKappa,97.5);
end


%% =====================================================================
%%  시각화 함수 (Nature급)
%% =====================================================================
function plotScatter(trueV, predV, predStd, met, saveDir, nEns)
    fig = figure("Visible","off","Position",[100 100 600 550]);

    errorbar(trueV, predV, predStd, 'o', ...
        'MarkerSize',3, 'Color',[0.3 0.5 0.8], ...
        'MarkerFaceColor',[0.3 0.5 0.8], 'LineWidth',0.3, ...
        'CapSize',0, 'MarkerEdgeColor','none');
    hold on;
    plot([0.5 5.5],[0.5 5.5],'r--','LineWidth',1.5);

    for v = [1.5 2.5 3.5 4.5]
        xline(v,':','Color',[.7 .7 .7]);
        yline(v,':','Color',[.7 .7 .7]);
    end
    for c = 1:5
        text(c, 0.7, sprintf('D%d',c), 'HorizontalAlignment','center', ...
            'FontSize',9, 'Color',[.4 .4 .4]);
    end

    xlabel('Actual Weathering Grade','FontSize',12);
    ylabel('Predicted Weathering Grade','FontSize',12);
    title(sprintf('ResNet-50 Ensemble (N=%d)\nRMSE=%.3f  R^2=%.3f  \\kappa=%.3f  \\rho_s=%.3f', ...
        nEns, met.RMSE, met.R2, met.Kappa, met.Spearman), 'FontSize',11);
    xlim([0.5 5.5]); ylim([0.5 5.5]); axis equal; grid on;
    legend('Prediction ± \sigma','1:1 Line','Location','northwest');

    exportgraphics(fig, fullfile(saveDir,"scatter_ensemble.png"), "Resolution",300);
    try
        savefig(fig, fullfile(saveDir,"scatter_ensemble.fig"));
    catch
    end
    close(fig);
end

function plotCM(trueV, predV, met, saveDir)
    tC = categorical(max(1,min(5,round(trueV))));
    pC = categorical(max(1,min(5,round(predV))));

    fig = figure("Visible","off","Position",[100 100 500 450]);
    cm = confusionchart(tC, pC, ...
        "ColumnSummary","column-normalized", "RowSummary","row-normalized");
    cm.Title = sprintf('Confusion Matrix (Acc=%.1f%%, \\kappa=%.3f)', met.Accuracy, met.Kappa);
    cm.XLabel = 'Predicted Class';
    cm.YLabel = 'Actual Class';

    exportgraphics(fig, fullfile(saveDir,"confusion_matrix.png"), "Resolution",300);
    try
        savefig(fig, fullfile(saveDir,"confusion_matrix.fig"));
    catch
    end
    close(fig);
end

function plotErr(trueV, predV, saveDir)
    err = predV - trueV;
    fig = figure("Visible","off","Position",[100 100 550 400]);

    histogram(err, 60, 'FaceColor',[0.3 0.5 0.8], 'EdgeColor','w', 'Normalization','pdf');
    hold on;
    xline(0,'r--','LineWidth',1.5);
    xline(mean(err),'b-','LineWidth',1);

    % 정규분포 피팅 오버레이
    xr = linspace(min(err)-0.5, max(err)+0.5, 200);
    plot(xr, normpdf(xr, mean(err), std(err)), 'k-', 'LineWidth',1.5);

    xlabel('Prediction Error','FontSize',12);
    ylabel('Probability Density','FontSize',12);
    title(sprintf('Error Distribution\n\\mu=%.3f  \\sigma=%.3f  |max|=%.2f', ...
        mean(err), std(err), max(abs(err))), 'FontSize',12);
    legend('Histogram','Zero','Mean','Normal fit','Location','northeast');
    grid on;

    exportgraphics(fig, fullfile(saveDir,"error_distribution.png"), "Resolution",300);
    close(fig);
end

function plotUncertainty(trueV, predV, predStd, saveDir)
    fig = figure("Visible","off","Position",[100 100 550 400]);
    absErr = abs(predV - trueV);
    scatter(predStd, absErr, 8, trueV, 'filled', 'MarkerFaceAlpha',0.5);
    hold on;

    % 선형 회귀선
    p = polyfit(predStd, absErr, 1);
    xfit = linspace(min(predStd), max(predStd), 100);
    plot(xfit, polyval(p, xfit), 'r-', 'LineWidth',1.5);

    r = corr(predStd, absErr);
    colorbar; colormap(turbo);
    xlabel('Ensemble \sigma (Uncertainty)','FontSize',12);
    ylabel('|Prediction Error|','FontSize',12);
    title(sprintf('Uncertainty Calibration (r=%.3f)', r), 'FontSize',12);
    legend('Data', sprintf('Linear fit (slope=%.2f)', p(1)), 'Location','northwest');
    grid on;

    exportgraphics(fig, fullfile(saveDir,"uncertainty_calibration.png"), "Resolution",300);
    close(fig);
end

function plotBlandAltman(trueV, predV, saveDir)
% Bland-Altman plot (Method agreement)
    fig = figure("Visible","off","Position",[100 100 600 450]);

    meanV = (trueV + predV) / 2;
    diffV = predV - trueV;
    mu    = mean(diffV);
    sd    = std(diffV);
    LOA_hi = mu + 1.96*sd;
    LOA_lo = mu - 1.96*sd;

    scatter(meanV, diffV, 10, [0.3 0.5 0.8], 'filled', 'MarkerFaceAlpha',0.4);
    hold on;
    yline(mu, 'b-', 'LineWidth',1.5);
    yline(LOA_hi, 'r--', 'LineWidth',1.2);
    yline(LOA_lo, 'r--', 'LineWidth',1.2);

    text(max(meanV)*0.95, mu+0.15, sprintf('Mean=%.3f',mu), ...
        'HorizontalAlignment','right', 'Color','b', 'FontSize',9);
    text(max(meanV)*0.95, LOA_hi+0.15, sprintf('+1.96SD=%.3f',LOA_hi), ...
        'HorizontalAlignment','right', 'Color','r', 'FontSize',9);
    text(max(meanV)*0.95, LOA_lo-0.15, sprintf('-1.96SD=%.3f',LOA_lo), ...
        'HorizontalAlignment','right', 'Color','r', 'FontSize',9);

    xlabel('Mean of Actual and Predicted','FontSize',12);
    ylabel('Predicted − Actual','FontSize',12);
    title('Bland-Altman Plot','FontSize',13);
    grid on;

    exportgraphics(fig, fullfile(saveDir,"bland_altman.png"), "Resolution",300);
    close(fig);
end

function plotCalibration(trueV, predV, saveDir)
% Calibration plot: 예측값 구간별 실제 평균
    fig = figure("Visible","off","Position",[100 100 500 450]);

    edges = 0.5:0.5:5.5;
    nBins = numel(edges)-1;
    binMeanPred = zeros(nBins,1);
    binMeanTrue = zeros(nBins,1);
    binCount    = zeros(nBins,1);

    for b = 1:nBins
        mask = predV >= edges(b) & predV < edges(b+1);
        binCount(b) = sum(mask);
        if binCount(b) > 0
            binMeanPred(b) = mean(predV(mask));
            binMeanTrue(b) = mean(trueV(mask));
        end
    end

    valid = binCount > 5;
    plot([0.5 5.5],[0.5 5.5],'k--','LineWidth',1); hold on;
    scatter(binMeanPred(valid), binMeanTrue(valid), binCount(valid)*2+20, ...
        'filled', 'MarkerFaceColor',[0.3 0.5 0.8], 'MarkerFaceAlpha',0.7);

    xlabel('Mean Predicted Value (binned)','FontSize',12);
    ylabel('Mean Actual Value','FontSize',12);
    title('Calibration Plot (bubble size ∝ N)','FontSize',12);
    xlim([0.5 5.5]); ylim([0.5 5.5]); grid on;

    exportgraphics(fig, fullfile(saveDir,"calibration_plot.png"), "Resolution",300);
    close(fig);
end

function plotLearningCurve(info, saveDir)
    fig = figure("Visible","off","Position",[100 100 650 400]);
    trLoss = info.TrainingLoss;
    x = 1:numel(trLoss);
    plot(x, trLoss, 'b-', 'LineWidth',0.5);
    hold on;
    if isfield(info,'ValidationLoss')
        vLoss = info.ValidationLoss;
        vIdx  = find(~isnan(vLoss));
        plot(vIdx, vLoss(vIdx), 'ro-', 'LineWidth',1.2, 'MarkerSize',4);
        legend('Training Loss','Validation Loss','Location','northeast');
    end
    xlabel('Iteration','FontSize',12);
    ylabel('Loss (MSE)','FontSize',12);
    title('Learning Curve','FontSize',12);
    grid on;
    exportgraphics(fig, fullfile(saveDir,"learning_curve.png"), "Resolution",300);
    close(fig);
end

function plotGradCAM(net, Tte, cfg, saveDir)
    nSamp = min(8, height(Tte));
    idx = randperm(height(Tte), nSamp);
    layerNames = string({net.Layers.Name});
    convIdx = find(arrayfun(@(L) contains(class(L),'Convolution2D'), net.Layers), 1, 'last');
    gradLayer = layerNames(convIdx);

    fig = figure("Visible","off","Position",[100 100 nSamp*180 380]);
    t = tiledlayout(2, nSamp, "TileSpacing","compact","Padding","compact");

    for i = 1:nSamp
        row = Tte(idx(i),:);
        try
            img = imread(char(row.ImageFile));
            if ismatrix(img), img = repmat(img,1,1,3); end
            imgR = imresize(img, cfg.imageSize(1:2));

            nexttile(i);
            imshow(imgR);
            title(sprintf('%s (%.1f)', char(row.rw_raw), row.rw_value), 'FontSize',7);

            nexttile(nSamp+i);
            imgIn = imresize(im2single(img), cfg.imageSize(1:2));
            if size(imgIn,3)==1, imgIn = repmat(imgIn,1,1,3); end
            try
                scoreMap = gradCAM(net, imgIn, 1, "FeatureLayer", gradLayer);
                imshow(imgR); hold on;
                imagesc(scoreMap, "AlphaData", 0.5);
                colormap jet; hold off;
            catch
                imshow(imgR); title('CAM err','FontSize',7);
            end
        catch
            nexttile(i); title('err');
            nexttile(nSamp+i);
        end
    end
    title(t, 'Grad-CAM Visualization', 'FontSize',11);
    exportgraphics(fig, fullfile(saveDir,"gradcam_samples.png"), "Resolution",300);
    close(fig);
end

function plotTSNE(net, Tte, cfg, OPT, saveDir)
% t-SNE: 마지막 GAP 레이어 특징 공간 시각화
    fprintf("  t-SNE 특징 추출 중...\n");

    layerNames = string({net.Layers.Name});
    gapIdx = find(contains(lower(layerNames),"avg") | contains(lower(layerNames),"pool"), 1, 'last');
    featLayer = layerNames(gapIdx);

    dsPred = buildPredDs(Tte.ImageFile, cfg);
    feat = activations(net, dsPred, featLayer, ...
        "ExecutionEnvironment", cfg.execEnv, ...
        "OutputAs","rows", ...
        "MiniBatchSize", OPT.MB_SIZE);

    fprintf("  t-SNE 계산 중 (feature dim=%d)...\n", size(feat,2));
    Y2d = tsne(double(feat), 'Perplexity', min(30, floor(size(feat,1)/4)));

    fig = figure("Visible","off","Position",[100 100 600 500]);
    scatter(Y2d(:,1), Y2d(:,2), 12, Tte.rw_value, 'filled', 'MarkerFaceAlpha',0.6);
    colorbar;
    colormap(turbo);
    clim([1 5]);
    xlabel('t-SNE 1','FontSize',12);
    ylabel('t-SNE 2','FontSize',12);
    title('t-SNE Feature Space (colored by weathering grade)','FontSize',12);
    grid on;

    exportgraphics(fig, fullfile(saveDir,"tsne_features.png"), "Resolution",300);
    close(fig);
end

function plotAblation(results, saveDir)
    fig = figure("Visible","off","Position",[100 100 700 400]);

    names = [results.Condition];
    rmse  = [results.RMSE];
    r2    = [results.R2];
    kappa = [results.Kappa];

    x = 1:numel(names);
    b = bar(x, [rmse; r2; kappa]', 'grouped');
    b(1).FaceColor = [0.85 0.33 0.10];
    b(2).FaceColor = [0.00 0.45 0.74];
    b(3).FaceColor = [0.47 0.67 0.19];

    set(gca, 'XTick', x, 'XTickLabel', names, 'XTickLabelRotation', 15);
    ylabel('Score','FontSize',12);
    title('Ablation Study','FontSize',13);
    legend('RMSE','R^2','Cohen \kappa','Location','best');
    grid on;

    exportgraphics(fig, fullfile(saveDir,"ablation_study.png"), "Resolution",300);
    close(fig);
end


%% =====================================================================
%%  유틸리티
%% =====================================================================
function printDistribution(T)
    fprintf("\n  [풍화도 분포]\n");
    uraw = unique(T.rw_raw);
    for i = 1:numel(uraw)
        fprintf("    %-6s → %.1f : %d장\n", uraw(i), rwToVal(uraw(i)), sum(T.rw_raw==uraw(i)));
    end
    fprintf("\n  [반올림 등급]\n");
    for c = 1:5
        fprintf("    D%d : %d장\n", c, sum(T.rw_class==c));
    end
end

function roots = normRoots(x)
    if isempty(x), roots = strings(0,1); return; end
    if iscell(x), x = string(x); end
    x = string(x); x = x(:); x = x(strlength(x)>0);
    roots = x;
end

function mkd(d)
    d = char(string(d));
    if ~exist(d,'dir')
        mkdir(d);
    end
end

function env = resolveEnv(req)
    req = lower(string(req));
    if req == "cpu", env = "cpu"; return; end
    if req == "gpu", env = "gpu"; return; end
    try
        if gpuDeviceCount("available") > 0
            env = "gpu";
        else
            env = "cpu";
        end
    catch
        env = "cpu";
    end
end

function hdr(msg)
    fprintf("──────────────────────────────────────────\n");
    fprintf("  %s\n", msg);
    fprintf("──────────────────────────────────────────\n");
end

function files = collectFiles(roots, exts, recursive)
    roots = string(roots);
    exts  = string(exts);
    allPaths = {};
    for r = 1:numel(roots)
        root = roots(r);
        if ~isfolder(root)
            continue;
        end
        for e = 1:numel(exts)
            ext = exts(e);
            if ~startsWith(ext, ".")
                ext = "." + ext;
            end
            if recursive
                D = dir(fullfile(char(root), "**", "*" + char(ext)));
            else
                D = dir(fullfile(char(root), "*" + char(ext)));
            end
            for i = 1:numel(D)
                if ~D(i).isdir
                    allPaths{end+1,1} = fullfile(D(i).folder, D(i).name); %#ok<AGROW>
                end
            end
        end
    end
    if isempty(allPaths)
        files = strings(0,1);
    else
        files = unique(string(allPaths), "stable");
    end
end
