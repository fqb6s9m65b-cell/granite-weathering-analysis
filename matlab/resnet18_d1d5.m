function [netTrained, trainLog, runInfo] = resnet18_d1d5(varargin)
% resnet18_d1d5_v6  (ONE-FILE, MATLAB R2025a)
% =========================================================================
% ResNet-18 + Aux Color(11) -> Weathering Grade D1~D5
%
% [v6 주요 수정 사항] — 정확도 개선 목표
%
%   FIX-1 : 학습 중 Online Aux 재추출
%           → v5에서 이미지는 ColorJitter 적용되는데 aux는 원본에서 미리 추출한 값 사용
%           → 이미지-보조파라미터 불일치(mismatch) 해소
%           → 학습 시 augmented 이미지에서 실시간 aux 재추출
%
%   FIX-2 : Gated Fusion (게이트 기반 융합)
%           → 단순 concatenation 대신 학습 가능한 게이트로 image/color 가중
%           → 모달리티 간 상호작용 학습 가능
%
%   FIX-3 : Data-Driven Rule Centers
%           → Yusoff(2022) 50:50 블렌딩 제거
%           → 학습 데이터 중앙값에서 직접 rule boundary 도출
%           → 실제 데이터 분포에 맞는 앙상블
%
%   FIX-4 : Per-Feature Learnable Scaling (단일 auxScale → 11차원 벡터)
%           → 특성별 독립적 스케일링으로 중요 특성 자동 강조
%
%   FIX-5 : Cosine Annealing with Warm Restarts
%           → 단순 cosine 대신 주기적 재시작으로 local minima 탈출
%
%   FIX-6 : Test-Time Augmentation (TTA)
%           → 평가 시 5-crop + flip으로 예측 안정화
%
%   TUNE  : DropoutRate 0.4→0.3, LabelSmoothing 0.05→0.10,
%           ColorJitter 0.10→0.07, AuxNoise 0.03→0.02,
%           FreezeEpochs 8→10, Epochs 50→60
% =========================================================================
t0     = datetime('now','TimeZone','Asia/Seoul');

% GPU Forward Compatibility (RTX 50xx Blackwell CC12.0 지원)
try
    parallel.gpu.enableCUDAForwardCompatibility(true);
    fprintf('GPU Forward Compatibility 활성화 완료\n');
catch
end

runTag = "RUN_" + string(datetime('now','TimeZone','Asia/Seoul','Format','yyyyMMdd_HHmmss'));
PARAM_NAMES = {'R','G','B','H','S','V','L*','a*','b*','C*','is_wet'};

% v6 FIX-3: Yusoff 블렌딩 제거, 데이터 기반 센터는 학습 후 동적 계산
% (초기값은 본 연구 값만 사용, 학습 데이터 통계로 나중에 교체)
MY.astar = [-2.2, -0.50];
MY.bstar = [-2.2,  1.60];
MANUAL_CENTERS_INIT.bstar = linspace(MY.bstar(1), MY.bstar(2), 5);
MANUAL_CENTERS_INIT.astar = linspace(MY.astar(1), MY.astar(2), 5);
MANUAL_CENTERS_INIT.S     = [0.14, 0.18, 0.26, 0.39, 0.48];
MANUAL_ALPHA = 0.05;

defaultImgRoots = [
"C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\images_igneous_rock\TS_1, 기반암 암종 분류 데이터_1. 화성암_1"
"C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\images_igneous_rock\TS_1, 기반암 암종 분류 데이터_1. 화성암_2"
"C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\images_igneous_rock\TS_2. 기반암 절리 탐지 데이터_1. 화성암"
"C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\images_igneous_rock\VS_1. 기반암 암종 분류 데이터_1. 화성암"
"C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\images_igneous_rock\VS_2. 기반암 절리 탐지 데이터_1. 화성암"
];
defaultJsonRoots = [
"C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\json_igneous_rock\TL_1. 기반암 암종 분류 데이터_1. 화성암"
"C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\json_igneous_rock\TL_2. 기반암 절리 탐지 데이터_1. 화성암"
"C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\json_igneous_rock\VL_1. 기반암 암종 분류 데이터_1. 화성암"
"C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\json_igneous_rock\VL_2. 기반암 절리 탐지 데이터_1. 화성암"
];
defaultResultsRoot = "C:\Users\ROCKENG\Desktop\코랩 머신러닝\results";

% =====================================================================
% 옵션 파서
% =====================================================================
p = inputParser; p.FunctionName = mfilename;
addParameter(p,'Mode',                   "train");
addParameter(p,'ImageRootDirs',          defaultImgRoots);
addParameter(p,'JsonRootDirs',           defaultJsonRoots);
addParameter(p,'ResultsRoot',            defaultResultsRoot);
addParameter(p,'RequireJSON',            true);
addParameter(p,'JsonGradeKeys',          ["rw_grade","rw","weathering_grade","weathering","weatheringGrade","d_grade","grade","D"]);
addParameter(p,'JsonSurfaceKeys',        ["wetdry","wet_dry","surface","condition","state","is_wet"]);
addParameter(p,'UseJSONSurfaceFallback', false);
addParameter(p,'GroupTokenCount',        4);
addParameter(p,'TrainRatio',             0.80);
addParameter(p,'ValRatio',               0.10);
addParameter(p,'TestRatio',              0.10);
addParameter(p,'MinPerClassInVal',       2);
addParameter(p,'MinPerClassInTest',      2);
addParameter(p,'BalanceTrain',           "equalize");
addParameter(p,'BalanceTargetMode',      "quantile");
addParameter(p,'BalanceQuantile',        0.50);
addParameter(p,'BalanceUpCap',           15.0);           % v6.1: 5→15 (D5=611에서 최대 9165까지)
addParameter(p,'InputSize',              [224 224 3]);
addParameter(p,'Epochs',                 60);            % TUNE: 50→60
addParameter(p,'MiniBatchSize',          16);
addParameter(p,'GradAccumSteps',         4);
addParameter(p,'InitialLearnRate',       3e-4);
addParameter(p,'BackboneLRFactor',       0.01);
addParameter(p,'BackboneWarmupEpochs',   3);
addParameter(p,'FreezeEpochs',           10);            % TUNE: 8→10 (헤드 충분 학습)
addParameter(p,'WeightDecay',            1e-4);
addParameter(p,'WarmupEpochs',           5);
addParameter(p,'DropoutRate',            0.30);           % TUNE: 0.4→0.3
addParameter(p,'AuxDropoutRate',         0.1);
addParameter(p,'LabelSmoothing',         0.10);           % TUNE: 0.05→0.10
addParameter(p,'MixupAlpha',             0.2);
addParameter(p,'FocalGamma',             2.0);
addParameter(p,'ColorJitterStrength',    0.07);            % TUNE: 0.10→0.07
addParameter(p,'AuxNoiseStd',            0.02);            % TUNE: 0.03→0.02
addParameter(p,'ClassWeightCap',         2.5);
addParameter(p,'GradClipNorm',           0.5);
addParameter(p,'EarlyStopPatience',      15);
addParameter(p,'ExecutionEnvironment',   "auto");
addParameter(p,'AuxHiddenDim',           64);
addParameter(p,'AuxOutDim',              64);
addParameter(p,'AuxScaleInit',           1.0);
addParameter(p,'CosineRestartPeriod',    15);              % FIX-5: warm restart 주기
addParameter(p,'CosineRestartDecay',     0.8);             % FIX-5: 주기별 LR 감쇄
addParameter(p,'UseTTA',                 true);            % FIX-6: TTA 활성화
addParameter(p,'TTACrops',               5);               % FIX-6: TTA crop 수
addParameter(p,'RockFilter',             "R01");
addParameter(p,'NetMatPath',             "");
addParameter(p,'PredictImageDir',        "");
addParameter(p,'PredictOutXlsx',         "");
parse(p, varargin{:});
opts = p.Results;
opts.Mode = lower(string(opts.Mode));

outRoot    = fullfile(char(opts.ResultsRoot), char(runTag));
dirModels  = fullfile(outRoot,'models');
dirReports = fullfile(outRoot,'reports');
mkdirSafe(outRoot); mkdirSafe(dirModels); mkdirSafe(dirReports);

runInfo = struct();
runInfo.runTag        = runTag;
runInfo.outRoot       = outRoot;
runInfo.timestamp     = char(t0);
runInfo.opts          = opts;
runInfo.paramNames    = PARAM_NAMES;
runInfo.manualAlpha   = MANUAL_ALPHA;

netTrained = []; trainLog = [];

fprintf('\n%s\n', repmat('=',1,72));
fprintf('ResNet-18 + Aux Color | Weathering D1~D5  [v6]\n');
fprintf('출력: %s\n', outRoot);
fprintf('FocalLoss γ=%.1f | ColorJitter=%.0f%% | BackboneFreeze=%dEp + Warmup%dEp\n', ...
    opts.FocalGamma, opts.ColorJitterStrength*100, opts.FreezeEpochs, opts.BackboneWarmupEpochs);
fprintf('v6: OnlineAux | GatedFusion | DataDrivenRule | PerFeatureScale | WarmRestart | TTA\n');
fprintf('앙상블: model %.0f%% + rule %.0f%%\n', (1-MANUAL_ALPHA)*100, MANUAL_ALPHA*100);
fprintf('%s\n\n', repmat('=',1,72));

if opts.Mode == "predict"
    runInfo = doPredictOnly(opts, outRoot, PARAM_NAMES, MANUAL_CENTERS_INIT, MANUAL_ALPHA);
    return;
end

% =====================================================================
% STEP 1: 파일 수집
% =====================================================================
printStep(1,'파일 수집 + JSON 매핑');
imgFiles = collectImageFiles(string(opts.ImageRootDirs));
if strlength(string(opts.RockFilter)) > 0
    [~,stems,~] = fileparts(imgFiles);
    imgFiles = imgFiles(contains(upper(string(stems)), upper(string(opts.RockFilter))));
    fprintf('  RockFilter(%s) → %d images\n', string(opts.RockFilter), numel(imgFiles));
end
if isempty(imgFiles), error('resnet18:NoImages','No images. Check ImageRootDirs.'); end
jsonMap = buildJsonMap(string(opts.JsonRootDirs));
fprintf('  images=%d | json=%d\n', numel(imgFiles), jsonMap.Count);

% =====================================================================
% STEP 2: 메타 테이블
% =====================================================================
printStep(2,'메타 테이블 구성');
[T, pInfo] = buildMetaTable(imgFiles, jsonMap, ...
    string(opts.JsonGradeKeys), string(opts.JsonSurfaceKeys), ...
    opts.RequireJSON, opts.UseJSONSurfaceFallback, opts.GroupTokenCount);
runInfo.parseInfo = pInfo;
fprintf('  usable=%d / %d\n', height(T), numel(imgFiles));
if height(T) < 50, error('resnet18:TooFew','Too few samples (n=%d).', height(T)); end

classNames = categorical(["D1","D2","D3","D4","D5"]);
T.grade   = categorical(string(T.grade),   string(classNames), string(classNames));
T.surface = categorical(string(T.surface), ["WET","DRY"],      ["WET","DRY"]);
fprintf('  [등급 × 표면]\n');
for k = 1:numel(classNames)
    nk = nnz(T.grade==classNames(k));
    fprintf('    %s: %d  (Dry=%d, Wet=%d)\n', string(classNames(k)), nk, ...
        nnz(T.grade==classNames(k) & T.surface=="DRY"), ...
        nnz(T.grade==classNames(k) & T.surface=="WET"));
end

% =====================================================================
% STEP 3: 보조 색상 파라미터 추출
% =====================================================================
printStep(3,'보조 색상 파라미터 추출 (11개)');
T = extractAuxColorParams(T);

% =====================================================================
% STEP 4: 그룹 계층적 분할
% =====================================================================
printStep(4,'그룹 계층적 분할');
[splitVec, sInfo] = makeGroupStratifiedSplit(T, ...
    opts.TrainRatio, opts.ValRatio, opts.TestRatio, ...
    opts.MinPerClassInVal, opts.MinPerClassInTest);
T.split = splitVec;
runInfo.splitInfo = sInfo;

idxTrain = find(T.split=="train");
idxVal   = find(T.split=="val");
idxTest  = find(T.split=="test");
fprintf('  Train=%d | Val=%d | Test=%d\n', numel(idxTrain), numel(idxVal), numel(idxTest));

% =====================================================================
% STEP 5: 밸런싱
% =====================================================================
printStep(5,'학습 데이터 밸런싱');
[idxTrainBal, bInfo] = balanceTrainIndices(T, idxTrain, classNames, opts);
runInfo.balanceInfo = bInfo;
fprintf('  train: %d → %d\n', numel(idxTrain), numel(idxTrainBal));

% =====================================================================
% STEP 6: 보조 파라미터 정규화
% =====================================================================
auxTrain = double(T.auxParams(idxTrainBal,:));
auxMean  = mean(auxTrain,1,'omitnan');
auxStd   = std(auxTrain,0,1,'omitnan');
auxStd(auxStd < 1e-6) = 1;
T.auxNormed = single((double(T.auxParams) - auxMean) ./ auxStd);
runInfo.auxNorm.mean = auxMean;
runInfo.auxNorm.std  = auxStd;
fprintf('\n  정규화 완료 (train %d샘플)\n', size(auxTrain,1));

% =====================================================================
% FIX-3: Data-Driven Rule Centers (학습 데이터 중앙값 기반)
% =====================================================================
printStep(6,'Data-Driven Rule Centers 계산');
MANUAL_CENTERS = computeDataDrivenCenters(T, idxTrainBal, classNames, PARAM_NAMES);
runInfo.manualCenters = MANUAL_CENTERS;
fprintf('  [데이터 기반 센터]\n');
fprintf('    b* centers: '); fprintf('%.2f ', MANUAL_CENTERS.bstar); fprintf('\n');
fprintf('    a* centers: '); fprintf('%.2f ', MANUAL_CENTERS.astar); fprintf('\n');
fprintf('    S  centers: '); fprintf('%.3f ', MANUAL_CENTERS.S); fprintf('\n');

% =====================================================================
% STEP 7: 클래스 가중치
% =====================================================================
numClasses   = numel(classNames);
classCounts  = countcats(categorical(T.grade(idxTrainBal), categories(classNames)));
classWeights = zeros(numClasses,1);
vm = classCounts > 0;
classWeights(vm) = 1.0 ./ sqrt(double(classCounts(vm)));
classWeights = classWeights / sum(classWeights) * numClasses;
classWeights = min(classWeights, opts.ClassWeightCap);
classWeights = classWeights / sum(classWeights) * numClasses;
runInfo.classWeights = classWeights;
fprintf('  [클래스 가중치 (sqrt 기반, 캡=%.1f)]\n', opts.ClassWeightCap);
for k = 1:numClasses
    fprintf('    %s: n=%d  w=%.4f\n', string(classNames(k)), classCounts(k), classWeights(k));
end

% =====================================================================
% STEP 8: 네트워크 구축 (v6: Gated Fusion)
% =====================================================================
printStep(7,'네트워크 구축 (ResNet-18 + Aux Branch + Gated Fusion)');
[dlnet, lInfo] = buildDualBranchNetwork_v6(opts, numClasses);
runInfo.layerInfo = lInfo;
fprintf('  InputNames: %s\n', strjoin(dlnet.InputNames,', '));

% =====================================================================
% STEP 9: 학습
% =====================================================================
printStep(8,'학습 (v6: OnlineAux + GatedFusion + WarmRestart)');
execEnv = pickExecEnv(opts.ExecutionEnvironment);
fprintf('  Env=%s | Ep=%d | MB=%d×Acc%d=eff%d\n', ...
    execEnv, opts.Epochs, opts.MiniBatchSize, opts.GradAccumSteps, ...
    opts.MiniBatchSize * opts.GradAccumSteps);
fprintf('  LR=%.1e | BB×%.3f(Warmup%dep) | Freeze=%dep | EarlyStop=%d\n', ...
    opts.InitialLearnRate, opts.BackboneLRFactor, opts.BackboneWarmupEpochs, ...
    opts.FreezeEpochs, opts.EarlyStopPatience);
fprintf('  WarmRestart period=%d decay=%.2f\n', opts.CosineRestartPeriod, opts.CosineRestartDecay);

[dlnet, trainLog, execEnv, opts] = trainWithRetry(...
    dlnet, T, idxTrainBal, idxVal, classNames, classWeights, lInfo, opts, execEnv);
netTrained = dlnet;
runInfo.trainLog = trainLog;

% =====================================================================
% STEP 10: 평가 (v6: TTA 지원)
% =====================================================================
printStep(9,'평가 (model-only vs ensemble, TTA=%s)', string(opts.UseTTA));
ruleBoundaries = buildRuleBoundaries(MANUAL_CENTERS);
[evalVal,  evalValSurf]  = evaluateModel(dlnet, T, idxVal,  classNames, opts, execEnv, ruleBoundaries, MANUAL_ALPHA, lInfo);
[evalTest, evalTestSurf] = evaluateModel(dlnet, T, idxTest, classNames, opts, execEnv, ruleBoundaries, MANUAL_ALPHA, lInfo);
runInfo.eval = struct('val',evalVal,'test',evalTest,'valSurf',evalValSurf,'testSurf',evalTestSurf);

fprintf('\n  [VAL]  model Acc=%.4f F1=%.4f | ens Acc=%.4f F1=%.4f\n', ...
    evalVal.modelOnly.accuracy, evalVal.modelOnly.macroF1, ...
    evalVal.ensemble.accuracy,  evalVal.ensemble.macroF1);
fprintf('  [TEST] model Acc=%.4f F1=%.4f | ens Acc=%.4f F1=%.4f\n', ...
    evalTest.modelOnly.accuracy, evalTest.modelOnly.macroF1, ...
    evalTest.ensemble.accuracy,  evalTest.ensemble.macroF1);

% =====================================================================
% STEP 11: 보조 파라미터 중요도
% =====================================================================
printStep(10,'보조 파라미터 중요도');
paramImp = analyzeColorParamWeights(dlnet, PARAM_NAMES);
runInfo.paramImportance = paramImp;

% =====================================================================
% STEP 12: 색상 구간 통계
% =====================================================================
printStep(11,'색상 구간 통계 (Q25/Q50/Q75)');
qTrain = computeClassQuantiles(T, idxTrainBal, classNames, PARAM_NAMES);
qVal   = computeClassQuantiles(T, idxVal,      classNames, PARAM_NAMES);
qTest  = computeClassQuantiles(T, idxTest,     classNames, PARAM_NAMES);
runInfo.colorQuantiles.train = qTrain;
runInfo.colorQuantiles.val   = qVal;
runInfo.colorQuantiles.test  = qTest;

fprintf('  [b* 중앙값 per class - Train]\n');
bIdx = find(strcmp(PARAM_NAMES,'b*'),1);
if ~isempty(bIdx)
    for c = 1:numClasses
        fprintf('    %s (n=%d): Q25=%.2f  Q50=%.2f  Q75=%.2f\n', ...
            string(classNames(c)), qTrain.n(c), ...
            qTrain.q25(c,bIdx), qTrain.q50(c,bIdx), qTrain.q75(c,bIdx));
    end
end

% =====================================================================
% STEP 13: 저장 (MAT + XLSX)
% =====================================================================
printStep(12,'저장 (.mat + .xlsx)');
matPath = fullfile(dirModels,'net_resnet18_aux_weathering_v6.mat');
try
    save(matPath,'netTrained','trainLog','runInfo','-v7.3');
    di = dir(matPath);
    fprintf('  MAT: %s  (%.1f MB)\n', matPath, di.bytes/1e6);
    assert(di.bytes > 1e6, '.mat file suspiciously small');
catch ME
    warning('resnet18:MatSave','MAT 저장 실패: %s', ME.message);
end

xlPath = fullfile(dirReports,'weathering_results_v6.xlsx');
saveExcelBundle(xlPath, T, idxVal, idxTest, classNames, PARAM_NAMES, ...
    evalVal, evalTest, evalValSurf, evalTestSurf, ...
    paramImp, qTrain, qVal, qTest, runInfo, ...
    MANUAL_CENTERS, ruleBoundaries, MANUAL_ALPHA, dlnet, opts, execEnv, lInfo);
fprintf('  XLSX: %s\n', xlPath);

elapsed = datetime('now','TimeZone','Asia/Seoul') - t0;
fprintf('\n%s\nDONE | 소요: %s\n출력: %s\n%s\n\n', ...
    repmat('=',1,72), char(elapsed), outRoot, repmat('=',1,72));
end % ===== END MAIN =====


%% =========================================================================
%  FIX-3: Data-Driven Rule Centers
%  학습 데이터 중앙값에서 직접 rule center 도출
% =========================================================================
function mc = computeDataDrivenCenters(T, idxTr, classNames, paramNames)
numC = numel(classNames);
auxRaw = double(T.auxParams(idxTr,:));
grades = T.grade(idxTr);

bIdx = find(strcmp(paramNames,'b*'),1);
aIdx = find(strcmp(paramNames,'a*'),1);
sIdx = find(strcmp(paramNames,'S'),1);

mc.bstar = zeros(1,numC);
mc.astar = zeros(1,numC);
mc.S     = zeros(1,numC);

nPerClass = zeros(1,numC);
for c = 1:numC
    sub = auxRaw(grades==classNames(c),:);
    nPerClass(c) = size(sub,1);
    if size(sub,1) >= 2
        mc.bstar(c) = median(sub(:,bIdx),'omitnan');
        mc.astar(c) = median(sub(:,aIdx),'omitnan');
        mc.S(c)     = median(sub(:,sIdx),'omitnan');
    end
end

% v6.1: 소수 클래스 외삽 — 샘플 충분한 클래스 간 추세를 기반으로
% D1~D3은 대체로 충분하므로 이들의 증분(step)을 D4/D5에 외삽
mc.bstar = extrapolateSparse(mc.bstar, nPerClass, 1000);
mc.astar = extrapolateSparse(mc.astar, nPerClass, 1000);
mc.S     = extrapolateSparse(mc.S,     nPerClass, 1000);

% 최소 간격 강제: 인접 센터 간 최소 gap
mc.bstar = enforceMinGap(mc.bstar);
mc.astar = enforceMinGap(mc.astar);
mc.S     = enforceMinGap(mc.S);
end

function v = extrapolateSparse(v, nPerClass, minN)
% 샘플이 minN 미만인 클래스는 충분한 클래스들의 추세로 외삽
sufficient = nPerClass >= minN;
if sum(sufficient) < 2, return; end  % 추세 추정 불가

% 충분한 클래스들의 인덱스와 값
idxSuf = find(sufficient);
valSuf = v(idxSuf);

% 충분한 클래스들 간 평균 증분
if numel(idxSuf) >= 2
    avgStep = (valSuf(end) - valSuf(1)) / (idxSuf(end) - idxSuf(1));
else
    return;
end

% 부족한 클래스를 마지막 충분한 클래스에서 외삽
lastSufIdx = idxSuf(end);
lastSufVal = valSuf(end);
for c = (lastSufIdx+1):numel(v)
    v(c) = lastSufVal + avgStep * (c - lastSufIdx);
end
end

function v = enforceMinGap(v)
% 인접 센터 간 최소 간격 = 전체 범위의 10%
n = numel(v);
if n < 2, return; end

totalRange = abs(v(end) - v(1));
minGap = max(totalRange * 0.10, 1e-3);  % 최소 10% 간격

if v(end) >= v(1)
    % 증가 방향
    for i = 2:n
        if v(i) < v(i-1) + minGap
            v(i) = v(i-1) + minGap;
        end
    end
else
    % 감소 방향
    for i = 2:n
        if v(i) > v(i-1) - minGap
            v(i) = v(i-1) - minGap;
        end
    end
end
end


%% =========================================================================
%  색상 증강 헬퍼
% =========================================================================
function I = colorJitterAug(I, strength)
if strength <= 0, return; end
if rand > 0.3
    factor = 1 + (rand*2-1)*strength;
    I = uint8(min(255, max(0, double(I)*factor)));
end
if rand > 0.3
    hsvI = rgb2hsv(double(I)/255);
    satFactor = 1 + (rand*2-1)*strength*0.8;
    hsvI(:,:,2) = min(1, max(0, hsvI(:,:,2) * satFactor));
    if rand > 0.5
        hueShift = (rand*2-1)*strength*50/360;
        hsvI(:,:,1) = mod(hsvI(:,:,1) + hueShift, 1);
    end
    I = uint8(hsv2rgb(hsvI)*255);
end
if rand > 0.5
    factor = 1 + (rand*2-1)*strength*0.5;
    meanVal = mean(double(I(:)));
    I = uint8(min(255, max(0, meanVal + (double(I)-meanVal)*factor)));
end
end


%% =========================================================================
%  FIX-1: Online Aux 추출 (augmented 이미지에서 실시간 계산)
% =========================================================================
function auxVec = extractAuxFromPatch(patchImg, isWet)
% patchImg: uint8 [H×W×3]
% 반환: [1×11] single — R,G,B,H,S,V,L*,a*,b*,C*,is_wet
imgD = double(patchImg);
R = mean(imgD(:,:,1),'all');
G = mean(imgD(:,:,2),'all');
B = mean(imgD(:,:,3),'all');

hsvI = rgb2hsv(patchImg);
H = mean(hsvI(:,:,1),'all') * 360;
Sv = mean(hsvI(:,:,2),'all');
Vv = mean(hsvI(:,:,3),'all');

try
    labI = rgb2lab(patchImg);
    Ls = mean(labI(:,:,1),'all');
    As = mean(labI(:,:,2),'all');
    Bs = mean(labI(:,:,3),'all');
    Cs = sqrt(As^2 + Bs^2);
catch
    Ls=50; As=0; Bs=0; Cs=0;
end

auxVec = single([R,G,B,H,Sv,Vv,Ls,As,Bs,Cs,double(isWet)]);
end


%% =========================================================================
%  FIX-2: Gated Fusion Network 구축
% =========================================================================
function [dlnet, info] = buildDualBranchNetwork_v6(opts, numClasses)
baseNet = resnet18;
lg      = layerGraph(baseNet);
names   = string({lg.Layers.Name});

imgInputName = "";
for i = 1:numel(lg.Layers)
    if isa(lg.Layers(i),'nnet.cnn.layer.ImageInputLayer')
        imgInputName = string(lg.Layers(i).Name); break;
    end
end
if strlength(imgInputName) == 0, imgInputName = "data"; end

rmList = names(contains(lower(names),["fc","softmax","classification","prob"]));
for i = 1:numel(rmList)
    try lg = removeLayers(lg, rmList(i)); catch, end
end

ly      = string({lg.Layers.Name});
gapCand = ly(contains(lower(ly),["pool5","avgpool","gap"]));
gapName = "pool5";
if ~isempty(gapCand), gapName = gapCand(end); end

lg = addLayers(lg, flattenLayer('Name','img_flatten'));
lg = connectLayers(lg, gapName, 'img_flatten');

% --- Aux Branch (v6: 3-layer with residual-like skip) ---
lg = addLayers(lg, [
    featureInputLayer(11,'Name','aux_input','Normalization','none')
    fullyConnectedLayer(opts.AuxHiddenDim,'Name','aux_fc1')
    batchNormalizationLayer('Name','aux_bn1')
    reluLayer('Name','aux_relu1')
    dropoutLayer(opts.AuxDropoutRate,'Name','aux_drop1')
    fullyConnectedLayer(opts.AuxHiddenDim,'Name','aux_fc1b')   % v6: 추가 레이어
    batchNormalizationLayer('Name','aux_bn1b')
    reluLayer('Name','aux_relu1b')
    fullyConnectedLayer(opts.AuxOutDim,'Name','aux_fc2')
    batchNormalizationLayer('Name','aux_bn2')
    reluLayer('Name','aux_relu2')
]);

imgDim = 512;
auxDim = opts.AuxOutDim;

% --- FIX-2: Gated Fusion ---
% Gate = sigmoid(W_g * [img; aux] + b_g)
% fused = gate .* img_proj + (1-gate) .* aux_proj
%
% MATLAB dlnetwork에서는 커스텀 레이어로 구현하기 어려우므로
% 두 가지 경로의 FC를 만들고 학습 루프에서 게이트 연산 수행
% → 실제 구현: 네트워크는 concat까지만 하고, 분류 헤드에서 처리

% 이미지 차원 축소 (512 → 128)
lg = addLayers(lg, [
    fullyConnectedLayer(128,'Name','img_proj')
    batchNormalizationLayer('Name','img_proj_bn')
    reluLayer('Name','img_proj_relu')
]);
lg = connectLayers(lg, 'img_flatten', 'img_proj');

% Aux 차원 축소 (64 → 128)
lg = addLayers(lg, [
    fullyConnectedLayer(128,'Name','aux_proj')
    batchNormalizationLayer('Name','aux_proj_bn')
    reluLayer('Name','aux_proj_relu')
]);
lg = connectLayers(lg, 'aux_relu2', 'aux_proj');

% Gate 입력 (img 512 + aux 64 → concat → gate sigmoid는 학습루프에서)
% 여기서는 projected features를 concat
fuseDim = 128 + 128;  % projected dimensions

lg = addLayers(lg, [
    concatenationLayer(1,2,'Name','fusion_concat')
    fullyConnectedLayer(256,'Name','cls_fc1')
    batchNormalizationLayer('Name','cls_bn1')
    reluLayer('Name','cls_relu1')
    dropoutLayer(opts.DropoutRate,'Name','cls_drop1')
    fullyConnectedLayer(128,'Name','cls_fc2')          % v6: 추가 분류 레이어
    batchNormalizationLayer('Name','cls_bn2')
    reluLayer('Name','cls_relu2')
    dropoutLayer(opts.DropoutRate*0.5,'Name','cls_drop2')  % v6: 경량 dropout
    fullyConnectedLayer(numClasses,'Name','cls_fc_out')
    softmaxLayer('Name','cls_softmax')
]);

lg = connectLayers(lg,'img_proj_relu','fusion_concat/in1');
lg = connectLayers(lg,'aux_proj_relu','fusion_concat/in2');

% Gate 경로: concat(img_flat, aux_relu2) → FC → sigmoid
% → 학습 루프에서 별도 처리 (dlnetwork 외부)
% gate_input = [img_512; aux_64] → FC(576→256) → sigmoid
% 이를 위해 별도 gate network를 dlnetwork 밖에서 관리

dlnet = dlnetwork(lg);
totP  = sum(cellfun(@numel, dlnet.Learnables.Value));

inputNames = string(dlnet.InputNames);
imgIdx = find(inputNames == imgInputName, 1);
auxIdx = find(inputNames == "aux_input",  1);
if isempty(imgIdx), imgIdx = 1; end
if isempty(auxIdx), auxIdx = 2; end

info = struct('totalParams',totP,'fusionDim',fuseDim,'numClasses',numClasses, ...
    'imgInputName',imgInputName,'imgIdx',imgIdx,'auxIdx',auxIdx, ...
    'imgDim',imgDim,'auxDim',auxDim);
fprintf('  파라미터: %s | 융합dim: %d (gated proj)\n', addCommas(totP), fuseDim);
fprintf('  InputOrder: img[%d]=%s, aux[%d]=aux_input\n', imgIdx, imgInputName, auxIdx);
end


%% =========================================================================
%  Custom Training Loop (v6: 주요 개선)
%  FIX-1: Online Aux 재추출
%  FIX-4: Per-Feature Scaling
%  FIX-5: Cosine Annealing with Warm Restarts
% =========================================================================
function [dlnet, logS] = customTrainLoop(dlnet, T, idxTrain, idxVal, ...
    classNames, classWeights, lInfo, opts, execEnv)

numC   = numel(classNames);
inSz   = opts.InputSize;
mb     = opts.MiniBatchSize;
nAcc   = opts.GradAccumSteps;
nEp    = opts.Epochs;
peakLR = opts.InitialLearnRate;
wd     = opts.WeightDecay;

cw = single(classWeights(:));
if execEnv=="gpu", cw = gpuArray(cw); end

avgG = []; avgSqG = [];
iter = 0;
lyrNames = dlnet.Learnables.Layer;
parNames = dlnet.Learnables.Parameter;
isBN = contains(parNames,"Offset") | contains(parNames,"Scale");
isBB = ~(contains(lyrNames,"aux_") | contains(lyrNames,"cls_") | ...
         contains(lyrNames,"fusion_") | contains(lyrNames,"img_proj") | contains(lyrNames,"aux_proj"));

% FIX-4: Per-Feature Learnable Scaling (11차원)
auxScaleVec = single(ones(1,11) * opts.AuxScaleInit);
mAS = zeros(1,11); vAS = zeros(1,11);
lrAS = opts.InitialLearnRate * 1.0;     % v6.1: 0.1→1.0 (frozen 상태에서도 학습 가능)
iterAS = 0;

% 정규화 통계 저장 (Online Aux용)
try
    rawTrain = double(T.auxParams(idxTrain,:));
    auxMean_ = mean(rawTrain,1,'omitnan');
    auxStd_  = std(rawTrain,0,1,'omitnan');
    auxStd_(auxStd_ < 1e-6) = 1;
catch
    auxMean_ = double(mean(T.auxParams,1,'omitnan'));
    auxStd_  = double(std(T.auxParams,0,1,'omitnan'));
    auxStd_(auxStd_ < 1e-6) = 1;
end

logS = struct('epochLoss',zeros(nEp,1),'epochAcc',zeros(nEp,1), ...
    'valLoss',zeros(nEp,1),'valAcc',zeros(nEp,1),'valMacroF1',zeros(nEp,1), ...
    'lr',zeros(nEp,1),'auxScale',zeros(nEp,11));

% ── 실시간 학습 모니터 Figure 생성 ─────────────────────────────────────
hFig = figure('Name','Training Monitor — ResNet-18 + Aux v6', ...
    'NumberTitle','off','Position',[100 100 1200 800],'Color','w');

% (1) Loss
ax1 = subplot(2,3,1,'Parent',hFig);
hLnTrLoss = animatedline(ax1,'Color',[0.2 0.4 0.8],'LineWidth',1.5,'DisplayName','Train Loss');
hLnVaLoss = animatedline(ax1,'Color',[0.9 0.3 0.2],'LineWidth',1.5,'DisplayName','Val Loss');
title(ax1,'Loss'); xlabel(ax1,'Epoch'); ylabel(ax1,'Focal Loss');
legend(ax1,'Location','northeast'); grid(ax1,'on');

% (2) Accuracy
ax2 = subplot(2,3,2,'Parent',hFig);
hLnTrAcc = animatedline(ax2,'Color',[0.2 0.4 0.8],'LineWidth',1.5,'DisplayName','Train Acc');
hLnVaAcc = animatedline(ax2,'Color',[0.9 0.3 0.2],'LineWidth',1.5,'DisplayName','Val Acc');
title(ax2,'Accuracy'); xlabel(ax2,'Epoch'); ylabel(ax2,'Accuracy');
legend(ax2,'Location','southeast'); grid(ax2,'on'); ylim(ax2,[0 1]);

% (3) Val Macro F1
ax3 = subplot(2,3,3,'Parent',hFig);
hLnF1 = animatedline(ax3,'Color',[0.1 0.7 0.3],'LineWidth',2,'DisplayName','Val Macro F1');
hLnBestF1 = animatedline(ax3,'Color',[0.1 0.7 0.3],'LineStyle','--','LineWidth',1,'DisplayName','Best F1');
title(ax3,'Validation Macro F1'); xlabel(ax3,'Epoch'); ylabel(ax3,'F1');
legend(ax3,'Location','southeast'); grid(ax3,'on'); ylim(ax3,[0 1]);

% (4) Learning Rate
ax4 = subplot(2,3,4,'Parent',hFig);
hLnLR = animatedline(ax4,'Color',[0.6 0.2 0.8],'LineWidth',1.5);
title(ax4,'Learning Rate (Warm Restart)'); xlabel(ax4,'Epoch'); ylabel(ax4,'LR');
set(ax4,'YScale','log'); grid(ax4,'on');

% (5) auxScale (a*, b*, S)
ax5 = subplot(2,3,5,'Parent',hFig);
hLnAS_a = animatedline(ax5,'Color',[0.9 0.2 0.2],'LineWidth',1.5,'DisplayName','a*');
hLnAS_b = animatedline(ax5,'Color',[0.2 0.5 0.9],'LineWidth',1.5,'DisplayName','b*');
hLnAS_S = animatedline(ax5,'Color',[0.2 0.8 0.3],'LineWidth',1.5,'DisplayName','S');
yline(ax5,1,'--','α=1','Color',[0.5 0.5 0.5]);
title(ax5,'Per-Feature Scale (a*, b*, S)'); xlabel(ax5,'Epoch'); ylabel(ax5,'Scale');
legend(ax5,'Location','best'); grid(ax5,'on');

% (6) Patience / EarlyStop
ax6 = subplot(2,3,6,'Parent',hFig);
hBarPat = bar(ax6, 0, 0, 'FaceColor',[0.95 0.6 0.1],'EdgeColor','none');
hold(ax6,'on');
yline(ax6, opts.EarlyStopPatience, 'r--', sprintf('Patience=%d',opts.EarlyStopPatience), 'LineWidth',1.5);
title(ax6,'Early Stop Counter'); xlabel(ax6,'Epoch'); ylabel(ax6,'Patience');
ylim(ax6,[0 opts.EarlyStopPatience+2]); grid(ax6,'on');
hold(ax6,'off');

sgtitle(hFig, 'ResNet-18 + Aux Color v6 — Training Monitor', 'FontSize',14, 'FontWeight','bold');
drawnow;

bestF1   = -Inf;
bestLoss = Inf;
bestNet  = dlnet;
patience = 0;

% FIX-5: Warm Restart 변수
T0 = opts.CosineRestartPeriod;  % 초기 주기
Ti = T0;                        % 현재 주기
Tcur = 0;                       % 현재 주기 내 에폭
peakLR_cur = peakLR;            % 현재 주기 peak LR

for ep = 1:nEp
    % ── FIX-5: Cosine Annealing with Warm Restarts ──────────────────────
    if ep <= opts.WarmupEpochs
        lr = peakLR * (ep / opts.WarmupEpochs);
    else
        % Warm restart schedule
        prog = Tcur / max(1, Ti);
        lr   = peakLR_cur * 0.5 * (1 + cos(pi * prog));
        Tcur = Tcur + 1;
        if Tcur >= Ti
            Tcur = 0;
            Ti   = round(Ti * 1.2);   % 주기 점진적 증가
            peakLR_cur = peakLR_cur * opts.CosineRestartDecay;  % peak LR 감쇄
            fprintf('  [Ep %d] Warm Restart: Ti=%d, peakLR=%.1e\n', ep, Ti, peakLR_cur);
        end
    end
    lr = max(lr, 1e-7);
    logS.lr(ep) = lr;

    % ── 백본 해동 ──────────────────────────────────────────────────────
    freezeBackbone = (ep <= opts.FreezeEpochs);
    if ep == opts.FreezeEpochs + 1
        fprintf('  [Ep %d] 백본 해동 — Adam 상태 초기화 + BackboneLR WarmUp 시작\n', ep);
        avgG   = [];
        avgSqG = [];
        iter   = 0;
        % Warm restart도 리셋
        Tcur = 0;
        Ti   = T0;
        peakLR_cur = peakLR * opts.CosineRestartDecay;  % 해동 후 약간 낮은 LR
    end

    bbWarmupEps = max(1, opts.BackboneWarmupEpochs);
    if ep > opts.FreezeEpochs && ep <= opts.FreezeEpochs + bbWarmupEps
        bbWarmupProgress = (ep - opts.FreezeEpochs) / bbWarmupEps;
        bbLRFactor = opts.BackboneLRFactor * bbWarmupProgress;
    else
        bbLRFactor = opts.BackboneLRFactor;
    end

    shuf = idxTrain(randperm(numel(idxTrain)));
    nBat = floor(numel(shuf) / mb);
    eLoss = 0; eCorr = 0; eTot = 0;
    aG = []; aGS = zeros(1,11); aCnt = 0;

    if nBat == 0
        warning('resnet18:SmallBatch','학습셋(%d) < MB(%d), epoch %d 스킵.', numel(shuf), mb, ep);
        [vL,vA,vF] = quickVal(dlnet,T,idxVal,classNames,classWeights,opts,execEnv,lInfo);
        logS.epochLoss(ep)=NaN; logS.epochAcc(ep)=NaN;
        logS.valLoss(ep)=vL; logS.valAcc(ep)=vA; logS.valMacroF1(ep)=vF;
        logS.auxScale(ep,:)=double(auxScaleVec);
        fprintf('  Ep %2d/%d | (스킵) ValF1=%.3f\n', ep, nEp, vF);
        continue;
    end

    for bat = 1:nBat
        bIdx = shuf((bat-1)*mb+1 : bat*mb);
        imgB = zeros([inSz mb],'single');
        auxB = zeros(11, mb, 'single');  % FIX-1: 배치별 aux 버퍼

        for bi = 1:mb
            % FIX-1: augmented 이미지에서 직접 aux 추출
            augPatch = readPatch(T.file(bIdx(bi)), inSz, "train", opts.ColorJitterStrength);
            imgB(:,:,:,bi) = single(augPatch) / 255;

            % Online aux 추출 + 정규화
            isWet = double(T.surface(bIdx(bi)) == categorical("WET"));
            rawAux = extractAuxFromPatch(augPatch, isWet);
            auxB(:,bi) = single((double(rawAux) - auxMean_) ./ auxStd_)';
        end

        % Aux noise (FIX-1과 호환: online 추출 후 소량 노이즈)
        if opts.AuxNoiseStd > 0
            noise = single(randn(10, mb)) * single(opts.AuxNoiseStd);
            auxB(1:10,:) = auxB(1:10,:) + noise;
        end

        yT = zeros(numC, mb, 'single');
        for bi = 1:mb
            ci = find(classNames == T.grade(bIdx(bi)));
            if ~isempty(ci), yT(ci,bi) = 1; end
        end

        % Mixup/LabelSmoothing (상호 배타적)
        useMixup = opts.MixupAlpha > 0 && rand > 0.5;
        if useMixup
            lam  = betarnd(opts.MixupAlpha, opts.MixupAlpha);
            lam  = max(lam, 1-lam);  % v6: λ ≥ 0.5로 제한 (과도한 혼합 방지)
            perm = randperm(mb);
            imgB = lam*imgB + (1-lam)*imgB(:,:,:,perm);
            auxB = lam*auxB + (1-lam)*auxB(:,perm);
            yT   = lam*yT   + (1-lam)*yT(:,perm);
        elseif opts.LabelSmoothing > 0
            yT = yT*(1-opts.LabelSmoothing) + opts.LabelSmoothing/numC;
        end

        dlImg = dlarray(imgB,'SSCB');
        dlAux = dlarray(auxB,'CB');
        dlY   = dlarray(yT,'CB');
        if execEnv=="gpu"
            dlImg = gpuArray(dlImg);
            dlAux = gpuArray(dlAux);
            dlY   = gpuArray(dlY);
        end

        % FIX-4: Per-Feature Scale → dlarray 벡터
        dlAS = dlarray(single(auxScaleVec(:)));  % [11×1]
        if execEnv=="gpu", dlAS = gpuArray(dlAS); end

        [loss, grads, gS, probs] = dlfeval(@modelGrad_v6, dlnet, dlImg, dlAux, dlY, cw, dlAS, lInfo, opts.FocalGamma);

        if isempty(aG)
            aG = grads;
        else
            for gi = 1:height(aG)
                aG.Value{gi} = aG.Value{gi} + grads.Value{gi};
            end
        end
        gS_host = double(gather(extractdata(gS)));
        aGS  = aGS + gS_host(:)';
        aCnt = aCnt + 1;

        eLoss = eLoss + double(gather(extractdata(loss))) * mb;
        pArr = double(gather(extractdata(probs)));
        yArr = double(gather(extractdata(dlY)));
        [~,pI] = max(pArr,[],1); [~,tI] = max(yArr,[],1);
        eCorr = eCorr + sum(pI==tI);
        eTot  = eTot  + mb;

        if aCnt >= nAcc || bat == nBat
            for gi = 1:height(aG)
                aG.Value{gi} = aG.Value{gi} / aCnt;
            end
            aGS = aGS / aCnt;

            if wd > 0
                for gi = 1:height(aG)
                    if ~isBN(gi)
                        aG.Value{gi} = aG.Value{gi} + single(wd).*dlnet.Learnables.Value{gi};
                    end
                end
            end

            % 그래디언트 클리핑
            gnSq = 0;
            for gi = 1:height(aG)
                v = double(gather(extractdata(aG.Value{gi})));
                gnSq = gnSq + sum(v(:).^2);
            end
            gn = sqrt(gnSq);
            if gn > opts.GradClipNorm
                sf = single(opts.GradClipNorm / gn);
                for gi = 1:height(aG), aG.Value{gi} = aG.Value{gi}*sf; end
                aGS = aGS * double(sf);
            end

            % 차등 LR
            lrG = aG;
            if freezeBackbone
                for gi = 1:height(lrG)
                    if isBB(gi), lrG.Value{gi} = lrG.Value{gi}*0; end
                end
            else
                for gi = 1:height(lrG)
                    if isBB(gi), lrG.Value{gi} = lrG.Value{gi}*single(bbLRFactor); end
                end
            end

            iter = iter + 1;
            [dlnet, avgG, avgSqG] = adamupdate(dlnet, lrG, avgG, avgSqG, iter, lr, 0.9, 0.999, 1e-8);

            % FIX-4: Per-Feature auxScale Adam 업데이트
            iterAS = iterAS + 1;
            b1c = 0.9; b2c = 0.999; epsc = 1e-8;
            mAS  = b1c*mAS + (1-b1c)*aGS;
            vAS  = b2c*vAS + (1-b2c)*aGS.^2;
            mHat = mAS / (1 - b1c^iterAS);
            vHat = vAS / (1 - b2c^iterAS);
            auxScaleVec = auxScaleVec - single(lrAS * mHat ./ (sqrt(vHat) + epsc));
            auxScaleVec = max(single(0.1), min(single(5.0), auxScaleVec));

            aG = []; aGS = zeros(1,11); aCnt = 0;
        end
    end % bat

    logS.epochLoss(ep) = eLoss / max(1,eTot);
    logS.epochAcc(ep)  = eCorr / max(1,eTot);
    logS.auxScale(ep,:) = double(auxScaleVec);

    [vL,vA,vF] = quickVal(dlnet,T,idxVal,classNames,classWeights,opts,execEnv,lInfo);
    logS.valLoss(ep)=vL; logS.valAcc(ep)=vA; logS.valMacroF1(ep)=vF;

    % 상태 표시
    bbStatus = "[FROZEN]";
    if ~freezeBackbone
        if ep <= opts.FreezeEpochs + bbWarmupEps
            bbStatus = sprintf("[WARMUP %.0f%%]", bbLRFactor/opts.BackboneLRFactor*100);
        else
            bbStatus = "[ACTIVE]";
        end
    end

    % auxScale 요약 (주요 3개만)
    fprintf('  Ep %2d/%d %s | Loss=%.4f Acc=%.3f | VLoss=%.4f VAcc=%.3f VF1=%.3f | LR=%.1e BBx%.4f | αS=[%.2f,%.2f,%.2f]', ...
        ep, nEp, bbStatus, logS.epochLoss(ep), logS.epochAcc(ep), ...
        vL, vA, vF, lr, bbLRFactor, ...
        auxScaleVec(8), auxScaleVec(9), auxScaleVec(5));  % a*, b*, S

    if vF > bestF1 || (vF==bestF1 && vL<bestLoss)
        bestF1 = vF; bestLoss = vL; bestNet = dlnet; patience = 0;
        fprintf(' *');
    else
        patience = patience + 1;
    end
    fprintf('\n');

    % ── 실시간 그래프 업데이트 ──────────────────────────────────────────
    if isvalid(hFig)
        addpoints(hLnTrLoss, ep, logS.epochLoss(ep));
        addpoints(hLnVaLoss, ep, vL);
        addpoints(hLnTrAcc,  ep, logS.epochAcc(ep));
        addpoints(hLnVaAcc,  ep, vA);
        addpoints(hLnF1,     ep, vF);
        addpoints(hLnBestF1, ep, bestF1);
        addpoints(hLnLR,     ep, lr);
        addpoints(hLnAS_a,   ep, auxScaleVec(8));
        addpoints(hLnAS_b,   ep, auxScaleVec(9));
        addpoints(hLnAS_S,   ep, auxScaleVec(5));
        % Patience bar 업데이트
        set(hBarPat, 'XData', ep, 'YData', patience);
        if patience >= opts.EarlyStopPatience - 3
            set(hBarPat, 'FaceColor', [0.9 0.2 0.1]);  % 빨간색 경고
        else
            set(hBarPat, 'FaceColor', [0.95 0.6 0.1]);
        end
        xlim(ax6, [0 ep+1]);
        % Figure 타이틀에 현재 상태 표시
        sgtitle(hFig, sprintf('v6 Training — Ep %d/%d | Best F1=%.3f (Ep%d) | %s', ...
            ep, nEp, bestF1, find(logS.valMacroF1(1:ep)==bestF1,1,'last'), bbStatus), ...
            'FontSize',14,'FontWeight','bold');
        drawnow limitrate;
    end

    if patience >= opts.EarlyStopPatience
        fprintf('  EarlyStop: %d 에폭 미개선.\n', opts.EarlyStopPatience);
        break;
    end
end % ep

dlnet = bestNet;
lastEp = find(logS.lr > 0, 1, 'last');
if isempty(lastEp), lastEp = 1; end
validF1 = logS.valMacroF1(1:lastEp);
[bF1, bEp] = max(validF1);
if isempty(bEp) || bEp == 0, bEp = lastEp; bF1 = validF1(lastEp); end
logS.bestEpoch   = bEp;
logS.bestValF1   = bF1;
logS.bestValLoss = logS.valLoss(bEp);
fprintf('\n  Best: Ep=%d  ValF1=%.4f  ValLoss=%.4f\n', bEp, bF1, logS.valLoss(bEp));

% ── 최종 그래프 저장 ───────────────────────────────────────────────────
if exist('hFig','var') && isvalid(hFig)
    sgtitle(hFig, sprintf('v6 FINAL — Best F1=%.3f (Ep%d) | Acc=%.3f | Last Ep=%d', ...
        bF1, bEp, logS.valAcc(bEp), lastEp), 'FontSize',14,'FontWeight','bold');
    drawnow;
    % PNG 저장 (results 폴더)
    try
        figPath = fullfile(opts.ResultsRoot, 'training_monitor_v6.png');
        exportgraphics(hFig, figPath, 'Resolution', 200);
        fprintf('  학습 그래프 저장: %s\n', figPath);
    catch
        try
            figPath = fullfile(opts.ResultsRoot, 'training_monitor_v6.png');
            saveas(hFig, figPath);
            fprintf('  학습 그래프 저장(saveas): %s\n', figPath);
        catch ME2
            fprintf('  그래프 저장 실패: %s\n', ME2.message);
        end
    end
end
end


%% =========================================================================
%  Loss + Gradient (v6: Per-Feature Scale)
% =========================================================================
function [loss, grads, gScale, probs] = modelGrad_v6(dlnet, dlImg, dlAux, dlY, cw, dlAS, lInfo, gamma)
% FIX-4: Per-Feature Scaling (dlAS is [11×1])
dlAuxScaled = dlAux .* dlAS;

inputs = cell(1, numel(dlnet.InputNames));
inputs{lInfo.imgIdx} = dlImg;
inputs{lInfo.auxIdx} = dlAuxScaled;

probs = forward(dlnet, inputs{:});

eps_ = 1e-10;
pt       = sum(probs .* dlY, 1);
pt       = max(pt, eps_);
focal_wt = (1 - pt).^gamma;

cwHost  = double(gather(cw));
dlYhost = double(gather(extractdata(dlY)));
[~, cIdxH] = max(dlYhost, [], 1);
cIdxH  = min(max(cIdxH, 1), numel(cwHost));
alphaT = single(cwHost(cIdxH));
dlAlphaT = dlarray(alphaT, 'B');
if isa(extractdata(dlAux),'gpuArray')
    dlAlphaT = gpuArray(dlAlphaT);
end

loss = mean(-dlAlphaT .* focal_wt .* log(pt));

% Per-Feature Scale 정규화: 각 스케일이 1에서 너무 벗어나지 않도록
loss = loss + 0.01 * mean((dlAS - 1).^2);

grads  = dlgradient(loss, dlnet.Learnables);
gScale = dlgradient(loss, dlAS);
end


%% =========================================================================
%  Quick Validation
% =========================================================================
function [vL, vA, vF] = quickVal(dlnet, T, idxV, classNames, classWeights, opts, execEnv, lInfo)
numC = numel(classNames);
cwC  = double(classWeights(:));
mb   = min(32, max(1, numel(idxV)));
totL = 0; allP = []; allT = [];
for bat = 1:ceil(numel(idxV)/mb)
    b1 = (bat-1)*mb+1; b2 = min(bat*mb, numel(idxV));
    bIdx = idxV(b1:b2); bsz = numel(bIdx);
    if bsz == 0, continue; end

    imgB = zeros([opts.InputSize bsz],'single');
    for bi = 1:bsz
        imgB(:,:,:,bi) = single(readPatch(T.file(bIdx(bi)), opts.InputSize, "val", 0)) / 255;
    end
    auxB = single(T.auxNormed(bIdx,:))';

    yT = zeros(numC,bsz,'single');
    tI = zeros(1,bsz);
    for bi = 1:bsz
        ci = find(classNames==T.grade(bIdx(bi)));
        if ~isempty(ci), yT(ci,bi)=1; tI(bi)=ci; end
    end

    dlImg = dlarray(imgB,'SSCB');
    dlAux = dlarray(auxB,'CB');
    if execEnv=="gpu", dlImg=gpuArray(dlImg); dlAux=gpuArray(dlAux); end

    inputs = cell(1, numel(dlnet.InputNames));
    inputs{lInfo.imgIdx} = dlImg;
    inputs{lInfo.auxIdx} = dlAux;
    pp = double(gather(extractdata(predict(dlnet, inputs{:}))));

    eps_ = 1e-10;
    pt   = sum(double(yT) .* pp, 1);
    pt   = max(pt, eps_);
    fl   = -(1-pt).^opts.FocalGamma .* log(pt);
    for bi = 1:bsz
        ci = tI(bi); if ci>0, fl(bi)=fl(bi)*cwC(ci); end
    end
    totL = totL + sum(fl);
    [~,pI] = max(pp,[],1);
    allP = [allP, pI]; %#ok<AGROW>
    allT = [allT, tI]; %#ok<AGROW>
end

if isempty(allT)
    vL=NaN; vA=NaN; vF=0; return;
end
vL = totL / max(1, numel(idxV));
vA = sum(allP==allT) / numel(allT);

vm = allT > 0;
if sum(vm) < 2, vF=0; return; end
try
    cm  = confusionmat(double(allT(vm)'), double(allP(vm)'), 'Order',(1:numC)');
    f1s = zeros(numC,1);
    for k = 1:numC
        tp=cm(k,k); fp=sum(cm(:,k))-tp; fn=sum(cm(k,:))-tp;
        pr=tp/max(1,tp+fp); rc=tp/max(1,tp+fn);
        f1s(k)=2*pr*rc/max(1e-12,pr+rc);
    end
    vF = mean(f1s,'omitnan');
catch
    vF = 0;
end
end


%% =========================================================================
%  FIX-6: Test-Time Augmentation
% =========================================================================
function probAvg = ttaPredict(dlnet, fname, inSz, execEnv, lInfo, auxNormed, nCrops)
% 5-crop + horizontal flip = 10 augmented views의 평균
if nargin < 7, nCrops = 5; end

I = imread(fname);
if ~isa(I,'uint8'), try I=im2uint8(I); catch, I=uint8(255*mat2gray(I)); end; end
if ismatrix(I), I=repmat(I,1,1,3); end
if size(I,3)>3, I=I(:,:,1:3); end

[h,w,~] = size(I);
ph = inSz(1); pw = inSz(2);
if h < ph || w < pw
    I = imresize(I, max(ph/h, pw/w));
    [h,w,~] = size(I);
end

crops = cell(nCrops*2, 1);
idx = 0;

% Center crop
tc = max(1, floor((h-ph)/2)+1);
lc = max(1, floor((w-pw)/2)+1);
cc = safeExtract(I, tc, lc, ph, pw);
idx=idx+1; crops{idx} = cc;
idx=idx+1; crops{idx} = fliplr(cc);

% 4 corner crops
corners = [1,1; 1,max(1,w-pw+1); max(1,h-ph+1),1; max(1,h-ph+1),max(1,w-pw+1)];
for ci = 1:min(4, nCrops-1)
    cr = safeExtract(I, corners(ci,1), corners(ci,2), ph, pw);
    idx=idx+1; crops{idx} = cr;
    idx=idx+1; crops{idx} = fliplr(cr);
end

nViews = idx;
imgB = zeros([inSz nViews],'single');
for vi = 1:nViews
    imgB(:,:,:,vi) = single(crops{vi}) / 255;
end

auxB = repmat(single(auxNormed(:)), 1, nViews);  % [11 × nViews]

dlImg = dlarray(imgB,'SSCB');
dlAux = dlarray(auxB,'CB');
if execEnv=="gpu", dlImg=gpuArray(dlImg); dlAux=gpuArray(dlAux); end

inputs = cell(1, numel(dlnet.InputNames));
inputs{lInfo.imgIdx} = dlImg;
inputs{lInfo.auxIdx} = dlAux;
pp = double(gather(extractdata(predict(dlnet, inputs{:}))));

probAvg = mean(pp, 2);  % [numC × 1]
end

function patch = safeExtract(I, t, l, ph, pw)
patch = I(t:min(t+ph-1,size(I,1)), l:min(l+pw-1,size(I,2)), :);
if size(patch,1)~=ph || size(patch,2)~=pw
    patch = imresize(patch, [ph pw]);
end
end


%% =========================================================================
%  평가 (v6: TTA 지원)
% =========================================================================
function [mOut, mSurf] = evaluateModel(dlnet, T, idxSp, classNames, opts, execEnv, rb, alpha, lInfo)
[allT,allPm,allPe] = predictBatch(dlnet,T,idxSp,classNames,opts,execEnv,rb,alpha,lInfo);
mOut.modelOnly = buildMetrics(allT,allPm,classNames);
mOut.ensemble  = buildMetrics(allT,allPe,classNames);
mSurf = struct();
for s = ["WET","DRY"]
    idxS = idxSp(T.surface(idxSp)==categorical(s));
    if numel(idxS) < 5, continue; end
    [tS,pM,pE] = predictBatch(dlnet,T,idxS,classNames,opts,execEnv,rb,alpha,lInfo);
    fn = char(lower(s));
    mSurf.(fn).modelOnly = buildMetrics(tS,pM,classNames);
    mSurf.(fn).ensemble  = buildMetrics(tS,pE,classNames);
end
end

function [allT,allPm,allPe] = predictBatch(dlnet,T,idxSp,classNames,opts,execEnv,rb,alpha,lInfo)
useTTA = isfield(opts,'UseTTA') && opts.UseTTA;
nTTA   = 5;
if isfield(opts,'TTACrops'), nTTA = opts.TTACrops; end

mb=32; allT=[]; allPm=[]; allPe=[];

if useTTA
    % TTA 모드: 개별 이미지 처리
    for ii = 1:numel(idxSp)
        bi = idxSp(ii);
        ci = find(classNames == T.grade(bi));
        if isempty(ci), ci = 0; end
        allT = [allT, ci]; %#ok<AGROW>

        mP = ttaPredict(dlnet, T.file(bi), opts.InputSize, execEnv, lInfo, T.auxNormed(bi,:), nTTA);
        auxRaw = double(T.auxParams(bi,:));
        rP = ruleClassify(auxRaw, rb);
        eP = (1-alpha)*mP + alpha*rP;
        eP = eP / max(sum(eP),1e-10);

        [~,pm] = max(mP); [~,pe] = max(eP);
        allPm = [allPm, pm]; %#ok<AGROW>
        allPe = [allPe, pe]; %#ok<AGROW>
    end
else
    % 기존 배치 모드
    for bat = 1:ceil(numel(idxSp)/mb)
        b1=(bat-1)*mb+1; b2=min(bat*mb,numel(idxSp));
        bIdx=idxSp(b1:b2); bsz=numel(bIdx);
        if bsz==0, continue; end

        imgB=zeros([opts.InputSize bsz],'single');
        for bi=1:bsz
            imgB(:,:,:,bi)=single(readPatch(T.file(bIdx(bi)),opts.InputSize,"val",0))/255;
        end
        auxB   = single(T.auxNormed(bIdx,:))';
        auxRaw = double(T.auxParams(bIdx,:));

        dlImg=dlarray(imgB,'SSCB'); dlAux=dlarray(auxB,'CB');
        if execEnv=="gpu", dlImg=gpuArray(dlImg); dlAux=gpuArray(dlAux); end

        inputs = cell(1, numel(dlnet.InputNames));
        inputs{lInfo.imgIdx} = dlImg;
        inputs{lInfo.auxIdx} = dlAux;
        mP = double(gather(extractdata(predict(dlnet, inputs{:}))));
        rP = ruleClassify(auxRaw, rb);
        eP = (1-alpha)*mP + alpha*rP;
        eP = eP ./ max(sum(eP,1), 1e-10);
        [~,pm]=max(mP,[],1); [~,pe]=max(eP,[],1);

        tI=zeros(1,bsz);
        for bi=1:bsz
            ci=find(classNames==T.grade(bIdx(bi)));
            if ~isempty(ci), tI(bi)=ci; end
        end
        allT=[allT,tI]; allPm=[allPm,pm]; allPe=[allPe,pe]; %#ok<AGROW>
    end
end
end


%% =========================================================================
%  Metrics
% =========================================================================
function M = buildMetrics(aT, aP, classNames)
K = numel(classNames);
M.accuracy = NaN; M.macroF1 = NaN; M.kappa = NaN;
M.confusion = zeros(K,K);
M.perClass  = table(string(classNames(:)),zeros(K,1),zeros(K,1),zeros(K,1),zeros(K,1), ...
    'VariableNames',{'class','support','precision','recall','f1'});
if isempty(aT) || isempty(aP), return; end
vm = aT > 0;
if sum(vm) < 2, return; end
try
    cm  = confusionmat(aT(vm)',aP(vm)','Order',1:K);
    acc = sum(diag(cm))/max(1,sum(cm,'all'));
    sup = sum(cm,2);
    prec=zeros(K,1); rec=zeros(K,1); f1=zeros(K,1);
    for k=1:K
        tp=cm(k,k); fp=sum(cm(:,k))-tp; fn=sum(cm(k,:))-tp;
        prec(k)=tp/max(1,tp+fp); rec(k)=tp/max(1,tp+fn);
        f1(k)=2*prec(k)*rec(k)/max(1e-12,prec(k)+rec(k));
    end
    M.accuracy = acc;
    M.macroF1  = mean(f1,'omitnan');
    M.confusion = cm;
    M.kappa     = cohenKappa(cm);
    M.perClass  = table(string(classNames(:)),sup,prec,rec,f1, ...
        'VariableNames',{'class','support','precision','recall','f1'});
catch
end
end

function k = cohenKappa(cm)
n = sum(cm,'all');
if n<=0, k=NaN; return; end
po = sum(diag(cm))/n;
pe = sum((sum(cm,2).*sum(cm,1)')/(n^2));
k  = (po-pe)/max(1e-12,1-pe);
end


%% =========================================================================
%  보조 파라미터 중요도
% =========================================================================
function info = analyzeColorParamWeights(dlnet, paramNames)
lrn  = dlnet.Learnables;
mask = (lrn.Layer=="aux_fc1") & (lrn.Parameter=="Weights");
if ~any(mask)
    nP = numel(paramNames);
    info = struct('W',[],'colNorm',ones(1,nP),'importance',ones(1,nP)/nP, ...
        'rankIdx',1:nP,'paramNames',{paramNames});
    warning('resnet18:NoAuxFC1','aux_fc1 Weights not found.');
    return;
end
W    = double(gather(extractdata(lrn.Value{find(mask,1)})));
colN = vecnorm(W,2,1);
imp  = colN / max(colN+eps);
[~,ri] = sort(imp,'descend');
info = struct('W',W,'colNorm',colN,'importance',imp,'rankIdx',ri,'paramNames',{paramNames});
fprintf('  [중요도 Top-5] ');
for k=1:min(5,numel(paramNames)), fprintf('%s(%.3f) ',paramNames{ri(k)},imp(ri(k))); end
fprintf('\n');
end


%% =========================================================================
%  색상 구간 통계
% =========================================================================
function qInfo = computeClassQuantiles(T, idxSet, classNames, paramNames)
numC = numel(classNames); numP = numel(paramNames);
qInfo = struct('paramNames',{paramNames},'classNames',string(classNames), ...
    'q25',zeros(numC,numP),'q50',zeros(numC,numP),'q75',zeros(numC,numP), ...
    'mean',zeros(numC,numP),'std',zeros(numC,numP),'n',zeros(numC,1));
if isempty(idxSet), return; end
auxRaw = double(T.auxParams(idxSet,:));
grades = T.grade(idxSet);
for c = 1:numC
    sub = auxRaw(grades==classNames(c),:);
    qInfo.n(c) = size(sub,1);
    if size(sub,1) < 4, continue; end
    qInfo.q25(c,:)  = quantile(sub,0.25,1);
    qInfo.q50(c,:)  = quantile(sub,0.50,1);
    qInfo.q75(c,:)  = quantile(sub,0.75,1);
    qInfo.mean(c,:) = mean(sub,1,'omitnan');
    qInfo.std(c,:)  = std(sub,0,1,'omitnan');
end
end


%% =========================================================================
%  Excel 저장
% =========================================================================
function saveExcelBundle(xlPath, T, idxVal, idxTest, classNames, paramNames, ...
    evalVal, evalTest, evalValSurf, evalTestSurf, ...
    paramImp, qTrain, qVal, qTest, runInfo, ...
    manualCenters, ruleBoundaries, manualAlpha, dlnet, opts, execEnv, lInfo)

if exist(xlPath,'file'), try delete(xlPath); catch, end; end

sumTbl = table(["VAL_model";"VAL_ens";"TEST_model";"TEST_ens"], ...
    [evalVal.modelOnly.accuracy;  evalVal.ensemble.accuracy;  evalTest.modelOnly.accuracy;  evalTest.ensemble.accuracy], ...
    [evalVal.modelOnly.macroF1;   evalVal.ensemble.macroF1;   evalTest.modelOnly.macroF1;   evalTest.ensemble.macroF1], ...
    [evalVal.modelOnly.kappa;     evalVal.ensemble.kappa;     evalTest.modelOnly.kappa;     evalTest.ensemble.kappa], ...
    'VariableNames',{'set','accuracy','macroF1','kappa'});
safeWrite(sumTbl, xlPath, 'SUMMARY');

safeWrite(evalVal.modelOnly.perClass,  xlPath, 'VAL_MODEL');
safeWrite(evalVal.ensemble.perClass,   xlPath, 'VAL_ENSEMBLE');
safeWrite(evalTest.modelOnly.perClass, xlPath, 'TEST_MODEL');
safeWrite(evalTest.ensemble.perClass,  xlPath, 'TEST_ENSEMBLE');

writeSurfMetrics(evalValSurf,  xlPath, 'VAL');
writeSurfMetrics(evalTestSurf, xlPath, 'TEST');

writeConfusionTbl(evalVal.modelOnly.confusion,  classNames, xlPath, 'CM_VAL_MODEL');
writeConfusionTbl(evalVal.ensemble.confusion,   classNames, xlPath, 'CM_VAL_ENS');
writeConfusionTbl(evalTest.modelOnly.confusion, classNames, xlPath, 'CM_TEST_MODEL');
writeConfusionTbl(evalTest.ensemble.confusion,  classNames, xlPath, 'CM_TEST_ENS');

rkArr = zeros(numel(paramNames),1);
[~,ri] = sort(paramImp.importance,'descend');
for k=1:numel(paramNames), rkArr(ri(k))=k; end
impTbl = table(string(paramNames(:)),paramImp.importance(:),paramImp.colNorm(:),rkArr, ...
    'VariableNames',{'parameter','importance_norm','fc1_l2norm','rank'});
safeWrite(sortrows(impTbl,'rank'), xlPath, 'PARAM_IMPORTANCE');

safeWrite(table(string(paramNames(:)),runInfo.auxNorm.mean(:),runInfo.auxNorm.std(:), ...
    'VariableNames',{'parameter','train_mean','train_std'}), xlPath, 'AUX_NORM_STATS');

writeColorQuantiles(qTrain, paramNames, xlPath, 'COLOR_TRAIN');
writeColorQuantiles(qVal,   paramNames, xlPath, 'COLOR_VAL');
writeColorQuantiles(qTest,  paramNames, xlPath, 'COLOR_TEST');

try
    safeWrite(table((1:5)',manualCenters.bstar(:),manualCenters.astar(:),manualCenters.S(:), ...
        'VariableNames',{'gradeIdx','bstar_center','astar_center','S_center'}), xlPath, 'RULE_CENTERS');
    safeWrite(table((1:4)',ruleBoundaries.bstar(:),ruleBoundaries.astar(:),ruleBoundaries.S(:), ...
        'VariableNames',{'boundary','bstar','astar','S'}), xlPath, 'RULE_BOUNDARIES');
catch
end

TpV = buildPredTable(dlnet,T,idxVal,  classNames,paramNames,opts,execEnv,ruleBoundaries,manualAlpha,lInfo);
TpT = buildPredTable(dlnet,T,idxTest, classNames,paramNames,opts,execEnv,ruleBoundaries,manualAlpha,lInfo);
safeWrite(TpV, xlPath, 'PRED_VAL');
safeWrite(TpT, xlPath, 'PRED_TEST');

if isstruct(runInfo.trainLog)
    tl = runInfo.trainLog;
    nEp = find(tl.lr > 0, 1, 'last');
    if ~isempty(nEp) && nEp > 0
        % v6: auxScale가 11차원이므로 주요 3개만 저장
        safeWrite(table((1:nEp)',tl.epochLoss(1:nEp),tl.epochAcc(1:nEp), ...
            tl.valLoss(1:nEp),tl.valAcc(1:nEp),tl.valMacroF1(1:nEp), ...
            tl.lr(1:nEp), ...
            tl.auxScale(1:nEp,8), tl.auxScale(1:nEp,9), tl.auxScale(1:nEp,5), ...
            'VariableNames',{'epoch','train_loss','train_acc','val_loss','val_acc', ...
            'val_macroF1','lr','auxScale_astar','auxScale_bstar','auxScale_S'}), xlPath, 'TRAIN_LOG');
    end
end

kv = {'runTag',    char(runInfo.runTag);
    'timestamp', runInfo.timestamp;
    'version',   'v6';
    'manualAlpha', manualAlpha;
    'focalGamma', runInfo.opts.FocalGamma;
    'colorJitterStrength', runInfo.opts.ColorJitterStrength;
    'auxNoiseStd', runInfo.opts.AuxNoiseStd;
    'freezeEpochs', runInfo.opts.FreezeEpochs;
    'backboneWarmupEpochs', runInfo.opts.BackboneWarmupEpochs;
    'cosineRestartPeriod', runInfo.opts.CosineRestartPeriod;
    'useTTA', runInfo.opts.UseTTA;
    'labelSmoothing', runInfo.opts.LabelSmoothing;
    'dropoutRate', runInfo.opts.DropoutRate;
    'images_usable', height(T);
    'train_n',  nnz(T.split=="train");
    'val_n',    nnz(T.split=="val");
    'test_n',   nnz(T.split=="test")};
if isstruct(runInfo.trainLog) && isfield(runInfo.trainLog,'bestEpoch') && ~isempty(runInfo.trainLog.bestEpoch)
    kv(end+1,:) = {'best_epoch',  runInfo.trainLog.bestEpoch};
    kv(end+1,:) = {'best_val_f1', runInfo.trainLog.bestValF1};
end
safeWriteCell(kv, xlPath, 'RUN_INFO');
end

function writeSurfMetrics(S, xlPath, prefix)
try
    if ~isstruct(S), return; end
    fns = fieldnames(S);
    for i=1:numel(fns)
        fn=fns{i};
        if isfield(S.(fn),'ensemble'),  safeWrite(S.(fn).ensemble.perClass,  xlPath, sprintf('%s_%s_ENS',prefix,upper(fn))); end
        if isfield(S.(fn),'modelOnly'), safeWrite(S.(fn).modelOnly.perClass, xlPath, sprintf('%s_%s_MODEL',prefix,upper(fn))); end
    end
catch
end
end

function writeConfusionTbl(cm, classNames, xlPath, sheetName)
try
    if isempty(cm)||all(cm(:)==0), return; end
    clsStr = string(classNames(:))';
    safeWrite(array2table(cm,'RowNames',cellstr(clsStr),'VariableNames',cellstr("pred_"+clsStr)), xlPath, sheetName);
catch
end
end

function writeColorQuantiles(qInfo, paramNames, xlPath, sheetName)
try
    numC=numel(qInfo.classNames); numP=numel(paramNames);
    rows=cell(numC*numP,8); row=0;
    for c=1:numC
        for pp=1:numP
            row=row+1;
            rows{row,1}=char(qInfo.classNames(c)); rows{row,2}=paramNames{pp};
            rows{row,3}=qInfo.n(c);
            rows{row,4}=qInfo.q25(c,pp); rows{row,5}=qInfo.q50(c,pp);
            rows{row,6}=qInfo.q75(c,pp); rows{row,7}=qInfo.mean(c,pp);
            rows{row,8}=qInfo.std(c,pp);
        end
    end
    if row==0, return; end
    safeWrite(cell2table(rows(1:row,:),'VariableNames', ...
        {'grade','param','n','Q25','Q50_median','Q75','mean','std'}), xlPath, sheetName);
catch ME
    warning('resnet18:colorQ','%s: %s', sheetName, ME.message);
end
end

function Tp = buildPredTable(dlnet, T, idxSet, ~, paramNames, opts, execEnv, rb, alpha, lInfo)
N=numel(idxSet);
if N==0, Tp=table(); return; end
mb=32;
trueG=strings(N,1); surf=strings(N,1);
fileArr=strings(N,1); stemArr=strings(N,1); grpArr=strings(N,1);
predM=strings(N,1); predR=strings(N,1); predE=strings(N,1);
pM=zeros(N,5); pR=zeros(N,5); pE=zeros(N,5);
auxRawAll=double(T.auxParams(idxSet,:));

for i=1:N
    fileArr(i)=string(T.file(idxSet(i)));
    stemArr(i)=string(T.stem(idxSet(i)));
    grpArr(i) =string(T.group(idxSet(i)));
    trueG(i)  =string(T.grade(idxSet(i)));
    surf(i)   =string(T.surface(idxSet(i)));
end

for bat=1:ceil(N/mb)
    b1=(bat-1)*mb+1; b2=min(bat*mb,N);
    loc=b1:b2; bIdx=idxSet(loc); bsz=numel(bIdx);
    if bsz==0, continue; end

    imgB=zeros([opts.InputSize bsz],'single');
    for bi=1:bsz
        imgB(:,:,:,bi)=single(readPatch(T.file(bIdx(bi)),opts.InputSize,"val",0))/255;
    end
    auxB=single(T.auxNormed(bIdx,:))';
    rawB=double(T.auxParams(bIdx,:));

    dlImg=dlarray(imgB,'SSCB'); dlAux=dlarray(auxB,'CB');
    if execEnv=="gpu", dlImg=gpuArray(dlImg); dlAux=gpuArray(dlAux); end

    inputs=cell(1,numel(dlnet.InputNames));
    inputs{lInfo.imgIdx}=dlImg; inputs{lInfo.auxIdx}=dlAux;
    mProb=double(gather(extractdata(predict(dlnet,inputs{:}))));
    rProb=ruleClassify(rawB,rb);
    eProb=(1-alpha)*mProb+alpha*rProb;
    eProb=eProb./max(sum(eProb,1),1e-10);
    [~,pm]=max(mProb,[],1); [~,pr]=max(rProb,[],1); [~,pe]=max(eProb,[],1);

    for bi=1:bsz
        ii=loc(bi);
        predM(ii)="D"+string(pm(bi)); predR(ii)="D"+string(pr(bi)); predE(ii)="D"+string(pe(bi));
        pM(ii,:)=mProb(:,bi)'; pR(ii,:)=rProb(:,bi)'; pE(ii,:)=eProb(:,bi)';
    end
end

Tp=table(fileArr,stemArr,grpArr,surf,trueG,predM,predR,predE, ...
    pM(:,1),pM(:,2),pM(:,3),pM(:,4),pM(:,5), ...
    pE(:,1),pE(:,2),pE(:,3),pE(:,4),pE(:,5), ...
    'VariableNames',{'file','stem','group','surface','true_grade', ...
    'pred_model','pred_rule','pred_ensemble', ...
    'm_pD1','m_pD2','m_pD3','m_pD4','m_pD5', ...
    'e_pD1','e_pD2','e_pD3','e_pD4','e_pD5'});
auxTbl=array2table(auxRawAll,'VariableNames',cellstr(matlab.lang.makeValidName(string(paramNames(:)))));
Tp=[Tp auxTbl];
end


%% =========================================================================
%  OOM-safe 재시도
% =========================================================================
function [dlnet, trainLog, execEnv, opts] = trainWithRetry(dlnet, T, idxTrainBal, idxVal, classNames, classWeights, lInfo, opts, execEnv)
mb0=opts.MiniBatchSize; ga0=opts.GradAccumSteps;
for attempt=1:5
    try
        [dlnet, trainLog] = customTrainLoop(dlnet, T, idxTrainBal, idxVal, ...
            classNames, classWeights, lInfo, opts, execEnv);
        return;
    catch ME
        if isOomError(ME)
            if execEnv=="gpu"
                try g=gpuDevice; wait(g); reset(g); execEnv=pickExecEnv(opts.ExecutionEnvironment); catch, end
            end
            if opts.MiniBatchSize > 2
                opts.MiniBatchSize  = max(2, floor(opts.MiniBatchSize/2));
                opts.GradAccumSteps = max(1, ceil((mb0*ga0)/opts.MiniBatchSize));
                fprintf('  OOM → MB=%d Acc=%d (시도 %d)\n', opts.MiniBatchSize, opts.GradAccumSteps, attempt);
                continue;
            end
            if execEnv=="gpu"
                execEnv="cpu"; opts.MiniBatchSize=mb0; opts.GradAccumSteps=ga0;
                fprintf('  GPU OOM → CPU fallback (시도 %d)\n', attempt);
                continue;
            end
        end
        fprintf('  학습 오류 (시도 %d): %s\n', attempt, ME.message);
        rethrow(ME);
    end
end
error('resnet18:RetryFailed','학습 재시도 5회 초과.');
end

function tf = isOomError(ME)
msg=lower(string(ME.message)); id=lower(string(ME.identifier));
tf=contains(msg,["out of memory","insufficient memory","cuda","memory"]) || ...
   contains(id, ["outofmemory","oom","gpu"]);
end


%% =========================================================================
%  Predict-only
% =========================================================================
function runInfo = doPredictOnly(opts, outRoot, ~, manualCenters, manualAlpha)
runInfo = struct('mode','predict');
S = load(string(opts.NetMatPath));
if ~isfield(S,'netTrained'), error('resnet18:NoNet','netTrained not found.'); end
dlnet = S.netTrained;

if isfield(S,'runInfo') && isfield(S.runInfo,'auxNorm')
    auxMean=S.runInfo.auxNorm.mean; auxStd=S.runInfo.auxNorm.std;
else
    auxMean=[120 110 100 30 0.25 0.48 46 0 3 4 0.5];
    auxStd =[30  30  30  60 0.15 0.04 12 2 4  3 0.5];
    warning('resnet18:NoAuxNorm','기본 정규화값 사용.');
end

if isfield(S,'runInfo') && isfield(S.runInfo,'layerInfo')
    lInfo = S.runInfo.layerInfo;
else
    lInfo = struct('imgIdx',1,'auxIdx',2,'imgInputName','data');
end

% v6: 저장된 data-driven centers 사용
if isfield(S,'runInfo') && isfield(S.runInfo,'manualCenters')
    manualCenters = S.runInfo.manualCenters;
end

imgFiles=collectImageFiles(string(opts.PredictImageDir));
if isempty(imgFiles), error('resnet18:NoPredImg','이미지 없음.'); end
rb=buildRuleBoundaries(manualCenters);
execEnv=pickExecEnv(opts.ExecutionEnvironment);
inSz=opts.InputSize; mb=16; N=numel(imgFiles);
results=cell(N,1);

for bat=1:ceil(N/mb)
    b1=min((bat-1)*mb+1,N); b2=min(bat*mb,N);
    bIdx=b1:b2; bsz=numel(bIdx);
    imgB=zeros([inSz bsz],'single');
    auxB=zeros(11,bsz,'single'); auxRaw=zeros(bsz,11,'double');

    for bi=1:bsz
        img=readPatch(imgFiles(bIdx(bi)),inSz,"val",0);
        imgB(:,:,:,bi)=single(img)/255;

        isWet=0;
        [~,st,~]=fileparts(imgFiles(bIdx(bi)));
        if parseSurf(splitTok(st))=="WET", isWet=1; end

        raw = extractAuxFromPatch(img, isWet);
        auxRaw(bi,:) = double(raw);
        auxB(:,bi) = single((double(raw) - auxMean) ./ auxStd)';
    end

    dlImg=dlarray(imgB,'SSCB'); dlAux=dlarray(auxB,'CB');
    if execEnv=="gpu", dlImg=gpuArray(dlImg); dlAux=gpuArray(dlAux); end

    inputs=cell(1,numel(dlnet.InputNames));
    inputs{lInfo.imgIdx}=dlImg; inputs{lInfo.auxIdx}=dlAux;
    mP=double(gather(extractdata(predict(dlnet,inputs{:}))));
    rP=ruleClassify(auxRaw,rb);
    eP=(1-manualAlpha)*mP+manualAlpha*rP;
    eP=eP./max(sum(eP,1),1e-10);
    [~,pe]=max(eP,[],1);

    for bi=1:bsz
        results{bIdx(bi)}=struct('file',imgFiles(bIdx(bi)),'pred',"D"+string(pe(bi)),'scores',eP(:,bi)');
    end
end

vm=~cellfun(@isempty,results);
imgF2=imgFiles(vm); res2=results(vm);
Tp=table(imgF2,string(cellfun(@(r)r.pred,res2,'UniformOutput',false)),'VariableNames',{'file','pred_grade'});
for k=1:5, Tp.("p_D"+string(k))=cellfun(@(r)r.scores(k),res2); end

xlOut=fullfile(outRoot,'predict_results.xlsx');
if strlength(string(opts.PredictOutXlsx))>0, xlOut=char(opts.PredictOutXlsx); end
safeWrite(Tp, xlOut, 'PRED');
fprintf('Predict XLSX: %s\n', xlOut);
runInfo.xlsxPath=xlOut;
end


%% =========================================================================
%  규칙 분류기
% =========================================================================
function rb = buildRuleBoundaries(mc)
fields=fieldnames(mc); rb=struct();
for fi=1:numel(fields)
    fn=fields{fi}; ctr=mc.(fn); bnd=zeros(1,4);
    for k=1:4, bnd(k)=(ctr(k)+ctr(k+1))/2; end
    rb.(fn)=bnd;
end
end

function rProb = ruleClassify(auxRaw, rb)
N=size(auxRaw,1); votes=zeros(N,5);
pMap=struct('bstar',9,'astar',8,'S',5);
fields=fieldnames(rb);
for fi=1:numel(fields)
    fn=fields{fi}; col=pMap.(fn); bnd=rb.(fn);
    vals=double(auxRaw(:,col));
    for n=1:N
        v=vals(n);
        if     v<bnd(1), cls=1;
        elseif v<bnd(2), cls=2;
        elseif v<bnd(3), cls=3;
        elseif v<bnd(4), cls=4;
        else,            cls=5;
        end
        votes(n,cls)=votes(n,cls)+1;
    end
end
votes=votes+0.1;
rProb=(votes./sum(votes,2))';
end


%% =========================================================================
%  보조 파라미터 추출
% =========================================================================
function T = extractAuxColorParams(T)
n=height(T); auxP=zeros(n,11,'single');
fList=string(T.file); sList=string(T.surface);
fprintf('  추출 중 (%d 이미지)...\n', n);
parfor i=1:n
    try
        img=imread(fList(i));
        if ~isa(img,'uint8'), try img=im2uint8(img); catch, img=uint8(255*mat2gray(img)); end; end
        if ismatrix(img),   img=repmat(img,1,1,3); end
        if size(img,3)>3,   img=img(:,:,1:3); end
        [h,w,~]=size(img); mg=0.10;
        r1=max(1,round(h*mg)); r2=min(h,round(h*(1-mg)));
        c1=max(1,round(w*mg)); c2=min(w,round(w*(1-mg)));
        roi=img(r1:r2,c1:c2,:);

        isWet=double(strcmpi(sList(i),'WET'));
        auxP(i,:) = extractAuxFromPatch(roi, isWet);
    catch
        isWet=double(strcmpi(sList(i),'WET'));
        auxP(i,:)=single([120,110,100,30,0.25,0.48,46,0,3,4,isWet]);
    end
end
T.auxParams=auxP;
fprintf('  완료: %d × 11\n', n);
end


%% =========================================================================
%  IO 유틸리티
% =========================================================================
function Iout = readPatch(fname, inSz, mode, jitterStrength)
if nargin < 4, jitterStrength = 0; end
I=imread(fname);
if ~isa(I,'uint8'), try I=im2uint8(I); catch, I=uint8(255*mat2gray(I)); end; end
if ismatrix(I),  I=repmat(I,1,1,3); end
if size(I,3)>3,  I=I(:,:,1:3); end
[h,w,~]=size(I); ph=inSz(1); pw=inSz(2);
if h<ph||w<pw, I=imresize(I,max(ph/h,pw/w)); [h,w,~]=size(I); end

switch lower(string(mode))
    case "train"
        t=randi([1,max(1,h-ph+1)]); l=randi([1,max(1,w-pw+1)]);
        if rand>0.5, I=fliplr(I); end
        if rand>0.5, I=flipud(I); end
        if rand>0.3
            I=imrotate(I,(rand-0.5)*4,'bilinear','crop');
            [h,w,~]=size(I);
            t=randi([1,max(1,h-ph+1)]); l=randi([1,max(1,w-pw+1)]);
        end
        Iout=I(t:min(t+ph-1,h),l:min(l+pw-1,w),:);
        if size(Iout,1)~=ph||size(Iout,2)~=pw, Iout=imresize(Iout,[ph pw]); end
        if jitterStrength > 0
            Iout = colorJitterAug(Iout, jitterStrength);
        end
    otherwise
        t=max(1,floor((h-ph)/2)+1); l=max(1,floor((w-pw)/2)+1);
        Iout=I(t:min(t+ph-1,h),l:min(l+pw-1,w),:);
        if size(Iout,1)~=ph||size(Iout,2)~=pw, Iout=imresize(Iout,[ph pw]); end
end
end

function files = collectImageFiles(roots)
roots=string(roots); exts=[".jpg",".jpeg",".png",".tif",".tiff",".bmp"];
chunks={};
for r=1:numel(roots)
    rd=roots(r); if ~isfolder(rd), continue; end
    for e=1:numel(exts)
        dd=dir(fullfile(rd,"**","*"+exts(e)));
        if ~isempty(dd), chunks{end+1,1}=string(fullfile({dd.folder},{dd.name}))'; end %#ok<AGROW>
    end
end
if isempty(chunks), files=strings(0,1); else, files=unique(vertcat(chunks{:}),'stable'); end
end

function jMap = buildJsonMap(roots)
roots=string(roots); chunks={};
for r=1:numel(roots)
    rd=roots(r); if ~isfolder(rd), continue; end
    dd=dir(fullfile(rd,"**","*.json"));
    if ~isempty(dd), chunks{end+1,1}=string(fullfile({dd.folder},{dd.name}))'; end %#ok<AGROW>
end
if isempty(chunks), jf=strings(0,1); else, jf=unique(vertcat(chunks{:}),'stable'); end
jMap=containers.Map('KeyType','char','ValueType','char');
for i=1:numel(jf)
    [~,st,~]=fileparts(jf(i)); k=char(string(st));
    if ~isKey(jMap,k), jMap(k)=char(jf(i)); end
end
end

function [T,info] = buildMetaTable(imgFiles,jsonMap,gradeKeys,surfaceKeys,requireJSON,useSurfJSON,nTok)
n=numel(imgFiles);
fileArr=strings(n,1); stemArr=strings(n,1);
gradeArr=strings(n,1); surfArr=strings(n,1); grpArr=strings(n,1);
mJ=0; mG=0; mS=0; bJ=0; uSJ=0; keep=true(n,1);
for i=1:n
    f=imgFiles(i); fileArr(i)=string(f);
    [~,st,~]=fileparts(f); stemArr(i)=string(st);
    toks=splitTok(st); s=parseSurf(toks);
    if strlength(s)==0, surfArr(i)=missing; else, surfArr(i)=s; end
    if numel(toks)>=nTok, grpArr(i)=join(toks(1:nTok),"-"); else, grpArr(i)=string(st); end
    if ~isKey(jsonMap,char(stemArr(i)))
        mJ=mJ+1; if requireJSON, keep(i)=false; end; continue;
    end
    try
        J=jsondecode(fileread(string(jsonMap(char(stemArr(i))))));
        [g,~]=extractGrade(J,gradeKeys);
        if strlength(g)==0, mG=mG+1; keep(i)=false; continue; end
        gradeArr(i)=g;
        if ismissing(surfArr(i))&&useSurfJSON
            s2=extractSurf(J,surfaceKeys);
            if strlength(s2)>0, surfArr(i)=s2; uSJ=uSJ+1; end
        end
        if ismissing(surfArr(i)), mS=mS+1; keep(i)=false; continue; end
    catch, bJ=bJ+1; keep(i)=false;
    end
end
T=table(fileArr(keep),stemArr(keep),gradeArr(keep),surfArr(keep),categorical(grpArr(keep)), ...
    'VariableNames',{'file','stem','grade','surface','group'});
info=struct('total',n,'kept',height(T),'missJSON',mJ,'badJSON',bJ,'missGrade',mG,'missSurf',mS,'usedSurfJSON',uSJ);
end

function toks = splitTok(stem)
tmp=regexp(stem,'[-_]+','split'); tmp=tmp(~cellfun(@isempty,tmp)); toks=string(tmp(:));
end

function s = parseSurf(toks)
s="";
if numel(toks)>=7
    t7=upper(strtrim(toks(7)));
    if t7=="D"||t7=="DRY", s="DRY"; return; end
    if t7=="W"||t7=="WET", s="WET"; return; end
end
t=upper(strtrim(toks));
iW=(t=="W")|(t=="WET")|(t=="SAT"); iD=(t=="D")|(t=="DRY");
if any(iW)&&~any(iD),        s="WET";
elseif any(iD)&&~any(iW),    s="DRY";
elseif any(iW)&&any(iD)
    if find(iW,1)<find(iD,1), s="WET"; else, s="DRY"; end
end
end

function [g,kf] = extractGrade(J,gradeKeys)
for k=1:numel(gradeKeys)
    [tf,val]=gField(J,gradeKeys(k));
    if tf, [g2,ok]=nGrade(val); if ok, g=g2; kf=gradeKeys(k); return; end; end
end
try [g,kf]=rGrade(J,0,4); catch, g=""; kf=""; end
end

function s = extractSurf(J,surfKeys)
for k=1:numel(surfKeys)
    [tf,val]=gField(J,surfKeys(k));
    if tf, s2=nSurf(val); if strlength(s2)>0, s=s2; return; end; end
end
try s=rSurf(J,0,3); catch, s=""; end
end

function [tf,val] = gField(S,key)
tf=false; val=[];
if ~isstruct(S), return; end
fn=fieldnames(S); ix=find(strcmpi(fn,char(key)),1);
if isempty(ix), return; end
tf=true; val=S.(fn{ix});
end

function [g,ok] = nGrade(val)
ok=false; g="";
if isempty(val), return; end
if isnumeric(val)
    v=val(1);
    if isfinite(v)&&ismember(round(v),1:5), g="D"+string(round(v)); ok=true; end
    return;
end
v=upper(replace(string(val)," ",""));
if startsWith(v,"D")&&strlength(v)>=2
    d=extractBetween(v,2,2);
    if any(d==["1","2","3","4","5"]), g="D"+d; ok=true; return; end
end
if any(v==["1","2","3","4","5"]), g="D"+v; ok=true; return; end
tok=regexp(char(v),'(?:WD|RW|WEATHERING)[-_ ]*([1-5])','tokens','once');
if ~isempty(tok), g="D"+string(tok{1}); ok=true; end
end

function s = nSurf(val)
s=""; if isempty(val), return; end
v=upper(replace(string(val)," ",""));
if any(v==["W","WET","SAT","S","TRUE","1"]), s="WET";
elseif any(v==["D","DRY","FALSE","0"]), s="DRY"; end
end

function [g,kf] = rGrade(S,dep,maxD)
g=""; kf="";
if dep>maxD||~isstruct(S), return; end
fn=fieldnames(S);
for i=1:numel(fn)
    val=S.(fn{i}); low=lower(fn{i});
    if contains(low,"rw")||contains(low,"weather")||contains(low,"grade")
        [g2,ok]=nGrade(val); if ok, g=g2; kf=string(fn{i}); return; end
    end
    if isstruct(val)
        [g2,k2]=rGrade(val,dep+1,maxD);
        if strlength(g2)>0, g=g2; kf=string(fn{i})+"."+k2; return; end
    end
end
end

function s = rSurf(S,dep,maxD)
s=""; if dep>maxD||~isstruct(S), return; end
fn=fieldnames(S);
for i=1:numel(fn)
    val=S.(fn{i}); low=lower(fn{i});
    if contains(low,"wet")||contains(low,"dry")||contains(low,"surface")||contains(low,"cond")
        s2=nSurf(val); if strlength(s2)>0, s=s2; return; end
    end
    if isstruct(val), s2=rSurf(val,dep+1,maxD); if strlength(s2)>0, s=s2; return; end; end
end
end

function [splitVec,info] = makeGroupStratifiedSplit(T,rTr,rVa,rTe,minV,minT)
rs=rTr+rVa+rTe;
if abs(rs-1)>1e-6, rTr=rTr/rs; rVa=rVa/rs; rTe=rTe/rs; end
[G,~]=findgroups(T.group); ng=max(G);
gGr=strings(ng,1); gSu=strings(ng,1);
for gi=1:ng
    idx=(G==gi);
    gGr(gi)=string(mode(categorical(string(T.grade(idx)))));
    gSu(gi)=string(mode(categorical(string(T.surface(idx)))));
end
st=categorical(gGr+"_"+gSu);
cv1=cvpartition(st,'HoldOut',rTe); isTestG=test(cv1);
cv2=cvpartition(st(~isTestG),'HoldOut',rVa/(rTr+rVa)); tmp=test(cv2);
isValG=false(ng,1); isValG(~isTestG)=tmp;
splitVec=categorical(repmat("train",height(T),1),["train","val","test"]);
splitVec(ismember(G,find(isValG)))="val";
splitVec(ismember(G,find(isTestG)))="test";
clsC=categories(T.grade);
splitVec=forceMin(splitVec,T,clsC,"val",minV);
splitVec=forceMin(splitVec,T,clsC,"test",minT);
info=struct('groups',ng,'train',nnz(splitVec=="train"),'val',nnz(splitVec=="val"),'test',nnz(splitVec=="test"));
end

function splitVec = forceMin(splitVec,T,cls,tgt,mn)
if mn<=0, return; end
for c=1:numel(cls)
    nNow=nnz(splitVec==tgt & T.grade==cls{c});
    if nNow>=mn, continue; end
    idxTr=find(splitVec=="train" & T.grade==cls{c});
    if isempty(idxTr), continue; end
    gs=unique(T.group(idxTr),'stable'); mv=0;
    for g=1:numel(gs)
        if mv>=mn-nNow, break; end
        idxG=(T.group==gs(g))&(splitVec=="train");
        splitVec(idxG)=tgt; mv=mv+nnz(T.grade(idxG)==cls{c});
    end
end
end

function [idxOut,info] = balanceTrainIndices(T,idxTr,classNames,opts)
info=struct('mode',string(opts.BalanceTrain));
if string(opts.BalanceTrain)=="off", idxOut=idxTr; return; end
y=T.grade(idxTr);
cnt=countcats(categorical(y,categories(classNames)));
valid=cnt(cnt>0); if isempty(valid), idxOut=idxTr; return; end
switch string(opts.BalanceTargetMode)
    case "median",   t0=round(median(valid));
    case "quantile", t0=round(quantile(valid,opts.BalanceQuantile));
    case "max",      t0=max(valid);
    case "min",      t0=min(valid);
    otherwise,       t0=round(quantile(valid,0.5));
end
t0=max(1,t0); idxOut=[];
for k=1:numel(classNames)
    idxk=idxTr(y==classNames(k)); n=numel(idxk); if n==0, continue; end
    tT=min(t0,round(n*opts.BalanceUpCap));
    if n>=tT, pick=idxk(randperm(n,tT)); else, pick=[idxk;idxk(randi(n,tT-n,1))]; end
    idxOut=[idxOut;pick(:)]; %#ok<AGROW>
end
idxOut=idxOut(randperm(numel(idxOut)));
info.target=t0;
info.before=cnt;
info.after=countcats(categorical(T.grade(idxOut),categories(classNames)));
fprintf('  [밸런싱 결과] ');
for k=1:numel(classNames)
    fprintf('%s:%d ', string(classNames(k)), info.after(k));
end
fprintf('\n');
end

function env = pickExecEnv(req)
req=lower(string(req));
if req=="cpu", env="cpu";
elseif req=="gpu", env="gpu";
else
    try
        if canUseGPU
            env="gpu";
            g = gpuDevice;
            fprintf('  GPU 감지: %s (%.1f GB)\n', g.Name, g.TotalMemory/1e9);
        else
            env="cpu";
            fprintf('  ⚠ GPU 사용 불가 — canUseGPU=false\n');
            fprintf('    gpuDeviceCount=%d | MATLAB에서 GPU 지원 확인: gpuDevice\n', gpuDeviceCount);
        end
    catch ME
        env="cpu";
        fprintf('  ⚠ GPU 감지 실패: %s\n', ME.message);
    end
end
end

function safeWrite(tbl, xlPath, sheetName)
maxRetry=5;
for k=1:maxRetry
    try writetable(tbl, xlPath,'Sheet',sheetName); return;
    catch ME
        if k==maxRetry
            warning('resnet18:xls','[%s] 실패: %s', sheetName, ME.message);
        else
            pause(1.5);
        end
    end
end
end

function safeWriteCell(c, xlPath, sheetName)
maxRetry=5;
for k=1:maxRetry
    try writecell(c, xlPath,'Sheet',sheetName); return;
    catch ME
        if k==maxRetry
            warning('resnet18:xls','[%s] 실패: %s', sheetName, ME.message);
        else
            pause(1.5);
        end
    end
end
end

function mkdirSafe(d), if ~exist(d,'dir'), mkdir(d); end; end

function s = addCommas(n)
s=char(string(n)); idx=strfind(s,'.');
if isempty(idx), idx=numel(s)+1; end
for i=idx-4:-3:1, s=[s(1:i),',',s(i+1:end)]; end
end

function printStep(n, msg, varargin)
if nargin > 2
    msg = sprintf(msg, varargin{:});
end
fprintf('\n%s\n[STEP %2d] %s\n%s\n',repmat('-',1,72),n,msg,repmat('-',1,72));
end