function runInfo = rocktype_resnet18_train_3class(varargin)
% Granite_AE_ResNet50_Train  (ONE-FILE, R2025a/b)
% =========================================================================
%  ResNet-50 Autoencoder 기반 화강암 판별 모델
%
%  원리: 화강암 코어 이미지만으로 복원 학습 → 비화강암 입력 시 복원 오차 증가
%  활용: 타일 단위 화강암 확률 히트맵 → ROI 필터링
%
%  구조:
%    Encoder : ResNet-50 pretrained → conv4 (14×14×1024)
%    Bottleneck: 1×1 Conv (1024→256)
%    Decoder : TransposedConv 4단 (14→28→56→112→224) + sigmoid
%    Loss    : MSE
%    Metric  : SSIM (모니터링)
%
%  학습 (2-Phase, 총 30 epoch, 조기종료 없음):
%    Phase 1 — Encoder Frozen, Decoder only    :  8 epoch, LR 1e-3
%    Phase 2 — conv3+conv4 Unfreeze + Decoder  : 22 epoch, LR 5e-5/1e-4
%
%  Augmentation: RandomFlip(LR+UD), RandomRotation(±15°),
%                ColorJitter, GaussianNoise, RandomResizedCrop
%
%  GPU: RTX 5060 8GB, Batch 8 (OOM 시 4 자동 폴백)
% =========================================================================

%% ===== [0] 옵션 =====
OPT = parseOpts(varargin{:});

tstamp  = string(datetime('now','Format','yyyyMMdd_HHmmss'));
RUN_TAG = "Granite_AE_ResNet50_RUN_" + tstamp;
outRoot = fullfile(char(OPT.RESULTS_ROOT), char(RUN_TAG));
ckptDir = fullfile(outRoot,'checkpoints');
logDir  = fullfile(outRoot,'log');
figDir  = fullfile(outRoot,'figures');
ensureDirs({outRoot, ckptDir, logDir, figDir});

try diary off; end %#ok<TRYNC>
diary(fullfile(logDir,'run_log.txt')); diary on;

fprintf('\n%s\n',repmat('=',1,76));
fprintf('  Granite_AE_ResNet50_Train\n');
fprintf('  결과: %s\n', outRoot);
fprintf('%s\n\n',repmat('=',1,76));

%% ===== [1] GPU 초기화 =====
fprintf('[1] GPU 초기화\n');
[gpuOK, gpuName] = initGPU(OPT.USE_GPU);
execEnv = 'auto';
if ~gpuOK, execEnv = 'cpu'; end
fprintf('  GPU: %s | ExecEnv: %s\n\n', gpuName, execEnv);

%% ===== [2] 이미지 수집 =====
fprintf('[2] 이미지 수집 (화강암 전용)\n');
allFiles = collectImages(OPT.IMAGE_ROOTS, OPT.IMG_EXTS);
nTotal   = numel(allFiles);
fprintf('  합계: %d장\n\n', nTotal);
if nTotal == 0
    error('Granite_AE:NoImages','이미지를 찾지 못했습니다.');
end

%% ===== [3] Group-exclusive Split (시추공 단위) =====
fprintf('[3] Group-exclusive Split\n');
rng(OPT.SEED);
groups = cellfun(@(f) parseBorehole(f), allFiles, 'UniformOutput', false);
[trainIdx, valIdx, testIdx] = groupExclusiveSplit(groups, ...
    OPT.SPLIT_TRAIN, OPT.SPLIT_VAL, OPT.SPLIT_TEST);

trainFiles = allFiles(trainIdx);
valFiles   = allFiles(valIdx);
testFiles  = allFiles(testIdx);
nTrain = numel(trainFiles);
nVal   = numel(valFiles);
nTest  = numel(testFiles);

uGrpTrain = numel(unique(groups(trainIdx)));
uGrpVal   = numel(unique(groups(valIdx)));
uGrpTest  = numel(unique(groups(testIdx)));

fprintf('  Train: %d장 (%d groups)\n', nTrain, uGrpTrain);
fprintf('  Val  : %d장 (%d groups)\n', nVal,   uGrpVal);
fprintf('  Test : %d장 (%d groups)\n', nTest,  uGrpTest);

% split 보고서 저장
writeSplitReport(fullfile(logDir,'split_report.txt'), ...
    nTotal, nTrain, nVal, nTest, uGrpTrain, uGrpVal, uGrpTest);
fprintf('\n');

%% ===== [4] Encoder + Decoder 구성 =====
fprintf('[4] 네트워크 구성\n');
fprintf('  ResNet-50 로드 중...\n');
[encNet, encOutLayer, encOutCh] = buildEncoder();
fprintf('  Encoder: → %s [14×14×%d]\n', encOutLayer, encOutCh);

decNet = buildDecoder(encOutCh);
fprintf('  Decoder: %d layers\n', numel(decNet.Layers));
fprintf('  Encoder params: %s\n', formatParams(encNet));
fprintf('  Decoder params: %s\n\n', formatParams(decNet));

%% ===== [5] 학습 =====
fprintf('[5] 학습 시작 (Phase1: %d ep + Phase2: %d ep = 총 %d ep)\n', ...
    OPT.PHASE1_EPOCHS, OPT.PHASE2_EPOCHS, OPT.PHASE1_EPOCHS+OPT.PHASE2_EPOCHS);
fprintf('  Batch: %d | Augmentation: ON\n\n', OPT.BATCH_SIZE);

% --- 학습 상태 초기화 ---
history = struct('epoch',[],'phase',[],'trainLoss',[],'valLoss',[],'valSSIM',[]);
totalEpochs = OPT.PHASE1_EPOCHS + OPT.PHASE2_EPOCHS;
batchSize   = OPT.BATCH_SIZE;
globalEpoch = 0;

% Adam 상태 (decoder)
avgGradDec  = []; avgSqGradDec  = [];
% (encoder Adam 상태는 Phase 2 시작 시 초기화)

% Unfreeze 마스크 (Phase 2)
unfreezeIdx = findConv3Conv4Params(encNet);

% OOM 재시도 루프
trainOK = false;
while ~trainOK
    try
        % Datastore 생성
        imdsTrain = imageDatastore(trainFiles, ...
            'ReadFcn', @(f) trainReadFcn(f, [224 224]));
        imdsVal   = imageDatastore(valFiles, ...
            'ReadFcn', @(f) valReadFcn(f, [224 224]));

        mbqTrain = minibatchqueue(imdsTrain, ...
            'MiniBatchSize', batchSize, ...
            'PartialMiniBatch', 'discard', ...
            'OutputAsDlarray', true, ...
            'MiniBatchFormat', {'SSCB'});

        mbqVal = minibatchqueue(imdsVal, ...
            'MiniBatchSize', batchSize, ...
            'PartialMiniBatch', 'discard', ...
            'OutputAsDlarray', true, ...
            'MiniBatchFormat', {'SSCB'});

        itersPerEpoch = floor(nTrain / batchSize);
        iterGlobal = 0;
        t0 = tic;

        %% --- Phase 1: Encoder Frozen ---
        fprintf('\n  === Phase 1: Encoder Frozen (%d epochs, LR=%.0e) ===\n', ...
            OPT.PHASE1_EPOCHS, OPT.LR_PHASE1_DEC);

        for ep = 1:OPT.PHASE1_EPOCHS
            globalEpoch = globalEpoch + 1;
            shuffle(mbqTrain);
            epochLoss = 0; nBatch = 0;

            while hasdata(mbqTrain)
                X = next(mbqTrain);
                iterGlobal = iterGlobal + 1;

                [loss, gDec, stDec] = dlfeval( ...
                    @modelLossFrozen, encNet, decNet, X, encOutLayer);
                decNet.State = stDec;

                [decNet, avgGradDec, avgSqGradDec] = adamupdate( ...
                    decNet, gDec, avgGradDec, avgSqGradDec, ...
                    iterGlobal, OPT.LR_PHASE1_DEC);

                epochLoss = epochLoss + double(extractdata(loss));
                nBatch = nBatch + 1;
            end

            trainLoss = epochLoss / max(nBatch, 1);
            [valLoss, valSSIM] = validateEpoch(encNet, decNet, mbqVal, encOutLayer);

            history = appendHistory(history, globalEpoch, 1, trainLoss, valLoss, valSSIM);
            el = toc(t0);
            eta = el / globalEpoch * (totalEpochs - globalEpoch);

            fprintf('  [P1] Ep %2d/%d | TrainMSE %.5f | ValMSE %.5f | ValSSIM %.4f | %.0fs | ETA %.0fs\n', ...
                globalEpoch, totalEpochs, trainLoss, valLoss, valSSIM, el, eta);

            % 체크포인트
            if mod(globalEpoch, OPT.CKPT_EVERY) == 0
                saveCheckpoint(ckptDir, encNet, decNet, history, globalEpoch);
            end
        end

        %% --- Phase 2: Partial Unfreeze (conv3+conv4) ---
        fprintf('\n  === Phase 2: Unfreeze conv3+conv4 (%d epochs, EncLR=%.0e, DecLR=%.0e) ===\n', ...
            OPT.PHASE2_EPOCHS, OPT.LR_PHASE2_ENC, OPT.LR_PHASE2_DEC);

        % Adam 상태 리셋 (encoder용)
        avgGradEnc = []; avgSqGradEnc = [];
        iterPhase2 = 0;

        for ep = 1:OPT.PHASE2_EPOCHS
            globalEpoch = globalEpoch + 1;
            shuffle(mbqTrain);
            epochLoss = 0; nBatch = 0;

            while hasdata(mbqTrain)
                X = next(mbqTrain);
                iterGlobal = iterGlobal + 1;
                iterPhase2 = iterPhase2 + 1;

                [loss, gEnc, gDec, stDec] = dlfeval( ...
                    @modelLossUnfrozen, encNet, decNet, X, encOutLayer);
                decNet.State = stDec;

                % Encoder: conv3+conv4만 업데이트, 나머지 gradient 0
                gEnc = maskGradients(gEnc, unfreezeIdx);

                % Warmup (처음 3 epoch은 LR 선형 증가)
                warmupFrac = min(1.0, iterPhase2 / (3 * itersPerEpoch));
                encLR = OPT.LR_PHASE2_ENC * warmupFrac;

                [encNet, avgGradEnc, avgSqGradEnc] = adamupdate( ...
                    encNet, gEnc, avgGradEnc, avgSqGradEnc, ...
                    iterPhase2, encLR);

                [decNet, avgGradDec, avgSqGradDec] = adamupdate( ...
                    decNet, gDec, avgGradDec, avgSqGradDec, ...
                    iterGlobal, OPT.LR_PHASE2_DEC);

                epochLoss = epochLoss + double(extractdata(loss));
                nBatch = nBatch + 1;
            end

            trainLoss = epochLoss / max(nBatch, 1);
            [valLoss, valSSIM] = validateEpoch(encNet, decNet, mbqVal, encOutLayer);

            history = appendHistory(history, globalEpoch, 2, trainLoss, valLoss, valSSIM);
            el = toc(t0);
            eta = el / globalEpoch * (totalEpochs - globalEpoch);

            fprintf('  [P2] Ep %2d/%d | TrainMSE %.5f | ValMSE %.5f | ValSSIM %.4f | %.0fs | ETA %.0fs\n', ...
                globalEpoch, totalEpochs, trainLoss, valLoss, valSSIM, el, eta);

            if mod(globalEpoch, OPT.CKPT_EVERY) == 0
                saveCheckpoint(ckptDir, encNet, decNet, history, globalEpoch);
            end
        end

        trainOK = true;  % 학습 정상 완료

    catch ME
        isOOM = contains(lower(string(ME.message)), 'out of memory') || ...
                contains(lower(string(ME.message)), 'cuda');
        if isOOM && batchSize > OPT.BATCH_MIN
            newBatch = max(OPT.BATCH_MIN, floor(batchSize / 2));
            warning('Granite_AE:OOM', '%s', ...
                sprintf('OOM → Batch %d→%d 재시도', batchSize, newBatch));
            batchSize = newBatch;
            globalEpoch = 0;
            history = struct('epoch',[],'phase',[],'trainLoss',[],'valLoss',[],'valSSIM',[]);
            avgGradDec=[]; avgSqGradDec=[];
            continue;
        end
        rethrow(ME);
    end
end

totalTime = toc(t0);
fprintf('\n  학습 완료: %.1f min (Batch=%d)\n\n', totalTime/60, batchSize);

%% ===== [6] 임계값 계산 =====
fprintf('[6] 임계값 계산 (Train set 95th percentile)\n');
threshold = computeThreshold(encNet, decNet, trainFiles, encOutLayer, OPT);
fprintf('  Threshold (p95): %.6f\n\n', threshold);

%% ===== [7] Test 평가 =====
fprintf('[7] Test 평가\n');
[testMSE, testSSIM, testErrors] = evaluateSet( ...
    encNet, decNet, testFiles, encOutLayer, OPT);
aboveThresh = mean(testErrors > threshold) * 100;
fprintf('  Test MSE : %.5f\n', testMSE);
fprintf('  Test SSIM: %.4f\n', testSSIM);
fprintf('  Test 임계값 초과: %.1f%%\n\n', aboveThresh);

%% ===== [8] 시각화 =====
fprintf('[8] 시각화\n');

% 8-1. Loss curve
try
    plotLossCurve(history, figDir);
    fprintf('  loss_curve.png 저장\n');
catch ME
    warning(ME.identifier, '%s', sprintf('Loss curve 실패: %s', ME.message));
end

% 8-2. Reconstruction 비교
try
    plotReconSamples(encNet, decNet, testFiles, encOutLayer, figDir, OPT);
    fprintf('  reconstruction_samples.png 저장\n');
catch ME
    warning(ME.identifier, '%s', sprintf('Reconstruction plot 실패: %s', ME.message));
end

% 8-3. Error 히스토그램
try
    plotErrorHist(testErrors, threshold, figDir);
    fprintf('  error_histogram.png 저장\n');
catch ME
    warning(ME.identifier, '%s', sprintf('Error histogram 실패: %s', ME.message));
end

% 8-4. 타일 히트맵 데모
try
    tileHeatmapDemo(encNet, decNet, testFiles, encOutLayer, threshold, figDir, OPT);
    fprintf('  tile_heatmap 저장\n');
catch ME
    warning(ME.identifier, '%s', sprintf('Tile heatmap 실패: %s', ME.message));
end

fprintf('\n');

%% ===== [9] 결과 저장 =====
fprintf('[9] 결과 저장\n');
trySave(fullfile(outRoot,'encoderNet.mat'), 'encNet', encNet);
trySave(fullfile(outRoot,'decoderNet.mat'), 'decNet', decNet);
trySave(fullfile(outRoot,'training_history.mat'), 'history', history);
trySave(fullfile(outRoot,'threshold_info.mat'), 'threshold', threshold);

metrics = struct('testMSE',testMSE,'testSSIM',testSSIM, ...
    'threshold',threshold,'aboveThreshPct',aboveThresh, ...
    'batchUsed',batchSize,'totalTimeMin',totalTime/60, ...
    'nTrain',nTrain,'nVal',nVal,'nTest',nTest, ...
    'finalTrainLoss',history.trainLoss(end), ...
    'finalValLoss',history.valLoss(end), ...
    'finalValSSIM',history.valSSIM(end));
trySave(fullfile(outRoot,'test_metrics.mat'), 'metrics', metrics);

writeMetricsSummary(fullfile(logDir,'metrics_summary.txt'), metrics, OPT, gpuName);

%% 완료
fprintf('\n%s\n',repmat('=',1,76));
fprintf('  DONE | %.1f min | SSIM=%.4f | Threshold=%.6f\n', ...
    totalTime/60, testSSIM, threshold);
fprintf('  결과: %s\n', outRoot);
fprintf('%s\n\n',repmat('=',1,76));

try diary off; end %#ok<TRYNC>

runInfo = struct('outRoot',outRoot, 'testSSIM',testSSIM, ...
    'threshold',threshold, 'metrics',metrics, 'history',history);
end

%% =========================================================================
%% MODEL LOSS FUNCTIONS
%% =========================================================================
function [loss, gDec, stDec] = modelLossFrozen(encNet, decNet, X, outLayer)
Z = forward(encNet, X, 'Outputs', outLayer);
[Xhat, stDec] = forward(decNet, Z);
loss = mean((Xhat - X).^2, 'all');
gDec = dlgradient(loss, decNet.Learnables);
end

function [loss, gEnc, gDec, stDec] = modelLossUnfrozen(encNet, decNet, X, outLayer)
Z = forward(encNet, X, 'Outputs', outLayer);
[Xhat, stDec] = forward(decNet, Z);
loss = mean((Xhat - X).^2, 'all');
[gEnc, gDec] = dlgradient(loss, encNet.Learnables, decNet.Learnables);
end

function grad = maskGradients(grad, keepIdx)
for i = 1:height(grad)
    if ~keepIdx(i)
        grad.Value{i} = zeros(size(grad.Value{i}), 'like', grad.Value{i});
    end
end
end

%% =========================================================================
%% VALIDATION
%% =========================================================================
function [valLoss, valSSIM] = validateEpoch(encNet, decNet, mbqVal, outLayer)
shuffle(mbqVal);
lossSum = 0; ssimSum = 0; nB = 0; maxB = 50;
while hasdata(mbqVal) && nB < maxB
    X = next(mbqVal);
    Z = predict(encNet, X, 'Outputs', outLayer);
    Xhat = predict(decNet, Z);
    lossSum = lossSum + double(extractdata(mean((Xhat-X).^2,'all')));
    Xnp = gather(extractdata(X));
    Xhnp = gather(extractdata(Xhat));
    for b = 1:size(Xnp,4)
        ssimSum = ssimSum + ssim(Xhnp(:,:,:,b), Xnp(:,:,:,b));
    end
    nB = nB + 1;
end
nSamples = nB * size(Xnp, 4);
valLoss = lossSum / max(nB, 1);
valSSIM = ssimSum / max(nSamples, 1);
end

%% =========================================================================
%% THRESHOLD
%% =========================================================================
function threshold = computeThreshold(encNet, decNet, files, outLayer, OPT)
nSample = min(OPT.THRESHOLD_SAMPLES, numel(files));
rng(OPT.SEED);
idx = randperm(numel(files), nSample);
errors = zeros(nSample, 1);
for i = 1:nSample
    try
        I = valReadFcn(files{idx(i)}, [224 224]);
        dlX = dlarray(I, 'SSCB');
        Z = predict(encNet, dlX, 'Outputs', outLayer);
        Xhat = predict(decNet, Z);
        errors(i) = double(extractdata(mean((Xhat - dlX).^2, 'all')));
    catch
        errors(i) = NaN;
    end
end
errors = errors(~isnan(errors));
threshold = prctile(errors, 95);
end

%% =========================================================================
%% EVALUATE SET
%% =========================================================================
function [avgMSE, avgSSIM, errors] = evaluateSet(encNet, decNet, files, outLayer, OPT)
nSample = min(OPT.EVAL_SAMPLES, numel(files));
rng(OPT.SEED);
idx = randperm(numel(files), nSample);
errors = zeros(nSample, 1);
ssims  = zeros(nSample, 1);
for i = 1:nSample
    try
        I = valReadFcn(files{idx(i)}, [224 224]);
        dlX = dlarray(I, 'SSCB');
        Z = predict(encNet, dlX, 'Outputs', outLayer);
        Xhat = predict(decNet, Z);
        errors(i) = double(extractdata(mean((Xhat - dlX).^2, 'all')));
        ssims(i) = ssim(gather(extractdata(Xhat)), I);
    catch
        errors(i) = NaN; ssims(i) = NaN;
    end
end
ok = ~isnan(errors);
avgMSE  = mean(errors(ok));
avgSSIM = mean(ssims(ok));
errors  = errors(ok);
end

%% =========================================================================
%% NETWORK CONSTRUCTION
%% =========================================================================
function [encNet, outLayer, outCh] = buildEncoder()
net = resnet50;
lgraph = layerGraph(net);

% Classification head 제거
removeNames = {'ClassificationLayer_fc1000','fc1000_softmax','fc1000','avg_pool', ...
    'predictions','predictions_softmax','ClassificationLayer_predictions'};
allNames = {lgraph.Layers.Name};
for i = 1:numel(removeNames)
    if any(strcmp(allNames, removeNames{i}))
        lgraph = removeLayers(lgraph, removeNames{i});
        allNames = {lgraph.Layers.Name};
    end
end

encNet = dlnetwork(lgraph);

% conv4 출력 레이어 자동 탐지
[outLayer, outCh] = detectConv4Output(encNet);
end

function [layerName, nCh] = detectConv4Output(encNet)
candidates = {'activation_40_relu','conv4_block6_out','add_13','res4f_relu'};
layerNames = {encNet.Layers.Name};
dummyX = dlarray(randn(224,224,3,1,'single'), 'SSCB');

% 후보 우선 탐색
for i = 1:numel(candidates)
    if any(strcmp(layerNames, candidates{i}))
        try
            Z = predict(encNet, dummyX, 'Outputs', candidates{i});
            sz = size(extractdata(Z));
            if sz(1)==14 && sz(2)==14
                layerName = candidates{i}; nCh = sz(3); return;
            end
        catch
        end
    end
end

% 전체 탐색 (fallback)
for i = 1:numel(layerNames)
    try
        Z = predict(encNet, dummyX, 'Outputs', layerNames{i});
        sz = size(extractdata(Z));
        if numel(sz)>=3 && sz(1)==14 && sz(2)==14
            layerName = layerNames{i}; nCh = sz(3);
            fprintf('  Auto-detected: %s [14×14×%d]\n', layerName, nCh);
            return;
        end
    catch
    end
end
error('Granite_AE:NoConv4','conv4 output layer를 찾지 못했습니다.');
end

function decNet = buildDecoder(inCh)
layers = [
    imageInputLayer([14 14 inCh], 'Normalization','none', 'Name','dec_in')

    convolution2dLayer(1, 256, 'Padding','same', 'Name','bottleneck_conv')
    batchNormalizationLayer('Name','bottleneck_bn')
    reluLayer('Name','bottleneck_relu')

    transposedConv2dLayer(4, 128, 'Stride',2, 'Cropping','same', 'Name','dec_up1')
    batchNormalizationLayer('Name','dec_bn1')
    reluLayer('Name','dec_relu1')

    transposedConv2dLayer(4, 64, 'Stride',2, 'Cropping','same', 'Name','dec_up2')
    batchNormalizationLayer('Name','dec_bn2')
    reluLayer('Name','dec_relu2')

    transposedConv2dLayer(4, 32, 'Stride',2, 'Cropping','same', 'Name','dec_up3')
    batchNormalizationLayer('Name','dec_bn3')
    reluLayer('Name','dec_relu3')

    transposedConv2dLayer(4, 3, 'Stride',2, 'Cropping','same', 'Name','dec_up4')
    sigmoidLayer('Name','dec_sigmoid')
];
decNet = dlnetwork(layers);
end

function idx = findConv3Conv4Params(encNet)
tbl = encNet.Learnables;
idx = false(height(tbl), 1);
patterns = {'res3','bn3','conv3','res4','bn4','conv4', ...
    '_block3_','_block4_','_block5_','_block6_', ...
    '3a_','3b_','3c_','3d_','4a_','4b_','4c_','4d_','4e_','4f_'};
for i = 1:height(tbl)
    name = char(tbl.Layer(i));
    for p = 1:numel(patterns)
        if contains(name, patterns{p})
            idx(i) = true; break;
        end
    end
end
n = sum(idx);
fprintf('  Unfreeze 대상: %d / %d encoder params\n', n, height(tbl));
if n == 0
    warning('Granite_AE:NoUnfreeze','%s','conv3/conv4 미탐지 → 전체 unfreeze');
    idx = true(height(tbl), 1);
end
end

%% =========================================================================
%% VISUALIZATION
%% =========================================================================
function plotLossCurve(history, figDir)
fig = figure('Visible','off','Position',[100 100 900 400]);
yyaxis left;
plot(history.epoch, history.trainLoss, '-o', 'LineWidth',1.2); hold on;
plot(history.epoch, history.valLoss, '-s', 'LineWidth',1.2);
ylabel('MSE Loss');
yyaxis right;
plot(history.epoch, history.valSSIM, '-^', 'LineWidth',1.2);
ylabel('Val SSIM');
xlabel('Epoch');
legend({'Train MSE','Val MSE','Val SSIM'}, 'Location','best');
title('Training History');
grid on; box on;

% Phase 구분선
p1end = find(history.phase==1, 1, 'last');
if ~isempty(p1end)
    xline(history.epoch(p1end)+0.5, '--k', 'Phase1→2', ...
        'LabelHorizontalAlignment','center');
end
exportgraphics(fig, fullfile(figDir,'loss_curve.png'), 'Resolution',200);
close(fig);
end

function plotReconSamples(encNet, decNet, files, outLayer, figDir, OPT)
nShow = min(6, numel(files));
rng(OPT.SEED + 1);
idx = randperm(numel(files), nShow);

fig = figure('Visible','off','Position',[50 50 1800 600]);
tl = tiledlayout(fig, 3, nShow, 'Padding','compact','TileSpacing','compact');

for i = 1:nShow
    Iorig = valReadFcn(files{idx(i)}, [224 224]);
    dlX = dlarray(Iorig, 'SSCB');
    Z = predict(encNet, dlX, 'Outputs', outLayer);
    Xhat = gather(extractdata(predict(decNet, Z)));
    errMap = mean((Xhat - Iorig).^2, 3);

    % 원본
    nexttile(tl, i);
    imshow(Iorig); title('원본','FontSize',8);

    % 복원
    nexttile(tl, nShow + i);
    imshow(Xhat); title('복원','FontSize',8);

    % 오차맵
    nexttile(tl, 2*nShow + i);
    imagesc(errMap); axis image off;
    colormap(gca, hot(256)); clim([0 0.05]);
    mseVal = mean(errMap(:));
    title(sprintf('MSE=%.4f',mseVal),'FontSize',8);
end
exportgraphics(fig, fullfile(figDir,'reconstruction_samples.png'), 'Resolution',200);
close(fig);
end

function plotErrorHist(errors, threshold, figDir)
fig = figure('Visible','off','Position',[100 100 800 400]);
histogram(errors, 80, 'Normalization','probability', 'EdgeColor','none');
hold on;
xline(threshold, 'r--', 'LineWidth',1.5);
text(threshold*1.05, max(ylim)*0.9, sprintf('p95=%.5f', threshold), ...
    'Color','r', 'FontSize',10);
xlabel('Reconstruction MSE');
ylabel('Probability');
title('Test Error Distribution');
grid on; box on;
exportgraphics(fig, fullfile(figDir,'error_histogram.png'), 'Resolution',200);
close(fig);
end

function tileHeatmapDemo(encNet, decNet, files, outLayer, threshold, figDir, OPT)
nDemo = min(3, numel(files));
rng(OPT.SEED + 2);
idx = randperm(numel(files), nDemo);

for di = 1:nDemo
    I = imread(files{idx(di)});
    I = ensureRGB(I);

    [probMap, ~] = tileInference(encNet, decNet, I, outLayer, threshold, ...
        OPT.TILE_SIZE, OPT.TILE_STRIDE);

    fig = figure('Visible','off','Position',[100 100 1400 500]);
    tl = tiledlayout(fig, 1, 3, 'Padding','compact','TileSpacing','compact');

    % 원본
    nexttile(tl);
    imshow(I); title('원본','FontSize',10);

    % 히트맵
    nexttile(tl);
    imagesc(probMap); axis image off;
    colormap(gca, jet(256)); clim([0 1]); colorbar;
    title('화강암 확률 히트맵','FontSize',10);

    % 오버레이
    nexttile(tl);
    imshow(I); hold on;
    hm = imagesc(probMap); set(hm, 'AlphaData', 0.4);
    colormap(gca, jet(256)); clim([0 1]); colorbar;
    title('오버레이','FontSize',10);

    [~,stem,~] = fileparts(files{idx(di)});
    shortStem = stem; if strlength(shortStem)>30; shortStem=extractBefore(shortStem,31); end
    exportgraphics(fig, fullfile(figDir, sprintf('tile_heatmap_%d_%s.png', di, shortStem)), ...
        'Resolution',200);
    close(fig);
end
end

%% =========================================================================
%% TILE INFERENCE
%% =========================================================================
function [probMap, countMap] = tileInference(encNet, decNet, I, outLayer, threshold, tileSize, stride)
[H, W, ~] = size(I);
probMap  = zeros(H, W);
countMap = zeros(H, W);

rows = 1:stride:max(1, H - tileSize + 1);
cols = 1:stride:max(1, W - tileSize + 1);

for r = rows
    for c = cols
        rEnd = min(r + tileSize - 1, H);
        cEnd = min(c + tileSize - 1, W);
        tile = I(r:rEnd, c:cEnd, :);
        tile224 = imresize(tile, [224 224]);

        dlT = dlarray(single(tile224)/255, 'SSCB');
        Z = predict(encNet, dlT, 'Outputs', outLayer);
        Xhat = predict(decNet, Z);
        mseVal = double(extractdata(mean((Xhat - dlT).^2, 'all')));

        % 확률: 오차 낮을수록 1에 가까움
        prob = max(0, min(1, 1 - mseVal / (threshold * 2)));

        probMap(r:rEnd, c:cEnd)  = probMap(r:rEnd, c:cEnd) + prob;
        countMap(r:rEnd, c:cEnd) = countMap(r:rEnd, c:cEnd) + 1;
    end
end
countMap(countMap == 0) = 1;
probMap = probMap ./ countMap;
end

%% =========================================================================
%% DATA I/O
%% =========================================================================
function I = trainReadFcn(filename, targetSize)
I = imread(char(filename));
I = ensureRGB(I);

% RandomResizedCrop
[h, w, ~] = size(I);
scale = 0.8 + 0.2 * rand;
cropH = max(1, round(h * scale));
cropW = max(1, round(w * scale));
r = randi(max(1, h - cropH + 1));
c = randi(max(1, w - cropW + 1));
I = I(r:min(r+cropH-1,h), c:min(c+cropW-1,w), :);
I = imresize(I, targetSize);

% RandomFlip
if rand > 0.5, I = fliplr(I); end
if rand > 0.5, I = flipud(I); end

% RandomRotation ±15°
ang = (rand * 30) - 15;
I = imrotate(I, ang, 'bilinear', 'crop');
I = imresize(I, targetSize);  % imrotate 후 크기 보정

% ColorJitter (밝기 ±10%, 대비 ±10%, 채도 ±10%)
Ihsv = rgb2hsv(I);
Ihsv(:,:,2) = min(1, max(0, Ihsv(:,:,2) * (0.9 + 0.2*rand)));
Ihsv(:,:,3) = min(1, max(0, Ihsv(:,:,3) * (0.9 + 0.2*rand)));
I = hsv2rgb(Ihsv);

% GaussianNoise
if rand > 0.5
    I = min(1, max(0, I + randn(size(I)) * 0.02));
end

I = single(I);
if max(I(:)) > 1.1, I = I / 255; end  % uint8→single 변환 안전
end

function I = valReadFcn(filename, targetSize)
I = imread(char(filename));
I = ensureRGB(I);
I = imresize(I, targetSize);
I = single(I);
if max(I(:)) > 1.1, I = I / 255; end
end

function I = ensureRGB(I)
if isempty(I), error('Granite_AE:Empty','빈 이미지'); end
if ismatrix(I), I = repmat(I, [1 1 3]); end
if size(I, 3) > 3, I = I(:,:,1:3); end
if ~isa(I,'uint8'), I = im2uint8(I); end
end

%% =========================================================================
%% DATA COLLECTION + SPLIT
%% =========================================================================
function files = collectImages(roots, exts)
files = {};
for ri = 1:numel(roots)
    root = char(roots{ri});
    if ~exist(root, 'dir')
        warning('Granite_AE:ImgDir','%s',sprintf('폴더 없음: %s', root));
        continue;
    end
    dd = dir(fullfile(root, '**', '*'));
    nAdd = 0;
    for di = 1:numel(dd)
        if dd(di).isdir, continue; end
        [~,~,e] = fileparts(dd(di).name);
        if any(strcmpi(e, exts))
            files{end+1} = fullfile(dd(di).folder, dd(di).name); %#ok<AGROW>
            nAdd = nAdd + 1;
        end
    end
    fprintf('  %s → %d장\n', root, nAdd);
end
files = unique(files, 'stable')';
end

function grp = parseBorehole(filepath)
[~, stem, ~] = fileparts(filepath);
parts = strsplit(stem, '-');
if numel(parts) >= 2
    bh = regexprep(strtrim(parts{2}), '\(.*\)', '');
    grp = upper(strtrim(bh));
else
    grp = stem;
end
end

function [trainIdx, valIdx, testIdx] = groupExclusiveSplit(groups, rTr, rVa, ~)
uGroups = unique(groups, 'stable');
nG = numel(uGroups);
perm = randperm(nG);

nTr = max(1, round(nG * rTr));
nVa = max(1, round(nG * rVa));

trGroups = uGroups(perm(1:nTr));
vaGroups = uGroups(perm(nTr+1 : min(nTr+nVa, nG)));
teGroups = uGroups(perm(min(nTr+nVa+1, nG+1) : nG));

trainIdx = find(ismember(groups, trGroups));
valIdx   = find(ismember(groups, vaGroups));
testIdx  = find(ismember(groups, teGroups));

% 빈 split 방지
if isempty(testIdx) && numel(trainIdx) > 10
    testIdx = trainIdx(end-4:end);
    trainIdx = trainIdx(1:end-5);
end
if isempty(valIdx) && numel(trainIdx) > 10
    valIdx = trainIdx(end-4:end);
    trainIdx = trainIdx(1:end-5);
end
end

%% =========================================================================
%% HISTORY / CHECKPOINT / SAVE
%% =========================================================================
function h = appendHistory(h, epoch, phase, tLoss, vLoss, vSSIM)
h.epoch(end+1)     = epoch;
h.phase(end+1)     = phase;
h.trainLoss(end+1) = tLoss;
h.valLoss(end+1)   = vLoss;
h.valSSIM(end+1)   = vSSIM;
end

function saveCheckpoint(ckptDir, encNet, decNet, history, epoch)
try
    fname = fullfile(ckptDir, sprintf('ckpt_epoch%03d.mat', epoch));
    save(fname, 'encNet', 'decNet', 'history', 'epoch', '-v7.3');
catch ME
    warning(ME.identifier, '%s', sprintf('체크포인트 저장 실패: %s', ME.message));
end
end

function trySave(fpath, varName, varVal)
try
    S.(varName) = varVal; 
    save(fpath, '-struct', 'S', '-v7.3');
    fprintf('  %s 저장\n', fpath);
catch ME
    warning(ME.identifier, '%s', sprintf('저장 실패: %s | %s', fpath, ME.message));
end
end

%% =========================================================================
%% REPORTS
%% =========================================================================
function writeSplitReport(fpath, nTotal, nTr, nVa, nTe, gTr, gVa, gTe)
try
    fid = fopen(fpath, 'w');
    fprintf(fid, 'Group-Exclusive Split Report\n');
    fprintf(fid, '===========================\n');
    fprintf(fid, 'Total images : %d\n', nTotal);
    fprintf(fid, 'Train: %d (%.1f%%) — %d groups\n', nTr, 100*nTr/nTotal, gTr);
    fprintf(fid, 'Val  : %d (%.1f%%) — %d groups\n', nVa, 100*nVa/nTotal, gVa);
    fprintf(fid, 'Test : %d (%.1f%%) — %d groups\n', nTe, 100*nTe/nTotal, gTe);
    fprintf(fid, '\nSplit method: Borehole-based group-exclusive\n');
    fclose(fid);
catch
end
end

function writeMetricsSummary(fpath, m, OPT, gpuName)
try
    fid = fopen(fpath, 'w');
    fprintf(fid, 'Granite AE ResNet-50 Metrics Summary\n');
    fprintf(fid, '=====================================\n');
    fprintf(fid, 'GPU          : %s\n', gpuName);
    fprintf(fid, 'Batch        : %d\n', m.batchUsed);
    fprintf(fid, 'Epochs       : %d (P1=%d + P2=%d)\n', ...
        OPT.PHASE1_EPOCHS+OPT.PHASE2_EPOCHS, OPT.PHASE1_EPOCHS, OPT.PHASE2_EPOCHS);
    fprintf(fid, 'Train/Val/Test: %d / %d / %d\n', m.nTrain, m.nVal, m.nTest);
    fprintf(fid, 'Training time: %.1f min\n\n', m.totalTimeMin);
    fprintf(fid, 'Final Train MSE : %.6f\n', m.finalTrainLoss);
    fprintf(fid, 'Final Val MSE   : %.6f\n', m.finalValLoss);
    fprintf(fid, 'Final Val SSIM  : %.4f\n\n', m.finalValSSIM);
    fprintf(fid, 'Test MSE        : %.6f\n', m.testMSE);
    fprintf(fid, 'Test SSIM       : %.4f\n', m.testSSIM);
    fprintf(fid, 'Threshold (p95) : %.6f\n', m.threshold);
    fprintf(fid, 'Test > threshold: %.1f%%\n', m.aboveThreshPct);
    fclose(fid);
catch
end
end

%% =========================================================================
%% GPU / UTIL
%% =========================================================================
function [gpuOK, name] = initGPU(want)
gpuOK = false; name = '없음(CPU)';
if ~want, return; end
try parallel.gpu.enableCUDAForwardCompatibility(true); catch, end
try
    g = gpuDevice(1);
    fprintf('  %s | %.2f/%.2f GB | CC %s\n', ...
        g.Name, g.AvailableMemory/1e9, g.TotalMemory/1e9, string(g.ComputeCapability));
    gpuOK = true; name = g.Name;
catch ME
    warning(ME.identifier, '%s', sprintf('GPU 없음: %s', ME.message));
end
end

function s = formatParams(net)
n = sum(cellfun(@numel, net.Learnables.Value));
if n > 1e6
    s = sprintf('%.2fM', n/1e6);
else
    s = sprintf('%.1fK', n/1e3);
end
end

function ensureDirs(C)
for i = 1:numel(C)
    d = char(C{i});
    if ~exist(d, 'dir'), mkdir(d); end
end
end

%% =========================================================================
%% OPTIONS
%% =========================================================================
function OPT = parseOpts(varargin)
p = inputParser;

addParameter(p, 'IMAGE_ROOTS', {
    'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\images_igneous_rock\TS_1, 기반암 암종 분류 데이터_1. 화성암_1'
    'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\images_igneous_rock\TS_1, 기반암 암종 분류 데이터_1. 화성암_2'
    'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\images_igneous_rock\TS_2. 기반암 절리 탐지 데이터_1. 화성암'
    'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\images_igneous_rock\VS_1. 기반암 암종 분류 데이터_1. 화성암'
    'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\images_igneous_rock\VS_2. 기반암 절리 탐지 데이터_1. 화성암'
});
addParameter(p, 'RESULTS_ROOT', 'C:\Users\ROCKENG\Desktop\코랩 머신러닝\results');
addParameter(p, 'IMG_EXTS', {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'});

% Split
addParameter(p, 'SPLIT_TRAIN', 0.70);
addParameter(p, 'SPLIT_VAL',   0.15);
addParameter(p, 'SPLIT_TEST',  0.15);
addParameter(p, 'SEED', 42);

% Training
addParameter(p, 'BATCH_SIZE', 8);
addParameter(p, 'BATCH_MIN',  4);
addParameter(p, 'PHASE1_EPOCHS', 8);
addParameter(p, 'PHASE2_EPOCHS', 22);
addParameter(p, 'LR_PHASE1_DEC', 1e-3);
addParameter(p, 'LR_PHASE2_ENC', 5e-5);
addParameter(p, 'LR_PHASE2_DEC', 1e-4);
addParameter(p, 'CKPT_EVERY', 5);

% Eval
addParameter(p, 'THRESHOLD_SAMPLES', 5000);
addParameter(p, 'EVAL_SAMPLES', 5000);

% Tile inference
addParameter(p, 'TILE_SIZE', 224);
addParameter(p, 'TILE_STRIDE', 112);

% GPU
addParameter(p, 'USE_GPU', true);

parse(p, varargin{:});
OPT = p.Results;

if ischar(OPT.IMAGE_ROOTS) || isstring(OPT.IMAGE_ROOTS)
    OPT.IMAGE_ROOTS = cellstr(OPT.IMAGE_ROOTS);
end
end