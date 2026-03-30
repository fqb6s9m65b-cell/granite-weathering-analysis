function rocktype_resnet18_train_mat()
% rocktype_resnet18_train_mat (ONE-FILE COMPLETE, STRIP HEATMAP STYLE)
% -------------------------------------------------------------------------
% 출력 스타일: 사용자 예시처럼 "길쭉한 스트립 heatmap" (Class map + Confidence map)
% 핵심:
%  - MAT 네트워크 자동 로드
%  - ImageInputLayer 기반 InputSize 안전 추출 (1x1x512 같은 FC 입력Size 방지)
%  - 글로벌 예측 + 패치 슬라이딩(오버랩) 예측
%  - 배치 처리로 메모리 안정
%  - 불확실 패치(Top1<confThresh)는 Unknown 처리
%  - Excel + PNG + MAT 저장

clc;

fprintf("\n============================================\n");
fprintf("  Rock Type Single Image Analyzer\n");
fprintf("  (STRIP HEATMAP STYLE)\n");
fprintf("============================================\n\n");

%% ========================= User Parameters =========================
P = struct();

% "세세하게"의 핵심: tileSize를 줄이거나 overlap을 키우면 더 촘촘해짐
P.tileSize      = [112 112];   % [H W]  (추천: 112, 더 거칠게=224)
P.overlapRatio  = 0.50;        % 0.50(권장) / 0.75(아주 촘촘)
P.minStride     = 32;          % stride가 너무 작아 폭증하는 것 방지

P.miniBatchSize = 64;          % GPU면 64~256, CPU면 16~64
P.confThresh    = 0.60;        % Top1 확률이 이하면 Unknown 처리
P.printDigits   = 8;           % 100%처럼 보이는 반올림 완화

% Strip heatmap figure size (예시 느낌으로: 길쭉하게)
P.stripFigPos   = [100 100 1400 320];  % [x y w h]

unknownLabel = "Unknown";

%% ========================= 1) Load Network =========================
[fn_model, fp_model] = uigetfile('*.mat', '암종 분류 모델(.mat)을 선택하세요');
if isequal(fn_model,0); disp("취소됨."); return; end
modelPath = fullfile(fp_model, fn_model);
fprintf("[Info] 모델 로드: %s\n", modelPath);

S   = load(modelPath);
net = pickNetworkFromStruct_local(S);
if isempty(net)
    error("MAT 파일에서 네트워크(SeriesNetwork/DAGNetwork/dlnetwork)를 찾지 못했습니다.");
end
fprintf("[Info] 원본 네트워크 타입: %s\n", class(net));

% BatchNorm / dlnetwork 보정(필요시)
net = fixBatchNormStats_local(net);
fprintf("[Info] 보정 후 네트워크 타입: %s\n", class(net));

% 입력 크기(안전)
[inputSize, inputLayerName] = getNetInputSize_local(net);
netH = inputSize(1); netW = inputSize(2);
fprintf("[Info] InputLayer='%s', InputSize=[%d %d %d]\n", inputLayerName, inputSize);

% 클래스 이름(안전)
classNames = getClassNamesFromNet_local(net);  % may be empty
if ~isempty(classNames), classNames = string(classNames); end

%% ========================= 2) Load Image =========================
[fn_img, fp_img] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff','Image files'}, ...
    "분석할 암석 이미지 선택");
if isequal(fn_img,0); disp("취소됨."); return; end
imgPath = fullfile(fp_img, fn_img);
fprintf("[Info] 이미지 로드: %s\n", imgPath);

Iorig = imread(imgPath);
Iorig = ensure_rgb_local(Iorig);
[H,W,~] = size(Iorig);
fprintf("[Info] 원본 크기: %d x %d px\n", H, W);

% 중심/반경
coreCenter = [W/2, H/2];   % [cx, cy]
coreRadius = sqrt((W/2)^2 + (H/2)^2);

%% ========================= 3) Global Prediction =========================
Iglobal = im2single(imresize(Iorig, [netH netW]));
[globalLabel, globalScoreVec] = classify(net, Iglobal);
globalPredLabelStr = string(globalLabel);
globalScoreVec = squeeze(globalScoreVec);

if isempty(classNames)
    classNames = "Cls" + string(1:numel(globalScoreVec));
end

[globalProbSorted, ~] = sort(globalScoreVec, 'descend');
gTop1 = globalProbSorted(1);
gTop2 = globalProbSorted(min(2,end));
gMargin = gTop1 - gTop2;

if gMargin >= 0.20, gConfLevel="High";
elseif gMargin >= 0.10, gConfLevel="Medium";
else, gConfLevel="Low";
end

fmt = "%." + string(P.printDigits) + "f";
fprintf("\n===== Global Prediction =====\n");
fprintf(" Pred: %s\n", globalPredLabelStr);
fprintf(" Top1: %s\n", sprintf(fmt, gTop1));
fprintf(" Top2: %s\n", sprintf(fmt, gTop2));
fprintf(" Margin: %s (%s)\n", sprintf(fmt, gMargin), gConfLevel);

%% ========================= 4) Patch Grid (sliding + edge 포함) =========================
tileH = P.tileSize(1);
tileW = P.tileSize(2);

strideY = max(P.minStride, round(tileH*(1-P.overlapRatio)));
strideX = max(P.minStride, round(tileW*(1-P.overlapRatio)));

fprintf("\n[Info] Patch tile=%dx%d, overlap=%.2f, stride=%dx%d\n", ...
    tileH, tileW, P.overlapRatio, strideY, strideX);

% 이미지가 타일보다 작으면 padding
if H < tileH || W < tileW
    padY = max(0, tileH - H);
    padX = max(0, tileW - W);
    Iorig = padarray(Iorig, [padY padX], "replicate", "post");
    [H,W,~] = size(Iorig);
    fprintf("[Info] Padding applied -> New size: %d x %d\n", H, W);
end

ys = unique([1:strideY:(H - tileH + 1), (H - tileH + 1)]);
xs = unique([1:strideX:(W - tileW + 1), (W - tileW + 1)]);
nY = numel(ys); nX = numel(xs);
numPatches = nY * nX;

fprintf("[Info] Grid = %d(rows) x %d(cols) = %d patches\n", nY, nX, numPatches);

% 패치 메타
patchStartYX  = zeros(numPatches,2);
patchCenterYX = zeros(numPatches,2);
gridPos       = zeros(numPatches,2);

idx = 0;
for iy = 1:nY
    y = ys(iy);
    for ix = 1:nX
        x = xs(ix);
        idx = idx + 1;

        patchStartYX(idx,:) = [y x];
        gridPos(idx,:)      = [iy ix];

        cy = y + (tileH-1)/2;
        cx = x + (tileW-1)/2;
        patchCenterYX(idx,:) = [cy cx];
    end
end

%% ========================= 5) Patch classify (BATCH) =========================
execEnv = pickExecutionEnvironment_local();
fprintf("[Info] Patch classify (batch) env=%s, MiniBatch=%d\n", execEnv, P.miniBatchSize);

scoresPatch    = [];                     % (N x K) first batch에서 K 결정
YPatchStr      = strings(numPatches,1);
maxScorePatch  = zeros(numPatches,1,'single');
marginPatch    = zeros(numPatches,1,'single');  % Top1-Top2 (정밀 confidence)

b0 = 1;
while b0 <= numPatches
    b1 = min(numPatches, b0 + P.miniBatchSize - 1);
    nb = b1 - b0 + 1;

    I4 = zeros(netH, netW, 3, nb, 'single');

    for bi = 1:nb
        ii = b0 + bi - 1;
        y = patchStartYX(ii,1);
        x = patchStartYX(ii,2);

        patch = Iorig(y:y+tileH-1, x:x+tileW-1, :);

        % tile이 입력과 다르면 resize해서 네트워크 입력에 맞춤 (세세하게 분석할 때 핵심)
        I4(:,:,:,bi) = im2single(imresize(patch, [netH netW]));
    end

    [Yb, Sb] = classify(net, I4, 'ExecutionEnvironment', execEnv);

    if isempty(scoresPatch)
        baseK = size(Sb,2);
        scoresPatch = zeros(numPatches, baseK, 'single');

        if isempty(classNames)
            classNames = "Cls" + string(1:baseK);
        else
            classNames = string(classNames);
        end
    end

    scoresPatch(b0:b1,:) = single(Sb);
    YPatchStr(b0:b1)     = string(Yb);

    % Top1 score
    s = single(Sb);
    [s1, i1] = max(s, [], 2);
    s2 = s;
    s2(sub2ind(size(s2),(1:nb)', i1)) = -inf;
    s2 = max(s2, [], 2);

    maxScorePatch(b0:b1) = s1;
    marginPatch(b0:b1)   = s1 - s2;

    b0 = b1 + 1;
end

% 불확실 패치 Unknown 처리 (Top1 기준)
isUncertain = maxScorePatch < P.confThresh;
YPatchStr(isUncertain) = unknownLabel;

% classNames에 Unknown 추가(표/시각화 일관)
if ~any(classNames == unknownLabel)
    classNames = [classNames(:); unknownLabel];
end
classNames = string(classNames);



%% ========================= 6) Patch Maps (Class + Confidence) =========================
% class index 매핑
patchClassIdx = zeros(numPatches,1);
for i = 1:numPatches
    patchClassIdx(i) = find(classNames == YPatchStr(i), 1);
end

% grid map 생성
patchMap = zeros(nY, nX);
confMap  = zeros(nY, nX, 'single');   % Top1
margMap  = zeros(nY, nX, 'single');   % Top1-Top2

for i = 1:numPatches
    iy = gridPos(i,1); ix = gridPos(i,2);
    patchMap(iy,ix) = patchClassIdx(i);
    confMap(iy,ix)  = maxScorePatch(i);
    margMap(iy,ix)  = marginPatch(i);
end

%% ========================= 7) "예시 느낌" Strip Heatmap Figure =========================
figStrip = plotStripHeatmaps_local(patchMap, confMap, classNames, P.stripFigPos);
figStrip.Name = "Patch Strip Heatmaps";

% 글로벌 확률 bar
figure('Name','Rock Type Global Prediction','NumberTitle','off');
bar(globalScoreVec);
xticks(1:numel(classNames)-1); % Unknown은 글로벌에 없음이 보통
xticklabels(classNames(1:max(1,numel(globalScoreVec))));
ylabel('Probability');
title(sprintf("Global Prediction: %s (Margin=%.3f, %s)", globalPredLabelStr, gMargin, gConfLevel));
grid on;

%% ========================= 8) Summary Tables =========================
% patch summary
[grpName, ~, ic] = unique(YPatchStr, 'stable');
grpCnt = accumarray(ic, 1);
grpPct = grpCnt / numPatches * 100;

Tsummary = table(grpName(:), grpCnt(:), grpPct(:), ...
    'VariableNames', {'class','patch_count','patch_percent'});

[~, idxDom] = max(grpCnt);
dominantClass   = grpName(idxDom);
dominantPercent = grpPct(idxDom);

% radial
cx = patchCenterYX(:,2);
cy = patchCenterYX(:,1);
r  = sqrt((cx - coreCenter(1)).^2 + (cy - coreCenter(2)).^2);
rNorm = r / coreRadius;

% patch detail
patchID = (1:numPatches).';
patchInfo = table( ...
    patchID, ...
    patchStartYX(:,1), patchStartYX(:,2), ...
    patchCenterYX(:,1), patchCenterYX(:,2), ...
    r, rNorm, ...
    YPatchStr, maxScorePatch, marginPatch, ...
    'VariableNames', { ...
    'patchID','y_start','x_start','y_center','x_center', ...
    'radius_px','radius_norm','predClass','top1Score','top1Top2Margin'});

% score columns (baseK만)
baseK = size(scoresPatch,2);
for j = 1:baseK
    cname = matlab.lang.makeValidName("score_" + classNames(j));
    patchInfo.(cname) = scoresPatch(:,j);
end

% 색상 통계 (RGB+Lab)
Irgb = im2double(Iorig);
Ilab = rgb2lab(Irgb);

Rch = Irgb(:,:,1); Gch = Irgb(:,:,2); Bch = Irgb(:,:,3);
Lch = Ilab(:,:,1); ach = Ilab(:,:,2); bch = Ilab(:,:,3);

Tcolor = table( ...
    mean(Rch(:)), mean(Gch(:)), mean(Bch(:)), ...
    std(Rch(:)),  std(Gch(:)),  std(Bch(:)), ...
    mean(Lch(:)), mean(ach(:)), mean(bch(:)), ...
    'VariableNames', {'meanR','meanG','meanB','stdR','stdG','stdB','meanL','meana','meanb'});

edgesL   = [0 25 50 75 100];
[Lcnt,~] = histcounts(Lch(:), edgesL);
Lpct     = Lcnt / sum(Lcnt) * 100;
LbinName = ["L0_25","L25_50","L50_75","L75_100"].';
TcolorL  = table(LbinName, Lcnt.', Lpct.', 'VariableNames', {'L_bin','count','percent'});

% scalar summary
p = grpCnt / sum(grpCnt);
entropyPatch = -sum(p .* log2(max(p, eps)));
giniPatch    = 1 - sum(p.^2);

Tscalar = table( ...
    globalPredLabelStr, gTop1, gTop2, gMargin, gConfLevel, ...
    numPatches, ...
    dominantClass, dominantPercent, ...
    entropyPatch, giniPatch, ...
    P.tileSize(1), P.tileSize(2), P.overlapRatio, strideY, strideX, P.confThresh, ...
    'VariableNames', { ...
    'globalPred','globalTop1','globalTop2','globalMargin','globalConfLevel', ...
    'numPatches', ...
    'dominantPatchClass','dominantPatchPercent', ...
    'entropyPatch','giniPatch', ...
    'tileH','tileW','overlap','strideY','strideX','confThresh'});

%% ========================= 9) Save Outputs =========================
[~, baseName, ~] = fileparts(fn_img);
outDir = fullfile(fp_img, sprintf('%s_ROCKTYPE_STRIP_OUT', baseName));
if ~exist(outDir,'dir'); mkdir(outDir); end

timeTag  = char(datetime("now","Format","yyyyMMdd_HHmmss"));
xlsxPath = fullfile(outDir, 'rocktype_patch_analysis.xlsx');

writetable(Tsummary,  xlsxPath, 'Sheet','summary',        'WriteMode','overwritesheet','UseExcel',false);
writetable(patchInfo, xlsxPath, 'Sheet','patch_detail',  'WriteMode','overwritesheet','UseExcel',false);
writetable(Tscalar,   xlsxPath, 'Sheet','summary_scalar','WriteMode','overwritesheet','UseExcel',false);
writetable(Tcolor,    xlsxPath, 'Sheet','color_global',  'WriteMode','overwritesheet','UseExcel',false);
writetable(TcolorL,   xlsxPath, 'Sheet','color_L_hist',  'WriteMode','overwritesheet','UseExcel',false);

fprintf("[Save] Excel: %s\n", xlsxPath);

% Save figures
saveCurrentFigByName('Patch Strip Heatmaps',     fullfile(outDir, sprintf('patch_strip_%s.png', timeTag)));
saveCurrentFigByName('Rock Type Global Prediction', fullfile(outDir, sprintf('global_bar_%s.png', timeTag)));

% MAT 저장
analysisMeta = struct();
analysisMeta.imagePath   = imgPath;
analysisMeta.netMatPath  = modelPath;
analysisMeta.time        = datetime("now");
analysisMeta.inputSize   = inputSize;
analysisMeta.tileSize    = P.tileSize;
analysisMeta.overlap     = P.overlapRatio;
analysisMeta.stride      = [strideY strideX];
analysisMeta.confThresh  = P.confThresh;
analysisMeta.classNames  = classNames;

matPathOut = fullfile(outDir, sprintf('rocktype_strip_result_%s.mat', timeTag));
save(matPathOut, ...
    'patchMap','confMap','margMap', ...
    'YPatchStr','scoresPatch','patchInfo', ...
    'Tsummary','Tscalar','Tcolor','TcolorL', ...
    'analysisMeta', ...
    '-v7.3');

fprintf("[Save] MAT: %s\n", matPathOut);
fprintf("\n완료.\n");

end

%% ======================= Local Helpers =======================

function net = pickNetworkFromStruct_local(S)
net = [];
fn = fieldnames(S);
for k = 1:numel(fn)
    v = S.(fn{k});
    if isa(v,'SeriesNetwork') || isa(v,'DAGNetwork') || isa(v,'dlnetwork')
        net = v;
        fprintf("[Info] 네트워크 변수: %s\n", fn{k});
        return;
    end
end
end

function netOut = fixBatchNormStats_local(netIn)
net = netIn;

% dlnetwork -> DAG/Series로 변환
if isa(net,'dlnetwork')
    try
        lgraph = layerGraph(net);
        net = assembleNetwork(lgraph);
    catch
        % 변환 실패하면 그대로 사용(분석만 하는 경우 종종 문제 없음)
        netOut = netIn;
        return;
    end
end

% DAGNetwork에서 BatchNorm 통계 비어있으면 보정
if isa(net,'DAGNetwork')
    lgraph = layerGraph(net);
    layers = lgraph.Layers;
    for i = 1:numel(layers)
        if isa(layers(i),'nnet.cnn.layer.BatchNormalizationLayer')
            bn = layers(i);
            changed = false;
            if isempty(bn.TrainedMean)
                bn.TrainedMean = zeros(size(bn.Offset), 'like', bn.Offset);
                changed = true;
            end
            if isempty(bn.TrainedVariance)
                bn.TrainedVariance = ones(size(bn.Offset), 'like', bn.Offset);
                changed = true;
            end
            if changed
                lgraph = replaceLayer(lgraph, bn.Name, bn);
            end
        end
    end
    net = assembleNetwork(lgraph);
end

netOut = net;
end

function [inputSize, layerName] = getNetInputSize_local(net)
layers = net.Layers;
idx = [];

% 1) ImageInputLayer 최우선
for i = 1:numel(layers)
    if isa(layers(i), 'nnet.cnn.layer.ImageInputLayer')
        idx = i; break;
    end
end

% 2) fallback: InputSize 보유 + (H,W>1)
if isempty(idx)
    for i = 1:numel(layers)
        if isprop(layers(i),'InputSize')
            sz = layers(i).InputSize;
            if numel(sz)==3 && all(sz(1:2) > 1)
                idx = i; break;
            end
        end
    end
end

if isempty(idx)
    error("입력(ImageInputLayer/InputSize) 레이어를 찾지 못했습니다.");
end

inputSize  = layers(idx).InputSize;
layerName  = layers(idx).Name;
end

function classNames = getClassNamesFromNet_local(net)
classNames = strings(0,1);
layers = net.Layers;
for i = numel(layers):-1:1
    L = layers(i);

    if isprop(L,'Classes') && ~isempty(L.Classes)
        classNames = string(L.Classes);
        return;
    end

    if isprop(L,'Categories') && ~isempty(L.Categories)
        try
            classNames = string(categories(L.Categories));
        catch
            classNames = string(L.Categories);
        end
        return;
    end
end
end

function Iout = ensure_rgb_local(I)
if ismatrix(I)
    Iout = repmat(I,1,1,3);
else
    if size(I,3) == 1
        Iout = repmat(I,1,1,3);
    else
        Iout = I(:,:,1:3);
    end
end
end

function execEnv = pickExecutionEnvironment_local()
execEnv = "cpu";
try
    if canUseGPU
        try
            if gpuDeviceCount("available") > 0
                execEnv = "gpu";
            else
                execEnv = "cpu";
            end
        catch
            execEnv = "gpu";
        end
    end
catch
    execEnv = "cpu";
end
execEnv = char(execEnv);
end

function fig = plotStripHeatmaps_local(patchMap, confMap, classNames, figPos)
% 사용자 예시 느낌: 위=클래스 맵(이산), 아래=confidence 맵(연속)
K = numel(classNames);

fig = figure('Name','Patch Strip Heatmaps','NumberTitle','off', 'Position', figPos);
tiledlayout(2,1,'TileSpacing','compact','Padding','compact');

% --- 1) Class map
ax1 = nexttile;
imagesc(patchMap, [1 K]);
axis tight;
set(ax1,'YDir','normal');
title('Patch Class Map (Grid Heatmap)');
xlabel('Patch Column (X)'); ylabel('Patch Row (Y)');
cb1 = colorbar;
cb1.Ticks = 1:K;
cb1.TickLabels = classNames;

% --- 2) Confidence map (Top1 prob)
ax2 = nexttile;
imagesc(confMap, [0 1]);
axis tight;
set(ax2,'YDir','normal');
title('Patch Confidence Map (Top-1 Probability)');
xlabel('Patch Column (X)'); ylabel('Patch Row (Y)');
colorbar;

end

function saveCurrentFigByName(figName, outPath)
fh = findobj('Type','figure','Name',figName);
if ~isempty(fh)
    fh = fh(1);
    exportgraphics(fh, outPath, 'Resolution',300);
    fprintf("[Save] %s: %s\n", figName, outPath);
end
end
