function results = rock_color_gmm_labeling(varargin)
% rock_color_gmm_labeling_latest_onefile
% =========================================================================
% MATLAB R2025a/b 최신형 ONE-FILE 버전
%
% 핵심 변경
%   - resnet50() -> imagePretrainedNetwork("resnet50")
%   - layerActivations/activations -> minibatchpredict(..., Outputs=...)
%   - dlnetwork 기반 최신 추론 경로
%   - Color/Texture = CPU parfor
%   - Deep feature / PCA / Silhouette / CH / DB = GPU 우선
%   - CT 추출은 batch + checkpoint 구조
%   - 메모리 안전 우선
%
% 실행 예
%   results = rock_color_gmm_labeling_latest_onefile;
%
% 주의
%   - 기본 FeatureLayer = "avg_pool"
%   - 버전별 layer name 차이가 있으면 summary(net)로 확인 후 변경
% =========================================================================

%% ===== Parameters =====
p = inputParser;

addParameter(p,'ImageRoots',{
    'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\images_igneous_rock\TS_1, 기반암 암종 분류 데이터_1. 화성암_1'
    'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\images_igneous_rock\TS_1, 기반암 암종 분류 데이터_1. 화성암_2'
    'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\images_igneous_rock\TS_2. 기반암 절리 탐지 데이터_1. 화성암'
    'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\images_igneous_rock\VS_1. 기반암 암종 분류 데이터_1. 화성암'
    'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\images_igneous_rock\VS_2. 기반암 절리 탐지 데이터_1. 화성암'});

addParameter(p,'OutputDir',fullfile(pwd,...
    "ROCK_COLOR_GMM_LATEST_" + string(datetime('now','Format','yyyyMMdd_HHmmss'))));

addParameter(p,'UseParfor',true);
addParameter(p,'ResumeCheckpoint',true);
addParameter(p,'SaveFigures',true);
addParameter(p,'SaveExcel',true);
addParameter(p,'SaveDeepFeatInMat',false);
addParameter(p,'FigDPI',250);
addParameter(p,'ImageExts',{'.jpg','.jpeg','.png','.bmp','.tif','.tiff'});
addParameter(p,'RockFilter','');
addParameter(p,'RandomSeed',42);

addParameter(p,'UseFastMask',true);
addParameter(p,'UseGPUCompute',true);
addParameter(p,'CTBatchSize',2000);

addParameter(p,'Krange',2:8);
addParameter(p,'GMMOuterRepeats',3);
addParameter(p,'GMMReplicates',5);
addParameter(p,'GMMMaxIter',400);
addParameter(p,'MinClusterPct',1.0);
addParameter(p,'MergeSmallClusters',true);
addParameter(p,'SilhouetteSample',8000);

addParameter(p,'HistBinsHue',6);
addParameter(p,'ABEntropyBins',8);

addParameter(p,'FeatureLayer',"avg_pool");
addParameter(p,'MiniBatchSize',16);
addParameter(p,'MiniBatchMin',4);
addParameter(p,'DeepChunkSize',2048);
addParameter(p,'UseDeepRandomProj',true);
addParameter(p,'DeepProjDim',24);
addParameter(p,'DeepPCAMaxPC',24);

addParameter(p,'ColorPCAExplained',97);
addParameter(p,'TexturePCAExplained',95);
addParameter(p,'FusionPCAExplained',95);
addParameter(p,'ColorMaxPC',16);
addParameter(p,'TextureMaxPC',10);
addParameter(p,'FusionMaxPC',32);
addParameter(p,'ColorMinPC',4);
addParameter(p,'TextureMinPC',4);
addParameter(p,'FusionMinPC',6);

addParameter(p,'ColorWeight',2.50);
addParameter(p,'TextureWeight',1.00);
addParameter(p,'DeepWeight',0.50);

addParameter(p,'TrainRatio',0.80);
addParameter(p,'ValRatio',0.10);
addParameter(p,'TestRatio',0.10);
addParameter(p,'BootstrapN',20);
addParameter(p,'BootstrapRatio',0.70);

parse(p,varargin{:});
opts = p.Results;

if abs(opts.TrainRatio + opts.ValRatio + opts.TestRatio - 1) > 1e-8
    error('rock_color_gmm:InvalidRatio','TrainRatio+ValRatio+TestRatio 합은 1이어야 합니다.');
end
if opts.DeepChunkSize < 1
    error('rock_color_gmm:InvalidChunk','DeepChunkSize는 1 이상이어야 합니다.');
end
if opts.CTBatchSize < 1
    error('rock_color_gmm:InvalidCTBatch','CTBatchSize는 1 이상이어야 합니다.');
end
if ~license('test','Statistics_Toolbox')
    error('rock_color_gmm:NoStatsTbx','Statistics and Machine Learning Toolbox가 필요합니다.');
end

rng(opts.RandomSeed,'twister');
t0 = tic;

outDir = char(opts.OutputDir);
if ~exist(outDir,'dir')
    mkdir(outDir);
end

fprintf('\n%s\n',repmat('=',1,88));
fprintf('  rock_color_gmm_labeling_latest_onefile\n');
fprintf('  OutputDir        : %s\n',outDir);
fprintf('  UseParfor        : %s\n',string(opts.UseParfor));
fprintf('  UseGPUCompute    : %s\n',string(opts.UseGPUCompute));
fprintf('  FeatureLayer     : %s\n',string(opts.FeatureLayer));
fprintf('  UseDeepRandomProj: %s\n',string(opts.UseDeepRandomProj));
fprintf('  Krange           : %d ~ %d\n',min(opts.Krange),max(opts.Krange));
fprintf('  Weights  Color=%.2f  Texture=%.2f  Deep=%.2f\n', ...
    opts.ColorWeight,opts.TextureWeight,opts.DeepWeight);
fprintf('%s\n\n',repmat('=',1,88));

[gpuOK,~] = initGPU();
gpuCompute = gpuOK && opts.UseGPUCompute;
if gpuCompute
    fprintf('  GPU Compute: ON\n');
else
    fprintf('  GPU Compute: OFF\n');
end
checkSystemMemory(opts);

%% ===== [1] 파일 수집 =====
fprintf('--- [1] 파일 수집 ---\n');
imageFiles = collectFiles(opts.ImageRoots,opts.ImageExts);

if strlength(string(opts.RockFilter)) > 0
    [~,s0,~] = cellfun(@fileparts,imageFiles,'UniformOutput',false);
    imageFiles = imageFiles(contains(upper(string(s0)),upper(string(opts.RockFilter))));
end

if isempty(imageFiles)
    error('rock_color_gmm:NoImages','이미지 없음. ImageRoots 경로를 확인하세요.');
end
fprintf('  이미지: %s\n\n',fmtN(numel(imageFiles)));

%% ===== [2] Color + Texture 추출 =====
fprintf('--- [2] Color + Texture 추출 ---\n');

ckptCT = fullfile(outDir,'CKPT_colorTexture.mat');
colorFeatureNames   = buildColorFeatureNames(opts.HistBinsHue);
textureFeatureNames = buildTextureFeatureNames();

nC   = numel(colorFeatureNames);
nT   = numel(textureFeatureNames);
nImg = numel(imageFiles);

if opts.ResumeCheckpoint && exist(ckptCT,'file')
    fprintf('  [CT 체크포인트] 로드 중...\n');
    try
        ck = load(ckptCT,'colorFeat','textureFeat','doneFlags');
        colorFeat   = ck.colorFeat;
        textureFeat = ck.textureFeat;
        doneFlags   = ck.doneFlags;

        if size(colorFeat,1) ~= nImg || size(textureFeat,1) ~= nImg
            fprintf('  체크포인트 크기 불일치 -> 초기화\n');
            colorFeat   = nan(nImg,nC,'single');
            textureFeat = nan(nImg,nT,'single');
            doneFlags   = false(nImg,1);
        else
            fprintf('  재개: %s / %s 완료\n',fmtN(nnz(doneFlags)),fmtN(nImg));
        end
    catch ME
        safeWarn(ME,'CT 체크포인트 로드 실패 -> 초기화');
        colorFeat   = nan(nImg,nC,'single');
        textureFeat = nan(nImg,nT,'single');
        doneFlags   = false(nImg,1);
    end
else
    colorFeat   = nan(nImg,nC,'single');
    textureFeat = nan(nImg,nT,'single');
    doneFlags   = false(nImg,1);
end

todoIdx = find(~doneFlags);
fprintf('  추출 필요: %s장\n',fmtN(numel(todoIdx)));

if ~isempty(todoIdx)
    hEdges = linspace(0,360,opts.HistBinsHue+1);
    aEdges = linspace(-60,60,opts.ABEntropyBins+1);
    bEdges = linspace(-20,80,opts.ABEntropyBins+1);

    usePar = opts.UseParfor && checkParallelToolbox();
    fprintf('  병렬처리: %s\n',ternStr(usePar,'parfor','serial'));

    nTodo  = numel(todoIdx);
    nBatch = ceil(nTodo / opts.CTBatchSize);
    tCT    = tic;

    for bi = 1:nBatch
        s1 = (bi-1)*opts.CTBatchSize + 1;
        s2 = min(bi*opts.CTBatchSize,nTodo);
        idxBatch = todoIdx(s1:s2);
        nB = numel(idxBatch);

        fprintf('  Batch %d/%d  [%s ~ %s]\n',bi,nBatch,fmtN(s1),fmtN(s2));

        cBuf = zeros(nB,nC,'single');
        tBuf = zeros(nB,nT,'single');
        dBuf = false(nB,1);

        if usePar
            filesBatch = imageFiles(idxBatch);
            useFM = opts.UseFastMask;
            parfor ii = 1:nB
                cr = zeros(1,nC,'single');
                tr = zeros(1,nT,'single');
                ok = false;
                try
                    [cr,tr] = extractOneImage(filesBatch{ii},hEdges,aEdges,bEdges,nC,nT,useFM,false);
                    ok = true;
                catch
                end
                cBuf(ii,:) = cr;
                tBuf(ii,:) = tr;
                dBuf(ii)   = ok;
            end
        else
            useFM = opts.UseFastMask;
            warnCnt = 0;
            warnMax = 5;
            for ii = 1:nB
                try
                    [cBuf(ii,:),tBuf(ii,:)] = extractOneImage(imageFiles{idxBatch(ii)}, ...
                        hEdges,aEdges,bEdges,nC,nT,useFM,false);
                    dBuf(ii) = true;
                catch ME
                    warnCnt = warnCnt + 1;
                    if warnCnt <= warnMax
                        safeWarn(ME,sprintf('[CT %d/%d] 추출 실패',ii,nB));
                    elseif warnCnt == warnMax + 1
                        fprintf('    경고 %d건 초과 -> 억제\n',warnMax);
                    end
                end
            end
        end

        colorFeat(idxBatch,:)   = cBuf;
        textureFeat(idxBatch,:) = tBuf;
        doneFlags(idxBatch)     = dBuf;

        save(ckptCT,'colorFeat','textureFeat','doneFlags','-v7.3');
        fprintf('    누적 완료: %s / %s (%.0fs)\n',fmtN(nnz(doneFlags)),fmtN(nImg),toc(tCT));
    end
end

nOK_c = nnz(all(isfinite(colorFeat),2));
nOK_t = nnz(all(isfinite(textureFeat),2));
fprintf('  color 완전유효: %s  texture 완전유효: %s / %s\n', ...
    fmtN(nOK_c),fmtN(nOK_t),fmtN(nImg));

diagNaN(colorFeat,colorFeatureNames,'colorFeat');
diagNaN(textureFeat,textureFeatureNames,'textureFeat');

colorFeat   = imputeNaN(colorFeat);
textureFeat = imputeNaN(textureFeat);

validMask = doneFlags;
nFail = nnz(~validMask);
if nFail > 0
    fprintf('  추출 실패 %s장 -> 제외\n',fmtN(nFail));
end

imageFiles  = imageFiles(validMask);
colorFeat   = colorFeat(validMask,:);
textureFeat = textureFeat(validMask,:);
nImg        = size(colorFeat,1);

fprintf('  유효 이미지: %s\n',fmtN(nImg));
if nImg < 10
    error('rock_color_gmm:TooFewValidImages','유효 이미지 %d개 — 최소 10개 필요.',nImg);
end
fprintf('  Color  : %s x %d\n',fmtN(nImg),size(colorFeat,2));
fprintf('  Texture: %s x %d\n\n',fmtN(nImg),size(textureFeat,2));

%% ===== [3] ResNet-50 deep representation (Latest) =====
fprintf('--- [3] ResNet-50 deep representation [Latest] ---\n');
[net,~,inputSize] = loadResNet50Latest(); 
fprintf('  Layer=%s | Input=%dx%dx%d | MB=%d | Chunk=%d\n', ...
    string(opts.FeatureLayer),inputSize(1),inputSize(2),inputSize(3), ...
    opts.MiniBatchSize,opts.DeepChunkSize);

[deepFeat,deepScore,deepPCdim,deepInfo,usedMB,deepInputDim] = ...
    extractDeepChunkedLatest(net,imageFiles,opts,gpuOK,outDir,inputSize);
fprintf('  Deep input dim : %d\n',deepInputDim);
fprintf('  Deep output dim: %d\n',deepPCdim);
fprintf('  사용 MB        : %d\n\n',usedMB);

%% ===== [4] 메타 생성 =====
fprintf('--- [4] 메타 생성 ---\n');
metaCell  = cell(nImg,4);
luxVals   = nan(nImg,1);
groupKeys = strings(nImg,1);

for i = 1:nImg
    [~,st,~] = fileparts(imageFiles{i});
    m = parseImageMeta(st);
    metaCell(i,:) = {m.object_id,st,m.group_key,m.lux};
    luxVals(i)    = m.lux;
    groupKeys(i)  = string(m.group_key);
end
luxBin = makeLuxBins(luxVals);
fprintf('  완료\n\n');

%% ===== [5] PCA =====
fprintf('--- [5] PCA 차원 축소 (GPU=%s) ---\n',string(gpuCompute));

[colorScore,colorPCdim,colorPCAInfo] = blockPCA( ...
    colorFeat,opts.ColorPCAExplained,opts.ColorMaxPC,opts.ColorMinPC,gpuCompute);

[textureScore,texturePCdim,texturePCAInfo] = blockPCA( ...
    textureFeat,opts.TexturePCAExplained,opts.TextureMaxPC,opts.TextureMinPC,gpuCompute);

fprintf('  Color   PCA: %d dim (%.1f%%)\n',colorPCdim,colorPCAInfo.CumExplained(colorPCdim));
fprintf('  Texture PCA: %d dim (%.1f%%)\n',texturePCdim,texturePCAInfo.CumExplained(texturePCdim));
fprintf('  Deep repr  : %d dim (%s)\n',deepPCdim,string(deepInfo.Method));

fusionInput = [ ...
    opts.ColorWeight   * colorScore(:,1:colorPCdim), ...
    opts.TextureWeight * textureScore(:,1:texturePCdim), ...
    opts.DeepWeight    * deepScore(:,1:deepPCdim)];

fusionInputDim = size(fusionInput,2); %#ok<NASGU>

[fusedScore,fusedPCdim,fusionPCAInfo] = blockPCA( ...
    fusionInput,opts.FusionPCAExplained,opts.FusionMaxPC,opts.FusionMinPC,gpuCompute);

Xpca = fusedScore(:,1:fusedPCdim);
fprintf('  Fusion PCA: %d dim (%.1f%%)\n\n',fusedPCdim,fusionPCAInfo.CumExplained(fusedPCdim));

clear fusedScore fusionInput colorScore textureScore;

%% ===== [6] GMM K 탐색 =====
fprintf('--- [6] GMM K 탐색 ---\n');
[kSummaryTbl,repeatTbl,bestK_premerge,gmBestPremerge] = evaluateKCandidates(Xpca,opts,gpuCompute);

fprintf('\n  [K 결과]\n');
for i = 1:height(kSummaryTbl)
    mk = ternStr(kSummaryTbl.K(i)==bestK_premerge,'  ★ BEST','');
    fprintf('  K=%d | Score=%.4f | BIC=%.1f | Sil=%.4f | DB=%.4f%s\n', ...
        kSummaryTbl.K(i),kSummaryTbl.CompositeScore(i),kSummaryTbl.BIC_med(i), ...
        kSummaryTbl.Sil_med(i),kSummaryTbl.DB_med(i),mk);
end
fprintf('\n');

%% ===== [7] 클러스터링 + 병합 =====
fprintf('--- [7] 클러스터링 ---\n');
clusterIdx = cluster(gmBestPremerge,Xpca);
minSize    = max(1,round(nImg*opts.MinClusterPct/100));
mergeLog   = table();
bestK_final = bestK_premerge;

if opts.MergeSmallClusters
    [clusterIdx,bestK_final,mergeLog] = mergeSmallClusters(clusterIdx,Xpca,minSize);
end

counts = accumarray(clusterIdx,1,[bestK_final,1]);
for k = 1:bestK_final
    fprintf('  C%d: %s (%.2f%%)\n',k,fmtN(counts(k)),100*counts(k)/nImg);
end
fprintf('\n');

%% ===== [8] Bootstrap =====
fprintf('--- [8] Bootstrap ---\n');
[bootCentersPC,centersPC,centerStdPC] = bootstrapGMMStability(Xpca,clusterIdx,bestK_final,opts); 
fprintf('  N=%d 완료\n\n',opts.BootstrapN);

%% ===== [9] 색상 이름 =====
fprintf('--- [9] 색상 이름 배정 ---\n');
idxL50 = fIdx(colorFeatureNames,'L_p50');
idxa50 = fIdx(colorFeatureNames,'a_p50');
idxb50 = fIdx(colorFeatureNames,'b_p50');
idxC50 = fIdx(colorFeatureNames,'C_p50');

centersColor = zeros(bestK_final,4);
rgbCenters   = zeros(bestK_final,3);

rIdx = fIdx(colorFeatureNames,'R_mean');
gIdx = fIdx(colorFeatureNames,'G_mean');
bIdx = fIdx(colorFeatureNames,'B_mean');

for k = 1:bestK_final
    mk = (clusterIdx==k);
    centersColor(k,:) = [ ...
        median(colorFeat(mk,idxL50),'omitnan'), ...
        median(colorFeat(mk,idxa50),'omitnan'), ...
        median(colorFeat(mk,idxb50),'omitnan'), ...
        median(colorFeat(mk,idxC50),'omitnan') ];
    rgbCenters(k,:) = [ ...
        mean(colorFeat(mk,rIdx),'omitnan'), ...
        mean(colorFeat(mk,gIdx),'omitnan'), ...
        mean(colorFeat(mk,bIdx),'omitnan') ];
end

colorNames = assignColorNames(centersColor);
labelStr   = strings(nImg,1);
for k = 1:bestK_final
    labelStr(clusterIdx==k) = colorNames(k);
end

for k = 1:bestK_final
    fprintf('  %-14s | L=%6.2f a=%6.2f b=%6.2f C=%6.2f | RGB=[%.0f %.0f %.0f]\n', ...
        colorNames(k),centersColor(k,1),centersColor(k,2),centersColor(k,3),centersColor(k,4), ...
        rgbCenters(k,1),rgbCenters(k,2),rgbCenters(k,3));
end
fprintf('\n');

%% ===== [10] Silhouette =====
fprintf('--- [10] Silhouette (GPU=%s) ---\n',string(gpuCompute));
nSilSample = min(nImg,opts.SilhouetteSample);
rng(opts.RandomSeed+100,'twister');
silIdx     = randperm(nImg,nSilSample);
cIdxSil    = clusterIdx(silIdx);

if numel(unique(cIdxSil)) >= 2
    silVals = gpuSilhouette(Xpca(silIdx,:),cIdxSil,gpuCompute);
    fprintf('  mean silhouette = %.4f\n\n',mean(silVals,'omitnan'));
else
    silVals = nan(nSilSample,1);
    fprintf('  계산 생략\n\n');
end

%% ===== [11] Group-aware split =====
fprintf('--- [11] Group-aware split ---\n');
splitVec = groupAwareStratifiedSplit(groupKeys,labelStr,luxBin, ...
    opts.TrainRatio,opts.ValRatio,opts.TestRatio,opts.RandomSeed);
fprintf('  Train=%s | Val=%s | Test=%s\n\n', ...
    fmtN(nnz(splitVec=="train")),fmtN(nnz(splitVec=="val")),fmtN(nnz(splitVec=="test")));

%% ===== [12] 클러스터 요약 =====
clusterSummaryTbl = buildClusterSummaryTable( ...
    clusterIdx,colorNames,centersColor,rgbCenters,colorFeat,textureFeat, ...
    colorFeatureNames,textureFeatureNames,nImg);

%% ===== [13] Figure 저장 =====
if opts.SaveFigures
    fprintf('--- [13] Figure 저장 ---\n');
    try
        figs = makeFigures(kSummaryTbl,Xpca,colorFeat,clusterIdx,colorNames, ...
            silVals,cIdxSil,counts,nImg,colorFeatureNames);

        fNames = {'01_model_selection','02_pca_scatter','03_cielab_scatter', ...
                  '04_silhouette','05_distribution'};
        for fi = 1:numel(figs)
            fp = fullfile(outDir,[fNames{fi} '.png']);
            try
                exportgraphics(figs{fi},fp,'Resolution',opts.FigDPI,'ContentType','image');
            catch ME
                safeWarn(ME,sprintf('exportgraphics 실패: %s',fp));
                try
                    saveas(figs{fi},fp);
                catch ME2
                    safeWarn(ME2,'saveas 실패');
                end
            end
            fprintf('  %s\n',fp);
            try
                close(figs{fi});
            catch
            end
        end
    catch ME
        safeWarn(ME,'Figure 생성 실패');
    end
    fprintf('\n');
end

%% ===== [14] Excel 저장 =====
if opts.SaveExcel
    fprintf('--- [14] Excel 저장 ---\n');
    try
        saveAllExcel(outDir,imageFiles,metaCell,groupKeys,labelStr,clusterIdx,splitVec, ...
            colorFeat,colorFeatureNames,textureFeat,textureFeatureNames, ...
            clusterSummaryTbl,kSummaryTbl,repeatTbl,mergeLog, ...
            silVals,cIdxSil,bootCentersPC,centersPC, ...
            colorPCAInfo,texturePCAInfo,deepInfo,fusionPCAInfo, ...
            colorPCdim,texturePCdim,deepPCdim,fusedPCdim, ...
            bestK_premerge,bestK_final,gpuOK,usedMB,opts);
        fprintf('  Excel 저장 완료\n\n');
    catch ME
        safeWarn(ME,'Excel 저장 실패');
    end
end

%% ===== [15] MAT 저장 =====
fprintf('--- [15] MAT 저장 ---\n');
matPath = fullfile(outDir,'ROCK_COLOR_GMM_LATEST.mat');

deepFeatToSave = [];
if opts.SaveDeepFeatInMat && ~opts.UseDeepRandomProj
    deepFeatToSave = deepFeat;
end

save(matPath, ...
    'imageFiles','metaCell','groupKeys','luxVals','luxBin', ...
    'labelStr','clusterIdx','splitVec', ...
    'colorFeat','colorFeatureNames','textureFeat','textureFeatureNames', ...
    'deepFeatToSave','deepScore','Xpca','gmBestPremerge', ...
    'kSummaryTbl','repeatTbl','mergeLog', ...
    'colorNames','centersColor','rgbCenters','clusterSummaryTbl', ...
    'bootCentersPC','centersPC','centerStdPC', ...
    'colorPCAInfo','texturePCAInfo','deepInfo','fusionPCAInfo', ...
    'colorPCdim','texturePCdim','deepPCdim','fusedPCdim', ...
    'bestK_premerge','bestK_final','gpuOK','usedMB','opts','-v7.3');

fprintf('  MAT: %s\n',matPath);

% checkpoint cleanup
for ck = {ckptCT, fullfile(outDir,'CKPT_deep_latest.mat')}
    if exist(ck{1},'file')
        try
            delete(ck{1});
        catch
        end
    end
end
fprintf('  체크포인트 삭제 완료\n\n');

results = struct( ...
    'nImages',nImg, ...
    'bestK_premerge',bestK_premerge, ...
    'bestK_final',bestK_final, ...
    'imageFiles',{imageFiles}, ...
    'labelStr',labelStr, ...
    'clusterIdx',clusterIdx, ...
    'splitVec',splitVec, ...
    'colorFeat',colorFeat, ...
    'colorFeatureNames',{colorFeatureNames}, ...
    'textureFeat',textureFeat, ...
    'textureFeatureNames',{textureFeatureNames}, ...
    'deepFeat',deepFeat, ...
    'deepScore',deepScore, ...
    'Xpca',Xpca, ...
    'gmModelPremerge',gmBestPremerge, ...
    'colorNames',colorNames, ...
    'centersColor',centersColor, ...
    'outDir',outDir, ...
    'gpuOK',gpuOK, ...
    'usedMB',usedMB, ...
    'fusedPCdim',fusedPCdim, ...
    'deepInfo',deepInfo);

elapsed = toc(t0);
fprintf('%s\n',repmat('=',1,88));
fprintf('  DONE | %.1f min | %s img | K=%d/%d | FusionPCA=%d | GPU=%s\n', ...
    elapsed/60,fmtN(nImg),bestK_premerge,bestK_final,fusedPCdim,upper(string(gpuOK)));
fprintf('  출력: %s\n',outDir);
fprintf('%s\n\n',repmat('=',1,88));
end

%% =========================================================================
%%  GPU 초기화
%% =========================================================================
function [gpuOK,execEnv] = initGPU()
gpuOK   = false;
execEnv = 'cpu';

try
    parallel.gpu.enableCUDAForwardCompatibility(true);
catch
end

try
    setenv("MW_CUDA_FORWARD_COMPATIBILITY","1");
    setenv("CUDA_CACHE_MAXSIZE","536870912");
catch
end

try
    nGPU = gpuDeviceCount("available");
    if nGPU > 0
        g = gpuDevice(1);
        fprintf('GPU[1/%d]: %s | %.2f / %.2f GB VRAM (free/total) | CC %s\n', ...
            nGPU,g.Name,g.AvailableMemory/1e9,g.TotalMemory/1e9,string(g.ComputeCapability));
        gpuOK = true;
        execEnv = 'gpu';
    end
catch ME
    warning(ME.identifier,'%s',ME.message);
    gpuOK = false;
    execEnv = 'cpu';
end

if ~gpuOK
    fprintf('GPU 미사용 -> CPU\n');
end
end

%% =========================================================================
%%  시스템 메모리 확인
%% =========================================================================
function checkSystemMemory(opts)
try
    if ispc
        [~,mem] = memory;
        availGB = mem.PhysicalMemory.Available/1e9;
        fprintf('시스템 가용 RAM: %.1f GB\n',availGB);
        if availGB < 4
            fprintf('  [경고] 가용 RAM %.1f GB — 부족\n',availGB);
        end
    end
catch
end

if ~opts.SaveDeepFeatInMat
    fprintf('  [안내] SaveDeepFeatInMat=false -> MAT 크기 절감\n');
end
if opts.UseDeepRandomProj
    fprintf('  [안내] UseDeepRandomProj=true -> RAM 절감\n');
end
end

%% =========================================================================
%%  최신형 ResNet-50 로드
%% =========================================================================
function [net,classNames,inputSize] = loadResNet50Latest()
try
    [net,classNames] = imagePretrainedNetwork("resnet50");
catch ME
    error('rock_color_gmm:ResNet50Unavailable', ...
        'imagePretrainedNetwork("resnet50") 로드 실패: %s',ME.message);
end

if ~isa(net,'dlnetwork')
    error('rock_color_gmm:InvalidNetType','예상과 달리 dlnetwork가 아닙니다.');
end

try
    inputSize = net.Layers(1).InputSize;
catch
    inputSize = [224 224 3];
end
end

%% =========================================================================
%%  최신형 Deep feature 추출
%% =========================================================================
function [deepFeat,deepScore,deepPCdim,deepInfo,usedMB,featDim] = ...
    extractDeepChunkedLatest(net,imageFiles,opts,gpuOK,outDir,inputSize)

nImg   = numel(imageFiles);
ckptD  = fullfile(outDir,'CKPT_deep_latest.mat');
usedMB = adjustMiniBatchToVRAMLatest(opts.MiniBatchSize,opts.MiniBatchMin,gpuOK,inputSize);

chunkN = max(1,min(opts.DeepChunkSize,nImg));
nChunk = ceil(nImg/chunkN);

ix0 = 1:min(chunkN,nImg);
[firstFeat,usedMB] = forwardChunkWithRetryLatest( ...
    net,imageFiles(ix0),opts.FeatureLayer,usedMB,opts.MiniBatchMin,gpuOK,inputSize);

firstFeat = single(firstFeat);
featDim   = size(firstFeat,2);

startChunk = 2;
Rmat = [];

if opts.ResumeCheckpoint && exist(ckptD,'file')
    try
        dk = load(ckptD,'deepBuf','lastChunk','featDimCk','usedMBck','Rmat');
        if dk.featDimCk == featDim
            fprintf('  [Deep 체크포인트] chunk %d/%d 재개\n',dk.lastChunk,nChunk);
            deepBuf    = dk.deepBuf;
            startChunk = dk.lastChunk + 1;
            usedMB     = dk.usedMBck;
            if isfield(dk,'Rmat')
                Rmat = dk.Rmat;
            end
        end
    catch ME
        safeWarn(ME,'Deep 체크포인트 로드 실패 -> 초기화');
    end
end

if ~exist('deepBuf','var')
    if opts.UseDeepRandomProj
        rng(opts.RandomSeed+9999,'twister');
        Rmat = randn(featDim,opts.DeepProjDim,'single') / sqrt(single(opts.DeepProjDim));
        deepBuf = zeros(nImg,opts.DeepProjDim,'single');
        deepBuf(ix0,:) = firstFeat * Rmat;
    else
        deepBuf = zeros(nImg,featDim,'single');
        deepBuf(ix0,:) = firstFeat;
    end
end

fprintf('  chunk 1/%d  [%s/%s]\n',nChunk,fmtN(numel(ix0)),fmtN(nImg));

for ci = startChunk:nChunk
    s1 = (ci-1)*chunkN + 1;
    s2 = min(ci*chunkN,nImg);
    idxChunk = s1:s2;

    [featChunk,usedMB] = forwardChunkWithRetryLatest( ...
        net,imageFiles(idxChunk),opts.FeatureLayer,usedMB,opts.MiniBatchMin,gpuOK,inputSize);

    featChunk = single(featChunk);
    if size(featChunk,2) ~= featDim
        error('rock_color_gmm:DeepDimMismatch', ...
            'Deep feature 차원 불일치: chunk=%d expected=%d actual=%d', ...
            ci,featDim,size(featChunk,2));
    end

    if opts.UseDeepRandomProj
        deepBuf(idxChunk,:) = featChunk * Rmat;
    else
        deepBuf(idxChunk,:) = featChunk;
    end

    fprintf('  chunk %d/%d  [%s/%s]\n',ci,nChunk,fmtN(s2),fmtN(nImg));

    if mod(ci,10)==0 || ci==nChunk
        lastChunk = ci; featDimCk = featDim; usedMBck = usedMB; 
        save(ckptD,'deepBuf','lastChunk','featDimCk','usedMBck','Rmat','-v7.3');
    end
end

clear firstFeat;

if opts.UseDeepRandomProj
    deepScore = deepBuf;
    deepFeat  = [];
    deepPCdim = opts.DeepProjDim;
    deepInfo  = struct( ...
        'Method',"RandomProjection", ...
        'InputDim',featDim, ...
        'OutputDim',opts.DeepProjDim, ...
        'Coeff',Rmat, ...
        'Mean',zeros(1,featDim,'single'), ...
        'Std',ones(1,featDim,'single'), ...
        'Explained',(100/opts.DeepProjDim)*ones(opts.DeepProjDim,1), ...
        'CumExplained',linspace(100/opts.DeepProjDim,100,opts.DeepProjDim)');
else
    deepFeat = deepBuf;
    [deepScore,deepPCdim,deepInfo] = blockPCA_deepMemLatest(deepFeat,opts.DeepPCAMaxPC,opts.UseGPUCompute);
    deepInfo.Method = "PCA";
    deepInfo.InputDim = featDim;
    deepInfo.OutputDim = deepPCdim;
end
end

%% =========================================================================
%%  최신형 chunk 추론
%% =========================================================================
function [feat,usedMB] = forwardChunkWithRetryLatest(net,filesChunk,layerName,mbStart,mbMin,gpuOK,inputSize)
usedMB = mbStart;
accel = "none";
if gpuOK
    accel = "auto";
end

while usedMB >= mbMin
    try
        X = readImageBatch4D(filesChunk,inputSize);
        if gpuOK
            X = gpuArray(X);
        end

        Y = minibatchpredict(net,X, ...
            Outputs=string(layerName), ...
            MiniBatchSize=usedMB, ...
            Acceleration=accel);

        if isa(Y,'dlarray')
            Y = extractdata(Y);
        end
        if isa(Y,'gpuArray')
            Y = gather(Y);
        end

        feat = reshapeFeatureOutputToRows(Y);
        feat = single(feat);
        return;

    catch ME
        safeWarn(ME,sprintf('Deep chunk 실패 -> MB=%d 재시도',floor(usedMB/2)));
        usedMB = floor(usedMB/2);
        if gpuOK
            try
                g = gpuDevice;
                reset(g);
            catch
            end
        end
    end
end

error('rock_color_gmm:DeepChunkFail','Deep feature chunk 추출 실패 (최소 MB=%d)',mbMin);
end

%% =========================================================================
%%  이미지 batch -> 4D single
%% =========================================================================
function X = readImageBatch4D(filesChunk,inputSize)
n = numel(filesChunk);
H = inputSize(1); W = inputSize(2); C = inputSize(3);

X = zeros(H,W,C,n,'single');
for i = 1:n
    img = imread(filesChunk{i});
    img = ensureRGBuint8(img);
    if size(img,1) ~= H || size(img,2) ~= W
        img = imresize(img,[H W]);
    end
    X(:,:,:,i) = single(img);
end
end

%% =========================================================================
%%  출력 reshape
%% =========================================================================
function feat = reshapeFeatureOutputToRows(Y)
sz = size(Y);

if ismatrix(Y)
    if sz(1) > sz(2)
        feat = Y';
    else
        feat = Y;
    end
    return;
end

if ndims(Y) == 4
    % typical: 1x1xC x N
    if sz(1)==1 && sz(2)==1
        feat = squeeze(Y);
        if isvector(feat)
            feat = feat(:)';
        elseif size(feat,1) > size(feat,2)
            feat = feat';
        end
        return;
    end
end

feat = squeeze(Y);
if isvector(feat)
    feat = feat(:)';
elseif size(feat,1) > size(feat,2)
    feat = feat';
end
end

%% =========================================================================
%%  MB 자동 조정
%% =========================================================================
function usedMB = adjustMiniBatchToVRAMLatest(mbStart,mbMin,gpuOK,inputSize)
usedMB = mbStart;
if ~gpuOK
    return;
end

try
    g = gpuDevice;
    freeGB = g.AvailableMemory/1e9;
    estGB = usedMB * inputSize(1) * inputSize(2) * inputSize(3) * 4 / 1e9 + 0.35;

    if freeGB < estGB*2.5
        safeMB = max(mbMin, floor(usedMB * freeGB / max(estGB*3,1e-6)));
        if safeMB < usedMB
            fprintf('  VRAM %.1f GB -> MB: %d -> %d\n',freeGB,usedMB,safeMB);
            usedMB = safeMB;
        end
    end
catch
end
end

%% =========================================================================
%%  Deep PCA 최신형
%% =========================================================================
function [score,nPC,info] = blockPCA_deepMemLatest(deepFeat,maxPC,gpuOK)
X = double(deepFeat);
mu = mean(X,1,'omitnan');
sd = std(X,0,1,'omitnan');
sd(sd<1e-8) = 1;

nComp = min([maxPC,size(X,2),size(X,1)-1]);
fprintf('    Deep PCA: %sx%d -> %d comp (GPU=%s)\n', ...
    fmtN(size(X,1)),size(X,2),nComp,string(gpuOK));

Xz = (X-mu)./sd;
clear X

if gpuOK && numel(Xz) > 1e5
    try
        [coeff,score,latent,explained] = gpuPCA(Xz,nComp);
    catch ME
        safeWarn(ME,'Deep GPU PCA fallback -> CPU');
        try
            g = gpuDevice;
            reset(g);
        catch
        end
        [coeff,score,latent,~,explained] = pca(Xz,'NumComponents',nComp,'Algorithm','svd','Rows','complete');
    end
else
    [coeff,score,latent,~,explained] = pca(Xz,'NumComponents',nComp,'Algorithm','svd','Rows','complete');
end

cumExp = cumsum(explained);
ix = find(cumExp >= 95,1,'first');
if isempty(ix)
    nPC = nComp;
else
    nPC = ix;
end
nPC = min(max(nPC,4),nComp);

info = struct('Mean',mu,'Std',sd,'Coeff',coeff,'Latent',latent, ...
    'Explained',explained,'CumExplained',cumExp);
end

%% =========================================================================
%%  단일 이미지 특징 추출
%% =========================================================================
function [cRow,tRow] = extractOneImage(imgPath,hEdges,aEdges,bEdges,nC,nT,useFastMask,useGPU)
if nargin<8
    useGPU = false;
end

cRow = zeros(1,nC,'single');
tRow = zeros(1,nT,'single');

img = imread(imgPath);
img = ensureRGBuint8(img);
if isempty(img)
    return;
end

if useGPU
    imgD_g = gpuArray(im2double(img));
    gray_g = rgb2gray(imgD_g);
    gray   = gather(gray_g);
    t = graythresh(gray);
    mask = gray > max(0.10,min(0.90,0.7*t));
    if nnz(mask) < 50
        mask = true(size(gray));
    end

    imgD = gather(imgD_g);
    lab  = rgb2lab(imgD);

    grayVt = gray(mask);
    if numel(grayVt) < 4
        grayVt = gray(:);
    end

    grayMfill = gray;
    grayMfill(~mask) = median(grayVt,'omitnan');

    grayFill_g = gpuArray(grayMfill);
    [Gmag_g,~] = imgradient(grayFill_g);
    F_g = abs(fftshift(fft2(grayFill_g)));

    Gmag = gather(Gmag_g);
    Fabs = gather(F_g);

    try
        E = edge(grayMfill,'canny');
    catch
        E = [];
    end
else
    imgD = im2double(img);
    gray = rgb2gray(imgD);

    if useFastMask
        mask = fastMask(gray);
    else
        mask = fullMask(gray);
    end
    if nnz(mask) < 50
        mask = true(size(gray));
    end

    lab = rgb2lab(imgD);

    grayVt = gray(mask);
    if numel(grayVt) < 4
        grayVt = gray(:);
    end

    grayMfill = gray;
    grayMfill(~mask) = median(grayVt,'omitnan');

    Gmag = [];
    Fabs = [];
    E = [];
end

R = double(img(:,:,1));
G = double(img(:,:,2));
B = double(img(:,:,3));

Rm = zeroIfNaN(mean(R(mask),'omitnan'));
Gm = zeroIfNaN(mean(G(mask),'omitnan'));
Bm = zeroIfNaN(mean(B(mask),'omitnan'));

Lc = double(lab(:,:,1));
ac = double(lab(:,:,2));
bc = double(lab(:,:,3));
Cc = hypot(ac,bc);
hc = atan2d(bc,ac);
hc(hc<0) = hc(hc<0) + 360;

Lv = Lc(mask); av = ac(mask); bv = bc(mask); Cv = Cc(mask); hv = hc(mask);
if numel(Lv) < 4
    Lv = Lc(:); av = ac(:); bv = bc(:); Cv = Cc(:); hv = hc(:);
end

Lp = zeroIfNaNVec(prctile(Lv,[5 50 95]));
ap = zeroIfNaNVec(prctile(av,[5 50 95]));
bp = zeroIfNaNVec(prctile(bv,[5 50 95]));
Cp = zeroIfNaNVec(prctile(Cv,[5 50 95]));

Liqr = zeroIfNaN(iqr(Lv));
aiqr = zeroIfNaN(iqr(av));
biqr = zeroIfNaN(iqr(bv));
Ciqr = zeroIfNaN(iqr(Cv));

hHist = histcounts(hv,hEdges,'Normalization','probability');
q1 = zeroIfNaN(mean(av>=0 & bv>=0));
q2 = zeroIfNaN(mean(av<0  & bv>=0));
q3 = zeroIfNaN(mean(av<0  & bv<0));
q4 = zeroIfNaN(mean(av>=0 & bv<0));

H2 = histcounts2(av,bv,aEdges,bEdges,'Normalization','probability');
H2 = H2(H2>0);
abEnt = 0;
if ~isempty(H2)
    abEnt = zeroIfNaN(-sum(H2 .* log2(H2)));
end

cVals = [Rm,Gm,Bm, ...
    Lp(1),Lp(2),Lp(3),Liqr, ...
    ap(1),ap(2),ap(3),aiqr, ...
    bp(1),bp(2),bp(3),biqr, ...
    Cp(1),Cp(2),Cp(3),Ciqr, ...
    hHist,q1,q2,q3,q4,abEnt];

% texture
grayTex = imresize(grayMfill,[256 256]);
maskTex = imresize(mask,[256 256],'nearest');

grayV = grayTex(maskTex);
if numel(grayV) < 4
    grayV = grayTex(:);
end

grayStd = zeroIfNaN(std(grayV,0,'omitnan'));
entVal  = zeroIfNaN(entropy(grayTex));

edgeDensity = 0;
try
    if isempty(E)
        Etmp = edge(grayTex,'canny');
    else
        Etmp = imresize(E,[256 256],'nearest');
    end
    edgeDensity = zeroIfNaN(mean(Etmp(maskTex),'omitnan'));
catch
end

g50 = 0; g95 = 0; giqr = 0;
try
    if isempty(Gmag)
        Gm2 = imgradient(grayTex);
    else
        Gm2 = imresize(Gmag,[256 256]);
    end
    gv = Gm2(maskTex);
    if numel(gv) < 4
        gv = Gm2(:);
    end
    g50  = zeroIfNaN(median(gv,'omitnan'));
    g95  = zeroIfNaN(prctile(gv,95));
    giqr = zeroIfNaN(iqr(gv));
catch
end

con = 0; cor = 0; ene = 0; hom = 0;
try
    I16  = uint8(floor(mat2gray(grayTex)*15));
    glcm = graycomatrix(I16,'NumLevels',16,'Offset',[0 1;-1 1;-1 0;-1 -1],'Symmetric',true);
    gp   = graycoprops(glcm,{'Contrast','Correlation','Energy','Homogeneity'});
    con  = zeroIfNaN(mean(gp.Contrast,'omitnan'));
    cor  = zeroIfNaN(mean(gp.Correlation,'omitnan'));
    ene  = zeroIfNaN(mean(gp.Energy,'omitnan'));
    hom  = zeroIfNaN(mean(gp.Homogeneity,'omitnan'));
catch
end

fftL = 0; fftM = 0; fftH = 0;
try
    if isempty(Fabs)
        [fftL,fftM,fftH] = fftBandRatios(grayTex);
    else
        [fftL,fftM,fftH] = fftBandRatios(imresize(Fabs,[256 256]));
    end
    fftL = zeroIfNaN(fftL);
    fftM = zeroIfNaN(fftM);
    fftH = zeroIfNaN(fftH);
catch
end

pMu = 0; pSig = 0;
try
    [pMu,pSig] = patchSimilarity(grayTex);
    pMu  = zeroIfNaN(pMu);
    pSig = zeroIfNaN(pSig);
catch
end

tVals = [grayStd,entVal,edgeDensity,g50,g95,giqr, ...
    con,cor,ene,hom,fftL,fftM,fftH,pMu,pSig];

if numel(cVals)~=nC || numel(tVals)~=nT
    return;
end

cRow = single(cVals);
tRow = single(tVals);
end

%% =========================================================================
%%  Mask / FFT / Patch
%% =========================================================================
function mask = fastMask(gray)
gray = mat2gray(gray);
t = graythresh(gray);
mask = gray > max(0.10,min(0.90,0.7*t));
if nnz(mask) < 100
    mask = true(size(gray));
end
end

function mask = fullMask(gray)
gray = mat2gray(gray);
t = graythresh(gray);
mask = gray > max(0.10,min(0.90,0.7*t));
mask = imfill(mask,'holes');
mask = bwareaopen(mask,100);

if nnz(mask) > 0
    cc = bwconncomp(mask);
    nP = cellfun(@numel,cc.PixelIdxList);
    [~,ix] = max(nP);
    m2 = false(size(mask));
    m2(cc.PixelIdxList{ix}) = true;
    mask = m2;
else
    mask = true(size(gray));
end
end

function [lR,mR,hR] = fftBandRatios(I)
I = im2double(I);

if ~ismatrix(I)
    I = mat2gray(I);
end

F = abs(fftshift(fft2(I)));
F = F / max(sum(F(:)),eps);

[m,n] = size(F);
cx = (m+1)/2; cy = (n+1)/2;
[X,Y] = ndgrid(1:m,1:n);
Rr = sqrt((X-cx).^2 + (Y-cy).^2) / max(hypot(m/2,n/2));

lR = sum(F(Rr<=0.15),'all');
mR = sum(F(Rr>0.15 & Rr<=0.45),'all');
hR = sum(F(Rr>0.45),'all');
end

function [mC,sC] = patchSimilarity(I)
I = imresize(im2double(I),[64 64]);

nB = 4;
bs = 16;
P = cell(nB,nB);

for r = 1:nB
    for c = 1:nB
        P{r,c} = I((r-1)*bs+(1:bs),(c-1)*bs+(1:bs));
    end
end

corrs = [];
for r = 1:nB
    for c = 1:nB
        p1 = P{r,c}(:);
        if c < nB
            corrs(end+1,1) = safeCorr(p1,P{r,c+1}(:)); %#ok<AGROW>
        end
        if r < nB
            corrs(end+1,1) = safeCorr(p1,P{r+1,c}(:)); %#ok<AGROW>
        end
    end
end

corrs = corrs(isfinite(corrs));
if isempty(corrs)
    mC = 0;
    sC = 0;
    return;
end

mC = mean(corrs,'omitnan');
sC = std(corrs,0,'omitnan');
end

function c = safeCorr(x,y)
if std(x,0,'omitnan') < 1e-12 || std(y,0,'omitnan') < 1e-12
    c = 0;
    return;
end

C = corrcoef(x,y);
if size(C,1) >= 2
    c = C(1,2);
else
    c = 0;
end
end

%% =========================================================================
%%  피처 이름
%% =========================================================================
function names = buildColorFeatureNames(nH)
names = {'R_mean','G_mean','B_mean', ...
         'L_p05','L_p50','L_p95','L_iqr', ...
         'a_p05','a_p50','a_p95','a_iqr', ...
         'b_p05','b_p50','b_p95','b_iqr', ...
         'C_p05','C_p50','C_p95','C_iqr'};
for i = 1:nH
    names{end+1} = sprintf('h_hist_%02d',i); %#ok<AGROW>
end
names = [names,{'ab_q1','ab_q2','ab_q3','ab_q4','ab_entropy'}];
end

function names = buildTextureFeatureNames()
names = {'gray_std','gray_entropy','edge_density','grad_p50','grad_p95','grad_iqr', ...
         'glcm_contrast','glcm_correlation','glcm_energy','glcm_homogeneity', ...
         'fft_low_ratio','fft_mid_ratio','fft_high_ratio','patch_corr_mean','patch_corr_std'};
end

%% =========================================================================
%%  NaN 유틸
%% =========================================================================
function v = zeroIfNaN(v)
if ~isfinite(v)
    v = 0;
end
end

function v = zeroIfNaNVec(v)
v(~isfinite(v)) = 0;
end

function X = imputeNaN(X)
X(~isfinite(X)) = 0;
end

function diagNaN(feat,names,tag)
cols = find(any(~isfinite(feat),1));
if isempty(cols)
    return;
end
fprintf('  [진단] %s NaN 피처:\n',tag);
for ci = cols
    r = mean(~isfinite(feat(:,ci)));
    fprintf('    %-28s NaN %.1f%%\n',names{ci},100*r);
end
end

%% =========================================================================
%%  PCA
%% =========================================================================
function [score,nPC,info] = blockPCA(X,targetExp,maxPC,minPC,gpuOK)
if nargin < 5
    gpuOK = false;
end

X  = double(X);
mu = mean(X,1,'omitnan');
sd = std(X,0,1,'omitnan');
sd(sd<1e-8) = 1;
Xz = (X-mu)./sd;

numComp = min([maxPC,size(Xz,2),size(Xz,1)-1]);
numComp = max(numComp,minPC);

if gpuOK && numel(Xz) > 1e5
    try
        [coeff,score,latent,explained] = gpuPCA(Xz,numComp);
    catch ME
        safeWarn(ME,'blockPCA GPU fallback -> CPU');
        try
            g = gpuDevice;
            reset(g);
        catch
        end
        [coeff,score,latent,~,explained] = pca(Xz,'NumComponents',numComp,'Algorithm','svd','Rows','complete');
    end
else
    [coeff,score,latent,~,explained] = pca(Xz,'NumComponents',numComp,'Algorithm','svd','Rows','complete');
end

cumExp = cumsum(explained);
ix = find(cumExp >= targetExp,1,'first');
if isempty(ix)
    nPC = numComp;
else
    nPC = ix;
end
nPC = min(max(nPC,minPC),numComp);

info = struct('Mean',mu,'Std',sd,'Coeff',coeff,'Latent',latent, ...
    'Explained',explained,'CumExplained',cumExp);
end

%% =========================================================================
%%  GPU PCA
%% =========================================================================
function [coeff,score,latent,explained] = gpuPCA(Xz,numComp)
[n,~] = size(Xz);
gXz = gpuArray(single(Xz));

[~,S,V] = svd(gXz,'econ');
svals   = gather(double(diag(S)));

latent = svals.^2 / (n-1);
latent = max(latent,0);

totalVar = sum(latent);
if totalVar > 0
    explained = 100 * latent / totalVar;
else
    explained = zeros(numel(latent),1);
end

V = gather(double(V));
for j = 1:min(numComp,size(V,2))
    [~,mxi] = max(abs(V(:,j)));
    if V(mxi,j) < 0
        V(:,j) = -V(:,j);
    end
end

coeff     = V(:,1:numComp);
latent    = latent(1:numComp);
explained = explained(1:numComp);
score     = gather(double(gXz * gpuArray(single(V(:,1:numComp)))));
end

%% =========================================================================
%%  GPU Silhouette
%% =========================================================================
function silVals = gpuSilhouette(X,idx,gpuOK)
if nargin < 3
    gpuOK = false;
end

N = size(X,1);
K = max(idx);
if K < 2
    silVals = zeros(N,1);
    return;
end

if gpuOK
    try
        if isa(X,'gpuArray')
            gX = X;
        else
            gX = gpuArray(single(X));
        end

        aVals = zeros(N,1,'single','gpuArray');
        bVals = inf(N,1,'single','gpuArray');

        for k = 1:K
            mk = (idx==k);
            nk = nnz(mk);
            if nk == 0
                continue;
            end

            Dk     = pdist2(gX,gX(mk,:));
            meanDk = mean(Dk,2);
            intra  = sum(Dk(mk,:),2) / max(nk-1,1);

            aVals(mk) = intra;

            others = ~mk;
            bVals(others) = min(bVals(others),meanDk(others));
        end

        denom = max(aVals,bVals);
        denom(denom < eps('single')) = eps('single');
        silVals = gather(double((bVals-aVals)./denom));
        return;
    catch ME
        safeWarn(ME,'gpuSilhouette fallback -> CPU');
        try
            g = gpuDevice; reset(g);
        catch
        end
    end
end

silVals = silhouette(X,idx);
end

%% =========================================================================
%%  GMM K 탐색
%% =========================================================================
function [kST,repTbl,bestK,bestGM] = evaluateKCandidates(Xpca,opts,gpuOK)
Krange = opts.Krange(:);
nK = numel(Krange);
N  = size(Xpca,1);

nSil = min(N,opts.SilhouetteSample);
rng(opts.RandomSeed+1,'twister');
silIdx = randperm(N,nSil);
Xsil = Xpca(silIdx,:);

if gpuOK
    try
        gXpca = gpuArray(single(Xpca));
        gXsil = gpuArray(single(Xsil));
    catch
        gXpca = Xpca;
        gXsil = Xsil;
        gpuOK = false;
        try
            g = gpuDevice; reset(g);
        catch
        end
    end
else
    gXpca = Xpca;
    gXsil = Xsil;
end

sOpts = statset('MaxIter',opts.GMMMaxIter,'Display','off');
maxRep = nK * opts.GMMOuterRepeats;
repRows = cell(maxRep,9); ri = 0;

Kcol = nan(nK,1);
BIC_med = nan(nK,1); AIC_med = nan(nK,1);
Sil_med = nan(nK,1); CH_med = nan(nK,1); DB_med = nan(nK,1);
MinPct_med = nan(nK,1); Dispersion = nan(nK,1);
Nvalid = zeros(nK,1); BestBIC = nan(nK,1); BestGM = cell(nK,1);

for ki = 1:nK
    K = Krange(ki);
    fprintf('  K=%d\n',K);
    rpM = nan(opts.GMMOuterRepeats,6);
    bBIC = inf;
    gmBK = [];

    for rr = 1:opts.GMMOuterRepeats
        fprintf('    r%d ... ',rr);
        t1 = tic;

        try
            rng(opts.RandomSeed + 1000*K + rr,'twister');
            gm = fitGMMrobust(Xpca,K,opts.GMMReplicates,sOpts);
            idxA = cluster(gm,Xpca);

            cnt = accumarray(idxA,1,[K 1]);
            minP = 100 * min(cnt) / N;

            silM = -1;
            if nSil >= 2 && numel(unique(idxA(silIdx))) >= 2
                silM = mean(gpuSilhouette(gXsil,idxA(silIdx),gpuOK),'omitnan');
            end

            ch = calcCH(gXpca,idxA,K,gpuOK);
            db = calcDB(gXpca,idxA,K,gpuOK);

            rpM(rr,:) = [gm.BIC,gm.AIC,silM,ch,db,minP];
            ri = ri + 1;
            repRows(ri,:) = {K,rr,gm.BIC,gm.AIC,silM,ch,db,minP,gm.Converged};

            if gm.BIC < bBIC
                bBIC = gm.BIC;
                gmBK = gm;
            end

            fprintf('ok (%.1fs)\n',toc(t1));
        catch ME
            safeWarn(ME,sprintf('GMM K=%d r=%d fail',K,rr));
            ri = ri + 1;
            repRows(ri,:) = {K,rr,NaN,NaN,NaN,NaN,NaN,NaN,false};
            fprintf('fail\n');
        end
    end

    vm = isfinite(rpM(:,1)) & isfinite(rpM(:,3)) & isfinite(rpM(:,4)) & isfinite(rpM(:,5));
    Nvalid(ki) = nnz(vm);
    Kcol(ki) = K;

    if any(vm)
        x = rpM(vm,:);
        BIC_med(ki) = median(x(:,1),'omitnan');
        AIC_med(ki) = median(x(:,2),'omitnan');
        Sil_med(ki) = median(x(:,3),'omitnan');
        CH_med(ki)  = median(x(:,4),'omitnan');
        DB_med(ki)  = median(x(:,5),'omitnan');
        MinPct_med(ki) = median(x(:,6),'omitnan');

        d1 = std(x(:,1),0,'omitnan') / max(abs(BIC_med(ki)),1e-9);
        d2 = std(x(:,3),0,'omitnan');
        d3 = std(x(:,5),0,'omitnan');
        Dispersion(ki) = d1 + d2 + d3;
        BestBIC(ki) = bBIC;
        BestGM{ki} = gmBK;
    end
end

repRows = repRows(1:ri,:);
if isempty(repRows)
    repTbl = table('Size',[0 9], ...
        'VariableTypes',{'double','double','double','double','double','double','double','double','logical'}, ...
        'VariableNames',{'K','Repeat','BIC','AIC','Silhouette','CH','DB','MinClusterPct','Converged'});
else
    repTbl = cell2table(repRows,'VariableNames', ...
        {'K','Repeat','BIC','AIC','Silhouette','CH','DB','MinClusterPct','Converged'});
end

bicN  = nMTZ(BIC_med,'lower');
silN  = nMTZ(Sil_med,'higher');
dbN   = nMTZ(DB_med,'lower');
chN   = nMTZ(CH_med,'higher');
minN  = nMTZ(MinPct_med,'higher');
dispN = nMTZ(Dispersion,'lower');

composite = 0.30*bicN + 0.25*silN + 0.20*dbN + 0.10*chN + 0.10*minN + 0.05*dispN;

vK = isfinite(Kcol) & isfinite(BIC_med) & isfinite(Sil_med) & isfinite(DB_med) & ...
     (Nvalid > 0) & ~cellfun(@isempty,BestGM);

if ~any(vK)
    error('rock_color_gmm:NoValidModel','유효 GMM 모델 없음');
end

sc = composite;
sc(~vK) = inf;
[~,bi] = min(sc);
bestK = Kcol(bi);
bestGM = BestGM{bi};

if isempty(bestGM)
    error('rock_color_gmm:EmptyBest','bestGM 비어 있음');
end

kST = table(Kcol,BIC_med,AIC_med,Sil_med,CH_med,DB_med,MinPct_med, ...
    Dispersion,Nvalid,BestBIC,composite,vK, ...
    'VariableNames',{'K','BIC_med','AIC_med','Sil_med','CH_med','DB_med','MinPct_med', ...
                     'Dispersion','Nvalid','BestRepBIC','CompositeScore','Valid'});
end

function gm = fitGMMrobust(X,K,reps,sOpts)
try
    gm = fitgmdist(X,K,'CovarianceType','full','SharedCovariance',false, ...
        'RegularizationValue',1e-4,'Replicates',reps,'Options',sOpts);
catch
    try
        gm = fitgmdist(X,K,'CovarianceType','diagonal','SharedCovariance',false, ...
            'RegularizationValue',1e-4,'Replicates',max(2,ceil(reps/2)),'Options',sOpts);
    catch
        gm = fitgmdist(X,K,'CovarianceType','diagonal','SharedCovariance',true, ...
            'RegularizationValue',1e-3,'Replicates',2,'Options',sOpts);
    end
end
end

function y = nMTZ(x,dir)
y = nan(size(x));
m = isfinite(x);
if ~any(m)
    return;
end
xm = x(m);
rv = max(xm) - min(xm);
if rv < 1e-12
    y(m) = 0;
elseif strcmpi(dir,'lower')
    y(m) = (xm - min(xm)) / rv;
else
    y(m) = (max(xm) - xm) / rv;
end
y(~m) = 1;
end

%% =========================================================================
%%  CH / DB
%% =========================================================================
function ch = calcCH(X,idx,K,gpuOK)
N = size(X,1);

if gpuOK
    try
        if isa(X,'gpuArray')
            gX = X;
        else
            gX = gpuArray(single(X));
        end

        oM = mean(gX,1);
        SSB = gpuArray(single(0));
        SSW = gpuArray(single(0));

        for k = 1:K
            m = (idx==k);
            nk = nnz(m);
            if nk == 0
                continue;
            end
            ck = mean(gX(m,:),1);
            SSB = SSB + nk*sum((ck-oM).^2);
            SSW = SSW + sum((gX(m,:)-ck).^2,'all');
        end

        SSB = gather(SSB);
        SSW = gather(SSW);

        if SSW <= 0 || K <= 1 || N <= K
            ch = 0;
            return;
        end
        ch = double((SSB/(K-1)) / (SSW/(N-K)));
        return;
    catch
        try
            g = gpuDevice; reset(g);
        catch
        end
    end
end

oM = mean(X,1);
SSB = 0; SSW = 0;
for k = 1:K
    m = (idx==k);
    nk = nnz(m);
    if nk == 0
        continue;
    end
    ck = mean(X(m,:),1);
    SSB = SSB + nk*sum((ck-oM).^2);
    SSW = SSW + sum(sum((X(m,:)-ck).^2));
end

if SSW <= 0 || K <= 1 || N <= K
    ch = 0;
    return;
end
ch = (SSB/(K-1)) / (SSW/(N-K));
end

function db = calcDB(X,idx,K,gpuOK)
if gpuOK
    try
        if isa(X,'gpuArray')
            gX = X;
        else
            gX = gpuArray(single(X));
        end

        ctrs = zeros(K,size(gX,2),'single','gpuArray');
        sctr = zeros(K,1,'single','gpuArray');

        for k = 1:K
            m = (idx==k);
            if nnz(m) == 0
                continue;
            end
            ctrs(k,:) = mean(gX(m,:),1);
            sctr(k) = mean(sqrt(sum((gX(m,:)-ctrs(k,:)).^2,2)));
        end

        D = pdist2(ctrs,ctrs);
        D(logical(eye(K,'gpuArray'))) = inf;
        R = (sctr + sctr') ./ max(D,eps('single'));
        R(logical(eye(K,'gpuArray'))) = 0;
        db = double(gather(mean(max(R,[],2))));
        return;
    catch
        try
            g = gpuDevice; reset(g);
        catch
        end
    end
end

ctrs = zeros(K,size(X,2));
sctr = zeros(K,1);

for k = 1:K
    m = (idx==k);
    if nnz(m)==0
        continue;
    end
    ctrs(k,:) = mean(X(m,:),1);
    sctr(k) = mean(sqrt(sum((X(m,:)-ctrs(k,:)).^2,2)));
end

R = zeros(K,K);
for i = 1:K
    for j = i+1:K
        d = norm(ctrs(i,:) - ctrs(j,:));
        if d > 0
            R(i,j) = (sctr(i)+sctr(j)) / d;
            R(j,i) = R(i,j);
        end
    end
end

db = mean(max(R,[],2),'omitnan');
end

%% =========================================================================
%%  클러스터 병합 / Bootstrap / 이름
%% =========================================================================
function [cIdx,Kf,log] = mergeSmallClusters(cIdx,X,minSz)
rows = cell(0,3);
changed = true;

while changed
    changed = false;
    [cIdx,Kc] = relabelSeq(cIdx);
    cnt = accumarray(cIdx,1,[Kc,1]);
    sm = find(cnt < minSz);

    if isempty(sm) || Kc <= 2
        break;
    end

    ctr = zeros(Kc,size(X,2));
    for k = 1:Kc
        mk = (cIdx==k);
        if any(mk)
            ctr(k,:) = mean(X(mk,:),1);
        end
    end

    for ii = 1:numel(sm)
        k = sm(ii);
        if cnt(k) >= minSz || Kc <= 2
            continue;
        end
        d = pdist2(ctr(k,:),ctr);
        d(k) = inf;
        if all(~isfinite(d))
            continue;
        end
        [~,nk] = min(d);
        cIdx(cIdx==k) = nk;
        rows(end+1,:) = {k,nk,cnt(k)}; %#ok<AGROW>
        changed = true;
        break;
    end
end

[cIdx,Kf] = relabelSeq(cIdx);
if isempty(rows)
    log = table('Size',[0 3],'VariableTypes',{'double','double','double'}, ...
        'VariableNames',{'FromCluster','ToCluster','MovedN'});
else
    log = cell2table(rows,'VariableNames',{'FromCluster','ToCluster','MovedN'});
end
end

function [i2,K] = relabelSeq(idx)
u = unique(idx(:));
i2 = zeros(size(idx));
for i = 1:numel(u)
    i2(idx==u(i)) = i;
end
K = numel(u);
end

function [bCP,cP,cSP] = bootstrapGMMStability(Xpca,cIdx,K,opts)
N = size(Xpca,1);
nPC = size(Xpca,2);

nBoot = opts.BootstrapN;
nSub  = max(K*5,round(N*optsBootstrapRatio(opts)));

cP = zeros(K,nPC);
for k = 1:K
    mk = (cIdx==k);
    if any(mk)
        cP(k,:) = mean(Xpca(mk,:),1);
    end
end

bCP = zeros(nBoot,K,nPC);
sO = statset('MaxIter',max(200,round(opts.GMMMaxIter*0.7)),'Display','off');

for bi = 1:nBoot
    try
        ix = randperm(N,min(nSub,N));
        gmB = fitGMMrobust(Xpca(ix,:),K,max(2,min(3,opts.GMMReplicates)),sO);
        match = greedyMatch(pdist2(gmB.mu,cP));
        bCP(bi,:,:) = gmB.mu(match,:);
    catch ME
        safeWarn(ME,sprintf('Bootstrap %d 실패',bi));
        bCP(bi,:,:) = cP;
    end
end

cSP = squeeze(std(bCP,0,1));
end

function r = optsBootstrapRatio(opts)
r = opts.BootstrapRatio;
end

function colorNames = assignColorNames(centersColor)
% L*a*b*C 중심값으로 [밝기접두어]_[색상명] 자동 생성
% centersColor: Kx4 [L, a, b, C]
%
% 색상 판단 기준 (CIELAB):
%   a > 0 : 적색 계열,  a < 0 : 녹색 계열
%   b > 0 : 황색 계열,  b < 0 : 청색 계열
%   C(채도) 낮으면 무채색(회색/흰색/검정)
%   L(명도) 0=검정, 100=흰색

K = size(centersColor,1);
colorNames = strings(K,1);

for k = 1:K
    L = centersColor(k,1);
    a = centersColor(k,2);
    b = centersColor(k,3);
    C = centersColor(k,4);

    % --- 1) 기본 색상명 결정 ---
    if C < 5
        % 무채색: 채도가 매우 낮음
        if L > 85
            baseName = "White";
        elseif L > 65
            baseName = "LightGray";
        elseif L > 45
            baseName = "MidGray";
        elseif L > 25
            baseName = "DarkGray";
        else
            baseName = "Black";
        end
    else
        % 유채색: 색상각(hue angle)으로 판단
        hue = atan2d(b, a);
        if hue < 0
            hue = hue + 360;
        end

        if hue < 20 || hue >= 345
            baseName = "Red";
        elseif hue < 45
            baseName = "RedOrange";
        elseif hue < 70
            baseName = "Orange";
        elseif hue < 90
            baseName = "YellowOrange";
        elseif hue < 105
            baseName = "Yellow";
        elseif hue < 150
            baseName = "YellowGreen";
        elseif hue < 170
            baseName = "Green";
        elseif hue < 200
            baseName = "Cyan";
        elseif hue < 260
            baseName = "Blue";
        elseif hue < 290
            baseName = "Violet";
        elseif hue < 330
            baseName = "Pink";
        else
            baseName = "RedPink";
        end

        % 채도 낮으면 ~ish Gray 계열로 표기
        if C < 12
            baseName = baseName + "ishGray";
        end
    end

    % --- 2) 밝기 접두어 ---
    if C >= 5
        if L > 75
            prefix = "Light";
        elseif L > 55
            prefix = "Mid";
        elseif L > 35
            prefix = "Dark";
        else
            prefix = "VeryDark";
        end
    else
        prefix = "";
    end

    % --- 3) 조합 ---
    if strlength(prefix) > 0
        colorNames(k) = prefix + "_" + baseName;
    else
        colorNames(k) = baseName;
    end
end

% --- 4) 중복 이름 처리: 같은 이름이면 밝기순으로 _1, _2 붙임 ---
uNames = unique(colorNames);
for i = 1:numel(uNames)
    idx = find(colorNames == uNames(i));
    if numel(idx) > 1
        % L값 기준 내림차순 정렬 (밝은것부터)
        Lvals = centersColor(idx, 1);
        [~, ord] = sort(Lvals, 'descend');
        for j = 1:numel(idx)
            colorNames(idx(ord(j))) = uNames(i) + "_" + string(j);
        end
    end
end
end

function matched = greedyMatch(D)
K = size(D,1);
matched = zeros(K,1);

for iter = 1:K
    [~,mi] = min(D(:));
    [ri,ci] = ind2sub(size(D),mi);
    matched(ri) = ci;
    D(ri,:) = inf;
    D(:,ci) = inf;
end

used = matched(matched>0);
for i = 1:K
    if matched(i)==0
        avail = setdiff(1:K,used);
        if isempty(avail)
            avail = 1:K;
        end
        matched(i) = avail(1);
        used(end+1) = avail(1); %#ok<AGROW>
    end
end
end

%% =========================================================================
%%  메타 / split
%% =========================================================================
function m = parseImageMeta(stem)
m = struct('object_id',stem,'group_key',deriveImageGroupKey(stem),'lux',NaN);
t = regexp(upper(stem),'LUX[_-]?(\d+(?:\.\d+)?)','tokens','once');
if ~isempty(t)
    m.lux = str2double(t{1});
end
end

function g = deriveImageGroupKey(stem)
g = upper(stem);
g = regexprep(g,'LUX[_-]?\d+(\.\d+)?','');
g = regexprep(g,'DEPTH[_-]?\d+(\.\d+)?','');
g = regexprep(g,'(^|[_-])(W|D|WET|DRY)($|[_-])','_');
g = regexprep(g,'(^|[_-])IMG\d+($|[_-])','_');
g = regexprep(g,'(^|[_-])\d{1,6}($|[_-])','_');
g = regexprep(g,'_+','_');
g = regexprep(g,'^-+|_+$|^_+|-+$','');
if isempty(g)
    g = upper(stem);
end
end

function luxBin = makeLuxBins(v)
luxBin = strings(numel(v),1);
luxBin(:) = "NA";

vf = v(isfinite(v));
if isempty(vf)
    return;
end

vmin = min(vf);
vmax = max(vf);

if vmax <= vmin
    luxBin(isfinite(v)) = "MID";
    return;
end

edges = linspace(vmin,vmax,4);
for i = 1:numel(v)
    if ~isfinite(v(i))
        continue;
    elseif v(i) < edges(2)
        luxBin(i) = "LOW";
    elseif v(i) < edges(3)
        luxBin(i) = "MID";
    else
        luxBin(i) = "HIGH";
    end
end
end

function sv = groupAwareStratifiedSplit(gKeys,lblStr,luxBin,tr,va,te,seed)
rng(seed,'twister');
gKeys = string(gKeys);
lblStr = string(lblStr);
luxBin = string(luxBin);

uG = unique(gKeys);
nG = numel(uG);

gLbl = strings(nG,1);
gLux = strings(nG,1);

for i = 1:nG
    mk = (gKeys==uG(i));
    gLbl(i) = majStr(lblStr(mk));
    gLux(i) = majStr(luxBin(mk));
end

gSt = gLbl + "|" + gLux;
u0 = unique(gSt);

for i = 1:numel(u0)
    ix = find(gSt==u0(i));
    if numel(ix) < 3
        gSt(ix) = gLbl(ix);
    end
end

spGrp = strings(nG,1);
uS = unique(gSt);

for si = 1:numel(uS)
    ix = find(gSt==uS(si));
    ix = ix(randperm(numel(ix)));
    [nT,nV,nTe] = allocSplit(numel(ix),tr,va,te);
    spGrp(ix(1:nT)) = "train";
    spGrp(ix(nT+1:nT+nV)) = "val";
    spGrp(ix(nT+nV+1:nT+nV+nTe)) = "test";
end

spGrp(spGrp=="") = "train";

sv = strings(numel(gKeys),1);
for i = 1:nG
    sv(gKeys==uG(i)) = spGrp(i);
end
end

function s = majStr(x)
x = string(x);
x = x(strlength(strtrim(x))>0);
if isempty(x)
    s = "";
    return;
end
u = unique(x);
cnt = zeros(numel(u),1);
for i = 1:numel(u)
    cnt(i) = nnz(x==u(i));
end
[~,ix] = max(cnt);
s = u(ix);
end

function [nT,nV,nTe] = allocSplit(n,tr,va,te)
if n <= 0
    nT = 0; nV = 0; nTe = 0; return;
elseif n == 1
    nT = 1; nV = 0; nTe = 0; return;
elseif n == 2
    nT = 1; nV = 0; nTe = 1; return;
end

raw = [n*tr,n*va,n*te];
cnt = floor(raw);
remN = n - sum(cnt);
[~,ord] = sort(raw-cnt,'descend');

for i = 1:remN
    cnt(ord(i)) = cnt(ord(i)) + 1;
end

while sum(cnt) > n
    [~,ix] = max(cnt);
    cnt(ix) = cnt(ix) - 1;
end

cnt = max(cnt,[1 1 1]);

while sum(cnt) > n
    [~,ix] = max(cnt-[1 1 1]);
    cnt(ix) = cnt(ix) - 1;
end

nT = cnt(1); nV = cnt(2); nTe = cnt(3);
end

%% =========================================================================
%%  요약 테이블
%% =========================================================================
function tbl = buildClusterSummaryTable(cIdx,colorNames,centersColor,rgbCenters, ...
    colorFeat,textureFeat,cFN,tFN,nImg)

K = numel(colorNames);
iL = fIdx(cFN,'L_p50');
ia = fIdx(cFN,'a_p50');
ib = fIdx(cFN,'b_p50');
iE = fIdx(cFN,'ab_entropy');
iEd = fIdx(tFN,'edge_density');
iG = fIdx(tFN,'grad_p95');
iP = fIdx(tFN,'patch_corr_mean');

rows = cell(K,15);
for k = 1:K
    mk = (cIdx==k);
    rows(k,:) = {k,char(colorNames(k)),nnz(mk),100*nnz(mk)/nImg, ...
        centersColor(k,1),centersColor(k,2),centersColor(k,3),centersColor(k,4), ...
        rgbCenters(k,1),rgbCenters(k,2),rgbCenters(k,3), ...
        median(colorFeat(mk,iE),'omitnan'), ...
        median(textureFeat(mk,iEd),'omitnan'), ...
        median(textureFeat(mk,iG),'omitnan'), ...
        median(textureFeat(mk,iP),'omitnan')};
end

tbl = cell2table(rows,'VariableNames',{'cluster_id','color_name','n','pct', ...
    'L50_center','a50_center','b50_center','C50_center','R_mean','G_mean','B_mean', ...
    'ab_entropy_med','edge_density_med','grad_p95_med','patch_corr_med'});

tbl.L50_median = accM(cIdx,colorFeat(:,iL),K);
tbl.a50_median = accM(cIdx,colorFeat(:,ia),K);
tbl.b50_median = accM(cIdx,colorFeat(:,ib),K);
end

function v = accM(idx,x,K)
v = nan(K,1);
for k = 1:K
    v(k) = median(x(idx==k),'omitnan');
end
end

%% =========================================================================
%%  Figures
%% =========================================================================
function figs = makeFigures(kST,Xpca,colorFeat,cIdx,colorNames,silVals,cIdxSil,counts,nImg,cFN)
K  = numel(colorNames);
cm = lines(K);
si = 1:max(1,floor(size(Xpca,1)/50000)):size(Xpca,1);

iL = fIdx(cFN,'L_p50');
ia = fIdx(cFN,'a_p50');
ib = fIdx(cFN,'b_p50');
iC = fIdx(cFN,'C_p50');

% fig1
fig1 = figure('Color','w','Position',[50 50 1400 400]);
fds = {'BIC_med','Sil_med','CH_med','CompositeScore'};
ttl = {'BIC','Silhouette','CH','Composite'};
[~,bestBI] = min(kST.CompositeScore);
for mi = 1:4
    subplot(1,4,mi);
    plot(kST.K,kST.(fds{mi}),'-o','LineWidth',2,'MarkerFaceColor','w');
    grid on;
    xlabel('K');
    title(ttl{mi});
    xline(kST.K(bestBI),'r--','LineWidth',1.2);
end
sgtitle(sprintf('Model Selection (best K=%d)',kST.K(bestBI)),'FontWeight','bold');

% fig2
fig2 = figure('Color','w','Position',[60 60 1500 450]);
pp = [1 2;1 3;2 3];
for p = 1:3
    ax = subplot(1,3,p); hold(ax,'on');
    hs = gobjects(K,1);
    for k = 1:K
        mk = (cIdx(si)==k);
        hs(k) = scatter(Xpca(si(mk),pp(p,1)),Xpca(si(mk),pp(p,2)), ...
            4,cm(k,:),'filled','MarkerFaceAlpha',0.25);
    end
    xlabel(sprintf('PC%d',pp(p,1)));
    ylabel(sprintf('PC%d',pp(p,2)));
    grid(ax,'on');
    hold(ax,'off');
    if p == 1
        legend(hs,cellstr(colorNames),'Location','bestoutside');
    end
end
sgtitle(sprintf('Fusion PCA (K=%d)',K),'FontWeight','bold');

% fig3
fig3 = figure('Color','w','Position',[70 70 1500 450]);
pd = [ia ib; iL iC; iL ib];
pn = {'a*_p50','b*_p50'; 'L*_p50','C*_p50'; 'L*_p50','b*_p50'};
for p = 1:3
    ax = subplot(1,3,p); hold(ax,'on');
    hs = gobjects(K,1);
    for k = 1:K
        mk = (cIdx(si)==k);
        hs(k) = scatter(colorFeat(si(mk),pd(p,1)),colorFeat(si(mk),pd(p,2)), ...
            4,cm(k,:),'filled','MarkerFaceAlpha',0.25);
    end
    xlabel(pn{p,1});
    ylabel(pn{p,2});
    grid(ax,'on');
    if p == 1
        xline(0,'--','Color',[.5 .5 .5]);
        yline(0,'--','Color',[.5 .5 .5]);
        legend(hs,cellstr(colorNames),'Location','bestoutside');
    end
    hold(ax,'off');
end
sgtitle('CIELAB Color Space','FontWeight','bold');

% fig4
fig4 = figure('Color','w','Position',[80 80 800 450]);
if any(isfinite(silVals))
    g = categorical(cIdxSil,1:K,cellstr(colorNames));
    boxchart(g,silVals);
    hold on;
    yline(mean(silVals,'omitnan'),'r--','LineWidth',1.5);
    hold off;
    title(sprintf('Silhouette (mean=%.4f)',mean(silVals,'omitnan')));
    grid on;
else
    text(.5,.5,'N/A','HorizontalAlignment','center');
    axis off;
end

% fig5
fig5 = figure('Color','w','Position',[90 90 1550 430]);
subplot(1,3,1);
bh = bar(categorical(colorNames,colorNames),counts,'FaceColor','flat');
bh.CData = cm;
ylabel('N');
title('Distribution');
grid on;

subplot(1,3,2);
hold on;
for k = 1:K
    histogram(colorFeat(cIdx==k,ib),60,'Normalization','pdf', ...
        'DisplayStyle','stairs','LineWidth',1.5,'EdgeColor',cm(k,:));
end
xlabel('b*_p50');
ylabel('PDF');
legend(cellstr(colorNames),'Location','best');
grid on;
hold off;

subplot(1,3,3);
hold on;
for k = 1:K
    histogram(colorFeat(cIdx==k,iL),60,'Normalization','pdf', ...
        'DisplayStyle','stairs','LineWidth',1.5,'EdgeColor',cm(k,:));
end
xlabel('L*_p50');
ylabel('PDF');
legend(cellstr(colorNames),'Location','best');
grid on;
hold off;
sgtitle(sprintf('n=%s',fmtN(nImg)),'FontWeight','bold');

figs = {fig1,fig2,fig3,fig4,fig5};
end

%% =========================================================================
%%  Excel
%% =========================================================================
function saveAllExcel(outDir,imageFiles,metaCell,groupKeys,labelStr,cIdx,splitVec, ...
    colorFeat,cFN,textureFeat,tFN,clST,kST,repTbl,mergeLog, ...
    silVals,cIdxSil,bCP,centersPC,cPi,tPi,dPi,fPi, ...
    cPd,tPd,dPd,fPd,bestKp,bestKf,gpuOK,usedMB,opts)

maxRow = 1000000;
luxVals = cell2mat(metaCell(:,4));
luxBin  = makeLuxBins(luxVals);

mTbl = table(string(imageFiles(:)),string(metaCell(:,2)),string(metaCell(:,1)), ...
    string(groupKeys),luxVals,luxBin,labelStr,cIdx,splitVec, ...
    'VariableNames',{'file','stem','object_id','group_key','lux','lux_bin', ...
                     'color_label','cluster_id','split'});

lTbl = [mTbl, ...
    array2table(double(colorFeat),'VariableNames',matlab.lang.makeValidName(cFN)), ...
    array2table(double(textureFeat),'VariableNames',matlab.lang.makeValidName(tFN))];

xl1 = fullfile(outDir,'COLOR_LABELS.xlsx');
if height(lTbl) <= maxRow
    writetable(lTbl,xl1,'Sheet','LABELS');
else
    for i = 1:ceil(height(lTbl)/maxRow)
        r1 = (i-1)*maxRow+1;
        r2 = min(i*maxRow,height(lTbl));
        writetable(lTbl(r1:r2,:),fullfile(outDir,sprintf('COLOR_LABELS_part%03d.xlsx',i)),'Sheet','LABELS');
    end
end

xl2 = fullfile(outDir,'GMM_MODEL_COMPARE.xlsx');
writetable(kST,xl2,'Sheet','K_SUMMARY');
writetable(repTbl,xl2,'Sheet','K_REPEATS');
if ~isempty(mergeLog)
    writetable(mergeLog,xl2,'Sheet','MERGE_LOG');
end

xl3 = fullfile(outDir,'GMM_CLUSTER_STATS.xlsx');
writetable(clST,xl3,'Sheet','CLUSTER_SUMMARY');
wCT(xl3,'COLOR_x_LUXBIN',labelStr,luxBin);
wCT(xl3,'COLOR_x_SPLIT',labelStr,splitVec);
wCT(xl3,'COLOR_x_GROUPKEY',labelStr,string(groupKeys));

xl4 = fullfile(outDir,'GMM_VALIDATION.xlsx');
if any(isfinite(silVals))
    u = unique(cIdxSil);
    sr = cell(numel(u),8);
    for i = 1:numel(u)
        k = u(i);
        sv = silVals(cIdxSil==k);
        sr(i,:) = {k,numel(sv),mean(sv,'omitnan'),std(sv,0,'omitnan'), ...
            min(sv,[],'omitnan'),median(sv,'omitnan'),max(sv,[],'omitnan'),nnz(sv<0)};
    end
    writetable(cell2table(sr,'VariableNames', ...
        {'cluster_id','n','sil_mean','sil_std','sil_min','sil_median','sil_max','n_neg'}), ...
        xl4,'Sheet','SILHOUETTE');
end

nSPC = min(8,size(centersPC,2));
bR = cell(0,8);
for k = 1:size(centersPC,1)
    for pc = 1:nSPC
        vals = squeeze(bCP(:,k,pc));
        bR(end+1,:) = {k,sprintf('PC%d',pc),centersPC(k,pc), ...
            mean(vals,'omitnan'),std(vals,0,'omitnan'), ...
            min(vals,[],'omitnan'),max(vals,[],'omitnan'), ...
            100*std(vals,0,'omitnan')/max(abs(centersPC(k,pc)),1e-9)}; %#ok<AGROW>
    end
end
writetable(cell2table(bR,'VariableNames', ...
    {'cluster_id','PC','center','boot_mean','boot_std','boot_min','boot_max','CV_pct'}), ...
    xl4,'Sheet','BOOTSTRAP');

wPCA(xl4,'PCA_COLOR',cPi);
wPCA(xl4,'PCA_TEXTURE',tPi);
wPCA(xl4,'PCA_DEEP',dPi);
wPCA(xl4,'PCA_FUSION',fPi);

methodTbl = table( ...
    {'pipeline';'deep_method';'color_weight';'texture_weight';'deep_weight'; ...
     'color_pca';'texture_pca';'deep_dim';'fusion_pca'; ...
     'K_pre';'K_final';'gpu';'MB';'UseParfor';'UseFastMask'; ...
     'UseGPUCompute';'DeepChunkSize';'SaveDeepFeatInMat';'FeatureLayer'}, ...
    {sprintf('Color+Texture+ResNet50Latest->PCA->GMM'); ...
     char(string(dPi.Method)); ...
     num2str(opts.ColorWeight);num2str(opts.TextureWeight);num2str(opts.DeepWeight); ...
     num2str(cPd);num2str(tPd);num2str(dPd);num2str(fPd); ...
     num2str(bestKp);num2str(bestKf); ...
     char(string(gpuOK));num2str(usedMB); ...
     char(string(opts.UseParfor));char(string(opts.UseFastMask)); ...
     char(string(opts.UseGPUCompute));num2str(opts.DeepChunkSize); ...
     char(string(opts.SaveDeepFeatInMat));char(string(opts.FeatureLayer))}, ...
    'VariableNames',{'param','value'});
writetable(methodTbl,xl4,'Sheet','METHOD');

writetable(table((1:numel(cFN))',string(cFN(:)),'VariableNames',{'id','name'}),xl4,'Sheet','COLOR_FEATURES');
writetable(table((1:numel(tFN))',string(tFN(:)),'VariableNames',{'id','name'}),xl4,'Sheet','TEXTURE_FEATURES');
end

function wCT(path,sheet,colorLabel,categoryLabel)
colorLabel = string(colorLabel);
categoryLabel = string(categoryLabel);

mk = strlength(strtrim(colorLabel))>0 & strlength(strtrim(categoryLabel))>0;
colorLabel = colorLabel(mk);
categoryLabel = categoryLabel(mk);

if isempty(colorLabel)
    return;
end

uR = unique(colorLabel);
uC = unique(categoryLabel);
rows = cell(0,3);

for i = 1:numel(uR)
    for j = 1:numel(uC)
        n = nnz(colorLabel==uR(i) & categoryLabel==uC(j));
        if n > 0
            rows(end+1,:) = {char(uR(i)),char(uC(j)),n}; %#ok<AGROW>
        end
    end
end

if isempty(rows)
    return;
end

try
    writetable(cell2table(rows,'VariableNames',{'color_name','category','n_images'}),path,'Sheet',sheet);
catch
end
end

function wPCA(path,sheet,info)
if ~isstruct(info) || ~isfield(info,'Explained') || isempty(info.Explained)
    return;
end

n = min(20,numel(info.Explained));
writetable(table((1:n)',info.Explained(1:n),info.CumExplained(1:n), ...
    'VariableNames',{'PC','explained_pct','cumulative_pct'}),path,'Sheet',sheet);
end

%% =========================================================================
%%  IO / 유틸
%% =========================================================================
function files = collectFiles(roots,exts)
files = {};
if ~iscell(roots)
    return;
end

for r = 1:numel(roots)
    root = roots{r};
    if ~exist(root,'dir')
        continue;
    end
    dd = dir(fullfile(root,'**','*'));
    for i = 1:numel(dd)
        if dd(i).isdir
            continue;
        end
        [~,~,e] = fileparts(dd(i).name);
        if any(strcmpi(e,exts))
            files{end+1,1} = fullfile(dd(i).folder,dd(i).name); %#ok<AGROW>
        end
    end
end

files = unique(files,'stable');
end

function idx = fIdx(names,target)
idx = find(strcmp(names,target),1,'first');
if isempty(idx)
    error('rock_color_gmm:NotFound','Feature not found: %s',target);
end
end

function img = ensureRGBuint8(img)
if isempty(img)
    error('rock_color_gmm:EmptyImg','빈 이미지');
end
if ismatrix(img)
    img = repmat(img,1,1,3);
end
if size(img,3) > 3
    img = img(:,:,1:3);
end
if ~isa(img,'uint8')
    img = im2uint8(img);
end
end

function ok = checkParallelToolbox()
ok = false;
try
    ok = license('test','Distrib_Computing_Toolbox') && ~isempty(ver('parallel'));
catch
end
end

function s = ternStr(cond,a,b)
if cond
    s = a;
else
    s = b;
end
end

function s = fmtN(n)
n = round(n);
s = char(string(n));
idx = strfind(s,'.');
if isempty(idx)
    idx = numel(s)+1;
end
for i = idx-4:-3:1
    s = [s(1:i),',',s(i+1:end)]; 
end
end

function safeWarn(ME,prefix)
id = 'rock_color_gmm:runtime';
try
    if ~isempty(ME.identifier) && ~isempty(regexp(ME.identifier,'^\w+:\w+','once'))
        id = ME.identifier;
    end
catch
end

if nargin < 2 || isempty(prefix)
    warning(id,'%s',ME.message);
else
    warning(id,'%s: %s',prefix,ME.message);
end
end