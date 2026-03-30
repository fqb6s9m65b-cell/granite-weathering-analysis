function runInfo = CoreColorAnalyzer_rgb_hsv_cielab_JSON_ExcelOnly(varargin)
% CoreColorAnalyzer_rgb_hsv_cielab_JSON_ExcelOnly  (ONE-FILE, R2025a / RTX 5060 8 GB)
% =========================================================================
% 기본 경로 (준엽 연구실)
%   이미지: TS_1 화성암_1/2, TS_2 화성암, VS_1/2 화성암
%   JSON  : TL_1/2 화성암, VL_1/2 화성암
%   결과  : results\ColorStats_RUN_yyyyMMdd_HHmmss\
%
% 처리 방침
%   1. GPU  → rgb2hsv/rgb2lab 배치 가속 (OOM 시 BATCH 축소 후 CPU 자동 폴백)
%   2. CPU  → parfor 전체 코어 병렬 (읽기+통계)
%   3. Excel → 유일한 핵심 출력 (행/열 한계 초과 시 자동 파일 분할)
%   4. PNG  → 선택 출력 (기본 OFF)
%   - JSON 매칭 성공 이미지만 분석
%   - 읽기/통계 오류 → 경고 후 스킵, SKIP_LOG.csv 저장
%   - 풍화등급(rw) = JSON "rw" 필드만 사용
%   - 픽셀 = raw 전체, 보정 없음
%   ★ .mat 체크포인트 파일 없음 (결과는 메모리 직접 누적 → Excel 저장)
%
% Nature/Science급 통계 분석 파이프라인
% ────────────────────────────────────
%   [7.1] Kruskal-Wallis H test + Dunn's post-hoc (Bonferroni)
%   [7.2] Effect size: η² (eta-squared), Cohen's d pairwise
%   [7.3] Spearman/Pearson correlation + Bootstrap 95% CI + BH-FDR
%   [7.4] Feature importance ranking (ANOVA F-score + MI proxy)
%   [7.5] PCA (주성분 분석, 설명 분산, 로딩)
%   [7.6] Discriminability (Silhouette score + Mahalanobis distance)
%   [7.7] Bootstrap 95% CI for grade-level means
%   [7.8] Nature-level 시각화 (Violin, Heatmap, PCA biplot, Radar, Box)
%   → Excel 14개 추가 시트 자동 저장
% =========================================================================

%% ===== [0] 옵션 =====
OPT = parseOpts(varargin{:});

tstamp  = string(datetime('now','Format','yyyyMMdd_HHmmss'));
outRoot = fullfile(char(OPT.RESULTS_ROOT), char("ColorStats_RUN_"+tstamp));
xlsxDir = fullfile(outRoot,'xlsx');
csvDir  = fullfile(outRoot,'csv');
pngDir  = fullfile(outRoot,'png');
perDir  = fullfile(outRoot,'per_image');
logDir  = fullfile(outRoot,'log');
statDir = fullfile(outRoot,'stat_plots');
ensureDirs({outRoot,xlsxDir,csvDir,pngDir,perDir,logDir,statDir});
try diary off; diary(fullfile(logDir,'run_log.txt')); diary on;
catch
end

fprintf('\n%s\n',repmat('=',1,76));
fprintf('  CoreColorAnalyzer_rgb_hsv_cielab_JSON_ExcelOnly  [Nature Edition]\n');
fprintf('  MATLAB: %s\n', version);
fprintf('  결과: %s\n',outRoot);
fprintf('%s\n\n',repmat('=',1,76));

%% ===== [1] GPU / CPU =====
fprintf('[1] GPU/CPU 초기화\n');
[gpuOK, gpuName, gpuDev] = initGPU(OPT.USE_GPU);
nW = initParallel(OPT.USE_PAR);
fprintf('  GPU: %s | CPU workers: %d\n\n', gpuName, nW);
dynBatch = OPT.GPU_BATCH;

%% ===== [2] 이미지 수집 =====
fprintf('[2] 이미지 수집\n');
allCellFiles = collectImages(OPT.IMAGE_ROOTS, OPT.IMG_EXTS);
nTotal       = numel(allCellFiles);
fprintf('  합계: %d장\n\n', nTotal);
if nTotal == 0
    error('CoreColorAnalyzer:NoImages','이미지를 찾지 못했습니다.');
end

%% ===== [3] JSON 인덱싱 + 매칭 필터 =====
fprintf('[3] JSON 인덱싱\n');
jsonMap = buildJsonMap(OPT.JSON_ROOTS);
fprintf('  JSON 합계: %d개\n', jsonMap.Count);
matchedFiles   = {};
unmatchedFiles = {};
for fi = 1:nTotal
    [~,st,~] = fileparts(allCellFiles{fi});
    if jsonMap.isKey(lower(st))
        matchedFiles{end+1}   = allCellFiles{fi}; %#ok<AGROW>
    else
        unmatchedFiles{end+1} = allCellFiles{fi}; %#ok<AGROW>
    end
end
nMatched   = numel(matchedFiles);
nUnmatched = numel(unmatchedFiles);
fprintf('  매칭 성공: %d장 | 미매칭(스킵): %d장\n\n', nMatched, nUnmatched);
if nUnmatched > 0
    try
        writetable(table(string(unmatchedFiles(:)),'VariableNames',{'file'}), ...
            fullfile(logDir,'UNMATCHED_IMAGES.csv'));
    catch
    end
end
if nMatched == 0
    error('CoreColorAnalyzer:NoMatch','JSON 매칭 이미지 없음. 경로·파일명 확인 필요.');
end
cellFiles = matchedFiles;
nImg      = nMatched;

%% ===== [4] GPU 배치 변환 + CPU parfor 통계 =====
fprintf('[4] 통계 추출 (GPU=%s | workers=%d)\n', tf2s(gpuOK), nW);
t0      = tic;
skipLog = {};
refL    = OPT.REF_D1_L;
refa    = OPT.REF_D1_a;
refb    = OPT.REF_D1_b;
roiMode = OPT.ROI_MODE;
bgTh    = OPT.AUTO_BG_TH;

allRows = cell(nImg,1);
bStart = 1;
bNum   = 0;
while bStart <= nImg
    bNum = bNum + 1;
    bEnd = min(bStart + dynBatch - 1, nImg);
    bIdx = bStart:bEnd;
    nB   = numel(bIdx);

    % ── GPU 여유 메모리 사전 확인 ──
    if gpuOK && ~isempty(gpuDev)
        try
            if gpuDev.AvailableMemory < OPT.GPU_MEM_FLOOR
                fprintf('  [Mem] GPU 여유 %.2f GB → reset\n', ...
                    gpuDev.AvailableMemory/1e9);
                reset(gpuDev);
            end
        catch
        end
    end

    %% 이미지 읽기 (parfor)
    imgs    = cell(nB,1);
    stems   = cell(nB,1);
    fpaths  = cell(nB,1);
    readOK  = true(nB,1);
    readErr = cell(nB,1);
    bFiles  = cellFiles(bIdx);
    parfor ii = 1:nB
        f = bFiles{ii};
        try
            I = ensureRGB(imread(f));
            imgs{ii}   = I;
            fpaths{ii} = f;
            [~,st,~]   = fileparts(f);
            stems{ii}  = st;
        catch ME
            readOK(ii)  = false;
            readErr{ii} = ME.message;
        end
    end
    for ii = 1:nB
        if ~readOK(ii)
            skipLog{end+1} = sprintf('[읽기실패] %s | %s', ...
                cellFiles{bIdx(ii)}, readErr{ii}); %#ok<AGROW>
            warning('CoreColorAnalyzer:Read','%s', ...
                sprintf('읽기 실패 → 스킵: %s', cellFiles{bIdx(ii)}));
        end
    end

    %% GPU 색공간 변환 (OOM → BATCH 반감 → CPU 폴백)
    hsvC   = cell(nB,1);
    labC   = cell(nB,1);
    oomCnt = 0;
    for ii = 1:nB
        if ~readOK(ii); continue; end
        convOK = false;
        while ~convOK
            try
                if gpuOK
                    Ig       = gpuArray(single(imgs{ii})) / 255;
                    hsvC{ii} = double(gather(rgb2hsv(Ig)));
                    labC{ii} = double(gather(rgb2lab(Ig)));
                    clear Ig;
                else
                    Iflt     = im2single(imgs{ii});
                    hsvC{ii} = double(rgb2hsv(Iflt));
                    labC{ii} = double(rgb2lab(Iflt));
                end
                convOK = true;
            catch MEg
                isOOM = gpuOK && ...
                    (contains(MEg.message,'out of memory','IgnoreCase',true) || ...
                     contains(MEg.message,'CUDA','IgnoreCase',true));
                if isOOM && oomCnt < 2
                    oomCnt = oomCnt + 1;
                    try reset(gpuDev);
                    catch
                    end
                    if oomCnt >= 2
                        gpuOK = false;
                        warning('CoreColorAnalyzer:OOM','%s','GPU OOM 2회 → CPU 폴백');
                    else
                        dynBatch = max(1, floor(dynBatch/2));
                        warning('CoreColorAnalyzer:OOM','%s', ...
                            sprintf('GPU OOM → BATCH 축소: %d', dynBatch));
                    end
                else
                    try
                        Iflt     = im2single(imgs{ii});
                        hsvC{ii} = double(rgb2hsv(Iflt));
                        labC{ii} = double(rgb2lab(Iflt));
                        convOK   = true;
                    catch
                        readOK(ii) = false;
                        skipLog{end+1} = sprintf('[변환실패] %s', fpaths{ii}); %#ok<AGROW>
                        convOK = true;
                    end
                end
            end
        end
    end

    %% 통계 계산 (parfor)
    bRows   = cell(nB,1);
    calcErr = cell(nB,1);
    parfor ii = 1:nB
        if ~readOK(ii) || isempty(hsvC{ii}); continue; end
        try
            bRows{ii} = buildRow(imgs{ii}, hsvC{ii}, labC{ii}, ...
                stems{ii}, fpaths{ii}, bIdx(ii), ...
                refL, refa, refb, jsonMap, roiMode, bgTh);
        catch ME2
            calcErr{ii} = ME2.message;
        end
    end
    for ii = 1:nB
        if ~isempty(calcErr{ii})
            skipLog{end+1} = sprintf('[통계실패] %s | %s', ...
                fpaths{ii}, calcErr{ii}); %#ok<AGROW>
            warning('CoreColorAnalyzer:Stat','%s', ...
                sprintf('통계 실패 → 스킵: %s', stems{ii}));
        end
        allRows{bIdx(ii)} = bRows{ii};
    end

    %% 진행 출력
    el    = toc(t0);
    done  = bIdx(end);
    eta   = 0;
    if done > 0; eta = el / done * (nImg - done); end
    nSkip = nnz(cellfun(@isempty, bRows));
    freeTxt = '';
    if gpuOK && ~isempty(gpuDev)
        try freeTxt = sprintf(' | GPU-free %.2fGB', gpuDev.AvailableMemory/1e9);
        catch
        end
    end
    fprintf('  배치 %4d | %6d/%6d | BATCH=%-3d | %.0fs | ETA %.0fs | 스킵%d%s\n', ...
        bNum, done, nImg, dynBatch, el, eta, nSkip, freeTxt);
    bStart = bEnd + 1;
end

%% ===== [5] 스킵 로그 =====
if ~isempty(skipLog)
    try
        writetable(table(string(skipLog(:)),'VariableNames',{'skip_reason'}), ...
            fullfile(logDir,'SKIP_LOG.csv'));
        fprintf('\n  스킵 %d건 → %s\n', numel(skipLog), fullfile(logDir,'SKIP_LOG.csv'));
    catch
    end
end

%% ===== [6] 테이블 조립 =====
fprintf('\n[5] 테이블 조립\n');
validRows = allRows(~cellfun(@isempty, allRows));
nValid    = numel(validRows);
fprintf('  성공 %d / 매칭 %d / 전체 %d\n', nValid, nMatched, nTotal);
if nValid == 0
    error('CoreColorAnalyzer:NoValid','분석된 이미지가 없습니다.');
end

T = struct2table(vertcat(validRows{:}));
T = stringifyCols(T);

fprintf('  풍화등급 분포:\n');
[rwU,~,rwI] = unique(T.rw);
for ri = 1:numel(rwU)
    n = nnz(rwI == ri);
    fprintf('    rw=%-8s : %5d장 (%.1f%%)\n', rwU(ri), n, 100*n/nValid);
end
fprintf('\n');

%% ===== [7] 집계 =====
fprintf('[6] 집계\n');
numCols = getNumCols(T);
T_grade = gradeAgg(T, 'rw',       numCols);
T_humid = gradeAgg(T, 'humidity', numCols);
T_cross = crossTab(T, 'rw', 'humidity');
fprintf('  완료\n\n');

%% =====================================================================
%%  ★★★  [7.x] Nature/Science급 통계 분석 파이프라인  ★★★
%% =====================================================================
fprintf('%s\n',repmat('═',1,76));
fprintf('  Nature/Science급 통계 분석 시작\n');
fprintf('%s\n\n',repmat('═',1,76));

% ── 공통 특징 행렬 구축 ──
[X_feat, featNames, grpIdx, grpU, nGrp] = buildFeatureMatrix(T, numCols);
nFeat = numel(featNames);
fprintf('  특징 행렬: %d samples × %d features × %d grades\n\n', ...
    size(X_feat,1), nFeat, nGrp);

statSheets = struct();

%% ===== [7.1] Kruskal-Wallis H test + Dunn's post-hoc =====
fprintf('[7.1] Kruskal-Wallis H test + Dunn''s post-hoc\n');
[T_kw, T_dunn] = runKruskalWallis(X_feat, grpIdx, featNames, grpU);
statSheets.KW_TEST      = T_kw;
statSheets.DUNN_POSTHOC = T_dunn;
nSig = sum(T_kw.p_value < 0.05);
fprintf('  유의한 특징 (p<0.05): %d / %d\n\n', nSig, nFeat);

%% ===== [7.2] Effect size: η², Cohen's d =====
fprintf('[7.2] Effect size (eta-squared, Cohen''s d)\n');
[T_eta2, T_cohend] = computeEffectSizes(X_feat, grpIdx, featNames, grpU);
statSheets.EFFECT_ETA2   = T_eta2;
statSheets.EFFECT_COHEND = T_cohend;
nLarge = sum(T_eta2.eta_squared >= 0.14);
fprintf('  Large effect (η²≥0.14): %d / %d\n\n', nLarge, nFeat);

%% ===== [7.3] Spearman/Pearson 상관 + Bootstrap CI + BH-FDR =====
fprintf('[7.3] Spearman 상관분석 + Bootstrap CI + BH-FDR\n');
nTopCorr = min(OPT.CORR_TOP_K, nFeat);
[T_corr_rw, T_corr_mat, T_corr_fdr] = computeCorrelation( ...
    X_feat, grpIdx, featNames, OPT.N_BOOT_CORR, nTopCorr);
statSheets.CORR_RW       = T_corr_rw;
statSheets.CORR_MATRIX   = T_corr_mat;
statSheets.CORR_PVAL_FDR = T_corr_fdr;
fprintf('  상위 %d 특징 상관분석 완료\n\n', nTopCorr);

%% ===== [7.4] Feature importance ranking =====
fprintf('[7.4] Feature importance ranking\n');
T_importance = rankFeatureImportance(X_feat, grpIdx, featNames);
statSheets.FEATURE_RANK = T_importance;
fprintf('  Top-5: %s\n\n', strjoin(T_importance.feature(1:min(5,nFeat)),', '));

%% ===== [7.5] PCA =====
fprintf('[7.5] PCA (주성분 분석)\n');
[T_pca_var, T_pca_load, T_pca_score, pcaRes] = runPCA(X_feat, grpIdx, featNames, grpU, T);
statSheets.PCA_VARIANCE = T_pca_var;
statSheets.PCA_LOADINGS = T_pca_load;
statSheets.PCA_SCORES   = T_pca_score;
nPC95 = find(cumsum(pcaRes.explained) >= 95, 1, 'first');
if isempty(nPC95); nPC95 = nFeat; end
fprintf('  95%% 분산 설명 PC 수: %d\n\n', nPC95);

%% ===== [7.6] Discriminability (Silhouette + Mahalanobis) =====
fprintf('[7.6] Discriminability analysis\n');
[T_sil, T_mahal] = computeDiscriminability(pcaRes.score, grpIdx, grpU, nPC95);
statSheets.SILHOUETTE  = T_sil;
statSheets.MAHALANOBIS = T_mahal;
fprintf('  Mean silhouette: %.3f\n\n', mean(T_sil.mean_silhouette));

%% ===== [7.7] Bootstrap CI for grade-level means =====
fprintf('[7.7] Bootstrap 95%% CI (grade-level means)\n');
T_bootCI = bootstrapGradeMeans(X_feat, grpIdx, grpU, featNames, OPT.N_BOOT_GRADE);
statSheets.BOOTSTRAP_CI = T_bootCI;
fprintf('  완료 (n=%d)\n\n', OPT.N_BOOT_GRADE);

%% ===== [7.8] Nature-level 시각화 =====
fprintf('[7.8] Nature-level 시각화\n');
try
    plotViolin(X_feat, grpIdx, grpU, featNames, T_importance, statDir, OPT.STAT_DPI);
    fprintf('  [OK] Violin plot\n');
catch ME; fprintf('  [WARN] Violin: %s\n', ME.message);
end
try
    plotCorrHeatmap(T_corr_mat, statDir, OPT.STAT_DPI);
    fprintf('  [OK] Correlation heatmap\n');
catch ME; fprintf('  [WARN] Heatmap: %s\n', ME.message);
end
try
    plotPCABiplot(pcaRes, grpIdx, grpU, featNames, T_importance, statDir, OPT.STAT_DPI);
    fprintf('  [OK] PCA biplot\n');
catch ME; fprintf('  [WARN] PCA biplot: %s\n', ME.message);
end
try
    plotFeatureImportance(T_importance, statDir, OPT.STAT_DPI);
    fprintf('  [OK] Feature importance\n');
catch ME; fprintf('  [WARN] Feature importance: %s\n', ME.message);
end
try
    plotRadar(X_feat, grpIdx, grpU, featNames, T_importance, statDir, OPT.STAT_DPI);
    fprintf('  [OK] Radar chart\n');
catch ME; fprintf('  [WARN] Radar: %s\n', ME.message);
end
try
    plotBoxSig(X_feat, grpIdx, grpU, featNames, T_importance, T_dunn, statDir, OPT.STAT_DPI);
    fprintf('  [OK] Box + significance\n');
catch ME; fprintf('  [WARN] Box: %s\n', ME.message);
end
try
    plotGradeDistOverview(X_feat, grpIdx, grpU, featNames, T_importance, statDir, OPT.STAT_DPI);
    fprintf('  [OK] Grade distribution overview\n');
catch ME; fprintf('  [WARN] Grade dist: %s\n', ME.message);
end
fprintf('\n');

fprintf('%s\n',repmat('═',1,76));
fprintf('  Nature/Science급 분석 완료\n');
fprintf('%s\n\n',repmat('═',1,76));

%% ===== [8] ★ Excel 저장 =====
fprintf('[7] Excel 저장 ★\n');
baseXlsx  = fullfile(xlsxDir, 'COLOR_STATS');
xlsxFiles = writeExcelSplit(T, T_grade, T_humid, T_cross, statSheets, baseXlsx, ...
    OPT, nTotal, nMatched, nValid, toc(t0));

csvPath = fullfile(csvDir, 'COLOR_STATS_raw.csv');
try writetable(T, csvPath, 'Encoding','UTF-8');
catch
end
fprintf('  CSV: %s\n\n', csvPath);

%% ===== [9] PNG (선택, 기본 OFF) =====
if OPT.EXPORT_PNG
    fprintf('[8] PNG 생성\n');
    nOK = 0; nFail = 0;
    for ii = 1:nImg
        if isempty(allRows{ii}); continue; end
        try
            I0 = ensureRGB(imread(cellFiles{ii}));
            [~,stem,ext] = fileparts(cellFiles{ii});
            imageId = sprintf('%06d_%s%s', ii, stem, ext);
            iRoot   = fullfile(perDir, imageId);
            iPng    = fullfile(iRoot,'png');
            iCsv    = fullfile(iRoot,'csv');
            ensureDirs({iRoot, iPng, iCsv});
            exportPng(I0, allRows{ii}, imageId, iPng, iCsv, pngDir, OPT);
            nOK = nOK + 1;
        catch ME3
            nFail = nFail + 1;
            warning(ME3.identifier,'%s', sprintf('PNG 스킵[%d]: %s', ii, ME3.message));
        end
        if mod(ii,100) == 0
            fprintf('  PNG %d/%d (성공%d 실패%d)\n', ii, nImg, nOK, nFail);
        end
    end
    fprintf('  PNG 완료: 성공%d 실패%d\n\n', nOK, nFail);
end

%% ===== 완료 =====
elapsed = toc(t0);
fprintf('%s\n',repmat('=',1,76));
fprintf('  DONE | %.1f min | 분석%d / 매칭%d / 전체%d | GPU=%s | Workers=%d\n', ...
    elapsed/60, nValid, nMatched, nTotal, tf2s(gpuOK), nW);
for fi = 1:numel(xlsxFiles)
    fprintf('  Excel[%d]: %s\n', fi, xlsxFiles{fi});
end
fprintf('  통계 그래프: %s\n', statDir);
fprintf('%s\n\n',repmat('=',1,76));
try diary off;
catch
end

runInfo = struct('outRoot',outRoot, 'xlsxFiles',{xlsxFiles}, 'csvPath',csvPath, ...
    'nTotal',nTotal, 'nMatched',nMatched, 'nValid',nValid, ...
    'nSkip',nMatched-nValid, 'gpuOK',gpuOK, 'nWorkers',nW, ...
    'statSheets',statSheets, 'statDir',statDir);
end


%% =========================================================================
%% buildRow  (기존)
%% =========================================================================
function row = buildRow(I0, hsv3, lab3, stem, fpath, imgIdx, ...
    refL, refa, refb, jsonMap, roiMode, bgTh)

row = struct();
row.img_index = imgIdx;
row.file      = fpath;
row.stem      = stem;

fn = parseFilename(stem);
row.fn_site     = fn.site;
row.fn_borehole = fn.borehole;
row.fn_box      = fn.box_no;
row.fn_rock1    = fn.rock1;
row.fn_rock2    = fn.rock2;
row.fn_angle    = fn.angle;
row.fn_rotation = fn.rotation;
row.fn_humid    = fn.humid;
row.fn_lux_bin  = fn.lux_bin;
row.fn_seq      = fn.seq;

jm = readJson(lower(stem), jsonMap);
row.json_matched    = jm.matched;
row.rw              = jm.rw;
row.humidity        = jm.humidity;
row.object_id       = jm.object_id;
row.box_no          = jm.box_no;
row.year            = jm.year;
row.hole_num        = jm.hole_num;
row.rock_depth      = jm.rock_depth;
row.rock_strength   = jm.rock_strength;
row.tcr             = jm.tcr;
row.rqd             = jm.rqd;
row.rmr             = jm.rmr;
row.rq              = jm.rq;
row.rjr             = jm.rjr;
row.fm              = jm.fm;
row.geo_structure   = jm.geo_structure;
row.rock_type1      = jm.rock_type1;
row.rock_type2      = jm.rock_type2;
row.lux             = jm.lux;
row.iso             = jm.iso;
row.f_stop          = jm.f_stop;
row.camera_type     = jm.camera_type;
row.location        = jm.location;
row.drilling_place1 = jm.drilling_place1;
row.drilling_place2 = jm.drilling_place2;

[H_px, W_px, ~] = size(I0);
row.height_px = H_px;
row.width_px  = W_px;
row.nPixels   = H_px * W_px;

mask = makeMask(I0, roiMode, bgTh);
row.roi_pixels = nnz(mask);
row.roi_pct    = 100 * nnz(mask) / (H_px * W_px);

Rf   = double(I0(:,:,1));
Gf   = double(I0(:,:,2));
Bf   = double(I0(:,:,3));
Hflt = hsv3(:,:,1);
Sflt = hsv3(:,:,2);
Vflt = hsv3(:,:,3);
Lflt = lab3(:,:,1);
aflt = lab3(:,:,2);
bflt = lab3(:,:,3);
Cflt = hypot(aflt, bflt);

row = addStats(row,'R',Rf,mask);   row = addShape(row,'R',Rf,mask);
row = addStats(row,'G',Gf,mask);   row = addShape(row,'G',Gf,mask);
row = addStats(row,'B',Bf,mask);   row = addShape(row,'B',Bf,mask);

row = addStats(row,'H',Hflt,mask);
row = addStats(row,'S',Sflt,mask);
row = addStats(row,'V',Vflt,mask);
Vv = Vflt(mask);
row.dark_pct   = mean(Vv < 0.20) * 100;
row.mid_pct    = mean(Vv >= 0.20 & Vv < 0.80) * 100;
row.bright_pct = mean(Vv >= 0.80) * 100;

row = addStats(row,'L',Lflt,mask); row = addShape(row,'L',Lflt,mask);
row = addStats(row,'a',aflt,mask); row = addShape(row,'a',aflt,mask);
row = addStats(row,'b',bflt,mask); row = addShape(row,'b',bflt,mask);
row = addStats(row,'C',Cflt,mask);

Lm = row.L_mean; am = row.a_mean; bm = row.b_mean; Cm = row.C_mean;
row.LCh_h_mean = atan2d(bm, am);
hArr = atan2d(bflt(mask), aflt(mask));
row.LCh_h_p50  = median(hArr,'omitnan');
row.LCh_h_std  = std(hArr, 0, 'omitnan');

Ycc = 0.299*Rf + 0.587*Gf + 0.114*Bf;
row = addStats(row,'YCC_Y',  Ycc,    mask);
row = addStats(row,'YCC_Cb', Bf-Ycc, mask);
row = addStats(row,'YCC_Cr', Rf-Ycc, mask);

Rm = row.R_mean; Gm = row.G_mean; Bm = row.B_mean; sRGB = Rm+Gm+Bm;
row.RB_ratio = Rm / max(Bm,1);
row.RG_ratio = Rm / max(Gm,1);
row.R_norm   = Rm / max(sRGB,1);
row.G_norm   = Gm / max(sRGB,1);
row.B_norm   = Bm / max(sRGB,1);

[mV,mC,~,mHn]    = lab2munsell(Lm, am, bm);
row.munsell_V       = mV;
row.munsell_C       = mC;
row.munsell_H_name  = mHn;
row.munsell_neutral = double(Cm < 4.0);
[mVp,mCp,~,mHnp]  = lab2munsell(row.L_p50, row.a_p50, row.b_p50);
row.munsell_V_p50      = mVp;
row.munsell_C_p50      = mCp;
row.munsell_H_name_p50 = mHnp;

eps1 = 1e-6;
Sm = row.S_mean; Vm = row.V_mean; Lp = row.L_p50; bp = row.b_p50;
row.RI         = (Rm^2) / max(Bm*(Rm+Gm+Bm), eps1);
row.CI         = (Rm-Bm) / max(Rm+Bm, eps1);
row.CI_p50     = (row.R_p50-row.B_p50) / max(row.R_p50+row.B_p50, eps1);
row.NRI        = Rm / max(sRGB, eps1);
row.RBI        = (Rm-Bm) / 255;
row.SVI        = Sm * (1-Vm) * 100;
row.WHI        = bm / (abs(am)+1);
row.CWI        = (bp+10) / max(Lp/10, eps1);
row.YI         = bm / max(Lm, eps1) * 100;
row.SAI        = Cm / max(Lm, eps1) * 100;
row.GRI        = 1 - Cm/30;
row.delta_E_D1 = sqrt((Lm-refL)^2 + (am-refa)^2 + (bm-refb)^2);
row.HI         = row.LCh_h_mean - atan2d(refb, refa);
row.CrI        = row.YCC_Cr_mean / max(abs(row.YCC_Cb_mean), eps1);
end


%% =========================================================================
%%  ★★★ Nature급 통계 분석 함수 ★★★
%% =========================================================================

%% ── 공통: 특징 행렬 구축 ──
function [X, fNames, gIdx, gU, nGrp] = buildFeatureMatrix(T, numCols)
    fNames = numCols(:);
    X = zeros(height(T), numel(fNames));
    for j = 1:numel(fNames)
        col = fNames{j};
        if any(strcmp(T.Properties.VariableNames, col))
            X(:,j) = double(T.(col));
        end
    end
    X(~isfinite(X)) = NaN;
    rwCol = string(T.rw);
    valid = strlength(strtrim(rwCol)) > 0 & rwCol ~= "unknown";
    X = X(valid,:);
    rwCol = rwCol(valid);
    [gU, ~, gIdx] = unique(rwCol);
    gU = gU(:);
    nGrp = numel(gU);
end

%% ── [7.1] Kruskal-Wallis + Dunn's post-hoc ──
function [Tkw, Tdunn] = runKruskalWallis(X, gIdx, fNames, gU)
    nF = numel(fNames);
    nGrp = numel(gU);
    kwH = nan(nF,1); kwP = nan(nF,1);

    for j = 1:nF
        v = X(:,j); ok = isfinite(v);
        if sum(ok) < 10; continue; end
        try
            [kwP(j), ~, stats] = kruskalwallis(v(ok), gIdx(ok), 'off');
            kwH(j) = stats.chi2stat;
        catch
        end
    end

    sigFlag = kwP < 0.05;
    interpH = repmat("ns", nF, 1);
    interpH(kwP < 0.05) = "*";
    interpH(kwP < 0.01) = "**";
    interpH(kwP < 0.001) = "***";
    Tkw = table(string(fNames), kwH, kwP, sigFlag, interpH, ...
        'VariableNames', {'feature','H_stat','p_value','significant','stars'});

    % Dunn's post-hoc
    nPairs = nGrp*(nGrp-1)/2;
    dFeat = cell(nF*nPairs,1); dG1 = dFeat; dG2 = dFeat;
    dZ = nan(nF*nPairs,1); dPr = dZ; dPb = dZ;
    idx = 0;
    for j = 1:nF
        if ~sigFlag(j); continue; end
        v = X(:,j); ok = isfinite(v);
        vv = v(ok); gg = gIdx(ok);
        [~, rk] = sort(vv); ranks = zeros(size(vv));
        ranks(rk) = tiedrank_local(vv(rk));
        meanR = accumarray(gg, ranks, [nGrp,1], @mean);
        nk = accumarray(gg, ones(size(gg)), [nGrp,1]);
        Nok = numel(vv);
        % tie correction
        [~,~,tg] = unique(vv); tCnt = accumarray(tg, 1);
        tCorr = 1 - sum(tCnt.^3 - tCnt) / (Nok^3 - Nok);
        S2 = (Nok*(Nok+1)/12) * tCorr;
        for a = 1:nGrp-1
            for b = a+1:nGrp
                idx = idx + 1;
                dFeat{idx} = fNames{j};
                dG1{idx} = char(gU(a));
                dG2{idx} = char(gU(b));
                se = sqrt(S2 * (1/nk(a) + 1/nk(b)));
                if se > 0
                    z = (meanR(a) - meanR(b)) / se;
                else
                    z = 0;
                end
                dZ(idx) = z;
                dPr(idx) = 2 * (1 - normcdf_local(abs(z)));
                dPb(idx) = min(1, dPr(idx) * nPairs);
            end
        end
    end
    valid = ~cellfun(@isempty, dFeat);
    dSig = dPb < 0.05;
    Tdunn = table(string(dFeat(valid)), string(dG1(valid)), string(dG2(valid)), ...
        dZ(valid), dPr(valid), dPb(valid), dSig(valid), ...
        'VariableNames', {'feature','group1','group2','z_stat','p_raw','p_bonf','significant'});
end

%% ── [7.2] Effect sizes ──
function [Teta, Tcoh] = computeEffectSizes(X, gIdx, fNames, gU)
    nF = numel(fNames);
    nGrp = numel(gU);

    eta2 = nan(nF,1);
    for j = 1:nF
        v = X(:,j); ok = isfinite(v);
        if sum(ok) < 10; continue; end
        try
            [~,tbl] = kruskalwallis(v(ok), gIdx(ok), 'off');
            SSb = tbl{2,2}; SSt = tbl{4,2};
            if SSt > 0; eta2(j) = SSb / SSt; end
        catch
        end
    end
    interp = repmat("negligible", nF, 1);
    interp(eta2 >= 0.01) = "small";
    interp(eta2 >= 0.06) = "medium";
    interp(eta2 >= 0.14) = "large";
    Teta = table(string(fNames), eta2, interp, ...
        'VariableNames', {'feature','eta_squared','interpretation'});

    % Cohen's d pairwise
    nPairs = nGrp*(nGrp-1)/2;
    cFeat = cell(nF*nPairs,1); cG1 = cFeat; cG2 = cFeat;
    cD = nan(nF*nPairs,1);
    idx = 0;
    for j = 1:nF
        v = X(:,j);
        for a = 1:nGrp-1
            for b = a+1:nGrp
                idx = idx + 1;
                cFeat{idx} = fNames{j};
                cG1{idx} = char(gU(a));
                cG2{idx} = char(gU(b));
                va = v(gIdx==a); va = va(isfinite(va));
                vb = v(gIdx==b); vb = vb(isfinite(vb));
                if numel(va)<2 || numel(vb)<2; continue; end
                sp = sqrt(((numel(va)-1)*var(va) + (numel(vb)-1)*var(vb)) / ...
                    (numel(va)+numel(vb)-2));
                if sp > 0
                    cD(idx) = (mean(va)-mean(vb)) / sp;
                end
            end
        end
    end
    valid = ~cellfun(@isempty, cFeat);
    cInterp = repmat("negligible", sum(valid), 1);
    ad = abs(cD(valid));
    cInterp(ad >= 0.2) = "small";
    cInterp(ad >= 0.5) = "medium";
    cInterp(ad >= 0.8) = "large";
    cInterp(ad >= 1.2) = "very large";
    Tcoh = table(string(cFeat(valid)), string(cG1(valid)), string(cG2(valid)), ...
        cD(valid), cInterp, ...
        'VariableNames', {'feature','group1','group2','cohens_d','interpretation'});
end

%% ── [7.3] Correlation ──
function [Trw, Tmat, Tfdr] = computeCorrelation(X, gIdx, fNames, nBoot, topK)
    nF = numel(fNames);
    rhoRw = nan(nF,1); pRw = nan(nF,1);
    ciLo = nan(nF,1); ciHi = nan(nF,1);

    for j = 1:nF
        v = X(:,j); ok = isfinite(v);
        if sum(ok) < 10; continue; end
        [rhoRw(j), pRw(j)] = corr(v(ok), gIdx(ok), 'Type','Spearman');
        % bootstrap CI
        rng(42+j);
        bRho = nan(nBoot,1);
        vok = v(ok); gok = gIdx(ok); nOk = sum(ok);
        for bi = 1:nBoot
            idx = randi(nOk, nOk, 1);
            bRho(bi) = corr(vok(idx), gok(idx), 'Type','Spearman');
        end
        ciLo(j) = prctile(bRho, 2.5);
        ciHi(j) = prctile(bRho, 97.5);
    end

    % BH-FDR
    pAdj = bhFDR(pRw);
    sigFlag = pAdj < 0.05;
    Trw = table(string(fNames), rhoRw, pRw, pAdj, ciLo, ciHi, sigFlag, ...
        'VariableNames', {'feature','spearman_rho','p_value','p_fdr','ci_lo','ci_hi','significant'});
    Trw = sortrows(Trw, 'spearman_rho', 'descend', 'ComparisonMethod','abs');

    % Inter-feature correlation matrix (top-K)
    [~, sortI] = sort(abs(rhoRw), 'descend', 'MissingPlacement','last');
    topIdx = sortI(1:topK);
    Xsub = X(:, topIdx);
    fSub = fNames(topIdx);
    [rMat, pMat] = corr(Xsub, 'Type','Spearman', 'Rows','pairwise');
    Tmat = array2table(rMat, 'VariableNames', matlab.lang.makeValidName(fSub), ...
        'RowNames', fSub);

    pFlat = pMat(:);
    pAdjFlat = bhFDR(pFlat);
    pAdjMat = reshape(pAdjFlat, size(pMat));
    Tfdr = array2table(pAdjMat, 'VariableNames', matlab.lang.makeValidName(fSub), ...
        'RowNames', fSub);
end

%% ── [7.4] Feature importance ──
function Timp = rankFeatureImportance(X, gIdx, fNames)
    nF = numel(fNames);
    Fscore = nan(nF,1); Fp = nan(nF,1); MI = nan(nF,1);

    for j = 1:nF
        v = X(:,j); ok = isfinite(v);
        if sum(ok) < 10; continue; end
        try
            [Fp(j), tbl] = anova1(v(ok), gIdx(ok), 'off');
            Fscore(j) = tbl{2,5};
        catch
        end
        % MI proxy
        try
            vok = v(ok); gok = gIdx(ok);
            nBins = 20;
            edges = quantile(vok, linspace(0,1,nBins+1));
            edges = unique(edges);
            if numel(edges) < 3; continue; end
            bins = discretize(vok, edges);
            validB = isfinite(bins);
            bins = bins(validB); gokb = gok(validB);
            nT = numel(bins);
            nBact = max(bins);
            joint = accumarray([bins, gokb], 1, [nBact, max(gokb)]) / nT;
            pX = sum(joint,2); pY = sum(joint,1);
            mi = 0;
            for xi = 1:size(joint,1)
                for yi = 1:size(joint,2)
                    if joint(xi,yi) > 0 && pX(xi) > 0 && pY(yi) > 0
                        mi = mi + joint(xi,yi) * log2(joint(xi,yi) / (pX(xi)*pY(yi)));
                    end
                end
            end
            MI(j) = mi;
        catch
        end
    end

    [~, rankF] = sort(Fscore, 'descend', 'MissingPlacement','last');
    [~, rankMI] = sort(MI, 'descend', 'MissingPlacement','last');
    rF = nan(nF,1); rM = nan(nF,1);
    rF(rankF) = 1:nF; rM(rankMI) = 1:nF;
    combRank = (rF + rM) / 2;
    [~, sortI] = sort(combRank);

    Timp = table(string(fNames(sortI)), Fscore(sortI), Fp(sortI), MI(sortI), ...
        rF(sortI), rM(sortI), combRank(sortI), ...
        'VariableNames', {'feature','F_score','F_pval','MI_proxy','rank_F','rank_MI','rank_combined'});
end

%% ── [7.5] PCA ──
function [Tvar, Tload, Tscore, res] = runPCA(X, ~, fNames, ~, T)
    Xz = X;
    mu = mean(Xz, 1, 'omitnan');
    sd = std(Xz, 0, 1, 'omitnan');
    sd(sd < 1e-12) = 1;
    Xz = (Xz - mu) ./ sd;
    Xz(isnan(Xz)) = 0;

    [coeff, score, latent, ~, explained] = pca(Xz, 'Rows','complete');

    nPC = size(coeff,2);
    Tvar = table((1:nPC)', latent, explained, cumsum(explained), ...
        'VariableNames', {'PC','eigenvalue','variance_pct','cumulative_pct'});

    nShow = min(10, nPC);
    loadMat = coeff(:, 1:nShow);
    pcNames = arrayfun(@(x) sprintf('PC%d',x), 1:nShow, 'UniformOutput',false);
    Tload = array2table(loadMat, 'VariableNames', pcNames, 'RowNames', fNames);

    nS = min(nShow, size(score,2));
    scoreMat = score(:, 1:nS);
    Tscore = array2table(scoreMat, 'VariableNames', pcNames(1:nS));
    rwValid = string(T.rw);
    validMask = strlength(strtrim(rwValid)) > 0 & rwValid ~= "unknown";
    rwValid = rwValid(validMask);
    Tscore.rw = rwValid;

    res = struct('coeff',coeff, 'score',score, 'latent',latent, ...
        'explained',explained, 'mu',mu, 'sd',sd);
end

%% ── [7.6] Discriminability ──
function [Tsil, Tmahal] = computeDiscriminability(score, gIdx, gU, nPC)
    nGrp = numel(gU);
    k = min(nPC, size(score,2));
    S = score(:, 1:k);

    % Silhouette
    try
        silVals = silhouette(S, gIdx);
    catch
        silVals = zeros(size(gIdx));
    end
    silMean = accumarray(gIdx, silVals, [nGrp,1], @mean);
    silStd  = accumarray(gIdx, silVals, [nGrp,1], @std);
    silN    = accumarray(gIdx, 1, [nGrp,1]);
    Tsil = table(gU, silMean, silStd, silN, ...
        'VariableNames', {'grade','mean_silhouette','std_silhouette','n_samples'});

    % Mahalanobis
    dMat = nan(nGrp);
    groupMeans = zeros(nGrp, k);
    groupCovs  = cell(nGrp,1);
    for g = 1:nGrp
        Sg = S(gIdx==g, :);
        groupMeans(g,:) = mean(Sg, 1);
        if size(Sg,1) > k
            groupCovs{g} = cov(Sg);
        else
            groupCovs{g} = eye(k);
        end
    end
    for a = 1:nGrp
        for b = a+1:nGrp
            na = silN(a); nb = silN(b);
            Sp = ((na-1)*groupCovs{a} + (nb-1)*groupCovs{b}) / (na+nb-2);
            Sp = Sp + 1e-6 * eye(k) * trace(Sp)/k;
            diff = groupMeans(a,:) - groupMeans(b,:);
            try
                d = sqrt(diff / Sp * diff');
            catch
                d = NaN;
            end
            dMat(a,b) = d;
            dMat(b,a) = d;
        end
        dMat(a,a) = 0;
    end
    Tmahal = array2table(dMat, 'VariableNames', cellstr(gU), 'RowNames', cellstr(gU));
end

%% ── [7.7] Bootstrap CI for grade means ──
function Tboot = bootstrapGradeMeans(X, gIdx, gU, fNames, nBoot)
    nGrp = numel(gU); nF = numel(fNames);
    rows = cell(nGrp * nF, 1);
    idx = 0;
    rng(42);
    for g = 1:nGrp
        mask = (gIdx == g);
        Xg = X(mask, :);
        for j = 1:nF
            idx = idx + 1;
            v = Xg(:,j); v = v(isfinite(v));
            if numel(v) < 3
                rows{idx} = struct('grade',char(gU(g)), 'feature',fNames{j}, ...
                    'mean',NaN, 'ci_lo',NaN, 'ci_hi',NaN, 'ci_width',NaN, 'n',numel(v));
                continue;
            end
            bMeans = nan(nBoot,1);
            for bi = 1:nBoot
                bMeans(bi) = mean(v(randi(numel(v), numel(v), 1)));
            end
            lo = prctile(bMeans, 2.5);
            hi = prctile(bMeans, 97.5);
            rows{idx} = struct('grade',char(gU(g)), 'feature',fNames{j}, ...
                'mean',mean(v), 'ci_lo',lo, 'ci_hi',hi, 'ci_width',hi-lo, 'n',numel(v));
        end
    end
    Tboot = struct2table(vertcat(rows{:}));
    Tboot.grade   = string(Tboot.grade);
    Tboot.feature = string(Tboot.feature);
end


%% =========================================================================
%%  ★★★ Nature급 시각화 함수 ★★★
%% =========================================================================

%% ── Violin plot (Top-6 features) ──
function plotViolin(X, gIdx, gU, fNames, Timp, saveDir, dpi)
    topF = Timp.feature(1:min(6, height(Timp)));
    nPlot = numel(topF);
    nGrp = numel(gU);
    colors = lines(nGrp);

    fig = figure('Visible','off','Position',[50 50 350*nPlot 420]);
    tl = tiledlayout(fig, 1, nPlot, 'Padding','compact','TileSpacing','compact');

    for fi = 1:nPlot
        col = topF(fi);
        jj = find(strcmp(fNames, col), 1);
        nexttile(tl);
        hold on;
        for g = 1:nGrp
            v = X(gIdx==g, jj); v = v(isfinite(v));
            if numel(v) < 4; continue; end
            try
                [f_kde, xi] = ksdensity(v, 'NumPoints', 100);
            catch
                continue;
            end
            f_kde = f_kde / max(f_kde) * 0.38;
            fill([g + f_kde, g - fliplr(f_kde)], [xi, fliplr(xi)], ...
                colors(g,:), 'FaceAlpha', 0.5, 'EdgeColor', colors(g,:), 'LineWidth', 0.8);
            % quartile lines
            q = prctile(v, [25 50 75]);
            for qi = 1:3
                lw = 1; if qi == 2; lw = 2; end
                plot([g-0.15, g+0.15], [q(qi) q(qi)], 'k-', 'LineWidth', lw);
            end
        end
        set(gca, 'XTick', 1:nGrp, 'XTickLabel', cellstr(gU));
        xlabel('Grade'); ylabel(char(col));
        title(char(col), 'FontSize', 10, 'Interpreter','none');
        grid on; box on;
    end
    title(tl, 'Violin Plots (Top-6 Features by Importance)', 'FontSize', 12);
    expFig(fig, fullfile(saveDir, 'VIOLIN_top6.png'), dpi);
    close(fig);
end

%% ── Correlation heatmap ──
function plotCorrHeatmap(Tmat, saveDir, dpi)
    if isempty(Tmat); return; end
    rMat = table2array(Tmat);
    fNames = Tmat.Properties.RowNames;
    n = numel(fNames);

    fig = figure('Visible','off','Position',[50 50 max(600, n*38) max(550, n*34)]);
    imagesc(rMat); axis image;
    colormap(rwbColormap(256));
    clim([-1 1]); colorbar('FontSize', 8);
    set(gca, 'XTick', 1:n, 'XTickLabel', fNames, 'XTickLabelRotation', 55, ...
        'YTick', 1:n, 'YTickLabel', fNames, 'FontSize', 7);
    % annotate
    for i = 1:n
        for j = 1:n
            v = rMat(i,j);
            if abs(v) > 0.3
                clr = 'w'; if abs(v) < 0.6; clr = 'k'; end
                text(j, i, sprintf('%.2f',v), 'HorizontalAlignment','center', ...
                    'FontSize', 6, 'Color', clr);
            end
        end
    end
    title('Spearman Correlation Matrix (Top Features)', 'FontSize', 11);
    expFig(fig, fullfile(saveDir, 'CORR_HEATMAP.png'), dpi);
    close(fig);
end

%% ── PCA biplot ──
function plotPCABiplot(pcaRes, gIdx, gU, fNames, Timp, saveDir, dpi)
    score = pcaRes.score;
    coeff = pcaRes.coeff;
    expl  = pcaRes.explained;
    nGrp = numel(gU);
    colors = lines(nGrp);

    fig = figure('Visible','off','Position',[50 50 750 650]);
    hold on;

    % scatter by grade
    for g = 1:nGrp
        mask = (gIdx == g);
        scatter(score(mask,1), score(mask,2), 12, colors(g,:), 'filled', ...
            'MarkerFaceAlpha', 0.4, 'DisplayName', char(gU(g)));
        % 95% confidence ellipse
        S = score(mask, 1:2);
        if size(S,1) > 3
            mu = mean(S);
            C = cov(S);
            drawEllipse95(mu, C, colors(g,:));
        end
    end

    % loading arrows (top 8)
    nArr = min(8, size(coeff,1));
    topIdx = zeros(nArr,1);
    topF = Timp.feature(1:min(nArr, height(Timp)));
    for k = 1:numel(topF)
        idx = find(strcmp(fNames, topF(k)), 1);
        if ~isempty(idx); topIdx(k) = idx; end
    end
    topIdx = topIdx(topIdx > 0);
    scaleF = max(abs(score(:,1))) * 0.7 / max(abs(coeff(topIdx,1)) + eps);
    for k = 1:numel(topIdx)
        ii = topIdx(k);
        quiver(0, 0, coeff(ii,1)*scaleF, coeff(ii,2)*scaleF, 0, ...
            'Color', [0.2 0.2 0.2], 'LineWidth', 1.3, 'MaxHeadSize', 0.5, ...
            'HandleVisibility','off');
        text(coeff(ii,1)*scaleF*1.08, coeff(ii,2)*scaleF*1.08, ...
            fNames{ii}, 'FontSize', 7, 'Color', [0.1 0.1 0.1], ...
            'Interpreter','none');
    end

    xlabel(sprintf('PC1 (%.1f%%)', expl(1)), 'FontSize', 11);
    ylabel(sprintf('PC2 (%.1f%%)', expl(2)), 'FontSize', 11);
    title('PCA Biplot with 95% Confidence Ellipses', 'FontSize', 12);
    legend('Location','best', 'FontSize', 8);
    grid on; box on;
    expFig(fig, fullfile(saveDir, 'PCA_BIPLOT.png'), dpi);
    close(fig);
end

%% ── Feature importance bar chart ──
function plotFeatureImportance(Timp, saveDir, dpi)
    nShow = min(20, height(Timp));
    T = Timp(1:nShow, :);

    fig = figure('Visible','off','Position',[50 50 700 500]);
    tl = tiledlayout(fig, 1, 2, 'Padding','compact','TileSpacing','compact');

    % F-score
    nexttile(tl);
    barh(T.F_score(end:-1:1), 'FaceColor', [0.20 0.47 0.74]);
    set(gca, 'YTick', 1:nShow, 'YTickLabel', cellstr(T.feature(end:-1:1)), 'FontSize', 7);
    xlabel('ANOVA F-score'); title('F-score Ranking');
    grid on;

    % MI proxy
    nexttile(tl);
    barh(T.MI_proxy(end:-1:1), 'FaceColor', [0.85 0.33 0.10]);
    set(gca, 'YTick', 1:nShow, 'YTickLabel', cellstr(T.feature(end:-1:1)), 'FontSize', 7);
    xlabel('Mutual Information (proxy)'); title('MI Ranking');
    grid on;

    sgtitle(sprintf('Feature Importance (Top %d)', nShow), 'FontSize', 12);
    expFig(fig, fullfile(saveDir, 'FEATURE_IMPORTANCE.png'), dpi);
    close(fig);
end

%% ── Radar chart ──
function plotRadar(X, gIdx, gU, fNames, Timp, saveDir, dpi)
    nF = min(8, height(Timp));
    topF = Timp.feature(1:nF);
    nGrp = numel(gU);
    colors = lines(nGrp);

    vals = nan(nGrp, nF);
    for fi = 1:nF
        jj = find(strcmp(fNames, topF(fi)), 1);
        for g = 1:nGrp
            v = X(gIdx==g, jj); v = v(isfinite(v));
            if ~isempty(v); vals(g, fi) = mean(v); end
        end
    end
    % normalize 0-1
    mn = min(vals, [], 1); mx = max(vals, [], 1);
    rng_v = mx - mn; rng_v(rng_v < eps) = 1;
    vals = (vals - mn) ./ rng_v;

    angles = linspace(0, 2*pi, nF+1);

    fig = figure('Visible','off','Position',[50 50 650 600]);
    ax = polaraxes(fig);
    hold(ax, 'on');
    for g = 1:nGrp
        v = [vals(g,:), vals(g,1)];
        polarplot(ax, angles, v, '-o', 'Color', colors(g,:), ...
            'LineWidth', 1.5, 'MarkerSize', 4, 'MarkerFaceColor', colors(g,:), ...
            'DisplayName', char(gU(g)));
    end
    ax.ThetaTick = rad2deg(angles(1:end-1));
    ax.ThetaTickLabel = cellstr(topF);
    ax.RLim = [0 1.05];
    ax.FontSize = 8;
    legend(ax, 'Location','southoutside','Orientation','horizontal','FontSize',8);
    title(ax, 'Radar Chart: Grade-wise Feature Profile (Normalized)', 'FontSize', 11);
    expFig(fig, fullfile(saveDir, 'RADAR_GRADES.png'), dpi);
    close(fig);
end

%% ── Box plot with significance brackets ──
function plotBoxSig(X, gIdx, gU, fNames, Timp, Tdunn, saveDir, dpi)
    topF = Timp.feature(1:min(6, height(Timp)));
    nPlot = numel(topF);

    fig = figure('Visible','off','Position',[50 50 350*nPlot 480]);
    tl = tiledlayout(fig, 1, nPlot, 'Padding','compact','TileSpacing','compact');

    for fi = 1:nPlot
        col = topF(fi);
        jj = find(strcmp(fNames, col), 1);
        nexttile(tl);
        data = X(:,jj);
        boxchart(categorical(gU(gIdx)), data, 'MarkerStyle','.');
        hold on;
        ylabel(char(col), 'Interpreter','none');
        title(char(col), 'FontSize', 10, 'Interpreter','none');
        grid on;

        % significance brackets
        if ~isempty(Tdunn)
            sigRows = Tdunn(Tdunn.feature == col & Tdunn.significant == true, :);
            yMax = max(data(isfinite(data)));
            yRange = max(data(isfinite(data))) - min(data(isfinite(data)));
            dy = yRange * 0.06;
            for si = 1:min(5, height(sigRows))
                g1 = find(gU == sigRows.group1(si));
                g2 = find(gU == sigRows.group2(si));
                if isempty(g1) || isempty(g2); continue; end
                yBr = yMax + dy * (si + 0.5);
                plot([g1 g2], [yBr yBr], 'k-', 'LineWidth', 1);
                plot([g1 g1], [yBr-dy*0.2, yBr], 'k-', 'LineWidth', 0.8);
                plot([g2 g2], [yBr-dy*0.2, yBr], 'k-', 'LineWidth', 0.8);
                pv = sigRows.p_bonf(si);
                star = '*';
                if pv < 0.01; star = '**'; end
                if pv < 0.001; star = '***'; end
                text((g1+g2)/2, yBr + dy*0.15, star, ...
                    'HorizontalAlignment','center', 'FontSize', 10);
            end
        end
    end
    title(tl, 'Box Plots with Significance Brackets (Dunn''s post-hoc, Bonferroni)', 'FontSize', 11);
    expFig(fig, fullfile(saveDir, 'BOXPLOT_SIGNIFICANCE.png'), dpi);
    close(fig);
end

%% ── Grade distribution overview (histogram grid) ──
function plotGradeDistOverview(X, gIdx, gU, fNames, Timp, saveDir, dpi)
    topF = Timp.feature(1:min(9, height(Timp)));
    nPlot = numel(topF);
    nGrp = numel(gU);
    colors = lines(nGrp);

    nRow = ceil(nPlot/3); nCol = min(3, nPlot);
    fig = figure('Visible','off','Position',[50 50 nCol*380 nRow*300]);
    tl = tiledlayout(fig, nRow, nCol, 'Padding','compact','TileSpacing','compact');

    for fi = 1:nPlot
        jj = find(strcmp(fNames, topF(fi)), 1);
        nexttile(tl); hold on;
        for g = 1:nGrp
            v = X(gIdx==g, jj); v = v(isfinite(v));
            if numel(v) < 3; continue; end
            [fk, xi] = ksdensity(v, 'NumPoints', 80);
            plot(xi, fk, '-', 'Color', colors(g,:), 'LineWidth', 1.3, ...
                'DisplayName', char(gU(g)));
        end
        xlabel(char(topF(fi)), 'Interpreter','none');
        ylabel('Density');
        title(char(topF(fi)), 'FontSize', 9, 'Interpreter','none');
        grid on;
        if fi == 1
            legend('Location','best','FontSize',7);
        end
    end
    title(tl, 'Grade-wise Distribution (KDE)', 'FontSize', 12);
    expFig(fig, fullfile(saveDir, 'GRADE_DIST_OVERVIEW.png'), dpi);
    close(fig);
end


%% =========================================================================
%%  JSON  (기존)
%% =========================================================================
function jmap = buildJsonMap(roots)
jmap = containers.Map('KeyType','char','ValueType','char');
for ri = 1:numel(roots)
    root = char(roots{ri});
    if ~exist(root,'dir')
        warning('CoreColorAnalyzer:JsonDir','%s', ...
            sprintf('JSON 폴더 없음: %s', root));
        continue;
    end
    dd   = dir(fullfile(root,'**','*.json'));
    nAdd = 0;
    for di = 1:numel(dd)
        [~,st,~] = fileparts(dd(di).name);
        key = lower(st);
        if ~jmap.isKey(key)
            jmap(key) = fullfile(dd(di).folder, dd(di).name);
            nAdd = nAdd + 1;
        end
    end
    fprintf('  %s\n  → %d개\n', root, nAdd);
end
end

function jm = readJson(key, jmap)
jm = struct('matched',0,'object_id','','width',NaN,'height',NaN, ...
    'box_no',NaN,'year',NaN,'drilling_place1','','drilling_place2','', ...
    'hole_num','','row_num',NaN,'ers','','camera_type','', ...
    'camera_angle',NaN,'rotation_angle','','shoot_height','', ...
    'f_stop',NaN,'shutter_speed','','iso',NaN,'lux',NaN, ...
    'location','','humidity','','rock_depth',NaN,'rock_strength','', ...
    'tcr','','rqd','','rmr',NaN,'rq','','rjr','','fm','', ...
    'rw','unknown','geo_structure','','rock_type1','','rock_type2','');
if ~jmap.isKey(key); return; end
try
    j = jsondecode(fileread(jmap(key)));
    jm.matched = 1;
    sMap = {'object_id','object_id';'drilling_place1','drilling_place1';
        'drilling_place2','drilling_place2';'hole_num','hole_num';
        'ers','ers';'camera_type','camera_type';
        'rotation_angle','rotation_angle';'shoot_height','shoot_height';
        'shutter_speed','shutter_speed';'location','location';
        'humidity','humidity';'rock_strength','rock_strength';
        'tcr','tcr';'rqd','rqd';'rq','rq';'rjr','rjr';
        'fm','fm';'rw','rw';'geo_structure','geo_structure'};
    for fi = 1:size(sMap,1)
        sk = sMap{fi,1}; jk = sMap{fi,2};
        if isfield(j,jk) && ~isempty(j.(jk))
            jm.(sk) = char(string(j.(jk)));
        end
    end
    nMap = {'width','width';'height','height';'box_no','box_no';
        'year','year';'row_num','row_num';'camera_angle','camera_angle';
        'rock_depth','rock_depth';'rmr','rmr';'lux','lux'};
    for fi = 1:size(nMap,1)
        sk = nMap{fi,1}; jk = nMap{fi,2};
        if isfield(j,jk); jm.(sk) = toNum(j.(jk)); end
    end
    if isfield(j,'f_stop');    jm.f_stop = toNum(j.f_stop);
    elseif isfield(j,'fstop'); jm.f_stop = toNum(j.fstop); end
    if isfield(j,'ISO');       jm.iso = toNum(j.ISO);
    elseif isfield(j,'iso');   jm.iso = toNum(j.iso); end
    if isfield(j,'rock_type')
        rt = j.rock_type;
        if isstruct(rt); rt = num2cell(rt); end
        for ri = 1:numel(rt)
            e = rt{ri};
            if isfield(e,'rock_type1') && ~isempty(e.rock_type1)
                jm.rock_type1 = char(string(e.rock_type1)); end
            if isfield(e,'rock_type2') && ~isempty(e.rock_type2)
                jm.rock_type2 = char(string(e.rock_type2)); end
        end
    end
catch ME
    warning(ME.identifier,'%s', ...
        sprintf('JSON 파싱 오류[%s]: %s', key, ME.message));
    jm.matched = 0;
end
end

function v = toNum(raw)
if isnumeric(raw) && isscalar(raw); v = double(raw);
elseif ischar(raw) || isstring(raw); v = str2double(raw);
else; v = NaN;
end
end


%% =========================================================================
%%  통계 (기존)
%% =========================================================================
function row = addStats(row, ch, X, mask)
v = X(mask); v = v(isfinite(v));
if isempty(v); v = 0; end
row.([ch '_mean']) = mean(v);
row.([ch '_std'])  = std(v,0);
row.([ch '_min'])  = min(v);
row.([ch '_max'])  = max(v);
pcts = prctile(v,[5 25 50 75 95]);
row.([ch '_p05'])  = pcts(1);
row.([ch '_p25'])  = pcts(2);
row.([ch '_p50'])  = pcts(3);
row.([ch '_p75'])  = pcts(4);
row.([ch '_p95'])  = pcts(5);
row.([ch '_iqr'])  = pcts(4) - pcts(2);
row.([ch '_ipr'])  = pcts(5) - pcts(1);
end

function row = addShape(row, ch, X, mask)
v = X(mask); v = v(isfinite(v));
if numel(v) < 4
    row.([ch '_skew'])    = NaN;
    row.([ch '_kurt'])    = NaN;
    row.([ch '_entropy']) = NaN;
    return;
end
mu = mean(v); sig = std(v,0);
if sig < 1e-10
    row.([ch '_skew']) = 0; row.([ch '_kurt']) = 0;
else
    zv = (v-mu)/sig;
    row.([ch '_skew']) = mean(zv.^3);
    row.([ch '_kurt']) = mean(zv.^4) - 3;
end
edges = linspace(min(v), max(v)+1e-9, 65);
cnt   = histcounts(v, edges);
p     = cnt / max(sum(cnt),1); p = p(p>0);
row.([ch '_entropy']) = -sum(p .* log2(p));
end

function [V,C,H_code,H_name] = lab2munsell(L, a, b)
V = max(0, min(10, L/10)); C = max(0, hypot(a,b)/5.5);
if hypot(a,b) < 4.0; H_code = -1; H_name = 'N'; return; end
h = mod(atan2d(b,a), 360);
hMap = [0   27  2.5  '5R  '; 27  45  20   '5YR '; 45  63  54   '7.5Y';
        63  80  72   '10YR'; 80  99  90   '2.5Y'; 99  116 108  '5Y  ';
        116 207 126  '5GY '; 207 261 252  '5B  ';
        261 297 288  '5PB '; 297 360 306  '5P  '];
H_code = 2.5; H_name = '5YR';
for ti = 1:size(hMap,1)
    if h >= hMap(ti,1) && h < hMap(ti,2)
        H_code = hMap(ti,3); H_name = strtrim(hMap(ti,4)); return;
    end
end
end

function mask = makeMask(I0, roiMode, bgTh)
[H,W,~] = size(I0);
switch lower(string(roiMode))
    case 'none';  mask = true(H,W);
    case 'auto'
        hsv  = rgb2hsv(im2single(I0)); V = hsv(:,:,3);
        mask = V > bgTh;
        if nnz(mask) < 0.1*H*W; mask = true(H,W); end
    case 'manual'
        try
            figure('Visible','on'); imshow(I0);
            title('ROI: 드래그 후 더블클릭');
            hh = drawfreehand('Color','y'); wait(hh);
            mask = createMask(hh); close(gcf);
        catch
            mask = true(H,W);
        end
    otherwise; mask = true(H,W);
end
if nnz(mask) < 50; mask = true(H,W); end
end


%% =========================================================================
%%  집계 (기존)
%% =========================================================================
function numCols = getNumCols(T)
skip = {'img_index','file','stem','fn_','rw','humidity','json_','object_id', ...
    'box_no','year','hole_num','drilling_','ers','camera_','rotation_', ...
    'shoot_','f_stop','shutter_','iso','lux','location','rock_depth', ...
    'rock_strength','tcr','rqd','rmr','rq','rjr','fm','geo_','rock_type', ...
    'height_px','width_px','nPixels','munsell_H_name'};
allC = T.Properties.VariableNames; numCols = {};
for ci = 1:numel(allC)
    col = allC{ci}; s = false;
    for pi = 1:numel(skip)
        if startsWith(col, skip{pi}); s = true; break; end
    end
    if s; continue; end
    if isnumeric(T.(col)); numCols{end+1} = col; end %#ok<AGROW>
end
end

function Tagg = gradeAgg(T, grpCol, numCols)
if ~any(strcmp(T.Properties.VariableNames, grpCol)); Tagg = table(); return; end
grps = unique(T.(grpCol));
grps = sort(grps(strlength(strtrim(string(grps))) > 0));
out  = cell(numel(grps),1);
for gi = 1:numel(grps)
    g  = grps(gi); mk = (T.(grpCol) == g); sub = T(mk,:);
    r  = struct(); r.(grpCol) = char(string(g)); r.n = nnz(mk);
    for ci = 1:numel(numCols)
        col = numCols{ci};
        if ~any(strcmp(sub.Properties.VariableNames, col)); continue; end
        v = sub.(col); v = v(isfinite(v));
        if isempty(v)
            r.([col '_mean'])=NaN; r.([col '_std'])=NaN;
            r.([col '_p05'])=NaN;  r.([col '_p50'])=NaN; r.([col '_p95'])=NaN;
        else
            r.([col '_mean'])=mean(v); r.([col '_std'])=std(v,0);
            pcts=prctile(v,[5 50 95]);
            r.([col '_p05'])=pcts(1); r.([col '_p50'])=pcts(2); r.([col '_p95'])=pcts(3);
        end
    end
    out{gi} = r;
end
ok = ~cellfun(@isempty, out);
if ~any(ok); Tagg = table(); return; end
Tagg = struct2table(vertcat(out{ok}));
end

function Tc = crossTab(T, rv, cv)
if ~any(strcmp(T.Properties.VariableNames,rv)) || ...
   ~any(strcmp(T.Properties.VariableNames,cv)); Tc = table(); return; end
rV = unique(T.(rv)); rV = sort(rV(strlength(strtrim(string(rV))) > 0));
cV = unique(T.(cv)); cV = sort(cV(strlength(strtrim(string(cV))) > 0));
data = zeros(numel(rV), numel(cV));
for ri = 1:numel(rV)
    for ci = 1:numel(cV)
        data(ri,ci) = nnz(T.(rv)==rV(ri) & T.(cv)==cV(ci));
    end
end
Tc = array2table(data, 'RowNames', cellstr(string(rV)), ...
    'VariableNames', matlab.lang.makeValidName(cellstr(string(cV))));
Tc = [table(rV,'VariableNames',{rv}), Tc];
Tc.total = sum(data, 2);
end


%% =========================================================================
%%  Excel 저장 (확장: statSheets 추가)
%% =========================================================================
function xlsxFiles = writeExcelSplit(T, Tg, Th, Tc, statSheets, basePath, ...
    OPT, nTotal, nMatched, nValid, elapsed)
MAXR = 1048000; MAXC = 16000; xlsxFiles = {};
nRow = height(T); nCol = width(T); nRP = ceil(nRow/MAXR);

% stat 시트 이름 목록
statFields = fieldnames(statSheets);

if nCol <= MAXC
    for pi = 1:nRP
        r1 = (pi-1)*MAXR+1; r2 = min(pi*MAXR, nRow);
        fp = basePath;
        if nRP > 1; fp = basePath + sprintf('_part%03d',pi); end
        fp = char(fp) + '.xlsx';
        fprintf('  %s  (행%d~%d)\n', fp, r1, r2);
        wSheet(T(r1:r2,:), fp, 'PIXEL_STATS');
        if pi == 1
            wSheet(Tg, fp, 'GRADE_STATS');
            wSheet(Th, fp, 'HUMID_STATS');
            wSheet(Tc, fp, 'RW_x_HUMID');
            % ★ Nature급 통계 시트 추가
            for si = 1:numel(statFields)
                sName = statFields{si};
                sData = statSheets.(sName);
                if istable(sData) && ~isempty(sData)
                    sheetName = sName;
                    if strlength(sheetName) > 31
                        sheetName = extractBefore(sheetName, 32);
                    end
                    wSheet(sData, fp, char(sheetName));
                end
            end
        end
        wSheet(makeIdx(OPT,nTotal,nMatched,nValid,elapsed,pi,nRP), fp, 'INDEX');
        xlsxFiles{end+1} = fp; %#ok<AGROW>
    end
else
    metaC  = getMetaCols(T);
    dataC  = setdiff(T.Properties.VariableNames, metaC, 'stable');
    chunkC = MAXC - numel(metaC);
    nCP    = ceil(numel(dataC)/chunkC); fidx = 0;
    for ri = 1:nRP
        r1 = (ri-1)*MAXR+1; r2 = min(ri*MAXR, nRow);
        for ci = 1:nCP
            c1 = (ci-1)*chunkC+1; c2 = min(ci*chunkC, numel(dataC));
            fidx = fidx + 1;
            fp = char(basePath + sprintf('_r%03d_c%03d.xlsx', ri, ci));
            wSheet(T(r1:r2, [metaC, dataC(c1:c2)]), fp, 'PIXEL_STATS');
            if fidx == 1
                wSheet(Tg, fp, 'GRADE_STATS');
                wSheet(Th, fp, 'HUMID_STATS');
                wSheet(Tc, fp, 'RW_x_HUMID');
                for si = 1:numel(statFields)
                    sName = statFields{si};
                    sData = statSheets.(sName);
                    if istable(sData) && ~isempty(sData)
                        sheetName = sName;
                        if strlength(sheetName) > 31
                            sheetName = extractBefore(sheetName, 32);
                        end
                        wSheet(sData, fp, char(sheetName));
                    end
                end
            end
            wSheet(makeIdx(OPT,nTotal,nMatched,nValid,elapsed,fidx,nRP*nCP), fp,'INDEX');
            xlsxFiles{end+1} = fp; %#ok<AGROW>
            fprintf('  %s\n', fp);
        end
    end
end
end

function metaC = getMetaCols(T)
allC = T.Properties.VariableNames;
pfx  = {'img_index','file','stem','fn_','rw','humidity','json_','object_id', ...
    'box_no','year','hole_num','drilling_','ers','camera_','rotation_', ...
    'shoot_','f_stop','shutter_','iso','lux','location','rock_depth', ...
    'rock_strength','tcr','rqd','rmr','rq','rjr','fm','geo_','rock_type', ...
    'height_px','width_px','nPixels'};
metaC = {};
for ci = 1:numel(allC)
    col = allC{ci};
    for pi = 1:numel(pfx)
        if startsWith(col, pfx{pi}); metaC{end+1} = col; break; end %#ok<AGROW>
    end
end
metaC = unique(metaC,'stable');
end

function Ti = makeIdx(OPT, nTotal, nMatched, nValid, elapsed, part, nParts)
items = {'timestamp',     char(string(datetime('now','Format','yyyy-MM-dd HH:mm:ss')))
         'matlab_version', version
         'part',           sprintf('%d/%d', part, nParts)
         'n_total',        num2str(nTotal)
         'n_json_matched', num2str(nMatched)
         'n_analyzed',     num2str(nValid)
         'n_skipped',      num2str(nTotal-nValid)
         'elapsed_min',    num2str(elapsed/60,'%.1f')
         'use_gpu',        tf2s(OPT.USE_GPU)
         'use_parfor',     tf2s(OPT.USE_PAR)
         'gpu_batch_init', num2str(OPT.GPU_BATCH)
         'gpu_mem_floor',  sprintf('%.1f GB', OPT.GPU_MEM_FLOOR/1e9)
         'roi_mode',       char(string(OPT.ROI_MODE))
         'rw_source',      'JSON rw 필드만 사용'
         'pixel_policy',   'raw 전체 픽셀, 보정 없음'
         'skip_policy',    'JSON 미매칭 → 스킵, 오류 → 경고 후 스킵'
         'stat_analysis',  'KW + Dunn + eta2 + Cohen d + Spearman + PCA + Silhouette + Bootstrap'
         'ref_D1',         sprintf('L=%.2f a=%.2f b=%.2f', ...
                               OPT.REF_D1_L, OPT.REF_D1_a, OPT.REF_D1_b)
         'results_root',   char(string(OPT.RESULTS_ROOT))};
Ti = cell2table(items, 'VariableNames',{'parameter','value'});
end

function wSheet(T, fp, sh)
if isempty(T); return; end
try
    writetable(T, fp, 'Sheet', sh, 'WriteRowNames', true);
catch ME
    warning(ME.identifier,'%s', sprintf('[%s] 저장 실패: %s', sh, ME.message));
end
end


%% =========================================================================
%%  PNG (기본 OFF, 기존)
%% =========================================================================
function exportPng(I0, row, imageId, iPng, iCsv, pngDir, OPT)
I   = im2single(I0);
h3  = rgb2hsv(I); l3 = rgb2lab(I);
R   = double(I0(:,:,1)); G = double(I0(:,:,2)); B = double(I0(:,:,3));
Hc  = h3(:,:,1); Sc = h3(:,:,2); Vc = h3(:,:,3);
Lc  = l3(:,:,1); ac = l3(:,:,2); bc = l3(:,:,3);
mask = makeMask(I0, OPT.ROI_MODE, OPT.AUTO_BG_TH);
imwrite(I0, fullfile(iPng,'ORIGINAL.png'));
imwrite(uint8(mask)*255, fullfile(iPng,'MASK.png'));
saveRoi(I0, mask, fullfile(iPng,'ROI_OVERLAY.png'), OPT.REPORT_DPI);
cfg = buildCfg(R,G,B,Hc,Sc,Vc,Lc,ac,bc);
for k = 1:numel(cfg)
    c = cfg(k);
    saveChanMap(c,  mask, fullfile(iPng,'MAP_' +c.key+'.png'), OPT.REPORT_DPI);
    saveChanHist(c, mask, fullfile(iPng,'HIST_'+c.key+'.png'), ...
        fullfile(iCsv,'HIST_'+c.key+'.csv'), OPT.REPORT_DPI);
end
saveRgbHist(R,G,B, mask, fullfile(iPng,'HIST_RGB.png'), ...
    fullfile(iCsv,'HIST_RGB.csv'), OPT.REPORT_DPI);
saveAbSpace(ac, bc, mask, iPng, iCsv, OPT);
rp = fullfile(iPng,'REPORT_9CH.png');
saveReport(I0, mask, row, cfg, rp, OPT.REPORT_DPI);
try copyfile(rp, fullfile(pngDir, string(imageId)+'_REPORT_9CH.png'));
catch
end
end

function cfg = buildCfg(R,G,B,Hc,Sc,Vc,Lc,ac,bc)
mk = @(k,t,u,X,mn,mx,r,h,d) struct('key',string(k),'title',string(t), ...
    'units',string(u),'data',double(X),'vmin',mn,'vmax',mx, ...
    'render',string(r),'hue',double(h),'diverging',logical(d));
cfg = [mk('R','R','0-255',R,0,255,'huesat',0,false)
       mk('G','G','0-255',G,0,255,'huesat',1/3,false)
       mk('B','B','0-255',B,0,255,'huesat',2/3,false)
       mk('H','H','0-1',Hc,0,1,'hueonly',nan,false)
       mk('S','S','0-1',Sc,0,1,'scalar',nan,false)
       mk('V','V','0-1',Vc,0,1,'scalar',nan,false)
       mk('L','L*','0-100',Lc,0,100,'scalar',nan,false)
       mk('a','a*','-128~127',ac,-128,127,'scalar',nan,true)
       mk('b','b*','-128~127',bc,-128,127,'scalar',nan,true)];
end

function saveRoi(I0, mask, fp, dpi)
fig = figure('Visible','off','Position',[100 100 1100 500]);
imshow(I0); hold on;
ov = cat(3,ones(size(mask)),zeros(size(mask)),zeros(size(mask)));
h  = imshow(ov); set(h,'AlphaData',0.20*double(mask));
title('ROI overlay'); expFig(fig,fp,dpi); close(fig);
end

function saveChanMap(c, mask, fp, dpi)
X   = c.data; X(~mask) = NaN;
fig = figure('Visible','off','Position',[100 100 950 400]);
ax  = axes(fig);
switch lower(c.render)
    case 'huesat'
        xn = min(max((X-c.vmin)/max(eps,c.vmax-c.vmin),0),1);
        imshow(hsv2rgb(cat(3,ones(size(xn))*c.hue,xn,ones(size(xn)))),'Parent',ax);
        axis(ax,'image'); axis(ax,'off');
    case 'hueonly'
        H = min(max(X,0),1);
        imshow(hsv2rgb(cat(3,H,ones(size(H)),ones(size(H)))),'Parent',ax);
        axis(ax,'image'); axis(ax,'off');
    otherwise
        imagesc(ax,X); axis(ax,'image'); axis(ax,'off');
        if c.diverging; colormap(ax,divMap(256)); else; colormap(ax,parula(256)); end
        clim(ax,[c.vmin c.vmax]); colorbar(ax);
end
title(ax, sprintf('MAP %s (%s)',c.title,c.units),'Interpreter','none');
expFig(fig,fp,dpi); close(fig);
end

function saveChanHist(c, mask, fpPng, fpCsv, dpi)
v    = c.data(mask); v = v(isfinite(v)); if isempty(v); v=0; end
e    = chanEdges(c.key); cen = (e(1:end-1)+e(2:end))/2;
prob = histcounts(v, e, 'Normalization','probability');
pcts = prctile(v,[5 50 95]);
try
    writetable(table(cen(:),prob(:), ...
        repmat(pcts(1),numel(cen),1), ...
        repmat(pcts(2),numel(cen),1), ...
        repmat(pcts(3),numel(cen),1), ...
        'VariableNames',{'center','prob','p05','p50','p95'}), fpCsv);
catch
end
fig = figure('Visible','off','Position',[100 100 900 400]);
ax  = axes(fig); bar(ax,cen,prob,1.0,'EdgeColor','none'); grid(ax,'on');
xlabel(ax,'value'); ylabel(ax,'probability');
title(ax, sprintf('HIST %s',c.title),'Interpreter','none');
xline(ax,pcts(1),'--'); xline(ax,pcts(2),'-'); xline(ax,pcts(3),'--');
expFig(fig,fpPng,dpi); close(fig);
end

function saveRgbHist(R,G,B, mask, fpPng, fpCsv, dpi)
rv = R(mask); gv = G(mask); bv = B(mask);
rv = rv(isfinite(rv)); gv = gv(isfinite(gv)); bv = bv(isfinite(bv));
if isempty(rv);rv=0;end; if isempty(gv);gv=0;end; if isempty(bv);bv=0;end
e   = linspace(0,255,257); cen = (e(1:end-1)+e(2:end))/2;
pr  = histcounts(rv,e,'Normalization','probability');
pg  = histcounts(gv,e,'Normalization','probability');
pb  = histcounts(bv,e,'Normalization','probability');
try writetable(table(cen(:),pr(:),pg(:),pb(:), ...
    'VariableNames',{'center','R','G','B'}), fpCsv);
catch
end
fig = figure('Visible','off','Position',[100 100 900 400]);
ax  = axes(fig); hold(ax,'on');
plot(ax,cen,pr,'LineWidth',1.3); plot(ax,cen,pg,'LineWidth',1.3);
plot(ax,cen,pb,'LineWidth',1.3); grid(ax,'on');
legend(ax,["R","G","B"],'Location','northeast');
xlabel(ax,'value'); ylabel(ax,'probability'); title(ax,'RGB overlay');
expFig(fig,fpPng,dpi); close(fig);
end

function saveAbSpace(ac, bc, mask, iPng, iCsv, OPT)
a = ac(mask); b = bc(mask); ok = isfinite(a)&isfinite(b);
a = a(ok); b = b(ok); if isempty(a);a=0;b=0;end
ae = linspace(-20,20,81); be = linspace(-10,30,81);
N  = histcounts2(a,b,ae,be,'Normalization','probability');
try writetable(array2table(N), fullfile(iCsv,'LAB_AB_HIST2.csv'));
catch
end
fig = figure('Visible','off','Position',[100 100 1100 450]);
tl  = tiledlayout(fig,1,2,'Padding','compact','TileSpacing','compact');
nexttile(tl);
imagesc((be(1:end-1)+be(2:end))/2, (ae(1:end-1)+ae(2:end))/2, N);
axis image; set(gca,'YDir','normal');
xlabel('b*'); ylabel('a*'); title('a*-b* 2D hist'); colormap(parula(256)); colorbar;
nexttile(tl);
ns = min(OPT.MAX_SCATTER,numel(a)); rng(0); idx = randperm(numel(a),ns);
scatter(a(idx),b(idx),3,'.','MarkerEdgeAlpha',0.15);
axis equal; grid on; xlim([-20 20]); ylim([-10 30]);
xlabel('a*'); ylabel('b*'); title(sprintf('scatter n=%d',ns));
expFig(fig, fullfile(iPng,'LAB_AB_SPACE.png'), OPT.REPORT_DPI); close(fig);
end

function saveReport(I0, mask, row, cfg, fp, dpi)
fig = figure('Visible','off','Position',[60 60 2600 900]);
tl  = tiledlayout(fig,2,5,'Padding','compact','TileSpacing','compact');
nexttile(tl); imshow(I0); title('Original','Interpreter','none');
nexttile(tl); imshow(I0); hold on;
ov = cat(3,ones(size(mask)),zeros(size(mask)),zeros(size(mask)));
h  = imshow(ov); set(h,'AlphaData',0.22*double(mask));
title(sprintf('ROI %dpx',row.roi_pixels),'Interpreter','none'); axis image off;
chs = {'L','a','b','V','R','G','B','H'};
for mi = 1:8
    nexttile(tl); quickMap(cfg,chs{mi},mask);
    title(sprintf('%s map',chs{mi}),'Interpreter','none');
end
expFig(fig,fp,dpi); close(fig);
end

function quickMap(cfg, key, mask)
idx = find(string({cfg.key}) == string(key), 1);
if isempty(idx); imagesc(zeros(size(mask))); axis image off; return; end
c = cfg(idx); X = c.data; X(~mask) = NaN;
switch lower(c.render)
    case 'huesat'
        xn = min(max((X-c.vmin)/max(eps,c.vmax-c.vmin),0),1);
        imshow(hsv2rgb(cat(3,ones(size(xn))*c.hue,xn,ones(size(xn))))); axis image off;
    case 'hueonly'
        H = min(max(X,0),1);
        imshow(hsv2rgb(cat(3,H,ones(size(H)),ones(size(H))))); axis image off;
    otherwise
        imagesc(X); axis image off;
        if c.diverging; colormap(divMap(256)); else; colormap(parula(256)); end
        clim([c.vmin c.vmax]); colorbar;
end
end

function edges = chanEdges(key)
switch string(key)
    case {'R','G','B'};  edges = linspace(0,255,257);
    case {'H','S','V'};  edges = linspace(0,1,101);
    case 'L';            edges = linspace(0,100,101);
    case {'a','b'};      edges = linspace(-20,30,101);
    otherwise;           edges = linspace(0,1,101);
end
end

function cmap = divMap(n)
x = linspace(0,1,n)'; cmap = zeros(n,3);
i1 = x<=0.5; t1 = x(i1)/0.5;       cmap(i1,:) = [t1 t1 ones(sum(i1),1)];
i2 = x>0.5;  t2 = (x(i2)-0.5)/0.5; cmap(i2,:) = [ones(sum(i2),1) 1-t2 1-t2];
end

function expFig(fig, fp, dpi)
try exportgraphics(fig, char(fp), 'Resolution', dpi);
catch
    try print(fig, char(fp), '-dpng', sprintf('-r%d',dpi));
    catch
    end
end
end


%% =========================================================================
%%  파일명 파싱 (기존)
%% =========================================================================
function fn = parseFilename(stem)
fn = struct('site','','borehole','','box_no','','rock1','','rock2','', ...
    'angle',NaN,'rotation','','humid','','lux_bin',NaN,'seq',NaN);
pTok  = regexp(stem,'\(([^)]*)\)','tokens');
clean = regexprep(stem,'\([^)]*\)','(X)');
parts = strsplit(clean,'-');
if numel(parts) < 8; return; end
fn.site     = strtrim(parts{1});
fn.borehole = regexprep(strtrim(parts{2}),'\(X\)','');
if numel(pTok) >= 1; fn.box_no = pTok{1}{1}; end
fn.rock1 = strtrim(parts{3}); fn.rock2 = strtrim(parts{4});
aNum = regexp(strtrim(parts{5}),'\d+','match','once');
if ~isempty(aNum); fn.angle = str2double(aNum); end
if numel(pTok) >= 2; fn.rotation = pTok{2}{1}; end
h = upper(strtrim(parts{6}));
if strcmp(h,'W'); fn.humid='습윤'; elseif strcmp(h,'D'); fn.humid='건조'; else; fn.humid=h; end
fn.lux_bin = str2double(strtrim(parts{7}));
fn.seq     = str2double(strtrim(parts{8}));
end


%% =========================================================================
%%  유틸 (기존 + Nature 추가)
%% =========================================================================
function OPT = parseOpts(varargin)
p = inputParser;
addParameter(p,'IMAGE_ROOTS',{
    'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\images_igneous_rock\TS_1, 기반암 암종 분류 데이터_1. 화성암_1'
    'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\images_igneous_rock\TS_1, 기반암 암종 분류 데이터_1. 화성암_2'
    'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\images_igneous_rock\TS_2. 기반암 절리 탐지 데이터_1. 화성암'
    'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\images_igneous_rock\VS_1. 기반암 암종 분류 데이터_1. 화성암'
    'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\images_igneous_rock\VS_2. 기반암 절리 탐지 데이터_1. 화성암'
});
addParameter(p,'JSON_ROOTS',{
    'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\json_igneous_rock\TL_1. 기반암 암종 분류 데이터_1. 화성암'
    'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\json_igneous_rock\TL_2. 기반암 절리 탐지 데이터_1. 화성암'
    'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\json_igneous_rock\VL_1. 기반암 암종 분류 데이터_1. 화성암'
    'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\json_igneous_rock\VL_2. 기반암 절리 탐지 데이터_1. 화성암'
});
addParameter(p,'RESULTS_ROOT', 'C:\Users\ROCKENG\Desktop\코랩 머신러닝\results');
addParameter(p,'IMG_EXTS',     {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'});
addParameter(p,'ROI_MODE',     'none');
addParameter(p,'AUTO_BG_TH',   0.04);
addParameter(p,'EXPORT_PNG',   false);
addParameter(p,'REPORT_DPI',   220);
addParameter(p,'STAT_DPI',     300);       % ★ Nature 그래프 DPI
addParameter(p,'MAX_SCATTER',  50000);
addParameter(p,'USE_GPU',      true);
addParameter(p,'USE_PAR',      true);
addParameter(p,'GPU_BATCH',    32);
addParameter(p,'GPU_MEM_FLOOR',1.5e9);
addParameter(p,'SAVE_EVERY',   500);
addParameter(p,'REF_D1_L',     48.12);
addParameter(p,'REF_D1_a',     -1.77);
addParameter(p,'REF_D1_b',     -1.29);
addParameter(p,'N_BOOT_CORR',  1000);     % ★ 상관 Bootstrap 반복 수
addParameter(p,'N_BOOT_GRADE', 1000);     % ★ 등급 Bootstrap 반복 수
addParameter(p,'CORR_TOP_K',   25);       % ★ 상관 행렬 상위 K 특징
parse(p, varargin{:});
OPT = p.Results;
if ischar(OPT.IMAGE_ROOTS)||isstring(OPT.IMAGE_ROOTS)
    OPT.IMAGE_ROOTS = cellstr(OPT.IMAGE_ROOTS); end
if ischar(OPT.JSON_ROOTS)||isstring(OPT.JSON_ROOTS)
    OPT.JSON_ROOTS = cellstr(OPT.JSON_ROOTS); end
OPT.RESULTS_ROOT = string(OPT.RESULTS_ROOT);
OPT.ROI_MODE     = lower(string(OPT.ROI_MODE));
end

function cellFiles = collectImages(roots, exts)
cellFiles = {};
for ri = 1:numel(roots)
    root = char(roots{ri});
    if ~exist(root,'dir')
        warning('CoreColorAnalyzer:ImgDir','%s', ...
            sprintf('폴더 없음(스킵): %s', root));
        continue;
    end
    dd = dir(fullfile(root,'**','*')); nAdd = 0;
    for di = 1:numel(dd)
        if dd(di).isdir; continue; end
        [~,~,e] = fileparts(dd(di).name);
        if any(strcmpi(e, exts))
            cellFiles{end+1} = fullfile(dd(di).folder, dd(di).name); %#ok<AGROW>
            nAdd = nAdd + 1;
        end
    end
    fprintf('  %s\n  → %d장\n', root, nAdd);
end
cellFiles = unique(cellFiles,'stable')';
end

function [gpuOK, name, dev] = initGPU(want)
gpuOK = false; name = '없음(CPU)'; dev = [];
if ~want; return; end
try setenv('MW_CUDA_FORWARD_COMPATIBILITY','1');
catch
end
try parallel.gpu.enableCUDAForwardCompatibility(true);
catch
end
try
    g    = gpuDevice(1);
    fprintf('  GPU : %s\n',  g.Name);
    fprintf('  CC  : %s\n',  string(g.ComputeCapability));
    fprintf('  VRAM: %.2f / %.2f GB\n', g.AvailableMemory/1e9, g.TotalMemory/1e9);
    gpuOK = true; name = g.Name; dev = g;
catch ME
    warning(ME.identifier,'%s', sprintf('GPU 없음: %s', ME.message));
end
end

function nW = initParallel(want)
nW = 0; if ~want; return; end
try
    if license('test','Distrib_Computing_Toolbox')
        pp = gcp('nocreate');
        if isempty(pp); pp = parpool('Processes'); end
        nW = pp.NumWorkers;
        fprintf('  parpool: %d workers\n', nW);
    end
catch ME
    warning(ME.identifier,'%s', sprintf('parpool 실패: %s', ME.message));
end
end

function I = ensureRGB(I0)
if isempty(I0); error('CoreColorAnalyzer:Empty','빈 이미지'); end
if ismatrix(I0); I0 = repmat(I0,1,1,3);
elseif size(I0,3) == 4; I0 = I0(:,:,1:3); end
if ~isa(I0,'uint8'); I0 = im2uint8(I0); end
I = I0;
end

function T = stringifyCols(T)
pfx = {'file','stem','fn_','rw','humidity','json_','object_id','hole_num', ...
    'rock_strength','tcr','rqd','rq','rjr','fm','geo_','rock_type', ...
    'camera_','location','drilling_','ers','rotation_','shoot_', ...
    'shutter_','munsell_H_name'};
for ci = 1:width(T)
    col = T.Properties.VariableNames{ci};
    for pi = 1:numel(pfx)
        if startsWith(col, pfx{pi})
            try T.(col) = string(T.(col));
            catch
            end
            break;
        end
    end
end
end

function ensureDirs(C)
for i = 1:numel(C)
    d = char(C{i}); if ~exist(d,'dir'); mkdir(d); end
end
end

function s = tf2s(v)
if v; s = 'ON'; else; s = 'OFF'; end
end

%% ── 통계 유틸 (Nature 추가) ──
function ranks = tiedrank_local(x)
    n = numel(x);
    ranks = zeros(n,1);
    i = 1;
    while i <= n
        j = i;
        while j < n && x(j+1) == x(j)
            j = j + 1;
        end
        avgRank = (i + j) / 2;
        ranks(i:j) = avgRank;
        i = j + 1;
    end
end

function p = normcdf_local(z)
    p = 0.5 * erfc(-z / sqrt(2));
end

function pAdj = bhFDR(pVals)
    p = pVals(:);
    n = numel(p);
    pAdj = nan(n,1);
    validIdx = find(isfinite(p));
    if isempty(validIdx); return; end
    pv = p(validIdx);
    [pSorted, sortI] = sort(pv);
    m = numel(pv);
    adjSorted = nan(m,1);
    adjSorted(m) = pSorted(m);
    for i = m-1:-1:1
        adjSorted(i) = min(adjSorted(i+1), pSorted(i) * m / i);
    end
    adjSorted = min(adjSorted, 1);
    unsorted = nan(m,1);
    unsorted(sortI) = adjSorted;
    pAdj(validIdx) = unsorted;
end

function cmap = rwbColormap(n)
    x = linspace(0,1,n)';
    cmap = zeros(n,3);
    mid = 0.5;
    % blue → white
    mask1 = x <= mid;
    t1 = x(mask1) / mid;
    cmap(mask1,:) = [t1, t1, ones(sum(mask1),1)];
    % white → red
    mask2 = x > mid;
    t2 = (x(mask2) - mid) / (1 - mid);
    cmap(mask2,:) = [ones(sum(mask2),1), 1-t2, 1-t2];
end

function drawEllipse95(mu, C, color)
    try
        [V, D] = eig(C);
        theta = linspace(0, 2*pi, 100);
        r = sqrt(chi2inv(0.95, 2));
        xy = r * [cos(theta); sin(theta)];
        rotated = V * sqrt(D) * xy;
        plot(mu(1) + rotated(1,:), mu(2) + rotated(2,:), '-', ...
            'Color', [color, 0.5], 'LineWidth', 1.2, 'HandleVisibility','off');
    catch
    end
end
