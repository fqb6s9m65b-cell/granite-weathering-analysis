function results = RockPLY_ThreeStage_Weathering3D_R2025a(varargin)
% RockPLY_OneFile_StageFlow_R2025a
% =========================================================================
% ONE-FILE / MATLAB R2025a-b
%
% PLY 자체를 원천 3D 시각 데이터로 사용하여
%   1) PLY 로드
%   2) 9채널 색상 분석
%   3) 3D 형상 분석
%   4) PLY -> 2D raster 투영
%   5) 암종 MAT / 풍화도 MAT tile-wise 추론
%   6) 2D -> 3D 역매핑
%   7) 텍스처 / 통계 / 상관 / ΔE
%   8) 시각화 / 저장
%
% 특징:
%   - GPU 우선, CPU fallback
%   - 배경(validMask 외부) 예측 오염 방지
%   - dlnetwork logits -> softmax 보정
%   - tile-wise 분할 후 결합
%   - ONE-FILE
%
% 작성 기준:
%   - PLY 자체가 "원천 이미지/시각 데이터"
%   - image scaling은 별도 JPG/PNG가 아니라 raster grid 해상도 조절 개념
% =========================================================================

%% ------------------------------------------------------------------------
% [0] 옵션
%% ------------------------------------------------------------------------
p = inputParser;
p.FunctionName = mfilename;

addParameter(p,'PLY',               '',            @(x)ischar(x)||isstring(x));
addParameter(p,'RockTypeMAT',       '',            @(x)ischar(x)||isstring(x));
addParameter(p,'WeatherMAT',        '',            @(x)ischar(x)||isstring(x));
addParameter(p,'OutputDir',         '',            @(x)ischar(x)||isstring(x));

addParameter(p,'MaxPoints',         0,             @isnumeric);
addParameter(p,'GridSize',          1024,          @isnumeric);
addParameter(p,'TileSize',          112,           @isnumeric);
addParameter(p,'TileStride',        112,           @isnumeric);
addParameter(p,'MiniBatchSize',     64,            @isnumeric);

addParameter(p,'BackgroundFillRGB', [255 255 255], @isnumeric);
addParameter(p,'GradeEdges',        [0.2 0.4 0.6 0.8], @isnumeric);
addParameter(p,'ConfThreshold',     0.60,          @isnumeric);
addParameter(p,'KnnNeighbors',      30,            @isnumeric);
addParameter(p,'RoughnessPoints',   80000,         @isnumeric);
addParameter(p,'LoGSigma',          1.0,           @isnumeric);
addParameter(p,'TextureWins',       [3 5 9 15],    @isnumeric);
addParameter(p,'StatTestAlpha',     0.05,          @isnumeric);

addParameter(p,'SkipRockType',      false,         @islogical);
addParameter(p,'ForceWeathering',   false,         @islogical);
addParameter(p,'GraniteKeywords',   {'granite','화강암','Granite','GRANITE','granit'}, @iscell);

addParameter(p,'SaveMat',           true,          @islogical);
addParameter(p,'SaveXlsx',          true,          @islogical);
addParameter(p,'SaveCSV',           true,          @islogical);
addParameter(p,'SaveFigures',       true,          @islogical);
addParameter(p,'FigDPI',            300,           @isnumeric);
addParameter(p,'VisibleFigures',    'on',          @(x)ischar(x)||isstring(x));

parse(p,varargin{:});
opts = p.Results;

t0  = tic;
SEP = repmat('=',1,88);

fprintf('\n%s\n', SEP);
fprintf(' RockPLY One-File Analyzer (MATLAB R2025a/b)\n');
fprintf(' TileSize=%d | TileStride=%d | GridSize=%d\n', ...
    opts.TileSize, opts.TileStride, opts.GridSize);
fprintf('%s\n\n', SEP);

%% ------------------------------------------------------------------------
% [1] 환경 / GPU / 파일 선택
%% ------------------------------------------------------------------------
printSec(1,'GPU / 환경 초기화');
gpuInfo = initGPU();
execEnv = gpuInfo.execEnv;
fprintf('  GPU available : %s\n', string(gpuInfo.available));
if gpuInfo.available
    fprintf('  GPU device    : %s (CC %s, %.2f GB)\n', ...
        gpuInfo.name, gpuInfo.cc, gpuInfo.totalMem/1024^3);
end
fprintf('  Exec Env      : %s\n', execEnv);

[plyPath, rockMatPath, weatherMatPath, outDir] = chooseFilesAndOutput(opts);
fprintf('  PLY           : %s\n', plyPath);
if ~opts.SkipRockType
    fprintf('  RockType MAT  : %s\n', rockMatPath);
end
fprintf('  Weather MAT   : %s\n', weatherMatPath);
fprintf('  Output        : %s\n', outDir);

pngDir = fullfile(outDir,'png');
csvDir = fullfile(outDir,'csv');
mkdirSafe(outDir);
mkdirSafe(pngDir);
mkdirSafe(csvDir);

%% ------------------------------------------------------------------------
% [2] PLY 로드
%% ------------------------------------------------------------------------
printSec(2,'PLY 로드');
pc  = pcread(plyPath);
xyz = double(pc.Location);

if isempty(pc.Color)
    error('PLY:NoColor','PLY에 RGB Color 정보가 없습니다.');
end
rgb = double(pc.Color);

N0 = size(xyz,1);
if opts.MaxPoints > 0 && opts.MaxPoints < N0
    keepIdx = randperm(N0, opts.MaxPoints);
    xyz     = xyz(keepIdx,:);
    rgb     = rgb(keepIdx,:);
else
    keepIdx = (1:N0)';
end
N = size(xyz,1);

[faces, hasFaces] = parsePlyFacesSafe(plyPath);
if hasFaces && numel(keepIdx) ~= N0
    old2new = zeros(N0,1);
    old2new(keepIdx) = 1:N;
    faceKeep = all(ismember(faces, keepIdx), 2);
    faces    = faces(faceKeep,:);
    faces    = [old2new(faces(:,1)), old2new(faces(:,2)), old2new(faces(:,3))];
    hasFaces = ~isempty(faces);
end

fprintf('  Vertex        : %s\n', fmtN(N));
if hasFaces
    fprintf('  Faces         : %s\n', fmtN(size(faces,1)));
else
    fprintf('  Faces         : N/A\n');
end

%% ------------------------------------------------------------------------
% [3] 9채널 색상 분석
%% ------------------------------------------------------------------------
printSec(3,'9채널 색상 분석 (RGB/HSV/CIELAB)');
RGB_raw = uint8(rgb);
R = double(RGB_raw(:,1));
G = double(RGB_raw(:,2));
B = double(RGB_raw(:,3));

rgbU8 = reshape(RGB_raw, [N 1 3]);

if gpuInfo.available
    try
        rgbG   = gpuArray(im2single(rgbU8));
        hsvArr = gather(rgb2hsv(rgbG));
        labArr = gather(rgb2lab(rgbG));
    catch
        hsvArr = rgb2hsv(rgbU8);
        labArr = rgb2lab(rgbU8);
    end
else
    hsvArr = rgb2hsv(rgbU8);
    labArr = rgb2lab(rgbU8);
end

Hch = squeeze(hsvArr(:,1,1)) * 360;
Sch = squeeze(hsvArr(:,1,2));
Vch = squeeze(hsvArr(:,1,3));

Lch = squeeze(labArr(:,1,1));
ach = squeeze(labArr(:,1,2));
bch = squeeze(labArr(:,1,3));

Cch = sqrt(ach.^2 + bch.^2);
hab = mod(atan2d(bch,ach), 360);

normPrc = @(v,p1,p2) min(max((v-prctile(v,p1))./max(eps,prctile(v,p2)-prctile(v,p1)),0),1);
Ln    = normPrc(Lch,5,95);
Bn    = normPrc(bch,5,95);
dEref = sqrt((ach-median(ach,'omitnan')).^2 + (bch-median(bch,'omitnan')).^2);
dEn   = normPrc(dEref,5,95);

WHI = normPrc(0.50*(1-Ln) + 0.30*Bn + 0.20*dEn, 5, 95);
SVI = normPrc(0.55*normPrc(Sch,5,95) + 0.45*(1-normPrc(Vch,5,95)), 5, 95);

ge = opts.GradeEdges;
Dgrade_rule = ones(N,1,'single');
Dgrade_rule(WHI > ge(1)) = 2;
Dgrade_rule(WHI > ge(2)) = 3;
Dgrade_rule(WHI > ge(3)) = 4;
Dgrade_rule(WHI > ge(4)) = 5;

ch9Names = {'R','G','B','H','S','V','Lstar','astar','bstar'};
ch9Data  = {R,G,B,Hch,Sch,Vch,Lch,ach,bch};

ch9Stats = compute9ChStats(ch9Names, ch9Data);

fprintf('  L*a*b* p50    : %.3f / %.3f / %.3f\n', ...
    median(Lch,'omitnan'), median(ach,'omitnan'), median(bch,'omitnan'));
fprintf('  WHI p50       : %.4f\n', median(WHI,'omitnan'));

%% ------------------------------------------------------------------------
% [4] 3D 형상 분석
%% ------------------------------------------------------------------------
printSec(4,'3D 형상 분석');
shapeInfo = analyzeShape3D(xyz, opts);

fprintf('  Dims          : %.5f x %.5f x %.5f\n', ...
    shapeInfo.dims(1), shapeInfo.dims(2), shapeInfo.dims(3));
fprintf('  Vol           : %.5f\n', shapeInfo.vol);
fprintf('  SA/Vol        : %.5f\n', shapeInfo.saVol);
fprintf('  Rough mean    : %.6f\n', shapeInfo.roughnessMean);
fprintf('  Curv mean     : %.6f\n', shapeInfo.curvatureMean);
fprintf('  Sphericity    : %.6f\n', shapeInfo.sphericity);
fprintf('  Elongation    : %.6f\n', shapeInfo.elongation);

%% ------------------------------------------------------------------------
% [5] PLY -> 2D raster 투영
%% ------------------------------------------------------------------------
printSec(5,'PLY -> 2D raster 투영');
proj = rasterizePlyTopView(xyz, RGB_raw, WHI, Dgrade_rule, ...
    opts.GridSize, opts.BackgroundFillRGB);

Irgb        = proj.Irgb;
Iwhi        = proj.Iwhi;
Idrule      = proj.Idrule;
validMask2D = proj.validMask;
vertexRC    = proj.vertexPixelRC;

fprintf('  Raster        : %d x %d\n', size(Irgb,1), size(Irgb,2));
fprintf('  Valid pixels  : %s (%.2f%%)\n', fmtN(nnz(validMask2D)), ...
    100 * nnz(validMask2D) / numel(validMask2D));

%% ------------------------------------------------------------------------
% [6] 2D 텍스처 분석
%% ------------------------------------------------------------------------
printSec(6,'2D raster 텍스처 분석');
texResult = computeTexture2D(Irgb, validMask2D, opts);
fprintf('  gradL p50     : %.4f\n', texResult.gradL_p50);
fprintf('  entropy p50   : %.4f\n', texResult.ent_p50);
fprintf('  LoG p50       : %.4f\n', texResult.log_p50);

%% ------------------------------------------------------------------------
% [7] 암종 MAT 추론
%% ------------------------------------------------------------------------
rockTypeResult = struct( ...
    'enabled', false, ...
    'globalPred', "", ...
    'globalScores', [], ...
    'isGranite', true, ...
    'classNames', strings(0,1), ...
    'patchMap', [], ...
    'confMap', [], ...
    'scoreMap', [], ...
    'summary', table() );

if ~opts.SkipRockType && ~isempty(rockMatPath)
    printSec(7,'암종 판단 MAT 추론');
    rockModel = loadNetworkFromMat(rockMatPath);

    Iglobal = im2single(imresize(Irgb, rockModel.inputSize(1:2)));
    [rtGlobalLabel, rtGlobalScore] = classifyOneImage(rockModel, Iglobal, execEnv);

    rtTile = runTileInferenceMasked(Irgb, validMask2D, rockModel, ...
        opts.TileSize, opts.TileStride, execEnv, opts.MiniBatchSize);

    rockTypeResult.enabled      = true;
    rockTypeResult.globalPred   = rtGlobalLabel;
    rockTypeResult.globalScores = rtGlobalScore(:)';
    rockTypeResult.classNames   = rockModel.classNames;
    rockTypeResult.patchMap     = rtTile.IdMap;
    rockTypeResult.confMap      = rtTile.ConfMap;
    rockTypeResult.scoreMap     = rtTile.ScoreMap;

    gkw = string(opts.GraniteKeywords(:));
    isGranite = false;

    for k = 1:numel(gkw)
        if contains(string(rtGlobalLabel), gkw(k), 'IgnoreCase', true)
            isGranite = true;
            break;
        end
    end

    if ~isGranite
        validIds = rtTile.IdMap(validMask2D & ~isnan(rtTile.IdMap));
        if ~isempty(validIds)
            modeClass = mode(validIds);
            if modeClass >= 1 && modeClass <= numel(rockModel.classNames)
                domName = string(rockModel.classNames(modeClass));
                for k = 1:numel(gkw)
                    if contains(domName, gkw(k), 'IgnoreCase', true)
                        isGranite = true;
                        break;
                    end
                end
            end
        end
    end

    rockTypeResult.isGranite = isGranite;

    nRt = rockModel.nClasses;
    rtCnt = zeros(nRt,1);
    validIds = rtTile.IdMap(validMask2D & ~isnan(rtTile.IdMap));
    for k = 1:nRt
        rtCnt(k) = nnz(round(validIds) == k);
    end
    rtPct = 100 * rtCnt / max(1, sum(rtCnt));
    rockTypeResult.summary = table( ...
        string(rockModel.classNames(:)), rtCnt, rtPct, ...
        'VariableNames', {'RockType','TileCount','TilePercent'} );

    fprintf('  Global rock   : %s\n', rtGlobalLabel);
    fprintf('  Granite       : %s\n', string(isGranite));
    if ~isGranite && ~opts.ForceWeathering
        fprintf('  [WARNING] 화강암이 아닐 가능성 -> 풍화도 해석은 참고용\n');
    end
end

%% ------------------------------------------------------------------------
% [8] 풍화도 MAT 추론
%% ------------------------------------------------------------------------
printSec(8,'풍화도 판단 MAT 추론');
weatherModel = loadNetworkFromMat(weatherMatPath);

wdTile = runTileInferenceMasked(Irgb, validMask2D, weatherModel, ...
    opts.TileSize, opts.TileStride, execEnv, opts.MiniBatchSize);

Id_predTile = wdTile.IdMap;
ScoreTile   = wdTile.ScoreMap;
ConfTile    = wdTile.ConfMap;

nClasses = weatherModel.nClasses;

weights3d = reshape(1:nClasses, [1 1 nClasses]);
numArr    = sum(ScoreTile .* weights3d, 3, 'omitnan');
denArr    = sum(ScoreTile, 3, 'omitnan');
Igrad     = (numArr ./ max(denArr, eps) - 1) ./ max(nClasses-1, 1);
Igrad(~validMask2D) = NaN;

coveragePct = 100 * nnz(validMask2D & ~isnan(Id_predTile)) / max(1, nnz(validMask2D));

allGrades = Id_predTile(validMask2D & ~isnan(Id_predTile));
if isempty(allGrades)
    meanGradeAll = NaN;
    stdGradeAll  = NaN;
    ratioD45     = NaN;
else
    meanGradeAll = mean(allGrades,'omitnan');
    stdGradeAll  = std(allGrades,'omitnan');
    ratioD45     = mean(allGrades >= 4);
end

highConfMask = validMask2D & ~isnan(Id_predTile) & ConfTile >= opts.ConfThreshold;
fracHighConf = nnz(highConfMask) / max(1, nnz(validMask2D));
meanGradeConf = mean(Id_predTile(highConfMask), 'omitnan');

validGrad = Igrad(validMask2D & ~isnan(Igrad));
validConf = ConfTile(validMask2D & ~isnan(Igrad));
if ~isempty(validGrad) && sum(validConf,'omitnan') > 0
    wMeanGrad = sum(double(validGrad).*double(validConf), 'omitnan') / ...
                sum(double(validConf), 'omitnan');
else
    wMeanGrad = NaN;
end

fprintf('  Coverage      : %.2f%%\n', coveragePct);
fprintf('  Mean grade    : %.4f (std=%.4f)\n', meanGradeAll, stdGradeAll);
fprintf('  Weighted mean : %.4f\n', wMeanGrad);
fprintf('  D4-D5 ratio   : %.4f\n', ratioD45);
fprintf('  High-conf     : %.2f%%\n', fracHighConf * 100);

%% ------------------------------------------------------------------------
% [9] 2D -> 3D 역매핑
%% ------------------------------------------------------------------------
printSec(9,'2D -> 3D 역매핑');
vertexTileGrade  = nan(N,1);
vertexGradScore  = nan(N,1);
vertexConfidence = nan(N,1);

for i = 1:N
    rr = vertexRC(i,1);
    cc = vertexRC(i,2);
    if rr >= 1 && rr <= size(Id_predTile,1) && cc >= 1 && cc <= size(Id_predTile,2)
        if validMask2D(rr,cc)
            vertexTileGrade(i)  = Id_predTile(rr,cc);
            vertexGradScore(i)  = Igrad(rr,cc);
            vertexConfidence(i) = ConfTile(rr,cc);
        end
    end
end

if hasFaces
    faceTileGrade = mean([vertexTileGrade(faces(:,1)), vertexTileGrade(faces(:,2)), vertexTileGrade(faces(:,3))], 2, 'omitnan');
    faceGradScore = mean([vertexGradScore(faces(:,1)), vertexGradScore(faces(:,2)), vertexGradScore(faces(:,3))], 2, 'omitnan');
    faceConf      = mean([vertexConfidence(faces(:,1)), vertexConfidence(faces(:,2)), vertexConfidence(faces(:,3))], 2, 'omitnan');
else
    faceTileGrade = [];
    faceGradScore = [];
    faceConf      = [];
end

fprintf('  Vertex mapped : %s / %s\n', fmtN(nnz(~isnan(vertexTileGrade))), fmtN(N));

%% ------------------------------------------------------------------------
% [10] 통계 검정
%% ------------------------------------------------------------------------
printSec(10,'통계 검정');
statTests = runStatisticalTests(vertexTileGrade, Lch, ach, bch, WHI, Cch, opts.StatTestAlpha);

fprintf('  KS L*(D1,D5)  : %.4e\n', statTests.ks_L_pval);
fprintf('  KW WHI        : %.4e\n', statTests.kw_WHI_pval);
fprintf('  Spearman g-WHI: rho=%.4f, p=%.4e\n', ...
    statTests.spearman_grade_WHI_rho, statTests.spearman_grade_WHI_pval);

%% ------------------------------------------------------------------------
% [11] 등급별 9채널 통계
%% ------------------------------------------------------------------------
printSec(11,'등급별 9채널 통계');
gradeStats = computeGradeWise9ChStats(vertexTileGrade, ch9Data, ch9Names, nClasses);
if ~isempty(gradeStats)
    disp(gradeStats);
end

%% ------------------------------------------------------------------------
% [12] 상관 / ΔE
%% ------------------------------------------------------------------------
printSec(12,'상관 / ΔE');
corrMat = compute9ChCorrelation(ch9Data, min(N,300000));

dE76_stats = struct();
dE76_stats.mean = mean(dEref,'omitnan');
dE76_stats.std  = std(dEref,'omitnan');
dE76_stats.p05  = prctile(dEref,5);
dE76_stats.p50  = median(dEref,'omitnan');
dE76_stats.p95  = prctile(dEref,95);

fprintf('  dE76 p50      : %.4f\n', dE76_stats.p50);

%% ------------------------------------------------------------------------
% [13] 시각화
%% ------------------------------------------------------------------------
printSec(13,'시각화');
if opts.SaveFigures
    try
        saveFig01_Raster(Irgb, Iwhi, Idrule, pngDir, opts.FigDPI, opts.VisibleFigures);
    catch ME
        warning(ME.identifier,'%s',ME.message);
    end

    if rockTypeResult.enabled
        try
            saveFig02_RockType(Irgb, rockTypeResult, pngDir, opts.FigDPI, opts.VisibleFigures);
        catch ME
            warning(ME.identifier,'%s',ME.message);
        end
    end

    try
        saveFig03_Weathering2D(Irgb, Id_predTile, ConfTile, Igrad, validMask2D, ...
            weatherModel.classNames, pngDir, opts.FigDPI, opts.VisibleFigures);
    catch ME
        warning(ME.identifier,'%s',ME.message);
    end

    try
        saveFig04_Weathering3D(xyz, RGB_raw, vertexTileGrade, vertexGradScore, ...
            pngDir, opts.FigDPI, opts.VisibleFigures);
    catch ME
        warning(ME.identifier,'%s',ME.message);
    end

    try
        saveFig05_ColorStats(ch9Data, ch9Names, WHI, SVI, dEref, pngDir, opts.FigDPI, opts.VisibleFigures);
    catch ME
        warning(ME.identifier,'%s',ME.message);
    end

    try
        saveFig06_Texture(texResult, validMask2D, pngDir, opts.FigDPI, opts.VisibleFigures);
    catch ME
        warning(ME.identifier,'%s',ME.message);
    end

    try
        saveFig07_Correlation(corrMat, ch9Names, pngDir, opts.FigDPI, opts.VisibleFigures);
    catch ME
        warning(ME.identifier,'%s',ME.message);
    end

    try
        saveFig08_Summary(plyPath, outDir, N, hasFaces, faces, ...
            shapeInfo, meanGradeAll, stdGradeAll, wMeanGrad, ratioD45, ...
            WHI, SVI, Lch, ach, bch, rockTypeResult, weatherModel, statTests, ...
            t0, pngDir, opts.FigDPI, opts.VisibleFigures);
    catch ME
        warning(ME.identifier,'%s',ME.message);
    end
end

%% ------------------------------------------------------------------------
% [14] 저장
%% ------------------------------------------------------------------------
printSec(14,'XLSX / CSV / MAT 저장');

statsTbl = build9ChStatsTable(ch9Stats, WHI, SVI, Cch, hab, dEref, vertexGradScore);

rulePct  = gradePctVec(Dgrade_rule, 5);
modelPct = gradePctVec(vertexTileGrade, nClasses);
maxG     = max(5, nClasses);

summaryTbl = table((1:maxG)', nan(maxG,1), nan(maxG,1), ...
    'VariableNames', {'grade_id','rule_pct','model_pct'});
summaryTbl.rule_pct(1:5)         = rulePct(:);
summaryTbl.model_pct(1:nClasses) = modelPct(:);

scalarTbl = table( ...
    meanGradeAll, stdGradeAll, wMeanGrad, meanGradeConf, ratioD45, ...
    fracHighConf, opts.ConfThreshold, ...
    shapeInfo.roughnessMean, shapeInfo.curvatureMean, ...
    shapeInfo.sphericity, shapeInfo.elongation, ...
    dE76_stats.p50, dE76_stats.mean, ...
    statTests.kw_WHI_pval, statTests.spearman_grade_WHI_rho, ...
    'VariableNames', { ...
    'meanGrade','stdGrade','weightedMeanGrade','meanGrade_conf', ...
    'ratioD45','fracHighConf','confThreshold', ...
    'roughness_mean','curvature_mean','sphericity','elongation', ...
    'dE76_p50','dE76_mean','kw_WHI_pval','spearman_grade_WHI_rho'} );

weatherMeta = rmfield_safe(weatherModel, {'S','model'});
modelTbl = table( ...
    string(weatherMatPath), ...
    string(weatherMeta.modelVarName), ...
    weatherMeta.inputSize(1), weatherMeta.inputSize(2), weatherMeta.inputSize(3), ...
    string(strjoin(cellstr(string(weatherMeta.classNames(:)')),',')), ...
    'VariableNames', {'mat_path','model_var','input_h','input_w','input_c','classes'} );

xlsxPath = '';
if opts.SaveXlsx
    xlsxPath = fullfile(outDir, 'PLY_STAGE_ANALYSIS.xlsx');
    try
        writetable(statsTbl,   xlsxPath, 'Sheet', 'COLOR_STATS');
        writetable(summaryTbl, xlsxPath, 'Sheet', 'GRADE_SUMMARY');
        writetable(scalarTbl,  xlsxPath, 'Sheet', 'SCALAR_SUMMARY');
        writetable(modelTbl,   xlsxPath, 'Sheet', 'MODEL_INFO');
        if ~isempty(gradeStats)
            writetable(gradeStats, xlsxPath, 'Sheet', 'GRADE_9CH_STATS');
        end
        if rockTypeResult.enabled && ~isempty(rockTypeResult.summary)
            writetable(rockTypeResult.summary, xlsxPath, 'Sheet', 'ROCK_TYPE');
        end
        fprintf('  XLSX          : %s\n', xlsxPath);
    catch ME
        warning(ME.identifier,'%s',ME.message);
        xlsxPath = '';
    end
end

if opts.SaveCSV
    try
        writetable(statsTbl,   fullfile(csvDir,'COLOR_STATS.csv'));
        writetable(summaryTbl, fullfile(csvDir,'GRADE_SUMMARY.csv'));
        writetable(scalarTbl,  fullfile(csvDir,'SCALAR_SUMMARY.csv'));
        if ~isempty(gradeStats)
            writetable(gradeStats, fullfile(csvDir,'GRADE_9CH_STATS.csv'));
        end
        if rockTypeResult.enabled && ~isempty(rockTypeResult.summary)
            writetable(rockTypeResult.summary, fullfile(csvDir,'ROCK_TYPE.csv'));
        end
        fprintf('  CSV dir       : %s\n', csvDir);
    catch ME
        warning(ME.identifier,'%s',ME.message);
    end
end

matPath = '';
if opts.SaveMat
    matPath = fullfile(outDir, 'PLY_STAGE_ANALYSIS.mat');
    try
        save(matPath, ...
            'xyz','rgb','faces','hasFaces', ...
            'R','G','B','Hch','Sch','Vch','Lch','ach','bch','Cch','hab', ...
            'WHI','SVI','Dgrade_rule','dEref','dE76_stats', ...
            'proj','Id_predTile','ScoreTile','ConfTile','Igrad', ...
            'vertexTileGrade','vertexGradScore','vertexConfidence', ...
            'faceTileGrade','faceGradScore','faceConf', ...
            'shapeInfo','texResult','statTests','corrMat', ...
            'ch9Stats','ch9Names','gradeStats', ...
            'rockTypeResult','weatherMeta', ...
            'statsTbl','summaryTbl','scalarTbl','modelTbl', ...
            '-v7.3');
        fprintf('  MAT           : %s\n', matPath);
    catch ME
        warning(ME.identifier,'%s',ME.message);
        matPath = '';
    end
end

%% ------------------------------------------------------------------------
% [15] 결과 구조체
%% ------------------------------------------------------------------------
results = struct();
results.file      = plyPath;
results.outputDir = outDir;
results.N         = N;
results.xyz       = xyz;
results.rgb       = RGB_raw;
results.faces     = faces;
results.hasFaces  = hasFaces;

results.color = struct( ...
    'R',R,'G',G,'B',B, ...
    'H',Hch,'S',Sch,'V',Vch, ...
    'L',Lch,'a',ach,'b',bch,'C',Cch,'hab',hab );

results.indices = struct( ...
    'WHI',WHI,'SVI',SVI,'dE76',dEref,'dE76_stats',dE76_stats );

results.rule = struct('Dgrade',Dgrade_rule);

results.raster = struct( ...
    'Irgb',Irgb,'Iwhi',Iwhi,'Idrule',Idrule, ...
    'Id_predTile',Id_predTile,'ScoreTile',ScoreTile,'ConfTile',ConfTile,'Igrad',Igrad, ...
    'validMask',validMask2D,'vertexPixelRC',vertexRC );

results.rockType = rockTypeResult;
results.weatherModel = weatherMeta;

results.model = struct( ...
    'vertexTileGrade',vertexTileGrade, ...
    'vertexGradScore',vertexGradScore, ...
    'vertexConfidence',vertexConfidence, ...
    'faceTileGrade',faceTileGrade, ...
    'faceGradScore',faceGradScore, ...
    'faceConf',faceConf );

results.shape   = shapeInfo;
results.texture = texResult;
results.stats   = struct( ...
    'ch9Stats',ch9Stats, ...
    'gradeStats',gradeStats, ...
    'corrMat',corrMat, ...
    'statTests',statTests );

results.tables = struct( ...
    'statsTbl',statsTbl, ...
    'summaryTbl',summaryTbl, ...
    'scalarTbl',scalarTbl, ...
    'modelTbl',modelTbl );

results.files = struct( ...
    'xlsx',xlsxPath, ...
    'mat',matPath, ...
    'pngDir',pngDir, ...
    'csvDir',csvDir );

fprintf('\n%s\n', SEP);
fprintf(' DONE in %.2f sec\n', toc(t0));
fprintf(' Output : %s\n', outDir);
fprintf('%s\n', SEP);

end % main


%% =========================================================================
% 로컬 함수
%% =========================================================================

function gpuInfo = initGPU()
gpuInfo = struct('available',false,'execEnv','cpu','name','','cc','','totalMem',0);
try
    setenv("MW_CUDA_FORWARD_COMPATIBILITY","1");
    if exist("parallel.gpu.enableCUDAForwardCompatibility","file") == 2
        parallel.gpu.enableCUDAForwardCompatibility(true);
    end
catch
end
try
    if canUseGPU && gpuDeviceCount("available") > 0
        g = gpuDevice(1);
        gpuInfo.available = true;
        gpuInfo.execEnv   = 'gpu';
        gpuInfo.name      = g.Name;
        gpuInfo.cc        = string(g.ComputeCapability);
        gpuInfo.totalMem  = g.TotalMemory;

        A = gpuArray.rand(256,256,'single');
        B = A * A; %#ok<NASGU>
        wait(g);
        clear A B;
    end
catch
    gpuInfo.available = false;
    gpuInfo.execEnv   = 'cpu';
end
end

function [plyPath, rockMatPath, weatherMatPath, outDir] = chooseFilesAndOutput(opts)
if isempty(char(opts.PLY))
    [fn,fd] = uigetfile({'*.ply','PLY';'*.*','All Files'}, 'PLY 파일 선택');
    if isequal(fn,0)
        error('Input:Canceled','PLY 선택이 취소되었습니다.');
    end
    plyPath = fullfile(fd,fn);
else
    plyPath = char(opts.PLY);
end

if ~opts.SkipRockType
    if isempty(char(opts.RockTypeMAT))
        [fn,fd] = uigetfile({'*.mat','MAT';'*.*','All Files'}, '[Stage 1] 암종 MAT 선택');
        if isequal(fn,0)
            rockMatPath = '';
        else
            rockMatPath = fullfile(fd,fn);
        end
    else
        rockMatPath = char(opts.RockTypeMAT);
    end
else
    rockMatPath = '';
end

if isempty(char(opts.WeatherMAT))
    [fn,fd] = uigetfile({'*.mat','MAT';'*.*','All Files'}, '[Stage 2] 풍화도 MAT 선택');
    if isequal(fn,0)
        error('Input:Canceled','풍화도 MAT 선택이 취소되었습니다.');
    end
    weatherMatPath = fullfile(fd,fn);
else
    weatherMatPath = char(opts.WeatherMAT);
end

if isempty(char(opts.OutputDir))
    [baseDir, stem, ~] = fileparts(plyPath);
    runTag = char(datetime('now','Format','yyyyMMdd_HHmmss'));
    outDir = fullfile(baseDir, [stem '_PLY_STAGE_' runTag]);
else
    outDir = char(opts.OutputDir);
end
end

function modelInfo = loadNetworkFromMat(matPath)
S  = load(matPath);
fn = fieldnames(S);

mdl = [];
modelVarName = '';

cand = {'net','trainedNet','netTrained','dlnet','model','bestNet','netFinal'};
for i = 1:numel(cand)
    if isfield(S, cand{i})
        mdl = S.(cand{i});
        modelVarName = cand{i};
        break;
    end
end

if isempty(mdl)
    for i = 1:numel(fn)
        v = S.(fn{i});
        if isa(v,'SeriesNetwork') || isa(v,'DAGNetwork') || isa(v,'dlnetwork')
            mdl = v;
            modelVarName = fn{i};
            break;
        end
    end
end

if isempty(mdl)
    error('Model:NotFound','네트워크를 MAT에서 찾지 못했습니다: %s', matPath);
end

mdl = fixBatchNormStats(mdl);

inputSize = [224 224 3];
if isfield(S,'inputSize') && isnumeric(S.inputSize) && numel(S.inputSize) >= 2
    tmp = double(S.inputSize(:)');
    inputSize(1:min(3,numel(tmp))) = tmp(1:min(3,numel(tmp)));
elseif isfield(S,'imageSize') && isnumeric(S.imageSize) && numel(S.imageSize) >= 2
    tmp = double(S.imageSize(:)');
    inputSize(1:min(3,numel(tmp))) = tmp(1:min(3,numel(tmp)));
else
    inputSize = getInputSizeFromNet(mdl, inputSize);
end
inputSize(3) = max(inputSize(3),1);

classNames = strings(0,1);
labelCand = {'classNames','classes','class_names','labels'};
for i = 1:numel(labelCand)
    if isfield(S, labelCand{i})
        classNames = string(S.(labelCand{i}));
        classNames = classNames(:);
        break;
    end
end
if isempty(classNames)
    classNames = getClassNamesFromNet(mdl);
end
if isempty(classNames)
    classNames = ["D1";"D2";"D3";"D4";"D5"];
end

mu = [];
sig = [];
if isfield(S,'mu'),   mu = S.mu; end
if isfield(S,'sig'),  sig = S.sig; end
if isempty(mu) && isfield(S,'mean'), mu = S.mean; end
if isempty(sig) && isfield(S,'std'), sig = S.std; end

modelInfo = struct( ...
    'S',S, ...
    'model',mdl, ...
    'modelVarName',modelVarName, ...
    'inputSize',double(inputSize), ...
    'classNames',classNames(:), ...
    'nClasses',numel(classNames), ...
    'mu',mu, ...
    'sig',sig );
end

function mdl = fixBatchNormStats(mdl)
if isa(mdl,'dlnetwork')
    try
        lg  = layerGraph(mdl);
        mdl = assembleNetwork(lg);
    catch
    end
end

if isa(mdl,'DAGNetwork')
    try
        lg = layerGraph(mdl);
        layers = lg.Layers;
        for i = 1:numel(layers)
            if isa(layers(i),'nnet.cnn.layer.BatchNormalizationLayer')
                bn = layers(i);
                changed = false;
                if isempty(bn.TrainedMean)
                    bn.TrainedMean = zeros(size(bn.Offset),'like',bn.Offset);
                    changed = true;
                end
                if isempty(bn.TrainedVariance)
                    bn.TrainedVariance = ones(size(bn.Offset),'like',bn.Offset);
                    changed = true;
                end
                if changed
                    lg = replaceLayer(lg, bn.Name, bn);
                end
            end
        end
        mdl = assembleNetwork(lg);
    catch
    end
end
end

function inputSize = getInputSizeFromNet(mdl, defaultSize)
inputSize = defaultSize;
try
    if isprop(mdl,'Layers') && ~isempty(mdl.Layers)
        for i = 1:numel(mdl.Layers)
            if isa(mdl.Layers(i),'nnet.cnn.layer.ImageInputLayer')
                inputSize = double(mdl.Layers(i).InputSize);
                return;
            end
        end
    end
catch
end
end

function classNames = getClassNamesFromNet(net)
classNames = strings(0,1);
try
    for i = numel(net.Layers):-1:1
        if isprop(net.Layers(i),'Classes') && ~isempty(net.Layers(i).Classes)
            classNames = string(net.Layers(i).Classes);
            classNames = classNames(:);
            return;
        end
    end
catch
end
end

function img = applyNormalization(img, modelInfo)
if ~isempty(modelInfo.mu) && ~isempty(modelInfo.sig)
    try
        mu = reshape(single(modelInfo.mu),1,1,[]);
        sg = reshape(single(modelInfo.sig),1,1,[]);
        sg(sg == 0) = 1;
        if size(mu,3) == size(img,3)
            img = (img - mu) ./ sg;
        end
    catch
    end
end
end

function sv = ensureProbVector(sv, nClasses)
sv = double(sv(:)');
if isempty(sv)
    sv = ones(1,nClasses) / max(nClasses,1);
    return;
end

if numel(sv) > nClasses
    sv = sv(1:nClasses);
elseif numel(sv) < nClasses
    tmp = zeros(1,nClasses);
    tmp(1:numel(sv)) = sv;
    sv = tmp;
end

if any(~isfinite(sv))
    sv = ones(1,nClasses) / max(nClasses,1);
    return;
end

if any(sv < 0) || abs(sum(sv) - 1) > 1e-3
    sv = sv - max(sv);
    sv = exp(sv);
end

ss = sum(sv);
if ss <= 0 || ~isfinite(ss)
    sv = ones(1,nClasses) / max(nClasses,1);
else
    sv = sv / ss;
end
end

function [label, scores] = classifyOneImage(modelInfo, img, execEnv)
img = im2single(imresize(img, modelInfo.inputSize(1:2)));

if size(img,3) == 1 && modelInfo.inputSize(3) == 3
    img = repmat(img,1,1,3);
elseif size(img,3) == 3 && modelInfo.inputSize(3) == 1
    img = rgb2gray(img);
end

img = applyNormalization(img, modelInfo);
mdl = modelInfo.model;

try
    if isa(mdl,'SeriesNetwork') || isa(mdl,'DAGNetwork')
        [Y,S]  = classify(mdl, img, 'ExecutionEnvironment', execEnv);
        scores = ensureProbVector(double(squeeze(S)), modelInfo.nClasses);
        if ~isempty(Y) && ~strcmp(string(Y),'')
            label = string(Y);
        else
            [~,idx] = max(scores);
            label = string(modelInfo.classNames(idx));
        end
    elseif isa(mdl,'dlnetwork')
        dlX = dlarray(single(img),'SSC');
        if strcmp(execEnv,'gpu')
            dlX = gpuArray(dlX);
        end
        dlY    = predict(mdl, dlX);
        raw    = gather(extractdata(dlY));
        scores = ensureProbVector(raw, modelInfo.nClasses);
        [~,idx] = max(scores);
        label = string(modelInfo.classNames(idx));
    else
        scores = ones(1,modelInfo.nClasses) / modelInfo.nClasses;
        label  = "Unknown";
    end
catch
    scores = ones(1,modelInfo.nClasses) / modelInfo.nClasses;
    label  = "Unknown";
end
end

function tileRes = runTileInferenceMasked(Irgb, validMask, modelInfo, tileSize, stride, execEnv, miniBatch)
H  = size(Irgb,1);
W  = size(Irgb,2);
nC = modelInfo.nClasses;

IdMap    = nan(H,W);
ConfMap  = nan(H,W);
ScoreMap = nan(H,W,nC);
CountMap = zeros(H,W,'single');

rows = unique([1:stride:max(1,H-tileSize+1), max(1,H-tileSize+1)]);
cols = unique([1:stride:max(1,W-tileSize+1), max(1,W-tileSize+1)]);

tileList = struct('r',{},'c',{},'m',{});
nTiles   = 0;

for r = rows
    for c = cols
        rr = r:min(r+tileSize-1,H);
        cc = c:min(c+tileSize-1,W);
        subMask = validMask(rr,cc);
        if nnz(subMask) == 0
            continue;
        end
        nTiles = nTiles + 1;
        tileList(nTiles).r = rr;
        tileList(nTiles).c = cc;
        tileList(nTiles).m = subMask;
    end
end

fprintf('  Tile count    : %d\n', nTiles);

netH  = modelInfo.inputSize(1);
netW  = modelInfo.inputSize(2);
netCh = modelInfo.inputSize(3);

b0 = 1;
while b0 <= nTiles
    b1 = min(nTiles, b0 + miniBatch - 1);
    nb = b1 - b0 + 1;

    I4 = zeros(netH, netW, netCh, nb, 'single');

    for bi = 1:nb
        ii = b0 + bi - 1;
        rr = tileList(ii).r;
        cc = tileList(ii).c;

        tile = Irgb(rr,cc,:);
        if size(tile,1) ~= tileSize || size(tile,2) ~= tileSize
            tile = padTileToSize(tile, tileSize, tileSize, uint8(255));
        end

        img = im2single(imresize(tile, [netH netW]));
        if size(img,3) == 1 && netCh == 3
            img = repmat(img,1,1,3);
        elseif size(img,3) == 3 && netCh == 1
            img = rgb2gray(img);
        end
        img = applyNormalization(img, modelInfo);
        I4(:,:,:,bi) = img;
    end

    scoresBatch = zeros(nb, nC, 'single');
    mdl = modelInfo.model;

    try
        if isa(mdl,'SeriesNetwork') || isa(mdl,'DAGNetwork')
            [~,Sb] = classify(mdl, I4, ...
                'MiniBatchSize', nb, ...
                'ExecutionEnvironment', execEnv);
            Sb = double(Sb);
            for bi = 1:nb
                scoresBatch(bi,:) = single(ensureProbVector(Sb(bi,:), nC));
            end
        elseif isa(mdl,'dlnetwork')
            try
                dlX = dlarray(single(I4),'SSCB');
                if strcmp(execEnv,'gpu')
                    dlX = gpuArray(dlX);
                end
                dlY = predict(mdl, dlX);
                raw = gather(extractdata(dlY));
                batchScores = reshapeDLBatchScores(raw, nC, nb);
                for bi = 1:nb
                    scoresBatch(bi,:) = single(ensureProbVector(batchScores(bi,:), nC));
                end
            catch
                for bi = 1:nb
                    dlX = dlarray(single(I4(:,:,:,bi)),'SSC');
                    if strcmp(execEnv,'gpu')
                        dlX = gpuArray(dlX);
                    end
                    dlY = predict(mdl, dlX);
                    raw = gather(extractdata(dlY));
                    scoresBatch(bi,:) = single(ensureProbVector(raw, nC));
                end
            end
        else
            scoresBatch = ones(nb, nC, 'single') / nC;
        end
    catch
        scoresBatch = ones(nb, nC, 'single') / nC;
    end

    for bi = 1:nb
        ii = b0 + bi - 1;
        rr = tileList(ii).r;
        cc = tileList(ii).c;
        mk = tileList(ii).m;

        sv   = double(scoresBatch(bi,:));
        sv   = ensureProbVector(sv, nC);
        conf = max(sv);

        cntSub = CountMap(rr,cc);
        for k = 1:nC
            old = ScoreMap(rr,cc,k);
            newv = old;

            targetMask = mk;
            oldLocal = old;

            fillMask = targetMask & isnan(oldLocal);
            newv(fillMask) = sv(k);

            updMask = targetMask & ~isnan(oldLocal);
            if any(updMask(:))
                newv(updMask) = (oldLocal(updMask).*cntSub(updMask) + sv(k)) ./ (cntSub(updMask) + 1);
            end
            ScoreMap(rr,cc,k) = newv;
        end

        oldC = ConfMap(rr,cc);
        newC = oldC;

        fillMask = mk & isnan(oldC);
        newC(fillMask) = conf;

        updMask = mk & ~isnan(oldC);
        if any(updMask(:))
            newC(updMask) = (oldC(updMask).*cntSub(updMask) + conf) ./ (cntSub(updMask) + 1);
        end
        ConfMap(rr,cc)  = newC;
        CountMap(rr,cc) = cntSub + single(mk);
    end

    b0 = b1 + 1;
end

for i = 1:H
    for j = 1:W
        if ~validMask(i,j)
            continue;
        end
        sv = squeeze(ScoreMap(i,j,:));
        if all(isnan(sv))
            continue;
        end
        [~,mx] = max(sv);
        IdMap(i,j) = mx;
    end
end

for k = 1:nC
    tmp = ScoreMap(:,:,k);
    tmp(~validMask) = NaN;
    ScoreMap(:,:,k) = tmp;
end
ConfMap(~validMask)  = NaN;
IdMap(~validMask)    = NaN;
CountMap(~validMask) = 0;

tileRes = struct('IdMap',IdMap,'ScoreMap',ScoreMap,'ConfMap',ConfMap,'CountMap',CountMap);
end

function batchScores = reshapeDLBatchScores(raw, nC, nb)
batchScores = zeros(nb, nC);

if isempty(raw)
    batchScores(:,:) = 1 / max(nC,1);
    return;
end

sz = size(raw);

if isvector(raw)
    batchScores(1,:) = ensureProbVector(raw, nC);
    return;
end

if numel(sz) == 2
    if sz(1) == nb
        for i = 1:nb
            batchScores(i,:) = ensureProbVector(raw(i,:), nC);
        end
        return;
    elseif sz(2) == nb
        for i = 1:nb
            batchScores(i,:) = ensureProbVector(raw(:,i), nC);
        end
        return;
    end
end

try
    tmp = squeeze(raw);
    sz2 = size(tmp);
    if isvector(tmp)
        batchScores(1,:) = ensureProbVector(tmp, nC);
        return;
    end
    if numel(sz2) == 2
        if sz2(1) == nb
            for i = 1:nb
                batchScores(i,:) = ensureProbVector(tmp(i,:), nC);
            end
            return;
        elseif sz2(2) == nb
            for i = 1:nb
                batchScores(i,:) = ensureProbVector(tmp(:,i), nC);
            end
            return;
        end
    end
catch
end

batchScores(:,:) = 1 / max(nC,1);
end

function proj = rasterizePlyTopView(xyz, RGB_raw, WHI, Dgrade, gridSize, bgRGB)
minXY = min(xyz(:,1:2),[],1);
maxXY = max(xyz(:,1:2),[],1);
spanXY = max(maxXY - minXY, eps);

col = min(gridSize, max(1, round((xyz(:,1)-minXY(1))./spanXY(1)*(gridSize-1))+1));
row = min(gridSize, max(1, round((1-(xyz(:,2)-minXY(2))./spanXY(2))*(gridSize-1))+1));

Irgb = zeros(gridSize,gridSize,3,'uint8');
Irgb(:,:,1) = uint8(bgRGB(1));
Irgb(:,:,2) = uint8(bgRGB(2));
Irgb(:,:,3) = uint8(bgRGB(3));

Iwhi      = nan(gridSize,gridSize);
Idrule    = nan(gridSize,gridSize);
validMask = false(gridSize,gridSize);

z = xyz(:,3);

lin = sub2ind([gridSize gridSize], row, col);
[~,~,grp] = unique(lin);
uLin = unique(lin);

pickIdx = zeros(numel(uLin),1);
for i = 1:numel(uLin)
    ids = find(grp == i);
    [~,mx] = max(z(ids));
    pickIdx(i) = ids(mx);
end

rr = row(pickIdx);
cc = col(pickIdx);

for i = 1:numel(pickIdx)
    Irgb(rr(i),cc(i),:) = RGB_raw(pickIdx(i),:);
    Iwhi(rr(i),cc(i))   = WHI(pickIdx(i));
    Idrule(rr(i),cc(i)) = Dgrade(pickIdx(i));
    validMask(rr(i),cc(i)) = true;
end

proj = struct( ...
    'Irgb',Irgb, ...
    'Iwhi',Iwhi, ...
    'Idrule',Idrule, ...
    'validMask',validMask, ...
    'vertexPixelRC',[row col], ...
    'minXY',minXY, ...
    'maxXY',maxXY );
end

function shapeInfo = analyzeShape3D(xyz, opts)
dims = max(xyz,[],1) - min(xyz,[],1);

vol   = NaN;
saHull = NaN;
saVol  = NaN;

try
    [K, vol] = convhull(xyz);
    saHull = 0;
    for i = 1:size(K,1)
        e1 = xyz(K(i,2),:) - xyz(K(i,1),:);
        e2 = xyz(K(i,3),:) - xyz(K(i,1),:);
        saHull = saHull + 0.5 * norm(cross(e1,e2));
    end
    saVol = saHull / max(vol, eps);
catch
end

xyzC = xyz - mean(xyz,1,'omitnan');
[pcaVecs, pcaVals] = eig(cov(xyzC,'omitrows'));
eigVals = diag(pcaVals);
eigVals = sort(eigVals,'descend');
eigRatio = eigVals / max(sum(eigVals), eps);

if numel(eigVals) >= 3
    sphericity = eigVals(3) / max(eigVals(1), eps);
    elongation = 1 - eigVals(2) / max(eigVals(1), eps);
else
    sphericity = NaN;
    elongation = NaN;
end

N = size(xyz,1);
nRough = min(N, opts.RoughnessPoints);

roughness = nan(nRough,1);
curvature = nan(nRough,1);

try
    idx = randperm(N, nRough);
    xyzS = xyz(idx,:);
    kdt = KDTreeSearcher(xyz);

    for i = 1:nRough
        [id,~] = knnsearch(kdt, xyzS(i,:), 'K', opts.KnnNeighbors+1);
        id = id(2:end);
        nb = xyz(id,:);
        nb0 = nb - mean(nb,1,'omitnan');

        C = cov(nb0,'omitrows');
        [V,D] = eig(C);
        d = diag(D);
        [dSort, ord] = sort(d, 'ascend');
        nrm = V(:,ord(1));

        roughness(i) = std(abs(nb0*nrm), 'omitnan');
        curvature(i) = dSort(1) / max(sum(dSort), eps);
    end
catch
end

shapeInfo = struct( ...
    'dims',dims, ...
    'vol',vol, ...
    'saHull',saHull, ...
    'saVol',saVol, ...
    'pcaVecs',pcaVecs, ...
    'pcaEigenRatio',eigRatio(:)', ...
    'roughness',roughness, ...
    'curvature',curvature, ...
    'roughnessMean',mean(roughness,'omitnan'), ...
    'roughnessStd',std(roughness,'omitnan'), ...
    'curvatureMean',mean(curvature,'omitnan'), ...
    'curvatureStd',std(curvature,'omitnan'), ...
    'sphericity',sphericity, ...
    'elongation',elongation );
end

function tex = computeTexture2D(Irgb, validMask, opts)
Lmap = im2single(rgb2gray(Irgb));

try
    gradL = single(imgradient(Lmap,'Sobel'));
catch
    dx = [diff(Lmap,1,2) zeros(size(Lmap,1),1,'single')];
    dy = [diff(Lmap,1,1); zeros(1,size(Lmap,2),'single')];
    gradL = hypot(dx,dy);
end
gradL(~validMask) = NaN;
gradLn = normPrcMap(gradL, validMask, 5, 95);

try
    ent = single(entropyfilt(uint8(255*min(max(Lmap,0),1)), true(9)));
catch
    ent = zeros(size(Lmap),'single');
end
ent(~validMask) = NaN;
entN = normPrcMap(ent, validMask, 5, 95);

try
    sm  = imgaussfilt(Lmap, opts.LoGSigma);
    logL = abs(del2(sm));
catch
    logL = zeros(size(Lmap),'single');
end
logL(~validMask) = NaN;
logN = normPrcMap(logL, validMask, 5, 95);

msStd = struct();
msRng = struct();

for ww = opts.TextureWins
    w = round(double(ww));
    if w < 3, w = 3; end
    if mod(w,2) == 0, w = w + 1; end
    key = sprintf('w%d',w);
    try
        tmp1 = single(stdfilt(Lmap, true(w)));
        tmp2 = single(rangefilt(Lmap, true(w)));
    catch
        tmp1 = zeros(size(Lmap),'single');
        tmp2 = zeros(size(Lmap),'single');
    end
    tmp1(~validMask) = NaN;
    tmp2(~validMask) = NaN;
    msStd.(key) = normPrcMap(tmp1, validMask, 5, 95);
    msRng.(key) = normPrcMap(tmp2, validMask, 5, 95);
end

tex = struct( ...
    'gradL',gradLn, ...
    'entropy',entN, ...
    'logL',logN, ...
    'msStd',msStd, ...
    'msRng',msRng, ...
    'gradL_p50',median(gradLn(validMask),'omitnan'), ...
    'ent_p50',median(entN(validMask),'omitnan'), ...
    'log_p50',median(logN(validMask),'omitnan') );
end

function Xn = normPrcMap(X, mask, p1, p2)
Xn = zeros(size(X),'single');
vals = X(mask & isfinite(X));
if numel(vals) < 10
    Xn(~mask) = NaN;
    return;
end
lo = prctile(vals,p1);
hi = prctile(vals,p2);
tmp = min(max((X-lo)./max(hi-lo,eps), 0), 1);
Xn = single(tmp);
Xn(~mask) = NaN;
end

function ch9Stats = compute9ChStats(names, data)
ch9Stats = struct();
for i = 1:numel(names)
    v = data{i};
    v = v(isfinite(v));
    s = struct();
    s.mean     = mean(v,'omitnan');
    s.std      = std(v,'omitnan');
    s.p05      = prctile(v,5);
    s.p25      = prctile(v,25);
    s.p50      = median(v,'omitnan');
    s.p75      = prctile(v,75);
    s.p95      = prctile(v,95);
    s.ipr      = s.p95 - s.p05;
    s.skewness = skewness(v);
    s.kurtosis = kurtosis(v);
    s.min      = min(v);
    s.max      = max(v);
    ch9Stats.(matlab.lang.makeValidName(names{i})) = s;
end
end

function T = computeGradeWise9ChStats(grades, ch9Data, ch9Names, nClasses)
rows = {};
for g = 1:nClasses
    mask = round(grades) == g;
    if nnz(mask) < 5
        continue;
    end
    row = table();
    row.grade = g;
    row.count = nnz(mask);

    for c = 1:numel(ch9Names)
        nm = matlab.lang.makeValidName(ch9Names{c});
        v = ch9Data{c}(mask);
        v = v(isfinite(v));
        row.([nm '_mean']) = mean(v,'omitnan');
        row.([nm '_std'])  = std(v,'omitnan');
        row.([nm '_p05'])  = prctile(v,5);
        row.([nm '_p50'])  = median(v,'omitnan');
        row.([nm '_p95'])  = prctile(v,95);
    end
    rows{end+1} = row; %#ok<AGROW>
end

if isempty(rows)
    T = table();
else
    T = vertcat(rows{:});
end
end

function corrMat = compute9ChCorrelation(ch9Data, maxSamp)
nCh = numel(ch9Data);
n = numel(ch9Data{1});
if n > maxSamp
    idx = randperm(n, maxSamp);
else
    idx = 1:n;
end
M = zeros(numel(idx), nCh);
for i = 1:nCh
    M(:,i) = ch9Data{i}(idx);
end
corrMat = corrcoef(M,'Rows','pairwise');
end

function tests = runStatisticalTests(grades, L, ~, ~, WHI, ~, alpha)
tests = struct();
tests.alpha = alpha;

valid = ~isnan(grades);
g  = round(grades(valid));
Lv = L(valid);
Wv = WHI(valid);

m1 = g == 1;
m5 = g == 5;

if nnz(m1) > 5 && nnz(m5) > 5
    try
        [~,p] = kstest2(Lv(m1), Lv(m5));
        tests.ks_L_pval = p;
    catch
        tests.ks_L_pval = NaN;
    end
else
    tests.ks_L_pval = NaN;
end

try
    [p,~,tbl] = kruskalwallis(Wv, g, 'off');
    tests.kw_WHI_pval = p;
    tests.kw_WHI_chi2 = tbl{2,5};
catch
    tests.kw_WHI_pval = NaN;
    tests.kw_WHI_chi2 = NaN;
end

try
    [rho,pv] = corr(double(g), double(Wv), 'Type', 'Spearman', 'Rows', 'complete');
    tests.spearman_grade_WHI_rho  = rho;
    tests.spearman_grade_WHI_pval = pv;
catch
    tests.spearman_grade_WHI_rho  = NaN;
    tests.spearman_grade_WHI_pval = NaN;
end

try
    [p,~,~] = anova1(Lv, g, 'off');
    tests.anova_L_pval = p;
catch
    tests.anova_L_pval = NaN;
end
end

function T = build9ChStatsTable(ch9Stats, WHI, SVI, ~, ~, dE, gradScore)
fn = fieldnames(ch9Stats);
extNames = {'WHI','SVI','dE76','GradientScore'};
extData  = {WHI,SVI,dE,gradScore};

nRow = numel(fn) + numel(extNames);
param = cell(nRow,1);
mu  = nan(nRow,1);
sd  = nan(nRow,1);
p05 = nan(nRow,1);
p50 = nan(nRow,1);
p95 = nan(nRow,1);
ipr = nan(nRow,1);
sk  = nan(nRow,1);
ku  = nan(nRow,1);

for i = 1:numel(fn)
    s = ch9Stats.(fn{i});
    param{i} = fn{i};
    mu(i)  = s.mean;
    sd(i)  = s.std;
    p05(i) = s.p05;
    p50(i) = s.p50;
    p95(i) = s.p95;
    ipr(i) = s.ipr;
    sk(i)  = s.skewness;
    ku(i)  = s.kurtosis;
end

for j = 1:numel(extNames)
    i = numel(fn) + j;
    v = extData{j};
    v = v(isfinite(v));
    param{i} = extNames{j};
    mu(i)  = mean(v,'omitnan');
    sd(i)  = std(v,'omitnan');
    p05(i) = prctile(v,5);
    p50(i) = median(v,'omitnan');
    p95(i) = prctile(v,95);
    ipr(i) = p95(i) - p05(i);
    sk(i)  = skewness(v);
    ku(i)  = kurtosis(v);
end

T = table(param, mu, sd, p05, p50, p95, ipr, sk, ku, ...
    'VariableNames', {'param','mean','std','p05','p50','p95','ipr','skewness','kurtosis'});
end

function out = overlayScalarOnRGB(Irgb, scalarMap, validMask)
base = im2double(Irgb);
out  = base;

S = scalarMap;
S(~validMask) = NaN;
vv = S(isfinite(S));
if isempty(vv)
    out = im2uint8(out);
    return;
end

cm  = turbo(256);
Sn  = min(max((S-min(vv))./max(max(vv)-min(vv),eps),0),1);
idx = max(1, min(256, round(Sn*255)+1));
alpha = 0.45;

for ch = 1:3
    tmp = out(:,:,ch);
    cch = reshape(cm(idx,ch), size(S));
    m = validMask & isfinite(S);
    tmp(m) = (1-alpha)*tmp(m) + alpha*cch(m);
    out(:,:,ch) = tmp;
end
out = im2uint8(out);
end

function out = overlayGradeOnRGB(Irgb, gradeMap, validMask, nClasses)
base = im2double(Irgb);
out  = base;
cm   = jet(max(nClasses,5));
alpha = 0.45;

for ch = 1:3
    tmp = out(:,:,ch);
    cch = nan(size(gradeMap));
    for k = 1:nClasses
        cch(round(gradeMap)==k) = cm(k,ch);
    end
    m = validMask & isfinite(cch);
    tmp(m) = (1-alpha)*tmp(m) + alpha*cch(m);
    out(:,:,ch) = tmp;
end
out = im2uint8(out);
end

function saveFig01_Raster(Irgb, Iwhi, Idrule, pngDir, dpi, visibleMode)
fig = figure('Color','w','Position',[50 50 1800 550],'Visible',char(visibleMode));

subplot(1,3,1);
imshow(Irgb);
title('PLY Raster RGB','FontWeight','bold');

subplot(1,3,2);
imagesc(Iwhi); axis image off; colorbar;
colormap(gca,'turbo');
title('WHI Map','FontWeight','bold');

subplot(1,3,3);
imagesc(Idrule,[1 5]); axis image off; colorbar;
colormap(gca,'jet');
title('Rule-based D-grade','FontWeight','bold');

saveFigSafe(fig, fullfile(pngDir,'01_raster.png'), dpi);
end

function saveFig02_RockType(Irgb, rtRes, pngDir, dpi, visibleMode)
fig = figure('Color','w','Position',[60 60 2000 550],'Visible',char(visibleMode));
nC = numel(rtRes.classNames);

subplot(1,4,1);
imshow(Irgb);
title('Raster RGB','FontWeight','bold');

subplot(1,4,2);
imagesc(rtRes.patchMap,[1 nC]); axis image off;
cb = colorbar;
cb.Ticks = 1:nC;
cb.TickLabels = cellstr(string(rtRes.classNames(:)'));
colormap(gca, lines(max(nC,2)));
title('Rock Type Tile Map','FontWeight','bold');

subplot(1,4,3);
imagesc(rtRes.confMap,[0 1]); axis image off; colorbar;
colormap(gca,'turbo');
title('Rock Type Confidence','FontWeight','bold');

subplot(1,4,4);
scores = rtRes.globalScores;
if isempty(scores)
    scores = zeros(1,nC);
end
bar(1:nC, 100*scores, 'FaceColor',[0.2 0.6 0.9]);
set(gca,'XTick',1:nC,'XTickLabel',cellstr(string(rtRes.classNames(:)')),'XTickLabelRotation',30);
ylabel('Probability (%)');
ylim([0 100]);
grid on;
title(sprintf('Global: %s', rtRes.globalPred),'FontWeight','bold');

saveFigSafe(fig, fullfile(pngDir,'02_rock_type.png'), dpi);
end

function saveFig03_Weathering2D(Irgb, Id_predTile, ConfTile, Igrad, validMask, classNames, pngDir, dpi, visibleMode)
fig = figure('Color','w','Position',[70 70 2000 900],'Visible',char(visibleMode));
nC = numel(classNames);

subplot(2,3,1);
imshow(Irgb);
title('Raster RGB','FontWeight','bold');

subplot(2,3,2);
imagesc(Id_predTile,[1 nC]); axis image off;
cb = colorbar;
cb.Ticks = 1:nC;
cb.TickLabels = cellstr(string(classNames(:)'));
colormap(gca,'jet');
title('Tile Grade Map','FontWeight','bold');

subplot(2,3,3);
imagesc(ConfTile,[0 1]); axis image off; colorbar;
colormap(gca,'turbo');
title('Confidence Map','FontWeight','bold');

subplot(2,3,4);
imagesc(Igrad); axis image off; colorbar;
colormap(gca,'turbo');
title('Gradient Score','FontWeight','bold');

subplot(2,3,5);
imshow(overlayGradeOnRGB(Irgb, Id_predTile, validMask, nC));
title('Grade Overlay','FontWeight','bold');

subplot(2,3,6);
imshow(overlayScalarOnRGB(Irgb, Igrad, validMask));
title('Gradient Overlay','FontWeight','bold');

saveFigSafe(fig, fullfile(pngDir,'03_weathering_2D.png'), dpi);
end

function saveFig04_Weathering3D(xyz, RGB_raw, vGrade, vGrad, pngDir, dpi, visibleMode)
fig = figure('Color','w','Position',[80 80 1800 900],'Visible',char(visibleMode));

nPlot = min(size(xyz,1), 250000);
idx = randperm(size(xyz,1), nPlot);

subplot(2,2,1);
pcshow(pointCloud(xyz(idx,:),'Color',RGB_raw(idx,:)),'MarkerSize',8);
title('Original 3D RGB','FontWeight','bold');
view(135,25);

subplot(2,2,2);
scatter3(xyz(idx,1), xyz(idx,2), xyz(idx,3), 2, vGrade(idx), 'filled');
colorbar; colormap(gca,'jet');
title('3D Vertex Grade','FontWeight','bold');
view(135,25);

subplot(2,2,3);
scatter3(xyz(idx,1), xyz(idx,2), xyz(idx,3), 2, vGrad(idx), 'filled');
colorbar; colormap(gca,'turbo');
title('3D Vertex Gradient','FontWeight','bold');
view(135,25);

subplot(2,2,4);
histogram(vGrade(~isnan(vGrade)),'BinMethod','integers','FaceAlpha',0.75);
xlabel('Grade'); ylabel('Count');
title('Vertex Grade Histogram','FontWeight','bold');
grid on;

saveFigSafe(fig, fullfile(pngDir,'04_weathering_3D.png'), dpi);
end

function saveFig05_ColorStats(ch9Data, ch9Names, WHI, SVI, dE, pngDir, dpi, visibleMode)
fig = figure('Color','w','Position',[90 90 2000 1200],'Visible',char(visibleMode));

for i = 1:9
    subplot(4,3,i);
    histogram(ch9Data{i}, 120, 'Normalization','pdf','FaceAlpha',0.7);
    hold on;
    xline(median(ch9Data{i},'omitnan'),'r--','LineWidth',1.2);
    title(ch9Names{i},'FontWeight','bold');
    grid on;
end

subplot(4,3,10);
histogram(WHI,120,'Normalization','pdf','FaceAlpha',0.7);
hold on; xline(median(WHI,'omitnan'),'r--','LineWidth',1.2);
title('WHI','FontWeight','bold'); grid on;

subplot(4,3,11);
histogram(SVI,120,'Normalization','pdf','FaceAlpha',0.7);
hold on; xline(median(SVI,'omitnan'),'r--','LineWidth',1.2);
title('SVI','FontWeight','bold'); grid on;

subplot(4,3,12);
histogram(dE,120,'Normalization','pdf','FaceAlpha',0.7);
hold on; xline(median(dE,'omitnan'),'r--','LineWidth',1.2);
title('\DeltaE76','FontWeight','bold'); grid on;

saveFigSafe(fig, fullfile(pngDir,'05_color_stats.png'), dpi);
end

function saveFig06_Texture(texResult, validMask, pngDir, dpi, visibleMode)
fig = figure('Color','w','Position',[100 100 1800 550],'Visible',char(visibleMode));

subplot(1,3,1);
imagesc(maskNaN(texResult.gradL, validMask)); axis image off; colorbar;
colormap(gca,'turbo');
title('Gradient(L)','FontWeight','bold');

subplot(1,3,2);
imagesc(maskNaN(texResult.entropy, validMask)); axis image off; colorbar;
colormap(gca,'turbo');
title('Entropy','FontWeight','bold');

subplot(1,3,3);
imagesc(maskNaN(texResult.logL, validMask)); axis image off; colorbar;
colormap(gca,'turbo');
title('LoG','FontWeight','bold');

saveFigSafe(fig, fullfile(pngDir,'06_texture.png'), dpi);
end

function saveFig07_Correlation(corrMat, ch9Names, pngDir, dpi, visibleMode)
fig = figure('Color','w','Position',[110 110 900 800],'Visible',char(visibleMode));
imagesc(corrMat,[-1 1]); colorbar;
colormap(gca, coolwarm_local());

set(gca,'XTick',1:numel(ch9Names),'XTickLabel',ch9Names,'XTickLabelRotation',45);
set(gca,'YTick',1:numel(ch9Names),'YTickLabel',ch9Names);
title('9-Channel Correlation','FontWeight','bold');

for i = 1:size(corrMat,1)
    for j = 1:size(corrMat,2)
        text(j,i,sprintf('%.2f',corrMat(i,j)), ...
            'HorizontalAlignment','center','FontSize',8);
    end
end

saveFigSafe(fig, fullfile(pngDir,'07_correlation.png'), dpi);
end

function saveFig08_Summary(plyPath, outDir, N, hasFaces, faces, shapeInfo, ...
    meanGradeAll, stdGradeAll, wMeanGrad, ratioD45, ...
    WHI, SVI, Lch, ach, bch, rockTypeResult, weatherModel, statTests, ...
    t0, pngDir, dpi, visibleMode)

fig = figure('Color','w','Position',[120 120 1100 950],'Visible',char(visibleMode));
axes('Position',[0.03 0.03 0.94 0.94]); axis off;

[~,stem,ext] = fileparts(plyPath);
if hasFaces
    nFaces = size(faces,1);
else
    nFaces = 0;
end

txt = {};
txt{end+1} = 'RockPLY One-File Analyzer Summary';
txt{end+1} = repmat('-',1,65);
txt{end+1} = sprintf('PLY              : %s%s', stem, ext);
txt{end+1} = sprintf('Output           : %s', outDir);
txt{end+1} = sprintf('Vertex           : %s', fmtN(N));
txt{end+1} = sprintf('Faces            : %s', fmtN(nFaces));
txt{end+1} = ' ';
txt{end+1} = '[Shape]';
txt{end+1} = sprintf('Dims             : %.5f x %.5f x %.5f', shapeInfo.dims(1),shapeInfo.dims(2),shapeInfo.dims(3));
txt{end+1} = sprintf('Vol              : %.5f', shapeInfo.vol);
txt{end+1} = sprintf('SA/Vol           : %.5f', shapeInfo.saVol);
txt{end+1} = sprintf('Rough mean       : %.6f', shapeInfo.roughnessMean);
txt{end+1} = sprintf('Curv mean        : %.6f', shapeInfo.curvatureMean);
txt{end+1} = sprintf('Sphericity       : %.6f', shapeInfo.sphericity);
txt{end+1} = sprintf('Elongation       : %.6f', shapeInfo.elongation);
txt{end+1} = ' ';
txt{end+1} = '[Color]';
txt{end+1} = sprintf('L* p05/p50/p95   : %.2f / %.2f / %.2f', prctile(Lch,5),median(Lch,'omitnan'),prctile(Lch,95));
txt{end+1} = sprintf('a* p05/p50/p95   : %.2f / %.2f / %.2f', prctile(ach,5),median(ach,'omitnan'),prctile(ach,95));
txt{end+1} = sprintf('b* p05/p50/p95   : %.2f / %.2f / %.2f', prctile(bch,5),median(bch,'omitnan'),prctile(bch,95));
txt{end+1} = sprintf('WHI p50          : %.4f', median(WHI,'omitnan'));
txt{end+1} = sprintf('SVI p50          : %.4f', median(SVI,'omitnan'));
txt{end+1} = ' ';
txt{end+1} = '[Rock Type]';
if rockTypeResult.enabled
    txt{end+1} = sprintf('Global           : %s', rockTypeResult.globalPred);
    txt{end+1} = sprintf('Granite          : %s', string(rockTypeResult.isGranite));
else
    txt{end+1} = 'Skipped';
end
txt{end+1} = ' ';
txt{end+1} = '[Weathering Model]';
txt{end+1} = sprintf('Model var        : %s', weatherModel.modelVarName);
txt{end+1} = sprintf('Classes          : %s', strjoin(cellstr(string(weatherModel.classNames(:)')), ', '));
txt{end+1} = sprintf('Mean grade       : %.4f', meanGradeAll);
txt{end+1} = sprintf('Std grade        : %.4f', stdGradeAll);
txt{end+1} = sprintf('Weighted mean    : %.4f', wMeanGrad);
txt{end+1} = sprintf('D4-D5 ratio      : %.4f', ratioD45);
txt{end+1} = ' ';
txt{end+1} = '[Statistics]';
txt{end+1} = sprintf('KS L*(D1,D5)     : %.4e', statTests.ks_L_pval);
txt{end+1} = sprintf('KW WHI           : %.4e', statTests.kw_WHI_pval);
txt{end+1} = sprintf('Spearman g-WHI   : rho=%.4f, p=%.4e', ...
    statTests.spearman_grade_WHI_rho, statTests.spearman_grade_WHI_pval);
txt{end+1} = ' ';
txt{end+1} = sprintf('Elapsed sec      : %.2f', toc(t0));

text(0.01, 0.99, strjoin(txt, '\n'), ...
    'VerticalAlignment','top', ...
    'FontName','Consolas', ...
    'FontSize',10, ...
    'Interpreter','none');

saveFigSafe(fig, fullfile(pngDir,'08_summary.png'), dpi);
end

function Y = maskNaN(X, validMask)
Y = X;
Y(~validMask) = NaN;
end

function [faces, hasFaces] = parsePlyFacesSafe(plyPath)
faces = [];
hasFaces = false;

try
    fid = fopen(plyPath,'rb');
    if fid < 0
        return;
    end
    c = onCleanup(@() fclose(fid));

    nVerts = 0;
    nFaces = 0;
    isBinary = false;
    isBigEndian = false;
    headerLines = {};
    faceCountType = 'uint8';
    faceIndexType = 'int32';

    while true
        line = fgetl(fid);
        if ~ischar(line)
            break;
        end
        line = strtrim(line);
        headerLines{end+1} = line; %#ok<AGROW>

        if startsWith(line,'format')
            if contains(line,'binary_big_endian')
                isBinary = true;
                isBigEndian = true;
            elseif contains(line,'binary_little_endian')
                isBinary = true;
            end
        end

        tok = strsplit(line,' ');
        if numel(tok) >= 3 && strcmp(tok{1},'element')
            if strcmp(tok{2},'vertex')
                nVerts = str2double(tok{3});
            elseif strcmp(tok{2},'face')
                nFaces = str2double(tok{3});
            end
        end

        if numel(tok) >= 5 && strcmp(tok{1},'property') && strcmp(tok{2},'list')
            if strcmp(tok{5},'vertex_indices') || strcmp(tok{5},'vertex_index')
                faceCountType = mapPlyType(tok{3});
                faceIndexType = mapPlyType(tok{4});
            end
        end

        if strcmp(line,'end_header')
            break;
        end
    end

    if nFaces <= 0
        return;
    end

    if ~isBinary
        for i = 1:nVerts
            fgetl(fid);
        end
        faceList = zeros(nFaces,3);
        ok = 0;
        for i = 1:nFaces
            ln = fgetl(fid);
            if ~ischar(ln)
                break;
            end
            nums = sscanf(ln,'%d');
            if numel(nums) >= 4 && nums(1) >= 3
                ok = ok + 1;
                faceList(ok,:) = double(nums(2:4))' + 1;
            end
        end
        faces = faceList(1:ok,:);
        hasFaces = ~isempty(faces);
    else
        if isBigEndian
            byteOrder = 'ieee-be';
        else
            byteOrder = 'ieee-le';
        end

        fseek(fid, estimateVertexBytes(headerLines) * nVerts, 'cof');

        faceList = zeros(nFaces,3);
        ok = 0;

        for i = 1:nFaces
            cnt = fread(fid,1,faceCountType,0,byteOrder);
            if isempty(cnt)
                break;
            end
            idxAll = fread(fid,double(cnt),faceIndexType,0,byteOrder);
            if numel(idxAll) >= 3
                ok = ok + 1;
                faceList(ok,:) = double(idxAll(1:3))' + 1;
            end
        end

        faces = faceList(1:ok,:);
        hasFaces = ~isempty(faces);
    end
catch
    faces = [];
    hasFaces = false;
end
end

function t = mapPlyType(tp)
tp = lower(strtrim(tp));
switch tp
    case {'char','int8'}
        t = 'int8';
    case {'uchar','uint8'}
        t = 'uint8';
    case {'short','int16'}
        t = 'int16';
    case {'ushort','uint16'}
        t = 'uint16';
    case {'int','int32'}
        t = 'int32';
    case {'uint','uint32'}
        t = 'uint32';
    case {'float','float32'}
        t = 'single';
    case {'double','float64'}
        t = 'double';
    otherwise
        t = 'int32';
end
end

function nb = estimateVertexBytes(headerLines)
nb = 0;
inVert = false;

typeMap = struct( ...
    'char',1,'uchar',1,'int8',1,'uint8',1, ...
    'short',2,'ushort',2,'int16',2,'uint16',2, ...
    'int',4,'uint',4,'int32',4,'uint32',4,'float',4,'float32',4, ...
    'double',8,'float64',8 );

for i = 1:numel(headerLines)
    ln = strtrim(headerLines{i});
    tok = strsplit(ln,' ');
    if isempty(tok)
        continue;
    end

    if strcmp(tok{1},'element')
        inVert = numel(tok) >= 2 && strcmp(tok{2},'vertex');
    elseif strcmp(tok{1},'property') && inVert
        if numel(tok) >= 3 && ~strcmp(tok{2},'list')
            tp = tok{2};
            fns = fieldnames(typeMap);
            matched = false;
            for k = 1:numel(fns)
                if strcmpi(tp, fns{k})
                    nb = nb + typeMap.(fns{k});
                    matched = true;
                    break;
                end
            end
            if ~matched
                nb = nb + 4;
            end
        end
    end
end

if nb <= 0
    nb = 27;
end
end

function cmap = coolwarm_local()
n = 256;
r = [linspace(0.23,1,n/2) linspace(1,0.71,n/2)]';
g = [linspace(0.30,1,n/2) linspace(1,0.015,n/2)]';
b = [linspace(0.75,1,n/2) linspace(1,0.15,n/2)]';
cmap = [r g b];
end

function tOut = padTileToSize(tIn, outH, outW, fillVal)
[h,w,c] = size(tIn);
tOut = repmat(fillVal, [outH outW c], 'like', tIn);
tOut(1:h,1:w,:) = tIn;
end

function pct = gradePctVec(grades, nClasses)
pct = zeros(1,nClasses);
g = grades(~isnan(grades));
if isempty(g)
    return;
end
for k = 1:nClasses
    pct(k) = 100 * nnz(round(g) == k) / numel(g);
end
end

function s = fmtN(n)
if ~isnumeric(n) || ~isfinite(n)
    s = 'N/A';
    return;
end
s = sprintf('%d', round(double(n)));
pos = numel(s) - 3;
while pos > 0
    s = [s(1:pos) ',' s(pos+1:end)]; 
    pos = pos - 3;
end
end

function printSec(n,msg)
fprintf('\n[%02d] %s\n', n, msg);
end

function mkdirSafe(d)
if ~exist(d,'dir')
    try
        mkdir(d);
    catch ME
        warning(ME.identifier,'%s',ME.message);
    end
end
end

function saveFigSafe(fig, fpath, dpi)
drawnow;
try
    exportgraphics(fig, fpath, 'Resolution', dpi);
catch
    try
        saveas(fig, fpath);
    catch ME
        warning(ME.identifier,'%s',ME.message);
    end
end
end

function S = rmfield_safe(S, names)
for i = 1:numel(names)
    if isfield(S, names{i})
        S = rmfield(S, names{i});
    end
end
end