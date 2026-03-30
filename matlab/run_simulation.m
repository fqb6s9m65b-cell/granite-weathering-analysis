function runInfo = run_simulation(varargin)
% run_simulation (ONE-FILE, R2025a/b)
% -----------------------------------------------------------------------------
% 목적
%   - IMG_ROOT 이미지 + JSON_ROOT ROI(vertices) 매칭(stem)
%   - ROI 마스크 생성
%   - (1) 균열 탐지
%   - (2) 균열 거리(distMap)
%   - (3) 색상값(Lab) + preset edges6 기반 픽셀 D1~D5 맵
%   - (4) (옵션) 1/5/10년 "색상구간 기반 시간진행" 가상 사진 + 등급 오버레이
%   - (5) 분포 기반(p05/p50/p95/IQR) 색상 통계 + (a*,b*) 2D hist 시각화
%
% 요구(권장)
%   Image Processing Toolbox: imguidedfilter, imbinarize, bwdist, poly2mask, bwareaopen, bwmorph, strel, imclose
%
% 출력
%   - Figure 표시
%   - SAVE_PNG=true면 outDir에 PNG/CSV 저장
% -----------------------------------------------------------------------------

clc;

%% -------------------- Options --------------------
opt = struct();

% 경로
opt.IMG_ROOT  = "C:\Users\ROCKENG\Desktop\Sample (5)\Sample\01.원천데이터\2. 기반암 절리 탐지 데이터\1. 화성암";
opt.JSON_ROOT = "C:\Users\ROCKENG\Desktop\Sample (5)\Sample\02.라벨링데이터\2. 기반암 절리 탐지 데이터";

% 스캔/매칭
opt.IMG_EXT   = ".jpg";
opt.NUM_SIMS  = 3;          % Inf면 전체
opt.SEED      = 0;

% 균열 탐지
opt.SENSITIVITY   = 0.55;
opt.MIN_CRACK_PX  = 50;
opt.DO_THIN       = true;
opt.CRACK_CLOSE_RADIUS = 1; % 0이면 skip

% 거리 기반 표시
opt.EARLY_DIST_PX = 15;
opt.MID_DIST_PX   = 40;
opt.CAXIS_MAX     = 120;

% 색상 기반 풍화 등급
opt.BIN_VERSION   = 2;          % 1/2/3 preset
opt.PRIMARY_VAR   = "L_mean";   % 현재는 L* 중심
opt.PRIMARY_DIR   = "auto";     % "auto"|"high_is_fresh"|"high_is_weathered"
opt.FUSE_MODE     = "max";      % crack influence fuse

% lux
opt.LUX_REF       = 100;
opt.LUX_EFFECT_L  = 0.0;        % L* 보정(촬영조건 보정). default 0
opt.FORCE_LUX     = NaN;        % 강제 lux 입력(파일명/JSON 무시)

% ---- [UPGRADE] 미래 진행(색상구간 기반) ----
opt.DO_FUTURE_SIM       = true;
opt.FUTURE_YEARS        = [1 5 10];
opt.TIME_STEP_YR        = 1.0;   % 내부 dt(년)

opt.BASE_TAU_YEARS      = 10.0;  % 기본 시간상수(멀리 떨어진 픽셀)
opt.DIST_SIGMA_PX       = 35.0;  % 균열 영향 거리 스케일(px)
opt.DIST_ACCEL          = 3.0;   % 균열 근처 진행 가속
opt.LUX_ACCEL_BETA      = 0.0;   % lux로 진행속도 가속(가상). 0=미사용

opt.AB_DRIFT_SCALE      = 1.0;   % 0이면 a,b 고정
opt.DLAB_PER_GRADE      = [-2.0, +0.4, +0.8]; % 등급 1단계 진행에 준하는 Lab 변화량(가상)

% ---- [UPGRADE] 분포/시각화 ----
opt.MAX_HIST_SAMPLES    = 2e6;   % L* hist 샘플 상한
opt.MAX_SCATTER_SAMPLES = 2e5;   % a*b scatter 샘플 상한
opt.HIST_BINS_L         = 256;
opt.HIST2_BINS_AB       = 120;   % a*b 2D hist bins

% 저장
opt.SAVE_PNG        = false;
opt.OUT_DIR         = fullfile(pwd, "_sim_out");
opt.MAKE_RUN_SUBDIR = true;
opt.DPI             = 200;

opt = wx_parse_nv(opt, varargin{:});

if opt.MAKE_RUN_SUBDIR
    ts = string(datetime('now','Format','yyyyMMdd_HHmmss'));
    outDir = string(fullfile(string(opt.OUT_DIR), "Run_" + ts));
else
    outDir = string(opt.OUT_DIR);
end
if opt.SAVE_PNG && ~exist(char(outDir),'dir')
    mkdir(char(outDir));
end

fprintf('=== [시뮬레이션 시작] 데이터 스캔 중... ===\n');

%% -------------------- Scan files --------------------
imgFiles  = dir(fullfile(char(opt.IMG_ROOT),  "**", "*" + string(opt.IMG_EXT)));
jsonFiles = dir(fullfile(char(opt.JSON_ROOT), "**", "*.json"));

if isempty(imgFiles),  error('이미지 파일이 없습니다. IMG_ROOT/IMG_EXT 확인'); end
if isempty(jsonFiles), error('JSON 파일이 없습니다. JSON_ROOT 확인'); end

jsonMap = containers.Map('KeyType','char','ValueType','char');
for i = 1:numel(jsonFiles)
    [~, nm, ~] = fileparts(jsonFiles(i).name);
    jsonMap(char(nm)) = fullfile(jsonFiles(i).folder, jsonFiles(i).name);
end

rng(opt.SEED);
if isinf(opt.NUM_SIMS)
    pick = 1:numel(imgFiles);
else
    nPick = min(double(opt.NUM_SIMS), numel(imgFiles));
    pick = randperm(numel(imgFiles), nPick);
end

presetMap  = wx_get_igneous_preset_edges(opt.BIN_VERSION);
dirPrimary = string(opt.PRIMARY_DIR);
if strcmpi(dirPrimary,"auto")
    dirPrimary = wx_default_direction_for_primary(opt.PRIMARY_VAR);
end

%% -------------------- Loop --------------------
rows = {};
nOK = 0;

for kk = 1:numel(pick)
    idx = pick(kk);

    imgPath = fullfile(imgFiles(idx).folder, imgFiles(idx).name);
    [~, stem, ~] = fileparts(imgFiles(idx).name);
    stem = char(stem);

    if ~isKey(jsonMap, stem)
        fprintf('[Skip] JSON 매칭 실패: %s\n', stem);
        continue;
    end
    jPath = jsonMap(stem);

    nOK = nOK + 1;
    try
        one = wx_run_one(imgPath, jPath, stem, presetMap, dirPrimary, outDir, opt);
        rows(end+1,:) = {one.stem, one.imgPath, one.jsonPath, one.lux, one.imageGrade, ...
                         one.midDamagePct, one.areaROI, one.areaCrack, ...
                         one.p05L, one.p50L, one.p95L, one.iqrL, ...
                         one.frac(1), one.frac(2), one.frac(3), one.frac(4), one.frac(5)}; %#ok<AGROW>
    catch ME
        warning(ME.identifier, '%s', sprintf("처리 실패: %s | %s", imgPath, ME.message));
    end
end

fprintf('=== 모든 시뮬레이션 완료 ===\n');
fprintf('Processed: %d / Picked: %d\n', nOK, numel(pick));

if ~isempty(rows)
    T = cell2table(rows, 'VariableNames', ...
        {'stem','imgPath','jsonPath','lux','imageGrade','midDamagePct','areaROI','areaCrack', ...
         'p05L','p50L','p95L','iqrL','fracD1','fracD2','fracD3','fracD4','fracD5'});
else
    T = table();
end

runInfo = struct();
runInfo.outDir = outDir;
runInfo.opt = opt;
runInfo.summaryTable = T;

if opt.SAVE_PNG && ~isempty(T)
    try
        writetable(T, fullfile(char(outDir), "SIM_SUMMARY.csv"));
    catch ME
        warning(ME.identifier, '%s', ME.message);
    end
end

end

%% =========================================================================
% Run one pair
%% =========================================================================
function one = wx_run_one(imgPath, jPath, stem, presetMap, dirPrimary, outDir, opt)

% --- load image ---
Iu8 = imread(imgPath);
if ismatrix(Iu8)
    Iu8 = repmat(Iu8, 1, 1, 3);
end
if size(Iu8,3) > 3
    Iu8 = Iu8(:,:,1:3);
end
Igray = rgb2gray(Iu8);

% --- load json ---
dat = wx_read_json(jPath);

% --- ROI mask ---
mask = wx_mask_from_json_vertices(dat, [size(Igray,1) size(Igray,2)]);
if isempty(mask) || ~isequal(size(mask), size(Igray))
    mask = true(size(Igray));
end

% --- masked RGB for display ---
Imasked = Iu8;
for c = 1:3
    t = Imasked(:,:,c);
    t(~mask) = 0;
    Imasked(:,:,c) = t;
end

% --- lux parse ---
lux = wx_parse_lux_from_name(stem);
if ~isfinite(lux)
    lux = wx_try_read_lux_from_json(dat);
end
if isfinite(opt.FORCE_LUX)
    lux = double(opt.FORCE_LUX);
end

% --- crack detection ---
Ifilt = imguidedfilter(Igray);
bw_crack = imbinarize(Ifilt, 'adaptive', ...
    'Sensitivity', opt.SENSITIVITY, 'ForegroundPolarity', 'dark');
bw_crack = bw_crack & mask;

if opt.CRACK_CLOSE_RADIUS > 0
    try
        se = strel('disk', max(1, round(double(opt.CRACK_CLOSE_RADIUS))), 0);
        bw_crack = imclose(bw_crack, se);
    catch ME
        warning(ME.identifier, '%s', ME.message);
    end
end

bw_crack = bwareaopen(bw_crack, opt.MIN_CRACK_PX);

if opt.DO_THIN
    try
        bw_crack = bwmorph(bw_crack, 'thin', Inf);
    catch ME
        warning(ME.identifier, '%s', ME.message);
    end
end

% --- distance map ---
if any(bw_crack(:))
    distMap = bwdist(bw_crack);
else
    distMap = inf(size(bw_crack));
end
distMap(~mask) = NaN;

zoneEarly = (distMap < double(opt.EARLY_DIST_PX)) & mask;
zoneMid   = (distMap < double(opt.MID_DIST_PX)) & mask;

% --- Lab ---
I = im2single(Iu8);
Lab = wx_srgb2lab_local(I);
Limg = Lab(:,:,1); aimg = Lab(:,:,2); bimg = Lab(:,:,3);

% lux correction (촬영 보정)
if isfinite(lux) && opt.LUX_EFFECT_L ~= 0
    k = double(opt.LUX_EFFECT_L);
    Limg = Limg + k * log10( max(1, double(lux)) / max(1, double(opt.LUX_REF)) );
end

% --- grade from edges6 ---
edges6 = wx_pick_edges6(string(opt.PRIMARY_VAR), presetMap);
gradeColor = wx_grade_map_from_scalar(Limg, edges6, dirPrimary, mask); % 0..5

% crack influence -> gradeCrack
gradeCrack = gradeColor;
gradeCrack(zoneMid)   = min(uint8(5), gradeCrack(zoneMid)   + uint8(1));
gradeCrack(zoneEarly) = min(uint8(5), gradeCrack(zoneEarly) + uint8(2));

switch lower(string(opt.FUSE_MODE))
    case "max"
        gradeFinal = max(gradeColor, gradeCrack);
    otherwise
        gradeFinal = max(gradeColor, gradeCrack);
end

% image-level grade: p50 (median)
Lvec = double(Limg(mask));
Lvec = Lvec(isfinite(Lvec));
if isempty(Lvec), Lmed = NaN; else, Lmed = median(Lvec); end
imageGrade = wx_grade_scalar(Lmed, edges6, dirPrimary);

fprintf('[OK] %s | lux=%s | %s\n', stem, wx_num2str_or_na(lux), imageGrade);

% --- distribution stats (p05/p50/p95/IQR) ---
[Lsamp, Asamp, Bsamp] = wx_sample_lab(Limg, aimg, bimg, mask, opt);
p05L = prctile(Lsamp, 5);
p50L = prctile(Lsamp, 50);
p95L = prctile(Lsamp, 95);
iqrL = p95L - p05L;

% --- areas ---
area_total = nnz(mask);
area_crack = nnz(bw_crack);
area_mid   = nnz(zoneMid);
damage_mid = 100 * (area_mid / max(1, area_total));

counts = zeros(1,5);
for g=1:5
    counts(g) = nnz(gradeFinal==g);
end
frac = counts / max(1, nnz(mask));

%% -------------------- Figure 1: main (6 panels) --------------------
f1 = figure('Color','w', 'Name', sprintf('Sim: %s', stem), 'Position',[60 60 1350 720]);

subplot(2,3,1);
imshow(Imasked);
title('1) 원본(ROI)');

subplot(2,3,2);
imshow(Imasked); hold on;
ovR = zeros(size(Imasked), 'uint8'); ovR(:,:,1) = 255;
h = imshow(ovR);
set(h, 'AlphaData', 0.85*double(bw_crack));
title(sprintf('2) 균열 탐지 (area=%d px)', area_crack));

subplot(2,3,3);
mx = max(distMap(:), [], 'omitnan');
if ~isfinite(mx), mx = double(opt.CAXIS_MAX); end
vis = distMap;
vis(~isfinite(vis)) = mx;
vis(isnan(vis)) = mx;
imagesc(vis); axis image off;
colormap(gca, flipud(jet(256)));
clim([0 double(opt.CAXIS_MAX)]); colorbar;
title('3) 거리맵(균열->거리)');

subplot(2,3,4);
imagesc(gradeColor); axis image off;
colormap(gca, flipud(jet(5))); clim([1 5]);
cb = colorbar; cb.Ticks = 1:5; cb.TickLabels = {'D1','D2','D3','D4','D5'};
title('4) 색상 기반 픽셀 등급');

subplot(2,3,5);
imshow(Imasked); hold on;
cmap = flipud(jet(5));
idxRGB = gradeFinal; idxRGB(~mask) = 1;
rgb = ind2rgb(uint8(idxRGB), cmap);
h2 = imshow(rgb);
alpha = 0.10 + 0.55*(double(gradeFinal)-1)/4;
alpha(~mask) = 0;
set(h2, 'AlphaData', alpha);
title(sprintf('5) 합성 등급 오버레이 | MidDamage=%.2f%%', damage_mid));

subplot(2,3,6);
bar(1:5, frac, 1.0);
xlim([0.5 5.5]); grid on;
set(gca,'XTick',1:5,'XTickLabel',{'D1','D2','D3','D4','D5'});
ylabel('area fraction');
title('6) 합성 등급 면적비');

sgtitle(sprintf('%s | lux=%s | ImageGrade(p50,L*)=%s | p50(L*)=%.2f', ...
    stem, wx_num2str_or_na(lux), imageGrade, p50L));

%% -------------------- Figure 2: color distribution (L* hist + ab 2D hist) --------------------
fC = figure('Color','w', 'Name', sprintf('ColorStats: %s', stem), 'Position',[80 80 1350 520]);

subplot(1,2,1);
wx_plot_L_hist_with_edges(Lsamp, edges6, p05L, p50L, p95L, opt);
title(sprintf('L* histogram (ROI) | IQR=%.2f', iqrL));
xlabel('L*'); ylabel('count');

subplot(1,2,2);
wx_plot_ab_2dhist(Asamp, Bsamp, opt);
title('a*-b* 2D hist (ROI)');
xlabel('a*'); ylabel('b*');

%% -------------------- Save PNGs --------------------
if opt.SAVE_PNG
    base = wx_sanitize_filename(stem);
    try
        exportgraphics(f1, fullfile(char(outDir), sprintf('%s_main.png', base)), 'Resolution', opt.DPI);
    catch ME
        warning(ME.identifier, '%s', ME.message);
    end
    try
        exportgraphics(fC, fullfile(char(outDir), sprintf('%s_colorstats.png', base)), 'Resolution', opt.DPI);
    catch ME
        warning(ME.identifier, '%s', ME.message);
    end
end

%% -------------------- Future simulation (color-bin driven) --------------------
if opt.DO_FUTURE_SIM && ~isempty(opt.FUTURE_YEARS)

    yrs = sort(double(opt.FUTURE_YEARS(:)'));
    [snapIu8, snapGrade, snapLstats] = wx_simulate_future_by_colorbins( ...
        Limg, aimg, bimg, mask, distMap, edges6, dirPrimary, lux, opt, yrs);

    nY = numel(yrs);
    fF = figure('Color','w', 'Name', sprintf('FutureSim: %s', stem), 'Position',[100 80 360*nY 650]);

    for i = 1:nY
        y = yrs(i);
        Iaged_u8 = snapIu8{i};
        gF = snapGrade{i};

        subplot(3,nY,i);
        imshow(Iaged_u8);
        title(sprintf('%dy (가상 사진)', y));

        subplot(3,nY,nY+i);
        imshow(Iaged_u8); hold on;
        cmap5 = flipud(jet(5));
        idx = gF; idx(~mask) = 1;
        rgb2 = ind2rgb(uint8(idx), cmap5);
        h3 = imshow(rgb2);
        alpha2 = 0.10 + 0.55*(double(gF)-1)/4;
        alpha2(~mask)=0; alpha2(gF==0)=0;
        set(h3,'AlphaData',alpha2);
        title(sprintf('%dy (등급 오버레이)', y));

        subplot(3,nY,2*nY+i);
        st = snapLstats(i,:);
        bar([st(1) st(2) st(3)], 1, 0.8); %#ok<NBRAK>
        set(gca,'XTick',1:3,'XTickLabel',{'p05','p50','p95'});
        ylim([0 100]); grid on;
        title(sprintf('%dy L* (p05/p50/p95)', y));
        ylabel('L*');
    end

    sgtitle(sprintf('%s | Future visualization (color-bins driven, uncalibrated)', stem));

    if opt.SAVE_PNG
        base = wx_sanitize_filename(stem);
        try
            exportgraphics(fF, fullfile(char(outDir), sprintf('%s_future.png', base)), 'Resolution', opt.DPI);
        catch ME
            warning(ME.identifier, '%s', ME.message);
        end
    end
end

fprintf('  - ROI area        : %d px\n', area_total);
fprintf('  - crack area      : %d px\n', area_crack);
fprintf('  - mid-risk area   : %d px\n', area_mid);
fprintf('  - mid damage ratio: %.2f%%\n', damage_mid);
fprintf('  - L*(ROI) p05/p50/p95: %.2f / %.2f / %.2f\n', p05L, p50L, p95L);
fprintf('---------------------------------------------------\n');

one = struct();
one.stem = string(stem);
one.imgPath = string(imgPath);
one.jsonPath = string(jPath);
one.lux = double(lux);
one.imageGrade = string(imageGrade);
one.midDamagePct = double(damage_mid);
one.areaROI = double(area_total);
one.areaCrack = double(area_crack);
one.frac = double(frac);

one.p05L = double(p05L);
one.p50L = double(p50L);
one.p95L = double(p95L);
one.iqrL = double(iqrL);

end

%% =========================================================================
% JSON reader
%% =========================================================================
function dat = wx_read_json(pth)
try
    txt = fileread(pth);
    dat = jsondecode(txt);
catch ME
    warning(ME.identifier, '%s', ME.message);
    dat = struct();
end
end

%% =========================================================================
% Robust ROI mask from JSON
%% =========================================================================
function mask = wx_mask_from_json_vertices(dat, szHW)
H = szHW(1); W = szHW(2);
mask = true(H,W);

polys = wx_extract_polygons_from_json(dat);
if isempty(polys)
    return;
end

mask = false(H,W);
try
    for p = 1:numel(polys)
        xy = polys{p};
        if isempty(xy) || size(xy,2) ~= 2 || size(xy,1) < 3, continue; end
        x = double(xy(:,1));
        y = double(xy(:,2));
        mask = mask | poly2mask(x, y, H, W);
    end
catch ME
    warning(ME.identifier, '%s', ME.message);
    mask = true(H,W);
end

if ~any(mask(:))
    mask = true(H,W);
end
end

function polys = wx_extract_polygons_from_json(dat)
polys = {};
try
    if ~isstruct(dat), return; end

    if isfield(dat,'vertices') && ~isempty(dat.vertices)
        polys = wx_polys_from_any(dat.vertices);
        if ~isempty(polys), return; end
    end

    if isfield(dat,'shapes') && ~isempty(dat.shapes)
        S = dat.shapes;
        if isstruct(S)
            for i=1:numel(S)
                if isfield(S(i),'points')
                    xy = wx_points_to_xy(S(i).points);
                    if size(xy,1) >= 3, polys{end+1} = xy; end %#ok<AGROW>
                end
            end
        end
        if ~isempty(polys), return; end
    end

    candFields = {'annotations','objects','roi','polygons'};
    for cf = 1:numel(candFields)
        f = candFields{cf};
        if isfield(dat, f) && ~isempty(dat.(f))
            polys = wx_polys_from_any(dat.(f));
            if ~isempty(polys), return; end
        end
    end
catch
end
end

function polys = wx_polys_from_any(v)
polys = {};
try
    if iscell(v)
        for i=1:numel(v)
            polys = [polys, wx_polys_from_any(v{i})]; %#ok<AGROW>
        end
        return;
    end

    if isstruct(v)
        for i=1:numel(v)
            vi = v(i);

            if isfield(vi,'x') && isfield(vi,'y')
                x = double([vi.x(:)]);
                y = double([vi.y(:)]);
                xy = [x(:), y(:)];
                if size(xy,1) >= 3, polys{end+1} = xy; end %#ok<AGROW>
                continue;
            end

            if isfield(vi,'points')
                xy = wx_points_to_xy(vi.points);
                if size(xy,1) >= 3, polys{end+1} = xy; end %#ok<AGROW>
                continue;
            end

            if isfield(vi,'vertices')
                polys = [polys, wx_polys_from_any(vi.vertices)]; %#ok<AGROW>
                continue;
            end
        end
        return;
    end

    if isnumeric(v)
        xy = wx_points_to_xy(v);
        if size(xy,1) >= 3, polys{end+1} = xy; end %#ok<AGROW>
        return;
    end
catch
end
end

function xy = wx_points_to_xy(pts)
xy = [];
try
    if isnumeric(pts) && size(pts,2) == 2
        xy = double(pts);
        return;
    end
    if iscell(pts)
        n = numel(pts);
        tmp = nan(n,2);
        ok = true;
        for i=1:n
            pi = pts{i};
            if isnumeric(pi) && numel(pi) >= 2
                tmp(i,1) = double(pi(1));
                tmp(i,2) = double(pi(2));
            else
                ok = false;
                break;
            end
        end
        if ok, xy = tmp; end
        return;
    end
catch
end
end

%% =========================================================================
% Lux helpers
%% =========================================================================
function lux = wx_parse_lux_from_name(stem)
lux = NaN;
try
    s = char(stem);
    tok = regexp(s, '-(\d+)\(NA\)-', 'tokens', 'once');
    if ~isempty(tok)
        lux = str2double(tok{1});
        return;
    end
    tok = regexp(s, '-(\d+)-', 'tokens', 'once');
    if ~isempty(tok)
        lux = str2double(tok{1});
        return;
    end
catch
end
end

function lux = wx_try_read_lux_from_json(dat)
lux = NaN;
if ~isstruct(dat), return; end
cands = ["lux","lux_raw","Lux","illum","illumination"];
for i=1:numel(cands)
    f = cands(i);
    if isfield(dat, char(f))
        try
            lux = double(dat.(char(f)));
            if isfinite(lux), return; end
        catch
        end
    end
end
end

function s = wx_num2str_or_na(x)
if isfinite(x)
    s = sprintf('%.0f', x);
else
    s = "NA";
end
end

%% =========================================================================
% Igneous preset edges
%% =========================================================================
function presetMap = wx_get_igneous_preset_edges(binVersion)
presetMap = containers.Map('KeyType','char','ValueType','any');
switch double(binVersion)
    case 1
        presetMap('L_mean') = [10.911426 25.831304 40.751181 55.671059 70.590937 85.510817];
    case 2
        presetMap('L_mean') = [27.511002 36.254763 44.998524 53.742284 62.486045 71.229805];
    case 3
        presetMap('L_mean') = [10.911426 37.745305 46.345101 54.277849 62.685462 85.510817];
    otherwise
        presetMap('L_mean') = [27.511002 36.254763 44.998524 53.742284 62.486045 71.229805];
end
end

function edges6 = wx_pick_edges6(primaryVar, presetMap)
primaryVar = string(primaryVar);
if isKey(presetMap, char(primaryVar))
    edges6 = presetMap(char(primaryVar));
else
    edges6 = presetMap('L_mean');
end
edges6 = double(edges6(:))';
end

function dir = wx_default_direction_for_primary(v)
v = lower(string(v));
if any(v == ["a_mean","b_mean"])
    dir = "high_is_weathered";
else
    dir = "high_is_fresh";
end
end

%% =========================================================================
% Grade mapping
%% =========================================================================
function g = wx_grade_scalar(x, edges6, direction)
x = double(x);
bin = discretize(x, edges6);
if isnan(bin)
    if x < edges6(1), bin = 1; end
    if x >= edges6(end), bin = 5; end
    if isnan(bin), bin = 3; end
end

direction = lower(string(direction));
switch direction
    case "high_is_fresh"
        gnum = 6 - bin;
    case "high_is_weathered"
        gnum = bin;
    otherwise
        gnum = 6 - bin;
end
g = "D" + string(gnum);
end

function gMap = wx_grade_map_from_scalar(X, edges6, direction, mask)
X = double(X);
gMap = zeros(size(X), 'uint8'); % ROI 밖 0
bin = discretize(X, edges6);
bin(X < edges6(1)) = 1;
bin(X >= edges6(end)) = 5;
bin(isnan(bin)) = 3;

direction = lower(string(direction));
switch direction
    case "high_is_fresh"
        g = 6 - bin;
    case "high_is_weathered"
        g = bin;
    otherwise
        g = 6 - bin;
end

g(~mask) = 0;
gMap(mask) = uint8(g(mask));
end

%% =========================================================================
% [UPGRADE] 샘플링 (분포 기반, 메모리 안전)
%% =========================================================================
function [Lsamp, Asamp, Bsamp] = wx_sample_lab(L, a, b, mask, opt)
Lvec = double(L(mask));
Avec = double(a(mask));
Bvec = double(b(mask));

Lvec = Lvec(isfinite(Lvec));
Avec = Avec(isfinite(Avec));
Bvec = Bvec(isfinite(Bvec));

n = min([numel(Lvec), numel(Avec), numel(Bvec)]);
Lvec = Lvec(1:n); Avec = Avec(1:n); Bvec = Bvec(1:n);

% 히스토그램 샘플 상한
mH = max(1000, round(double(opt.MAX_HIST_SAMPLES)));
if n > mH
    idx = randperm(n, mH);
    Lsamp = Lvec(idx); Asamp = Avec(idx); Bsamp = Bvec(idx);
else
    Lsamp = Lvec; Asamp = Avec; Bsamp = Bvec;
end

% scatter/2D hist용은 추가로 더 줄일 수 있음(내부에서 처리)
end

%% =========================================================================
% [UPGRADE] L* hist with edges6 & quantiles
%% =========================================================================
function wx_plot_L_hist_with_edges(Lsamp, edges6, p05, p50, p95, opt)
bins = max(32, round(double(opt.HIST_BINS_L)));
h = histogram(Lsamp, bins);
grid on;

yl = ylim;

% edges6 라인
for i=1:numel(edges6)
    x = edges6(i);
    line([x x], yl, 'LineStyle','--');
end

% p05/p50/p95 라인
line([p05 p05], yl, 'LineStyle','-');
line([p50 p50], yl, 'LineStyle','-');
line([p95 p95], yl, 'LineStyle','-');

legend({'hist','edges6','p05','p50','p95'}, 'Location','northeast');
end

%% =========================================================================
% [UPGRADE] a*-b* 2D hist
%% =========================================================================
function wx_plot_ab_2dhist(Asamp, Bsamp, opt)
% scatter sample limit
n = numel(Asamp);
mS = max(5000, round(double(opt.MAX_SCATTER_SAMPLES)));
if n > mS
    idx = randperm(n, mS);
    A = Asamp(idx); B = Bsamp(idx);
else
    A = Asamp; B = Bsamp;
end

nb = max(50, round(double(opt.HIST2_BINS_AB)));

% 2D hist
try
    [N, xe, ye] = histcounts2(A, B, nb, nb);
    imagesc(xe, ye, log10(N'+1)); axis xy;
    colorbar;
catch ME
    warning(ME.identifier, '%s', ME.message);
    scatter(A, B, 1, '.');
    grid on;
end
end

%% =========================================================================
% [UPGRADE] 미래 시각화(색상구간 기반 시간 진행)
%% =========================================================================
function [snapIu8, snapGrade, snapLstats] = wx_simulate_future_by_colorbins( ...
    L0, a0, b0, mask, distMap, edges6, dirPrimary, lux, opt, snapYears)

snapYears = double(snapYears(:)');
maxY = max(snapYears);

dt = double(opt.TIME_STEP_YR);
if dt <= 0, dt = 1; end

edges6 = double(edges6(:))';
binCenters = 0.5*(edges6(1:5) + edges6(2:6));

d = double(distMap);
d(~mask) = inf;
wDist = 1 + double(opt.DIST_ACCEL) .* exp(-d./max(1,double(opt.DIST_SIGMA_PX)));
wDist(~isfinite(wDist)) = 1;

wLux = 1;
if isfinite(lux) && opt.LUX_ACCEL_BETA ~= 0
    wLux = 1 + double(opt.LUX_ACCEL_BETA) * log10(max(1,double(lux))/max(1,double(opt.LUX_REF)));
    wLux = max(0.25, wLux);
end

tau0 = max(1e-3, double(opt.BASE_TAU_YEARS));

L = single(L0);
a = single(a0);
b = single(b0);

snapIu8   = cell(1, numel(snapYears));
snapGrade = cell(1, numel(snapYears));
snapLstats = nan(numel(snapYears), 3);

snapIdx = 1;

for y = 1:maxY

    gNow = wx_grade_map_fast_L(L, edges6, dirPrimary, mask); % 0..5
    gT   = min(uint8(5), gNow + uint8(1));
    gT(gNow==0) = 0;

    Ltarget = wx_grade_to_center_L(gT, binCenters, dirPrimary, mask);

    tau = tau0 ./ (wDist * wLux);
    alpha = min(1, dt ./ max(1e-3, tau));
    alpha = single(alpha);

    % 핵심: "다음 등급 구간 중심"으로 색상 이동
    L = L + alpha .* (Ltarget - L);

    if opt.AB_DRIFT_SCALE ~= 0
        dLab = single(double(opt.DLAB_PER_GRADE) .* [1, opt.AB_DRIFT_SCALE, opt.AB_DRIFT_SCALE]);
        a = a + alpha .* dLab(2);
        b = b + alpha .* dLab(3);
    end

    % clamp
    L = min(single(100), max(single(0), L));
    a = min(single(127), max(single(-128), a));
    b = min(single(127), max(single(-128), b));

    if snapIdx <= numel(snapYears) && y == snapYears(snapIdx)

        gSnap = wx_grade_map_fast_L(L, edges6, dirPrimary, mask);

        Iaged = wx_lab2srgb_local(double(L), double(a), double(b));
        Iu8 = uint8(255*min(1,max(0,Iaged)));
        for c=1:3
            t = Iu8(:,:,c);
            t(~mask) = 0;
            Iu8(:,:,c) = t;
        end

        % L stats (p05/p50/p95)
        Lvec = double(L(mask));
        Lvec = Lvec(isfinite(Lvec));
        if ~isempty(Lvec)
            snapLstats(snapIdx,:) = [prctile(Lvec,5), prctile(Lvec,50), prctile(Lvec,95)];
        end

        snapIu8{snapIdx}   = Iu8;
        snapGrade{snapIdx} = gSnap;

        snapIdx = snapIdx + 1;
    end
end

end

function gMap = wx_grade_map_fast_L(L, edges6, dirPrimary, mask)
e2 = edges6(2); e3 = edges6(3); e4 = edges6(4); e5 = edges6(5);
Ld = double(L);
bin = uint8(1) + uint8(Ld >= e2) + uint8(Ld >= e3) + uint8(Ld >= e4) + uint8(Ld >= e5);

dirPrimary = lower(string(dirPrimary));
switch dirPrimary
    case "high_is_fresh"
        g = uint8(6) - bin;
    case "high_is_weathered"
        g = bin;
    otherwise
        g = uint8(6) - bin;
end

g(~mask) = 0;
gMap = g;
end

function Lc = wx_grade_to_center_L(gD, binCenters, dirPrimary, mask)
gD = uint8(gD);
binCenters = single(binCenters(:)');

binIdx = zeros(size(gD), 'uint8');

dirPrimary = lower(string(dirPrimary));
switch dirPrimary
    case "high_is_fresh"
        binIdx(mask) = uint8(6) - gD(mask);
    case "high_is_weathered"
        binIdx(mask) = gD(mask);
    otherwise
        binIdx(mask) = uint8(6) - gD(mask);
end

binIdx(binIdx<1) = 1;
binIdx(binIdx>5) = 5;

Lc = zeros(size(gD), 'single');
for k = 1:5
    Lc(binIdx==k) = binCenters(k);
end
Lc(~mask) = 0;
end

%% =========================================================================
% Name-Value parser
%% =========================================================================
function opt = wx_parse_nv(opt, varargin)
if isempty(varargin), return; end
if mod(numel(varargin),2) ~= 0
    error("Name-Value 인자는 짝수 개여야 합니다.");
end
for k = 1:2:numel(varargin)
    name = string(varargin{k});
    val  = varargin{k+1};
    if ~isfield(opt, name)
        error("알 수 없는 옵션: %s", name);
    end
    opt.(name) = val;
end
end

function s = wx_sanitize_filename(s)
s = string(s);
s = regexprep(s, '[<>:"/\\|?*]', '_');
s = strtrim(s);
s = char(s);
end

%% =========================================================================
% sRGB <-> Lab (toolbox-free)
%% =========================================================================
function Lab = wx_srgb2lab_local(I)
I = double(I);
I = max(0,min(1,I));

aa = 0.055;
thr = 0.04045;
lin = zeros(size(I));
m = I <= thr;
lin(m)  = I(m)/12.92;
lin(~m) = ((I(~m)+aa)/(1+aa)).^2.4;

R = lin(:,:,1); G = lin(:,:,2); B = lin(:,:,3);

X = 0.4124564*R + 0.3575761*G + 0.1804375*B;
Y = 0.2126729*R + 0.7151522*G + 0.0721750*B;
Z = 0.0193339*R + 0.1191920*G + 0.9503041*B;

Xn = 0.95047; Yn = 1.00000; Zn = 1.08883;
x = X./Xn; y = Y./Yn; z = Z./Zn;

delta = 6/29;
delta3 = delta^3;

fx = wx_f_xyz_local(x, delta, delta3);
fy = wx_f_xyz_local(y, delta, delta3);
fz = wx_f_xyz_local(z, delta, delta3);

L = 116*fy - 16;
aC = 500*(fx - fy);
bC = 200*(fy - fz);

Lab = cat(3, L, aC, bC);
end

function f = wx_f_xyz_local(t, delta, delta3)
f = t.^(1/3);
m = t <= delta3;
f(m) = t(m)/(3*delta^2) + 4/29;
end

function I = wx_lab2srgb_local(L, aC, bC)
L  = double(L); aC = double(aC); bC = double(bC);

fy = (L + 16) / 116;
fx = fy + (aC / 500);
fz = fy - (bC / 200);

delta = 6/29;

xr = wx_f_inv_local(fx, delta);
yr = wx_f_inv_local(fy, delta);
zr = wx_f_inv_local(fz, delta);

Xn = 0.95047; Yn = 1.00000; Zn = 1.08883;
X = xr * Xn;
Y = yr * Yn;
Z = zr * Zn;

R =  3.2404542*X + (-1.5371385)*Y + (-0.4985314)*Z;
G = (-0.9692660)*X +  1.8760108*Y +  0.0415560*Z;
B =  0.0556434*X + (-0.2040259)*Y +  1.0572252*Z;

I = cat(3, wx_gamma_encode_local(R), wx_gamma_encode_local(G), wx_gamma_encode_local(B));
I = max(0, min(1, I));
end

function t = wx_f_inv_local(f, delta)
f3 = f.^3;
t = f3;
m = f3 <= delta^3;
t(m) = 3*(delta^2) * (f(m) - 4/29);
end

function s = wx_gamma_encode_local(lin)
lin = max(0, lin);
aa = 0.055;
thr = 0.0031308;
s = lin;
m = lin <= thr;
s(m)  = 12.92 * lin(m);
s(~m) = (1+aa) * lin(~m).^(1/2.4) - aa;
end
