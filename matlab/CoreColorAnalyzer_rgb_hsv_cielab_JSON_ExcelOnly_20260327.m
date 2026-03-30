function runInfo = CoreColorAnalyzer_rgb_hsv_cielab_JSON_ExcelOnly_20260327(varargin)
% CoreColorStats_FULL.m
% =========================================================================
% 핵심 원칙
%   1. 풍화등급(rw) = JSON "rw" 필드가 유일한 정답 (파일명 파싱 보조만)
%   2. 색상 분석   = 전체 픽셀 raw값 그대로, 보정·마스킹 없음
%   3. 파일명      = JSON 매칭 키 + 부가 메타 추출용
%
% 색공간  : RGB · HSV · HSL · CIELAB/LCh · XYZ · YCbCr
%           CMYK · Opponent · Log-chromaticity
% Munsell : CIELAB → V/C/H 근사
% 풍화지수 : RI CI NRI RBI SVI WHI CWI YI SAI GRI delta_E_D1 HI MWI CrI
% GPU     : rgb2hsv/rgb2lab GPU 가속 + OOM 자동 CPU 폴백
% CPU     : parfor 전체 코어 병렬
% Excel   : 행/열 한계 초과 시 자동 분할
%
% JSON 실제 필드명 (대소문자 정확히 일치)
%   object_id  width  height  box_no  year
%   drilling_place1  drilling_place2  hole_num  row_num  ers
%   camera_type  camera_angle  rotation_angle  shoot_height
%   f-stop  shutter_speed  ISO  lux  location  humidity
%   rock_depth  rock_strength  tcr  rqd  rmr  rq  rjr  fm
%   rw  geo_structure
%   rock_type[{rock_type1, rock_type2}]
%
% R2025a | ONE-FILE
% =========================================================================

%% ===== [0] 파라미터 =====
p = inputParser;
addParameter(p,'IMAGE_ROOTS',{
    'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\images_igneous_rock\TS_1, 기반암 암종 분류 데이터_1. 화성암_1'
    'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\images_igneous_rock\TS_1, 기반암 암종 분류 데이터_1. 화성암_2'
    'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\images_igneous_rock\TS_2. 기반암 절리 탐지 데이터_1. 화성암'
    'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\images_igneous_rock\VS_1. 기반암 암종 분류 데이터_1. 화성암'
    'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\images_igneous_rock\VS_2. 기반암 절리 탐지 데이터_1. 화성암'
});
addParameter(p,'JSON_ROOTS',{
    'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\json_igneous_rock\TL_1'
    'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\json_igneous_rock\TL_2'
    'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\json_igneous_rock\VL_1'
    'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\json_igneous_rock\VL_2'
});
addParameter(p,'OUT_DIR', fullfile( ...
    'C:\Users\ROCKENG\Desktop\코랩 머신러닝', ...
    "FULL_COLOR_" + string(datetime('now','Format','yyyyMMdd_HHmmss'))));
addParameter(p,'IMG_EXTS',   {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'});
addParameter(p,'SAVE_EVERY', 500);
addParameter(p,'USE_GPU',    true);
addParameter(p,'USE_PAR',    true);
addParameter(p,'GPU_BATCH',  64);
% D1 기준색 (delta_E_D1 용) — 이전 분석값
addParameter(p,'REF_D1_L',  48.12);
addParameter(p,'REF_D1_a',  -1.77);
addParameter(p,'REF_D1_b',  -1.29);
parse(p,varargin{:});
OPT = p.Results;

EXCEL_MAX_ROWS = 1048000;
EXCEL_MAX_COLS = 16000;

t0 = tic;
fprintf('\n%s\n', repmat('=',1,76));
fprintf('  CoreColorStats_FULL\n');
fprintf('  출력 : %s\n', OPT.OUT_DIR);
fprintf('  원칙 : 풍화등급=JSON rw 필드 / 픽셀=raw 전체 / 보정없음\n');
fprintf('%s\n\n', repmat('=',1,76));
if ~exist(OPT.OUT_DIR,'dir'); mkdir(OPT.OUT_DIR); end

%% ===== [1] GPU / CPU 초기화 =====
fprintf('[1] GPU/CPU 초기화\n');
[gpuOK, gpuName] = initGPU(OPT.USE_GPU);
nWorkers = initParallel(OPT.USE_PAR);
fprintf('  GPU : %s\n  CPU : %d workers\n\n', gpuName, nWorkers);

%% ===== [2] 이미지 파일 수집 =====
fprintf('[2] 이미지 파일 수집\n');
files = collectFiles(OPT.IMAGE_ROOTS, OPT.IMG_EXTS);
nImg  = numel(files);
fprintf('  총 %d장\n\n', nImg);
if nImg == 0
    error('CoreColorStats:NoImages', 'IMAGE_ROOTS 에서 이미지를 찾지 못했습니다.');
end

%% ===== [3] JSON 인덱스 구성 =====
fprintf('[3] JSON 인덱싱\n');
jsonMap = buildJsonMap(OPT.JSON_ROOTS);
fprintf('  %d개 인덱싱 완료\n', jsonMap.Count);

% JSON 매칭 통계
nMatched = 0;
for fi = 1:nImg
    [~,st,~] = fileparts(files{fi});
    if jsonMap.isKey(lower(st)); nMatched = nMatched+1; end
end
fprintf('  이미지 매칭: %d / %d (%.1f%%)\n\n', ...
    nMatched, nImg, 100*nMatched/nImg);

%% ===== [4] 체크포인트 =====
ckptPath = fullfile(OPT.OUT_DIR, 'CKPT_rows.mat');
allRows  = cell(nImg,1);
startIdx = 1;
if exist(ckptPath,'file')
    try
        ck = load(ckptPath,'allRows','lastIdx');
        if numel(ck.allRows) == nImg
            allRows  = ck.allRows;
            startIdx = ck.lastIdx + 1;
            nDone    = nnz(~cellfun(@isempty,allRows));
            fprintf('[CKPT] %d / %d 재개\n\n', nDone, nImg);
        end
    catch ME
        warning(ME.identifier,'%s', sprintf('CKPT 로드 실패: %s', ME.message));
    end
end

%% ===== [5] 배치 처리 (GPU 색공간 + CPU parfor 통계) =====
fprintf('[4] 색상 통계 추출\n');
fprintf('  GPU 색공간 변환 배치(%d장) + CPU parfor 통계\n\n', OPT.GPU_BATCH);

todoIdx  = startIdx:nImg;
batchSz  = OPT.GPU_BATCH;
nBatch   = ceil(numel(todoIdx)/batchSz);
tLoop    = tic;

refL = OPT.REF_D1_L;
refa = OPT.REF_D1_a;
refb = OPT.REF_D1_b;

for bi = 1:nBatch
    bs   = (bi-1)*batchSz + 1;
    be   = min(bi*batchSz, numel(todoIdx));
    bIdx = todoIdx(bs:be);
    nB   = numel(bIdx);

    %% ── [5-1] 이미지 읽기 (parfor) ──────────────────────────────
    imgs   = cell(nB,1);
    stems  = cell(nB,1);
    fpaths = cell(nB,1);
    valid  = true(nB,1);

    parfor ii = 1:nB
        f = files{bIdx(ii)};
        try
            I = ensureRGB_uint8(imread(f));
            imgs{ii}   = I;
            fpaths{ii} = f;
            [~,st,~]   = fileparts(f);
            stems{ii}  = st;
        catch
            valid(ii) = false;
        end
    end

    %% ── [5-2] GPU 색공간 변환 ────────────────────────────────────
    hsvCache = cell(nB,1);
    labCache = cell(nB,1);

    for ii = 1:nB
        if ~valid(ii); continue; end
        try
            if gpuOK
                Ig          = gpuArray(single(imgs{ii})) / 255;
                hsvCache{ii} = double(gather(rgb2hsv(Ig)));
                labCache{ii} = double(gather(rgb2lab(Ig)));
            else
                Iflt         = im2single(imgs{ii});
                hsvCache{ii} = double(rgb2hsv(Iflt));
                labCache{ii} = double(rgb2lab(Iflt));
            end
        catch MEg
            % OOM → CPU 폴백
            if gpuOK && contains(MEg.message,'out of memory')
                try reset(gpuDevice()); catch; end
                gpuOK = false;
                warning('CoreColorStats:GPUOOM','%s','GPU OOM → CPU 폴백');
            end
            try
                Iflt         = im2single(imgs{ii});
                hsvCache{ii} = double(rgb2hsv(Iflt));
                labCache{ii} = double(rgb2lab(Iflt));
            catch
                valid(ii) = false;
            end
        end
    end

    %% ── [5-3] 통계 계산 (parfor) ────────────────────────────────
    bRows = cell(nB,1);

    parfor ii = 1:nB
        if ~valid(ii) || isempty(hsvCache{ii}); continue; end
        try
            bRows{ii} = buildRow( ...
                imgs{ii}, hsvCache{ii}, labCache{ii}, ...
                stems{ii}, fpaths{ii}, bIdx(ii), ...
                refL, refa, refb, jsonMap);
        catch ME2
            warning(ME2.identifier,'%s', ...
                sprintf('통계 실패 [%s]: %s', stems{ii}, ME2.message));
        end
    end

    for ii = 1:nB
        allRows{bIdx(ii)} = bRows{ii};
    end

    %% ── 진행 상황 ────────────────────────────────────────────────
    elapsed = toc(tLoop);
    nDone   = bIdx(end) - startIdx + 1;
    eta     = 0;
    if nDone > 0
        eta = elapsed/nDone * (numel(todoIdx) - nDone);
    end
    fprintf('  배치 %3d/%d | %5d/%5d장 | %.0fs 경과 | ETA %.0fs\n', ...
        bi, nBatch, bIdx(end), nImg, elapsed, eta);

    %% ── 체크포인트 ───────────────────────────────────────────────
    if mod(bi, max(1,floor(OPT.SAVE_EVERY/batchSz))) == 0 || bi == nBatch
        lastIdx = bIdx(end);
        save(ckptPath,'allRows','lastIdx','-v7.3');
    end
end

%% ===== [6] 테이블 조립 =====
fprintf('\n[5] 테이블 조립\n');
validRows = allRows(~cellfun(@isempty, allRows));
nValid    = numel(validRows);
fprintf('  유효 : %d / %d\n', nValid, nImg);

T = struct2table(vertcat(validRows{:}));
T = stringifyCols(T);

% JSON 미매칭 이미지 경고
nNoJson = nnz(T.json_matched == 0);
if nNoJson > 0
    fprintf('  ⚠ JSON 미매칭 이미지: %d장 (rw=unknown)\n', nNoJson);
end

% rw 분포 출력
[rwVals,~,rwIdx] = unique(T.rw);
for ri = 1:numel(rwVals)
    cnt = nnz(rwIdx==ri);
    fprintf('  rw=%-8s : %5d장 (%.1f%%)\n', ...
        rwVals(ri), cnt, 100*cnt/nValid);
end
fprintf('\n');

%% ===== [7] 집계 =====
fprintf('[6] 풍화등급별 집계\n');
statCols = getStatCols(T);
T_grade  = gradeAggregate(T, 'rw',       statCols);
T_humid  = gradeAggregate(T, 'humidity', statCols);
T_cross  = crossTable(T,    'rw',        'humidity');
fprintf('  완료\n\n');

%% ===== [8] Excel 분할 저장 =====
fprintf('[7] Excel 저장\n');
baseXlsx  = fullfile(OPT.OUT_DIR, 'FULL_COLOR_STATS');
xlsxFiles = writeExcelSplit(T, T_grade, T_humid, T_cross, ...
    baseXlsx, EXCEL_MAX_ROWS, EXCEL_MAX_COLS, OPT, nImg, nValid, toc(t0));

%% 완료
if exist(ckptPath,'file'); delete(ckptPath); end
elapsed = toc(t0);
fprintf('\n%s\n', repmat('=',1,76));
fprintf('  DONE | %.1f min | %d/%d장 | GPU=%s | Workers=%d\n', ...
    elapsed/60, nValid, nImg, tf2s(gpuOK), nWorkers);
for fi = 1:numel(xlsxFiles)
    fprintf('  Excel[%d]: %s\n', fi, xlsxFiles{fi});
end
fprintf('%s\n\n', repmat('=',1,76));

runInfo = struct('outDir',OPT.OUT_DIR,'xlsxFiles',{xlsxFiles}, ...
    'nTotal',nImg,'nValid',nValid,'gpuOK',gpuOK, ...
    'nWorkers',nWorkers,'elapsedMin',elapsed/60);
end

%% =========================================================================
%% buildRow — 이미지 1장의 전체 통계 행 생성
%% =========================================================================
function row = buildRow(I0, hsv3, lab3, stem, fpath, imgIdx, ...
    refL, refa, refb, jsonMap)

row = struct();
row.img_index = imgIdx;
row.file      = fpath;
row.stem      = stem;

%% ── [A] 파일명 파싱 (JSON 매칭 키 + 부가 메타) ──────────────────
fn             = parseFilename(stem);
row.fn_site    = fn.site;
row.fn_borehole= fn.borehole;
row.fn_box     = fn.box_no;
row.fn_rock1   = fn.rock1;
row.fn_rock2   = fn.rock2;
row.fn_angle   = fn.angle;
row.fn_rotation= fn.rotation;
row.fn_humid   = fn.humid;
row.fn_lux_bin = fn.lux_bin;
row.fn_seq     = fn.seq;

%% ── [B] JSON 메타데이터 (풍화등급 포함) ──────────────────────────
jm = readJsonByKey(lower(stem), jsonMap);
row.json_matched = jm.matched;   % 1=매칭성공 0=실패

% ★ 핵심: 풍화등급은 JSON rw 필드만 사용
%   JSON 없으면 'unknown' 으로 표시 (파일명 추측 안 함)
row.rw            = jm.rw;          % D1~D5, D0~5 범위 등 원본 그대로

row.humidity      = jm.humidity;    % 건조/습윤
row.object_id     = jm.object_id;
row.box_no        = jm.box_no;
row.year          = jm.year;
row.drilling_place1 = jm.drilling_place1;
row.drilling_place2 = jm.drilling_place2;
row.hole_num      = jm.hole_num;
row.row_num       = jm.row_num;
row.ers           = jm.ers;
row.camera_type   = jm.camera_type;
row.camera_angle  = jm.camera_angle;
row.rotation_angle= jm.rotation_angle;
row.shoot_height  = jm.shoot_height;
row.f_stop        = jm.f_stop;
row.shutter_speed = jm.shutter_speed;
row.iso           = jm.iso;
row.lux           = jm.lux;
row.location      = jm.location;
row.rock_depth    = jm.rock_depth;
row.rock_strength = jm.rock_strength;
row.tcr           = jm.tcr;
row.rqd           = jm.rqd;
row.rmr           = jm.rmr;
row.rq            = jm.rq;
row.rjr           = jm.rjr;
row.fm            = jm.fm;
row.geo_structure = jm.geo_structure;
row.rock_type1    = jm.rock_type1;
row.rock_type2    = jm.rock_type2;

%% ── [C] 이미지 기본 정보 ─────────────────────────────────────────
[H_px, W_px, ~] = size(I0);
row.height_px = H_px;
row.width_px  = W_px;
row.nPixels   = H_px * W_px;

%% ── [D] 채널 분리 ────────────────────────────────────────────────
Rf   = double(I0(:,:,1));
Gf   = double(I0(:,:,2));
Bf   = double(I0(:,:,3));

Hflt = hsv3(:,:,1);  % 0~1
Sflt = hsv3(:,:,2);
Vflt = hsv3(:,:,3);

Lflt = lab3(:,:,1);  % L: 0~100
aflt = lab3(:,:,2);  % a: -128~127
bflt = lab3(:,:,3);  % b: -128~127
Cflt = hypot(aflt, bflt);

%% ── [E] RGB (raw, 0~255) ─────────────────────────────────────────
row = addStats(row,'R',Rf);
row = addStats(row,'G',Gf);
row = addStats(row,'B',Bf);
row = addShape(row,'R',Rf);
row = addShape(row,'G',Gf);
row = addShape(row,'B',Bf);

%% ── [F] HSV ─────────────────────────────────────────────────────
row = addStats(row,'H',Hflt);
row = addStats(row,'S',Sflt);
row = addStats(row,'V',Vflt);
row = addShape(row,'V',Vflt);

Vv = Vflt(:);
row.dark_pct   = mean(Vv < 0.20) * 100;   % 흑운모·각섬석 추정
row.mid_pct    = mean(Vv >= 0.20 & Vv < 0.80) * 100;
row.bright_pct = mean(Vv >= 0.80) * 100;  % 장석·석영 추정

%% ── [G] HSL ─────────────────────────────────────────────────────
[Hhsl, Shsl, Lhsl] = rgb2hsl_vec(Rf, Gf, Bf);
row = addStats(row,'HSL_H', Hhsl);
row = addStats(row,'HSL_S', Shsl);
row = addStats(row,'HSL_L', Lhsl);

%% ── [H] CIELAB + LCh ────────────────────────────────────────────
row = addStats(row,'L',  Lflt);
row = addStats(row,'a',  aflt);
row = addStats(row,'b',  bflt);
row = addStats(row,'C',  Cflt);
row = addShape(row,'L',  Lflt);
row = addShape(row,'a',  aflt);
row = addShape(row,'b',  bflt);

% LCh 색상각
Lm = row.L_mean;  am = row.a_mean;  bm = row.b_mean;
row.LCh_h_mean = atan2d(bm, am);
hArr = atan2d(bflt(:), aflt(:));
row.LCh_h_p50  = median(hArr, 'omitnan');
row.LCh_h_std  = std(hArr, 0, 'omitnan');

%% ── [I] CIE XYZ (D65) ───────────────────────────────────────────
Rlin = srgb2linear(Rf/255);
Glin = srgb2linear(Gf/255);
Blin = srgb2linear(Bf/255);
Xc = 0.4124*Rlin + 0.3576*Glin + 0.1805*Blin;
Yc = 0.2126*Rlin + 0.7152*Glin + 0.0722*Blin;
Zc = 0.0193*Rlin + 0.1192*Glin + 0.9505*Blin;
row = addStats(row,'XYZ_X', Xc);
row = addStats(row,'XYZ_Y', Yc);
row = addStats(row,'XYZ_Z', Zc);

%% ── [J] YCbCr ───────────────────────────────────────────────────
Ycc =  0.299*Rf + 0.587*Gf + 0.114*Bf;
Cb  = Bf - Ycc;
Cr  = Rf - Ycc;
row = addStats(row,'YCC_Y',  Ycc);
row = addStats(row,'YCC_Cb', Cb);
row = addStats(row,'YCC_Cr', Cr);
row = addShape(row,'YCC_Cr', Cr);  % Cr: 철산화 민감

%% ── [K] CMYK ────────────────────────────────────────────────────
Rn = Rf/255; Gn = Gf/255; Bn = Bf/255;
Kc = 1 - max(max(Rn,Gn),Bn);
dK = max(1-Kc, 1e-6);
row = addStats(row,'CMYK_C', (1-Rn-Kc)./dK);
row = addStats(row,'CMYK_M', (1-Gn-Kc)./dK);
row = addStats(row,'CMYK_Y', (1-Bn-Kc)./dK);
row = addStats(row,'CMYK_K', Kc);   % K: 암색 광물 비율

%% ── [L] Opponent Color ──────────────────────────────────────────
row = addStats(row,'OPP_RG', Rf-Gf);          % 적-녹
row = addStats(row,'OPP_BY', Bf-(Rf+Gf)/2);  % 청-황
row = addStats(row,'OPP_WB', Rf+Gf+Bf);       % 밝기합

%% ── [M] Log-chromaticity (조명 변화 강인) ───────────────────────
Rp = max(Rf,1); Gp = max(Gf,1); Bp = max(Bf,1);
lmean = (log(Rp)+log(Gp)+log(Bp))/3;
row = addStats(row,'LC_R', log(Rp)-lmean);
row = addStats(row,'LC_G', log(Gp)-lmean);
row = addStats(row,'LC_B', log(Bp)-lmean);

%% ── [N] RGB 파생 ────────────────────────────────────────────────
Rm = row.R_mean; Gm = row.G_mean; Bm = row.B_mean;
sRGB = Rm+Gm+Bm;
row.RB_ratio     = Rm / max(Bm,1);          % 풍화 핵심
row.RG_ratio     = Rm / max(Gm,1);
row.GB_ratio     = Gm / max(Bm,1);
row.RB_ratio_p50 = row.R_p50 / max(row.B_p50,1);
row.R_norm       = Rm / max(sRGB,1);
row.G_norm       = Gm / max(sRGB,1);
row.B_norm       = Bm / max(sRGB,1);

%% ── [O] 텍스처 (그레이 기반) ───────────────────────────────────
try
    gray = double(rgb2gray(I0));
    gv   = gray(:);
    row.gray_std      = std(gv, 0);
    row.gray_entropy  = entropy(uint8(gray));
    row.gray_contrast = (max(gv)-min(gv)) / max(max(gv)+min(gv),1);
catch
    row.gray_std=NaN; row.gray_entropy=NaN; row.gray_contrast=NaN;
end

%% ── [P] Munsell 근사 ────────────────────────────────────────────
[mV, mC, mH_code, mH_name] = lab2munsell(Lm, am, bm);
row.munsell_V      = mV;
row.munsell_C      = mC;
row.munsell_H_code = mH_code;
row.munsell_H_name = mH_name;

% p50 기반 Munsell
[mVp, mCp, ~, mHnp] = lab2munsell(row.L_p50, row.a_p50, row.b_p50);
row.munsell_V_p50      = mVp;
row.munsell_C_p50      = mCp;
row.munsell_H_name_p50 = mHnp;

row.munsell_neutral = double(row.C_mean < 4.0);  % 1=무채색(N)

%% ── [Q] 풍화지수 ────────────────────────────────────────────────
eps1 = 1e-6;
Sm = row.S_mean;  Vm = row.V_mean;  Cm = row.C_mean;
Lp = row.L_p50;   ap = row.a_p50;   bp = row.b_p50;

% RI: Redness Index (Birkeland 1974)
row.RI  = Rm^2 / max(Bm*(Rm+Gm+Bm), eps1);

% CI: Color Index — (R-B)/(R+B)
row.CI     = (Rm-Bm) / max(Rm+Bm, eps1);
row.CI_p50 = (row.R_p50-row.B_p50) / max(row.R_p50+row.B_p50, eps1);

% NRI: Normalised Redness
row.NRI = Rm / max(sRGB, eps1);

% RBI: R-B Index  (-1~+1)
row.RBI = (Rm-Bm) / 255;

% SVI: Saturation-based Weathering Index
row.SVI = Sm * (1-Vm) * 100;

% WHI: Weathering Hue Index
row.WHI = bm / (abs(am)+1);

% CWI: Color Weathering Index
row.CWI = (bp+10) / max(Lp/10, eps1);

% YI: Yellowness Index (%)
row.YI = bm / max(Lm, eps1) * 100;

% SAI: Saturation/Lightness Index (%)
row.SAI = Cm / max(Lm, eps1) * 100;

% GRI: Gray Rock Index  (신선암 ≈ 1)
row.GRI = 1 - Cm/30;

% delta_E_D1: CIE76 색차 vs D1 기준색
row.delta_E_D1 = sqrt((Lm-refL)^2 + (am-refa)^2 + (bm-refb)^2);

% HI: Hue angle shift vs D1
row.HI = row.LCh_h_mean - atan2d(refb, refa);

% MWI: Munsell-based Weathering Index
row.MWI = (mC-0.5) / max(10-mV, eps1);

% CrI: YCbCr Cr/Cb 비율 (철산화)
row.CrI = row.YCC_Cr_mean / max(abs(row.YCC_Cb_mean), eps1);
end

%% =========================================================================
%% JSON 읽기 — 실제 필드명 정확히 일치
%% =========================================================================
function jmap = buildJsonMap(roots)
jmap = containers.Map('KeyType','char','ValueType','char');
for ri = 1:numel(roots)
    root = roots{ri};
    if ~exist(root,'dir'); continue; end
    dd = dir(fullfile(root,'**','*.json'));
    for di = 1:numel(dd)
        [~,st,~] = fileparts(dd(di).name);
        key = lower(st);
        if ~jmap.isKey(key)
            jmap(key) = fullfile(dd(di).folder, dd(di).name);
        end
    end
end
end

function jm = readJsonByKey(key, jsonMap)
% 반환 구조체 초기화 (모든 필드 포함)
jm = struct( ...
    'matched',0, ...
    'object_id','', 'width',NaN, 'height',NaN, 'box_no',NaN, ...
    'year',NaN, 'drilling_place1','', 'drilling_place2','', ...
    'hole_num','', 'row_num',NaN, 'ers','', ...
    'camera_type','', 'camera_angle',NaN, 'rotation_angle','', ...
    'shoot_height','', 'f_stop',NaN, 'shutter_speed','', ...
    'iso',NaN, 'lux',NaN, 'location','', ...
    'humidity','', ...
    'rock_depth',NaN, 'rock_strength','', ...
    'tcr','', 'rqd','', 'rmr',NaN, 'rq','', 'rjr','', 'fm','', ...
    'rw','unknown', ...         % ← 기본값 'unknown'
    'geo_structure','', ...
    'rock_type1','', 'rock_type2','');

if ~jsonMap.isKey(key); return; end

try
    txt = fileread(jsonMap(key));
    j   = jsondecode(txt);
    jm.matched = 1;

    % ── 문자열 필드 ──────────────────────────────────────────────
    strFields = { ...
        'object_id',       'object_id'
        'drilling_place1', 'drilling_place1'
        'drilling_place2', 'drilling_place2'
        'hole_num',        'hole_num'
        'ers',             'ers'
        'camera_type',     'camera_type'
        'rotation_angle',  'rotation_angle'
        'shoot_height',    'shoot_height'
        'shutter_speed',   'shutter_speed'
        'location',        'location'
        'humidity',        'humidity'
        'rock_strength',   'rock_strength'
        'tcr',             'tcr'
        'rqd',             'rqd'
        'rq',              'rq'
        'rjr',             'rjr'
        'fm',              'fm'
        'rw',              'rw'         % ★ 풍화등급
        'geo_structure',   'geo_structure'
    };
    for fi = 1:size(strFields,1)
        jField = strFields{fi,2};
        sField = strFields{fi,1};
        if isfield(j, jField) && ~isempty(j.(jField))
            jm.(sField) = char(string(j.(jField)));
        end
    end

    % ── 숫자 필드 ────────────────────────────────────────────────
    numFields = { ...
        'width',        'width'
        'height',       'height'
        'box_no',       'box_no'
        'year',         'year'
        'row_num',      'row_num'
        'camera_angle', 'camera_angle'
        'rock_depth',   'rock_depth'
        'rmr',          'rmr'
        'lux',          'lux'
    };
    for fi = 1:size(numFields,1)
        jField = numFields{fi,2};
        sField = numFields{fi,1};
        if isfield(j, jField)
            v = j.(jField);
            if isnumeric(v) && isscalar(v)
                jm.(sField) = double(v);
            elseif ischar(v)||isstring(v)
                jm.(sField) = str2double(v);
            end
        end
    end

    % ── 특수 필드 (실제 JSON 필드명이 MATLAB 변수명과 다름) ──────
    % "f-stop"  → jm.f_stop
    if isfield(j,'f_stop')        % jsondecode 가 - 를 _ 로 변환
        jm.f_stop = safeNum(j.f_stop);
    elseif isfield(j,'fstop')
        jm.f_stop = safeNum(j.fstop);
    end

    % "ISO" 대문자
    if isfield(j,'ISO')
        jm.iso = safeNum(j.ISO);
    elseif isfield(j,'iso')
        jm.iso = safeNum(j.iso);
    end

    % ── rock_type 배열  [{rock_type1, rock_type2}] ───────────────
    if isfield(j,'rock_type')
        rt = j.rock_type;
        if isstruct(rt); rt = num2cell(rt); end
        for ri = 1:numel(rt)
            e = rt{ri};
            % 실제 JSON 필드명: "rock_type1", "rock_type2"
            if isfield(e,'rock_type1') && ~isempty(e.rock_type1)
                jm.rock_type1 = char(string(e.rock_type1));
            end
            if isfield(e,'rock_type2') && ~isempty(e.rock_type2)
                jm.rock_type2 = char(string(e.rock_type2));
            end
        end
    end

catch ME
    warning(ME.identifier,'%s', sprintf('JSON 파싱 오류 [%s]: %s', key, ME.message));
    jm.matched = 0;
end
end

function v = safeNum(raw)
if isnumeric(raw) && isscalar(raw); v = double(raw);
elseif ischar(raw)||isstring(raw);  v = str2double(raw);
else;                                v = NaN;
end
end

%% =========================================================================
%% 파일명 파싱 (JSON 매칭 보조 + 부가 메타)
%% =========================================================================
function fn = parseFilename(stem)
fn = struct('site','','borehole','','box_no','','rock1','','rock2','', ...
    'angle',NaN,'rotation','','humid','','lux_bin',NaN,'seq',NaN,'rw_guess','');

% 괄호 내용 추출
pTokens = regexp(stem, '\(([^)]*)\)', 'tokens');
clean   = regexprep(stem, '\([^)]*\)', '(X)');
parts   = strsplit(clean, '-');
if numel(parts) < 8; return; end

fn.site     = strtrim(parts{1});
fn.borehole = regexprep(strtrim(parts{2}), '\(X\)', '');
if numel(pTokens) >= 1; fn.box_no = pTokens{1}{1}; end
fn.rock1    = strtrim(parts{3});
fn.rock2    = strtrim(parts{4});

aStr = strtrim(parts{5});
aNum = regexp(aStr, '^\d+', 'match', 'once');
if ~isempty(aNum); fn.angle = str2double(aNum); end
if numel(pTokens) >= 2; fn.rotation = pTokens{2}{1}; end

h = upper(strtrim(parts{6}));
if strcmp(h,'W'); fn.humid='습윤'; elseif strcmp(h,'D'); fn.humid='건조'; else; fn.humid=h; end

fn.lux_bin = str2double(strtrim(parts{7}));
fn.seq      = str2double(strtrim(parts{8}));
end

%% =========================================================================
%% Munsell 근사 변환
%% =========================================================================
function [V, C, H_code, H_name] = lab2munsell(L, a, b)
% Value: L* → Munsell V (0~10)
V = max(0, min(10, L/10));

% Chroma: C* → Munsell C
C_star = hypot(a, b);
C = max(0, C_star / 5.5);

% 무채색
if C_star < 4.0
    H_code = -1; H_name = 'N'; return;
end

% Hue: hue angle → Munsell hue name
h = mod(atan2d(b, a), 360);

% 화강암 풍화 관련 주요 구간 포함
hueMap = [
%  hMin  hMax  code   name
     0    9     2.5   '5R'
     9   18     7.5   '10R'
    18   27    12.5   '2.5YR'
    27   45    20.0   '5YR'     % D4~D5 핵심 (철산화)
    45   63    54.0   '7.5YR'
    63   80    72.0   '10YR'    % D3~D4
    80   99    90.0   '2.5Y'
    99  116   108.0   '5Y'
   116  134   126.0   '10Y'
   134  152   144.0   '5GY'
   152  170   162.0   '10GY'
   170  188   180.0   '5G'
   188  207   198.0   '10G'
   207  225   216.0   '5BG'
   225  243   234.0   '10BG'
   243  261   252.0   '5B'
   261  279   270.0   '10B'
   279  297   288.0   '5PB'    % D1 신선 화강암 (청회색)
   297  315   306.0   '10PB'
   315  333   324.0   '5P'
   333  351   342.0   '10P'
   351  360     2.5   '5RP'
];

H_code = 2.5; H_name = '5R';
for ti = 1:size(hueMap,1)
    if h >= hueMap(ti,1) && h < hueMap(ti,2)
        H_code = hueMap(ti,3);
        H_name = hueMap(ti,4);
        return;
    end
end
end

%% =========================================================================
%% 색공간 변환
%% =========================================================================
function [H,S,L] = rgb2hsl_vec(R,G,B)
Rn=R/255; Gn=G/255; Bn=B/255;
Cmax=max(max(Rn,Gn),Bn); Cmin=min(min(Rn,Gn),Bn);
delta=Cmax-Cmin;
L=(Cmax+Cmin)/2;
S=zeros(size(L));
mk=delta>0;
S(mk)=delta(mk)./(1-abs(2*L(mk)-1));
H=zeros(size(L));
rm=mk&(Cmax==Rn); gm=mk&(Cmax==Gn); bm=mk&(Cmax==Bn);
H(rm)=mod((Gn(rm)-Bn(rm))./delta(rm),6)/6;
H(gm)=((Bn(gm)-Rn(gm))./delta(gm)+2)/6;
H(bm)=((Rn(bm)-Gn(bm))./delta(bm)+4)/6;
H=mod(H,1);
end

function out = srgb2linear(v)
out=zeros(size(v));
lo=v<=0.04045;
out(lo)=v(lo)/12.92;
out(~lo)=((v(~lo)+0.055)/1.055).^2.4;
end

%% =========================================================================
%% 채널 통계
%% =========================================================================
function row = addStats(row, ch, X)
v=X(:); v=v(isfinite(v));
if isempty(v); v=0; end
row.([ch '_mean'])=mean(v);
row.([ch '_std']) =std(v,0);
row.([ch '_min']) =min(v);
row.([ch '_max']) =max(v);
pcts=prctile(v,[5 25 50 75 95]);
row.([ch '_p05'])=pcts(1); row.([ch '_p25'])=pcts(2);
row.([ch '_p50'])=pcts(3); row.([ch '_p75'])=pcts(4);
row.([ch '_p95'])=pcts(5);
row.([ch '_iqr'])=pcts(4)-pcts(2);
row.([ch '_ipr'])=pcts(5)-pcts(1);
end

function row = addShape(row, ch, X)
v=X(:); v=v(isfinite(v));
if numel(v)<4
    row.([ch '_skew'])=NaN; row.([ch '_kurt'])=NaN;
    row.([ch '_entropy'])=NaN; return;
end
mu=mean(v); sig=std(v,0);
if sig<1e-10
    row.([ch '_skew'])=0; row.([ch '_kurt'])=0;
else
    zv=(v-mu)/sig;
    row.([ch '_skew'])=mean(zv.^3);
    row.([ch '_kurt'])=mean(zv.^4)-3;
end
nBin=64;
edges=linspace(min(v),max(v)+1e-9,nBin+1);
cnt=histcounts(v,edges); p=cnt/max(sum(cnt),1); p=p(p>0);
row.([ch '_entropy'])=-sum(p.*log2(p));
end

%% =========================================================================
%% 집계 / 교차표
%% =========================================================================
function statCols = getStatCols(T)
allCols=T.Properties.VariableNames;
skipPfx={'img_index','file','stem','fn_','rw','humidity','json_matched', ...
    'object_id','box_no','year','drilling_','hole_num','row_num','ers', ...
    'camera_','rotation_','shoot_','f_stop','shutter_','iso','lux', ...
    'location','rock_depth','rock_strength','tcr','rqd','rmr','rq', ...
    'rjr','fm','geo_','rock_type','height_px','width_px','nPixels', ...
    'munsell_H_name','munsell_neutral'};
statCols={};
for ci=1:numel(allCols)
    col=allCols{ci};
    skip=false;
    for pi=1:numel(skipPfx)
        if startsWith(col,skipPfx{pi}); skip=true; break; end
    end
    if skip; continue; end
    if ~isnumeric(T.(col)); continue; end
    statCols{end+1}=col; %#ok<AGROW>
end
end

function Tagg = gradeAggregate(T, groupCol, statCols)
if ~any(strcmp(T.Properties.VariableNames,groupCol)); Tagg=table(); return; end
grps=unique(T.(groupCol));
grps=grps(strlength(strtrim(string(grps)))>0); grps=sort(grps);
outRows=cell(numel(grps),1);
for gi=1:numel(grps)
    g=grps(gi); mk=(T.(groupCol)==g); sub=T(mk,:);
    r=struct(); r.(groupCol)=char(string(g)); r.n=nnz(mk);
    for ci=1:numel(statCols)
        col=statCols{ci};
        if ~any(strcmp(sub.Properties.VariableNames,col)); continue; end
        vals=sub.(col); vals=vals(isfinite(vals));
        if isempty(vals)
            r.([col '_mean'])=NaN; r.([col '_std'])=NaN;
            r.([col '_p05'])=NaN;  r.([col '_p50'])=NaN; r.([col '_p95'])=NaN;
        else
            r.([col '_mean'])=mean(vals); r.([col '_std'])=std(vals,0);
            pcts=prctile(vals,[5 50 95]);
            r.([col '_p05'])=pcts(1); r.([col '_p50'])=pcts(2); r.([col '_p95'])=pcts(3);
        end
    end
    outRows{gi}=r;
end
valid=~cellfun(@isempty,outRows);
if ~any(valid); Tagg=table(); return; end
Tagg=struct2table(vertcat(outRows{valid}));
end

function Tc = crossTable(T,rv,cv)
if ~any(strcmp(T.Properties.VariableNames,rv))||~any(strcmp(T.Properties.VariableNames,cv))
    Tc=table(); return; end
rV=unique(T.(rv)); rV=sort(rV(strlength(strtrim(string(rV)))>0));
cV=unique(T.(cv)); cV=sort(cV(strlength(strtrim(string(cV)))>0));
data=zeros(numel(rV),numel(cV));
for ri=1:numel(rV)
    for ci=1:numel(cV)
        data(ri,ci)=nnz(T.(rv)==rV(ri)&T.(cv)==cV(ci));
    end
end
Tc=array2table(data,'RowNames',cellstr(string(rV)), ...
    'VariableNames',matlab.lang.makeValidName(cellstr(string(cV))));
Tc=[table(rV,'VariableNames',{rv}),Tc]; Tc.total=sum(data,2);
end

%% =========================================================================
%% Excel 분할 저장
%% =========================================================================
function xlsxFiles = writeExcelSplit(T,Tg,Th,Tc, ...
    basePath,maxRows,maxCols,OPT,nImg,nValid,elapsed)

xlsxFiles={};
nRow=height(T); nCol=width(T);
nRowParts=ceil(nRow/maxRows);

if nCol<=maxCols
    for pi=1:nRowParts
        r1=(pi-1)*maxRows+1; r2=min(pi*maxRows,nRow);
        fp=basePath; if nRowParts>1; fp=basePath+sprintf("_part%03d",pi); end
        fp=fp+".xlsx";
        fprintf('  %s  (행 %d~%d, 열 %d개)\n',fp,r1,r2,nCol);
        wSheet(T(r1:r2,:),fp,'PER_IMAGE');
        if pi==1
            wSheet(Tg,fp,'GRADE_STATS');
            wSheet(Th,fp,'HUMID_STATS');
            wSheet(Tc,fp,'RW_x_HUMID');
        end
        wSheet(makeIndex(OPT,nImg,nValid,elapsed,pi,nRowParts),fp,'INDEX');
        xlsxFiles{end+1}=char(fp); %#ok<AGROW>
    end
else
    % 열도 분할
    metaC=getMetaCols(T);
    dataC=setdiff(T.Properties.VariableNames,metaC,'stable');
    chunkC=maxCols-numel(metaC);
    nColParts=ceil(numel(dataC)/chunkC);
    fidx=0;
    for ri=1:nRowParts
        r1=(ri-1)*maxRows+1; r2=min(ri*maxRows,nRow);
        for ci=1:nColParts
            c1=(ci-1)*chunkC+1; c2=min(ci*chunkC,numel(dataC));
            fidx=fidx+1;
            fp=basePath+sprintf("_r%03d_c%03d.xlsx",ri,ci);
            fprintf('  %s  (행%d~%d 열%d~%d)\n',fp,r1,r2,c1,c2);
            Tsub=T(r1:r2,[metaC,dataC(c1:c2)]);
            wSheet(Tsub,fp,'PER_IMAGE');
            if fidx==1
                wSheet(Tg,fp,'GRADE_STATS');
                wSheet(Th,fp,'HUMID_STATS');
                wSheet(Tc,fp,'RW_x_HUMID');
            end
            wSheet(makeIndex(OPT,nImg,nValid,elapsed,fidx, ...
                nRowParts*nColParts),fp,'INDEX');
            xlsxFiles{end+1}=char(fp); %#ok<AGROW>
        end
    end
end
end

function metaCols = getMetaCols(T)
allCols=T.Properties.VariableNames;
pfx={'img_index','file','stem','fn_','rw','humidity','json_matched', ...
    'object_id','box_no','year','drilling_','hole_num','row_num','ers', ...
    'camera_','rotation_','shoot_','f_stop','shutter_','iso','lux', ...
    'location','rock_depth','rock_strength','tcr','rqd','rmr','rq', ...
    'rjr','fm','geo_','rock_type','height_px','width_px','nPixels'};
metaCols={};
for ci=1:numel(allCols)
    col=allCols{ci};
    for pi=1:numel(pfx)
        if startsWith(col,pfx{pi}); metaCols{end+1}=col; break; end %#ok<AGROW>
    end
end
metaCols=unique(metaCols,'stable');
end

function Ti = makeIndex(OPT,nImg,nValid,elapsed,part,nParts)
items={
    'timestamp',   char(string(datetime('now','Format','yyyy-MM-dd HH:mm:ss')))
    'part',        sprintf('%d / %d',part,nParts)
    'n_images',    num2str(nImg)
    'n_valid',     num2str(nValid)
    'elapsed_min', num2str(elapsed/60,'%.1f')
    'use_gpu',     tf2s(OPT.USE_GPU)
    'use_parfor',  tf2s(OPT.USE_PAR)
    'gpu_batch',   num2str(OPT.GPU_BATCH)
    'rw_source',   'JSON rw 필드 (파일명 파싱 미사용)'
    'pixel_policy','raw 전체 픽셀 / 보정·마스킹 없음'
    'ref_D1',      sprintf('L=%.2f a=%.2f b=%.2f',OPT.REF_D1_L,OPT.REF_D1_a,OPT.REF_D1_b)
};
Ti=cell2table(items,'VariableNames',{'parameter','value'});
end

%% =========================================================================
%% 유틸
%% =========================================================================
function [gpuOK,name] = initGPU(want)
gpuOK=false; name='없음(CPU)';
if ~want; return; end
try
    parallel.gpu.enableCUDAForwardCompatibility(true);
catch; end
try
    g=gpuDevice(1);
    fprintf('  GPU: %s | %.2f / %.2f GB | CC %s\n', ...
        g.Name,g.AvailableMemory/1e9,g.TotalMemory/1e9,string(g.ComputeCapability));
    gpuOK=true; name=g.Name;
catch ME
    warning(ME.identifier,'%s',sprintf('GPU 없음: %s',ME.message));
end
end

function nW = initParallel(want)
nW=0;
if ~want; return; end
try
    if license('test','Distrib_Computing_Toolbox')
        p=gcp('nocreate');
        if isempty(p)
            p=parpool('Processes');
        end
        nW=p.NumWorkers;
        fprintf('  parpool: %d workers\n',nW);
    end
catch ME
    warning(ME.identifier,'%s',sprintf('parpool 실패: %s',ME.message));
end
end

function files = collectFiles(roots,exts)
files={};
for ri=1:numel(roots)
    root=roots{ri};
    if ~exist(root,'dir'); continue; end
    dd=dir(fullfile(root,'**','*'));
    for di=1:numel(dd)
        if dd(di).isdir; continue; end
        [~,~,e]=fileparts(dd(di).name);
        if any(strcmpi(e,exts))
            files{end+1,1}=fullfile(dd(di).folder,dd(di).name); %#ok<AGROW>
        end
    end
end
files=unique(files,'stable');
end

function I = ensureRGB_uint8(I)
if isempty(I); error('CoreColorStats:Empty','빈 이미지'); end
if ismatrix(I);      I=repmat(I,1,1,3);
elseif size(I,3)==4; I=I(:,:,1:3);
elseif size(I,3)~=3; error('CoreColorStats:BadCh','채널=%d',size(I,3)); end
if ~isa(I,'uint8'); I=im2uint8(I); end
end

function T = stringifyCols(T)
pfx={'file','stem','fn_','rw','humidity','json_','object_id','hole_num', ...
    'rock_strength','tcr','rqd','rq','rjr','fm','geo_','rock_type', ...
    'camera_','location','drilling_','ers','rotation_','shoot_', ...
    'shutter_','munsell_H_name'};
for ci=1:width(T)
    col=T.Properties.VariableNames{ci};
    for pi=1:numel(pfx)
        if startsWith(col,pfx{pi})
            try T.(col)=string(T.(col)); catch; end
            break;
        end
    end
end
end

function wSheet(T,fpath,sheet)
if isempty(T); return; end
try
    writetable(T,fpath,'Sheet',sheet,'WriteRowNames',false);
catch ME
    warning(ME.identifier,'%s',sprintf('[%s] 저장 실패: %s',sheet,ME.message));
end
end

function s = tf2s(v)
if v; s='ON'; else; s='OFF'; end
end