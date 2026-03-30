function runInfo = CoreWeatheringPaperKit_OneFile2028(varargin)
% CoreWeatheringPaperKit_OneFile2028 (ONE-FILE, image-only, RGB_raw + sRGB_norm, R2025a/b)
% ------------------------------------------------------------------------------
% 핵심 변경(요구 반영)
%   1) JSON 사용 안 함 (이미지만 분석)
%   2) RGB를 "두 체계"로 동시에 유지
%       - RGB_raw  : uint8(0~255)  (시각화/참고 통계)
%       - RGB_srgb : single(0~1)   (모든 정량 분석의 기준 입력)
%   3) Lab/HSV/ΔE/WHI/SVI/D-grade/texture는 RGB_srgb 기반으로만 계산
%   4) 분포 기반 통계(p05/p50/p95/IPR) 기본 제공
%
% 실행 예
%   runInfo = CoreWeatheringPaperKit_OneFile2028(); % UI로 선택 후 실행
%   runInfo = CoreWeatheringPaperKit_OneFile2028('MODE','folder','IMAGE_DIR',"C:\data\cores",'RECURSIVE',true);

OPT = local_parse_opts(varargin{:});

%% ---- setup out dirs ----
tstamp  = string(datetime('now','Format','yyyyMMdd_HHmmss'));
runName = string(OPT.RUN_NAME); if strlength(runName)==0, runName = "Run_" + tstamp; end
baseDir = string(OPT.OUT_BASEDIR); if strlength(baseDir)==0, baseDir = string(pwd); end

outDir = string(fullfile(baseDir, runName));
pngDir = string(fullfile(outDir, "png"));
matDir = string(fullfile(outDir, "mat"));
csvDir = string(fullfile(outDir, "csv"));
perDir = string(fullfile(outDir, "per_image"));
logDir = string(fullfile(outDir, "log"));

[outDir,pngDir,matDir,csvDir,perDir,logDir] = local_mkdir_or_fallback(outDir,pngDir,matDir,csvDir,perDir,logDir);

% diary log
try
    diary off;
    diary(char(fullfile(logDir, "run_log.txt")));
    diary on;
catch
end

fprintf("[INFO] Output root: %s\n", outDir);

%% ---- collect files ----
files = local_resolve_files_by_mode(OPT, outDir);
files = string(files(:));
files(files=="") = [];
files = unique(files, "stable");

if isempty(files)
    error("분석할 이미지가 없습니다.");
end
fprintf("[INFO] Found %d image(s).\n", numel(files));
disp(files(1:min(5,numel(files))));

%% ---- global accumulators ----
sumCell = cell(numel(files),1);
skippedFiles = strings(0,1);

%% ---- main loop ----
for ii = 1:numel(files)
    f = files(ii);
    [~, stem, ext] = fileparts(f);
    baseStem = string(stem);

    % 충돌 방지: index prefix
    itemId = string(sprintf("%06d_%s", ii, baseStem));
    itemRoot = string(fullfile(perDir, itemId));
    itemPng  = string(fullfile(itemRoot, "png"));
    itemCsv  = string(fullfile(itemRoot, "csv"));
    itemMat  = string(fullfile(itemRoot, "mat"));
    itemFig  = string(fullfile(itemRoot, "fig"));

    local_ensure_dirs({itemRoot,itemPng,itemCsv,itemMat,itemFig});

    fprintf("\n[INFO] (%d/%d) %s\n", ii, numel(files), f);

    % ---- read ----
    try
        I0 = imread(f);
    catch ME
        warning("CoreWeathering:imreadFail","%s", sprintf("읽기 실패 → 스킵: %s | %s", f, ME.message));
        skippedFiles(end+1,1) = f; %#ok<AGROW>
        continue;
    end

    % ---- standardize channel ----
    I0 = local_force_rgb(I0);

    % ---- standardize size (optional) ----
    if OPT.RESIZE_TARGETW > 0 && OPT.RESIZE_TARGETH > 0
        I0 = imresize(I0, [OPT.RESIZE_TARGETH OPT.RESIZE_TARGETW], "bilinear");
    elseif OPT.RESIZE_MAXW > 0 && size(I0,2) > OPT.RESIZE_MAXW
        scale = OPT.RESIZE_MAXW / size(I0,2);
        I0 = imresize(I0, scale, "bilinear");
    end

    % ---- full frame mask ----
    mask = true(size(I0,1), size(I0,2));

    % ---- RGB dual system (핵심) ----
    RGB_raw  = wx_rgb_uint8(I0);   % uint8 0-255 (시각/참고 통계)
    RGB_srgb = wx_srgb01(I0);      % single 0-1  (모든 정량 분석 입력)

    % ---- save original copy ----
    if OPT.SAVE_ORIGINAL_COPY
        try
            imwrite(RGB_raw, fullfile(itemPng, "original.png"));
        catch
        end
    end

    % ---- compute spaces (analysis uses RGB_srgb only) ----
    HSV = rgb2hsv(RGB_srgb);
    Hc = HSV(:,:,1); Sc = HSV(:,:,2); Vc = HSV(:,:,3);

    try
        Lab = rgb2lab(RGB_srgb);
    catch ME
        warning("CoreWeathering:rgb2labFail","%s", sprintf("rgb2lab 실패 → 스킵: %s | %s", f, ME.message));
        skippedFiles(end+1,1) = f; %#ok<AGROW>
        continue;
    end
    Lc = Lab(:,:,1);
    ac = Lab(:,:,2);
    bc = Lab(:,:,3);

    % ---- vectors ----
    Rraw = RGB_raw(:,:,1); Graw = RGB_raw(:,:,2); Braw = RGB_raw(:,:,3);
    Rrv = single(Rraw(mask)); Grv = single(Graw(mask)); Brv = single(Braw(mask)); % raw stats

    Rs = RGB_srgb(:,:,1); Gs = RGB_srgb(:,:,2); Bs = RGB_srgb(:,:,3);
    Rsv = single(Rs(mask)); Gsv = single(Gs(mask)); Bsv = single(Bs(mask));       % srgb stats

    Hv = single(Hc(mask)); Sv = single(Sc(mask)); Vv = single(Vc(mask));
    Lv = single(Lc(mask)); av = single(ac(mask)); bv = single(bc(mask));

    % ---- deltaE ab to (a50,b50) ----
    a50 = median(av,'omitnan');
    b50 = median(bv,'omitnan');
    dE = sqrt((ac - a50).^2 + (bc - b50).^2);
    dEv = single(dE(mask));

    % robust normalize dE (5-95)
    dE05 = prctile(dEv,5);
    dE95 = prctile(dEv,95);
    dEn = (dE - dE05) ./ max(eps, (dE95 - dE05));
    dEn = min(max(dEn,0),1);
    dEn(~mask) = 0;

    % ---- WHI / SVI (기준: RGB_srgb 기반) ----
    Ln   = local_norm_by_prc(Lc,  mask, 5,95);
    bn   = local_norm_by_prc(bc,  mask, 5,95);
    dEn2 = local_norm_by_prc(dEn, mask, 5,95);

    weatherScore = 0.50*(1 - Ln) + 0.30*bn + 0.20*dEn2;
    weatherScore(~mask) = 0;
    WHI = local_norm_by_prc(weatherScore, mask, 5,95);

    Sn = local_norm_by_prc(Sc, mask, 5,95);
    Vn = local_norm_by_prc(Vc, mask, 5,95);
    SVI = 0.55*Sn + 0.45*(1 - Vn);
    SVI = local_norm_by_prc(SVI, mask, 5,95);

    % ---- D-grade ----
    edges = OPT.GRADE_EDGES;
    D = ones(size(WHI),'single');
    D(WHI > edges(1)) = 2;
    D(WHI > edges(2)) = 3;
    D(WHI > edges(3)) = 4;
    D(WHI > edges(4)) = 5;
    D(~mask) = 0;

    % ---- tiles + textures ----
    tile = max(8, round(OPT.TILE));

    [tileMedL, tileIqrL] = local_tile_stats(Lc, mask, tile);
    [tileMedb, tileIqrb] = local_tile_stats(bc, mask, tile);
    [tileMedS, tileIqrS] = local_tile_stats(Sc, mask, tile);
    [tileMedV, tileIqrV] = local_tile_stats(Vc, mask, tile);

    gray = rgb2gray(RGB_srgb); % 분석용은 srgb 기반
    BWedge = edge(gray,"Sobel");
    edgeDensity = local_tile_mean(single(BWedge).*single(mask), mask, tile);
    localVar    = local_tile_mean(local_local_var(gray).*single(mask), mask, tile);

    % ---- per-pixel texture/derivative maps ----
    tex = local_texture_pack(Ln, bc, dEn, mask, OPT);

    tileGradL  = local_tile_mean(tex.gradL .* single(mask), mask, tile);
    tileGradb  = local_tile_mean(tex.gradb .* single(mask), mask, tile);
    tileGradDE = local_tile_mean(tex.graddEn .* single(mask), mask, tile);

    tileEntL9  = local_tile_mean(tex.entL .* single(mask), mask, tile);
    tileLoG    = local_tile_mean(tex.logL .* single(mask), mask, tile);

    wlist  = OPT.TEXTURE_WINS;
    wSmall = wlist(1);
    wLarge = wlist(end);

    tileStdSmall = local_tile_mean(tex.(sprintf("stdL_w%d",wSmall)) .* single(mask), mask, tile);
    tileStdLarge = local_tile_mean(tex.(sprintf("stdL_w%d",wLarge)) .* single(mask), mask, tile);
    tileRngLarge = local_tile_mean(tex.(sprintf("rngL_w%d",wLarge)) .* single(mask), mask, tile);

    %% ================= SAVE: per-image CSV =================
    % RGB_raw stats (uint8)
    Trgb_raw = local_stats_table_one("RGB_raw_uint8", ["R","G","B"], {Rrv,Grv,Brv});
    local_writetable_safe(Trgb_raw, fullfile(itemCsv,"rgb_raw_uint8_stats.csv"));

    % RGB_srgb stats (0-1)
    Trgb_srgb = local_stats_table_one("RGB_srgb_01", ["R","G","B"], {Rsv,Gsv,Bsv});
    local_writetable_safe(Trgb_srgb, fullfile(itemCsv,"rgb_srgb01_stats.csv"));

    % HSV stats
    Thsv = local_stats_table_one("HSV_from_srgb", ["H","S","V"], {Hv,Sv,Vv});
    local_writetable_safe(Thsv, fullfile(itemCsv,"hsv_stats.csv"));

    % Lab stats
    Tlab = local_stats_table_one("CIELAB_from_srgb", ["L","a","b"], {Lv,av,bv});
    local_writetable_safe(Tlab, fullfile(itemCsv,"cielab_stats.csv"));

    % Indices stats
    Tidx = local_stats_table_one("INDICES", ["dE","dEn","WHI","SVI","weatherScore"], ...
        {dEv, single(dEn(mask)), single(WHI(mask)), single(SVI(mask)), single(weatherScore(mask))});
    local_writetable_safe(Tidx, fullfile(itemCsv,"indices_stats.csv"));

    % Texture stats
    Ttex = local_stats_table_one("TEXTURE", ["gradL","gradb","graddEn","entL","logL"], ...
        {single(tex.gradL(mask)), single(tex.gradb(mask)), single(tex.graddEn(mask)), single(tex.entL(mask)), single(tex.logL(mask))});
    local_writetable_safe(Ttex, fullfile(itemCsv,"texture_stats.csv"));

    % Multi-scale std/range stats
    for ww = OPT.TEXTURE_WINS
        nameStd = sprintf("stdL_w%d", ww);
        nameRng = sprintf("rngL_w%d", ww);
        Tms = local_stats_table_one("MS_TEXTURE", [string(nameStd) string(nameRng)], ...
            {single(tex.(nameStd)(mask)), single(tex.(nameRng)(mask))});
        local_writetable_safe(Tms, fullfile(itemCsv, sprintf("texture_multiscale_w%d.csv", ww)));
    end

    % D-grade fractions
    Tdfrac = local_grade_fraction_table(D, mask, edges);
    local_writetable_safe(Tdfrac, fullfile(itemCsv,"dgrade_fraction.csv"));

    % tiles summary
    Ttile = local_tilegrid_summary_table(tileMedL,tileIqrL,tileMedb,tileIqrb,tileMedS,tileIqrS,tileMedV,tileIqrV,edgeDensity,localVar,tile);
    local_writetable_safe(Ttile, fullfile(itemCsv,"tiles_texture_summary.csv"));

    % Extra tile summary
    Ttile2 = local_texture_tilegrid_summary_table(tileGradL,tileGradb,tileGradDE,tileEntL9,tileLoG,tileStdSmall,tileStdLarge,tileRngLarge,tile,wSmall,wLarge);
    local_writetable_safe(Ttile2, fullfile(itemCsv,"tiles_texture_extra_summary.csv"));

    %% ================= SAVE: per-image PNG =================
    if OPT.SAVE_MAPS
        % RGB_raw maps
        local_save_map(single(Rraw), "R raw (uint8)", fullfile(itemPng,"MAP_R_raw.png"), [0 255]);
        local_save_map(single(Graw), "G raw (uint8)", fullfile(itemPng,"MAP_G_raw.png"), [0 255]);
        local_save_map(single(Braw), "B raw (uint8)", fullfile(itemPng,"MAP_B_raw.png"), [0 255]);

        % sRGB maps (0-1)
        local_save_map(single(Rs), "R sRGB (0-1)", fullfile(itemPng,"MAP_R_srgb.png"), [0 1]);
        local_save_map(single(Gs), "G sRGB (0-1)", fullfile(itemPng,"MAP_G_srgb.png"), [0 1]);
        local_save_map(single(Bs), "B sRGB (0-1)", fullfile(itemPng,"MAP_B_srgb.png"), [0 1]);

        % HSV/Lab/indices
        local_save_map(single(Hc), "H (0-1)", fullfile(itemPng,"MAP_H.png"), [0 1]);
        local_save_map(single(Sc), "S (0-1)", fullfile(itemPng,"MAP_S.png"), [0 1]);
        local_save_map(single(Vc), "V (0-1)", fullfile(itemPng,"MAP_V.png"), [0 1]);

        local_save_map(single(Lc), "L*", fullfile(itemPng,"MAP_L.png"), []);
        local_save_map(single(ac), "a*", fullfile(itemPng,"MAP_a.png"), []);
        local_save_map(single(bc), "b*", fullfile(itemPng,"MAP_b.png"), []);

        local_save_map(single(dE),  "dE (ab to median)", fullfile(itemPng,"MAP_dE.png"), []);
        local_save_map(single(dEn), "dE norm (0-1)",     fullfile(itemPng,"MAP_dEn.png"), [0 1]);
        local_save_map(single(WHI), "WHI (0-1)",         fullfile(itemPng,"MAP_WHI.png"), [0 1]);
        local_save_map(single(SVI), "SVI (0-1)",         fullfile(itemPng,"MAP_SVI.png"), [0 1]);
        local_save_map(single(D),   "D-grade (1-5)",     fullfile(itemPng,"MAP_Dgrade.png"), [0 5]);

        % tile grids
        local_save_map(tileMedL, "Tile median L*", fullfile(itemPng,"TILE_medL.png"), []);
        local_save_map(tileIqrL, "Tile IQR L*",    fullfile(itemPng,"TILE_iqrL.png"), []);
        local_save_map(tileMedb, "Tile median b*", fullfile(itemPng,"TILE_medb.png"), []);
        local_save_map(edgeDensity, "Tile edge density", fullfile(itemPng,"TILE_edgeDensity.png"), []);
        local_save_map(localVar,    "Tile local variance", fullfile(itemPng,"TILE_localVar.png"), []);

        % overlays
        local_save_overlay_scalar(RGB_raw, WHI, mask, fullfile(itemPng,"OVERLAY_WHI.png"), [0 1], "WHI overlay");
        local_save_overlay_scalar(RGB_raw, D,   mask, fullfile(itemPng,"OVERLAY_Dgrade.png"), [0 5], "D-grade overlay");

        % texture maps
        if OPT.SAVE_TEXTURE_MAPS
            local_save_map(tex.gradL,   "grad |Ln|",     fullfile(itemPng,"MAP_gradL.png"),   []);
            local_save_map(tex.gradb,   "grad |b*|",     fullfile(itemPng,"MAP_gradb.png"),   []);
            local_save_map(tex.graddEn, "grad |dEn|",    fullfile(itemPng,"MAP_graddEn.png"), []);
            local_save_map(tex.entL,    "entropy(Ln)",   fullfile(itemPng,"MAP_entropyL.png"), [0 1]);
            local_save_map(tex.logL,    "LoG(Ln)",       fullfile(itemPng,"MAP_LoG_L.png"),   [0 1]);

            local_save_map(tex.(sprintf("stdL_w%d",wSmall)), sprintf("std(Ln) w=%d",wSmall), ...
                fullfile(itemPng, sprintf("MAP_stdL_w%d.png",wSmall)), [0 1]);
            local_save_map(tex.(sprintf("stdL_w%d",wLarge)), sprintf("std(Ln) w=%d",wLarge), ...
                fullfile(itemPng, sprintf("MAP_stdL_w%d.png",wLarge)), [0 1]);
            local_save_map(tex.(sprintf("rngL_w%d",wLarge)), sprintf("range(Ln) w=%d",wLarge), ...
                fullfile(itemPng, sprintf("MAP_rngL_w%d.png",wLarge)), [0 1]);

            local_save_map(tileGradL,  "TILE mean gradL",  fullfile(itemPng,"TILE_gradL_mean.png"),  []);
            local_save_map(tileEntL9,  "TILE mean entL",   fullfile(itemPng,"TILE_entropyL_mean.png"), []);
            local_save_map(tileLoG,    "TILE mean LoG",    fullfile(itemPng,"TILE_LoG_mean.png"), []);
        end
    end

    %% ================= SAVE: per-image MAT =================
    if OPT.SAVE_MAT
        try
            S = struct();
            S.image = string(f);
            S.itemId = itemId;
            S.mask = mask;

            S.RGB_raw_uint8  = RGB_raw;
            S.RGB_srgb_01    = RGB_srgb;

            S.HSV = HSV;
            S.Lab = Lab;

            S.dE = dE; S.dEn = dEn;
            S.weatherScore = weatherScore;
            S.WHI = WHI;
            S.SVI = SVI;
            S.D = D;

            S.tile = tile;
            S.tileMedL = tileMedL; S.tileIqrL = tileIqrL;
            S.tileMedb = tileMedb; S.tileIqrb = tileIqrb;
            S.tileMedS = tileMedS; S.tileIqrS = tileIqrS;
            S.tileMedV = tileMedV; S.tileIqrV = tileIqrV;
            S.edgeDensity = edgeDensity;
            S.localVar = localVar;

            S.texture = tex;
            S.tileGradL = tileGradL;
            S.tileGradb = tileGradb;
            S.tileGradDE = tileGradDE;
            S.tileEntL = tileEntL9;
            S.tileLoG = tileLoG;
            S.tileStdSmall = tileStdSmall;
            S.tileStdLarge = tileStdLarge;
            S.tileRngLarge = tileRngLarge;
            S.textureWins = OPT.TEXTURE_WINS;

            save(fullfile(itemMat, "analysis.mat"), "-struct","S", "-v7.3");
            save(fullfile(matDir, itemId + ".mat"), "-struct","S", "-v7.3");
        catch ME
            warning("CoreWeathering:MatSaveFail","%s", sprintf("MAT 저장 실패: %s", ME.message));
        end
    end

    %% ================= GLOBAL SUMMARY ROW =================
    row = table();
    row.image = string(f);
    row.itemId = itemId;
    row.stem = baseStem + string(ext);
    row.H = size(RGB_raw,1); row.W = size(RGB_raw,2);
    row.nPix = nnz(mask);

    % --- raw rgb (uint8) summary ---
    row.Rraw_p50 = prctile(Rrv,50); row.Rraw_ipr = prctile(Rrv,95)-prctile(Rrv,5);
    row.Graw_p50 = prctile(Grv,50); row.Graw_ipr = prctile(Grv,95)-prctile(Grv,5);
    row.Braw_p50 = prctile(Brv,50); row.Braw_ipr = prctile(Brv,95)-prctile(Brv,5);

    % --- srgb(0-1) + lab indices summary ---
    row.Rsrgb_p50 = prctile(Rsv,50); row.Rsrgb_ipr = prctile(Rsv,95)-prctile(Rsv,5);
    row.Gsrgb_p50 = prctile(Gsv,50); row.Gsrgb_ipr = prctile(Gsv,95)-prctile(Gsv,5);
    row.Bsrgb_p50 = prctile(Bsv,50); row.Bsrgb_ipr = prctile(Bsv,95)-prctile(Bsv,5);

    row.L_p05 = prctile(Lv,5); row.L_p50 = prctile(Lv,50); row.L_p95 = prctile(Lv,95);
    row.a_p05 = prctile(av,5); row.a_p50 = prctile(av,50); row.a_p95 = prctile(av,95);
    row.b_p05 = prctile(bv,5); row.b_p50 = prctile(bv,50); row.b_p95 = prctile(bv,95);

    row.WHI_p50 = prctile(single(WHI(mask)),50);
    row.SVI_p50 = prctile(single(SVI(mask)),50);
    row.D_mode  = mode(double(D(mask)));

    % texture quick scalars
    row.gradL_p50 = prctile(single(tex.gradL(mask)),50);
    row.entL_p50  = prctile(single(tex.entL(mask)),50);
    row.logL_p50  = prctile(single(tex.logL(mask)),50);

    sumCell{ii} = row;
end

%% ---- stitch & save global summary ----
sumRows = vertcat(sumCell{~cellfun(@isempty,sumCell)});

try
    writetable(sumRows, fullfile(csvDir,"IMAGE_SUMMARY.csv"), "Encoding","UTF-8");
catch ME
    warning("CoreWeathering:CSVSaveFail","%s", sprintf("글로벌 CSV 저장 실패: %s", ME.message));
end

if OPT.SAVE_EXCEL
    try
        outXlsx = fullfile(outDir, "WEATHERING_ANALYSIS_PaperKit2028.xlsx");
        writetable(sumRows, outXlsx, "Sheet","SUMMARY");
        cfg = struct2table(OPT,"AsArray",true);
        writetable(cfg, outXlsx, "Sheet","CONFIG");
        fprintf("[OK] XLSX saved: %s\n", outXlsx);
    catch ME
        warning("CoreWeathering:XLSXSaveFail","%s", sprintf("XLSX 저장 실패: %s", ME.message));
    end
end

runInfo = struct();
runInfo.outDir = outDir;
runInfo.perImageDir = perDir;
runInfo.pngDir = pngDir;
runInfo.matDir = matDir;
runInfo.csvDir = csvDir;
runInfo.nFound = numel(files);
runInfo.nProcessed = height(sumRows);
runInfo.skippedFiles = skippedFiles;

fprintf("\n[DONE] nProcessed=%d | Output=%s\n", runInfo.nProcessed, outDir);

try
    diary off;
catch
end

end

%% =============================================================================
%                                LOCAL FUNCTIONS
%% =============================================================================

function OPT = local_parse_opts(varargin)
p = inputParser;
p.FunctionName = "CoreWeatheringPaperKit_OneFile2028";

addParameter(p,"MODE","select");          % select | single | folder
addParameter(p,"IMAGE_FILE","");
addParameter(p,"IMAGE_DIR","");
addParameter(p,"FILE_EXTS",[]);
addParameter(p,"RECURSIVE",false);
addParameter(p,"MAX_IMAGES",inf);
addParameter(p,"RNG",0);

addParameter(p,"OUT_BASEDIR",pwd);
addParameter(p,"RUN_NAME","");

% resize
addParameter(p,"RESIZE_TARGETW",0);
addParameter(p,"RESIZE_TARGETH",0);
addParameter(p,"RESIZE_MAXW",0);

% analysis
addParameter(p,"TILE",32);
addParameter(p,"GRADE_EDGES",[0.2 0.4 0.6 0.8]);

% texture expansion
addParameter(p,"TEXTURE_WINS",[3 5 9]);
addParameter(p,"LOG_SIGMA",1.0);
addParameter(p,"GRAD_METHOD","Sobel");    % Sobel | Prewitt

% saving
addParameter(p,"SAVE_ORIGINAL_COPY",true);
addParameter(p,"SAVE_MAPS",true);
addParameter(p,"SAVE_MAT",true);
addParameter(p,"SAVE_EXCEL",true);
addParameter(p,"SAVE_TEXTURE_MAPS",true);

parse(p,varargin{:});
OPT = p.Results;

OPT.MODE = lower(string(OPT.MODE));
OPT.GRAD_METHOD = char(string(OPT.GRAD_METHOD));

% texture wins sanitize
w = round(double(OPT.TEXTURE_WINS(:)'));
w = w(isfinite(w) & w>=3);
w = unique(w, "stable");
w(mod(w,2)==0) = w(mod(w,2)==0) + 1; % odd
if isempty(w), w = [3 5 9]; end
OPT.TEXTURE_WINS = w;

edges = double(OPT.GRADE_EDGES(:)');
if numel(edges) ~= 4 || any(~isfinite(edges)) || any(diff(edges)<=0) || edges(1)<=0 || edges(4)>=1
    OPT.GRADE_EDGES = [0.2 0.4 0.6 0.8];
else
    OPT.GRADE_EDGES = edges;
end
end

function files = local_resolve_files_by_mode(OPT, outDir)
mode = lower(string(OPT.MODE));

startPath = char(string(outDir));
if isempty(startPath) || ~isfolder(startPath)
    startPath = pwd;
end

patterns = local_normalize_patterns(OPT.FILE_EXTS);
if isempty(patterns)
    patterns = ["*.jpg","*.jpeg","*.png","*.tif","*.tiff","*.bmp"];
end

switch mode
    case "select"
        [fn, fp] = uigetfile( ...
            {'*.jpg;*.jpeg;*.png;*.tif;*.tiff;*.bmp','Image files';'*.*','All'}, ...
            '코어 이미지 선택(다중 선택 가능)', startPath, ...
            "MultiSelect","on");

        if isequal(fn,0)
            d = uigetdir(pwd, "코어 이미지 폴더 선택");
            if isequal(d,0)
                files = strings(0,1);
                return;
            end
            files = local_list_image_files(string(d), patterns, true);
        else
            if iscell(fn)
                tmp = strings(numel(fn),1);
                for i=1:numel(fn)
                    tmp(i) = string(fullfile(fp, fn{i}));
                end
                files = tmp;
            else
                files = string(fullfile(fp, fn));
            end
        end

    case "single"
        imgFile = string(OPT.IMAGE_FILE);
        if strlength(imgFile)==0
            [fn, fp] = uigetfile( ...
                {'*.jpg;*.jpeg;*.png;*.tif;*.tiff;*.bmp','Image files';'*.*','All'}, ...
                '코어 이미지 1장 선택', startPath);
            if isequal(fn,0)
                files = strings(0,1);
                return;
            end
            imgFile = string(fullfile(fp, fn));
        end
        if ~isfile(imgFile)
            error("IMAGE_FILE이 유효하지 않습니다: %s", imgFile);
        end
        files = imgFile;

    case "folder"
        imgDir = string(OPT.IMAGE_DIR);
        if strlength(imgDir)==0
            d = uigetdir(pwd, "코어 이미지 폴더 선택");
            if isequal(d,0)
                files = strings(0,1);
                return;
            end
            imgDir = string(d);
        end
        if ~isfolder(imgDir)
            error("IMAGE_DIR 폴더가 없습니다: %s", imgDir);
        end
        files = local_list_image_files(imgDir, patterns, logical(OPT.RECURSIVE));

    otherwise
        error("MODE는 select|single|folder 중 하나여야 합니다.");
end

files = string(files(:));
files(files=="") = [];
files = unique(files,"stable");

if isfinite(OPT.MAX_IMAGES) && OPT.MAX_IMAGES > 0 && OPT.MAX_IMAGES < numel(files)
    rng(double(OPT.RNG));
    idx = randperm(numel(files), OPT.MAX_IMAGES);
    files = files(idx);
end
end

function [outDir, pngDir, matDir, csvDir, perDir, logDir] = local_mkdir_or_fallback(outDir, pngDir, matDir, csvDir, perDir, logDir)
paths = {outDir,pngDir,matDir,csvDir,perDir,logDir};
okAll = true;

for i=1:numel(paths)
    d = string(paths{i});
    if ~isfolder(d)
        [ok, msg] = mkdir(d);
        if ~ok
            okAll = false;
            warning("CoreWeathering:mkdirFail","%s", sprintf("폴더 생성 실패: %s | %s", d, msg));
        end
    end
end

if okAll, return; end

td = string(fullfile(tempdir, "CoreWeatheringPaperKit_" + string(datetime('now','Format','yyyyMMdd_HHmmss'))));
outDir = td;
pngDir = fullfile(outDir,"png");
matDir = fullfile(outDir,"mat");
csvDir = fullfile(outDir,"csv");
perDir = fullfile(outDir,"per_image");
logDir = fullfile(outDir,"log");
local_ensure_dirs({outDir,pngDir,matDir,csvDir,perDir,logDir});
warning("CoreWeathering:mkdirFallback","%s", sprintf("출력 폴더를 tempdir로 변경: %s", outDir));
end

function local_ensure_dirs(C)
for i=1:numel(C)
    d = string(C{i});
    if ~isfolder(d)
        mkdir(d);
    end
end
end

function patterns = local_normalize_patterns(in)
if iscell(in)
    s = string(in(:));
else
    s = string(in);
end
if isempty(s), patterns = strings(0,1); return; end

if isscalar(s)
    one = strip(s(1));
    if contains(one,";"), s = split(one,";");
    elseif contains(one,","), s = split(one,",");
    else, s = one;
    end
end

s = lower(strip(s));
s(s=="") = [];
patterns = strings(0,1);

for i=1:numel(s)
    t = erase(s(i)," ");
    if startsWith(t,"*.")
        patterns(end+1,1) = t; %#ok<AGROW>
    else
        t = erase(t,"*");
        if ~startsWith(t,"."), t = "." + t; end
        patterns(end+1,1) = "*"+t; %#ok<AGROW>
    end
end
patterns = unique(patterns,"stable");
end

function files = local_list_image_files(imgDir, patterns, recursive)
imgDir = char(string(imgDir));
patterns = string(patterns(:));
files = strings(0,1);

for p=1:numel(patterns)
    pat = char(patterns(p));
    if recursive
        d = dir(fullfile(imgDir, "**", pat));
    else
        d = dir(fullfile(imgDir, pat));
    end
    if isempty(d), continue; end
    d = d(~[d.isdir]);
    fp = strings(numel(d),1);
    for k=1:numel(d)
        fp(k) = string(fullfile(d(k).folder, d(k).name));
    end
    files = [files; fp]; %#ok<AGROW>
end
files = unique(files,"stable");
end

function I0 = local_force_rgb(I0)
if ismatrix(I0)
    I0 = repmat(I0,1,1,3);
elseif size(I0,3)==1
    I0 = repmat(I0,1,1,3);
elseif size(I0,3)>3
    I0 = I0(:,:,1:3);
end
end

% ===================== 핵심: RGB dual system =====================
function Iu = wx_rgb_uint8(I0)
I0 = local_force_rgb(I0);
if isa(I0,'uint8')
    Iu = I0;
elseif isa(I0,'uint16')
    Iu = uint8(double(I0)/65535*255);
else
    Iu = uint8(max(0,min(255, round(double(I0)))));
end
end

function Is = wx_srgb01(I0)
% Always return sRGB (gamma-encoded) in [0,1]
I0 = local_force_rgb(I0);
if isa(I0,'uint8') || isa(I0,'uint16')
    Is = im2single(I0);
else
    Is = single(I0);
    mx = max(Is(:));
    if mx > 1.5
        Is = Is / 255; % float but looks like 0-255
    end
    Is = min(max(Is,0),1);
end
end

function Xn = local_norm_by_prc(X, mask, p1, p2)
if nnz(mask) < 50
    Xn = zeros(size(X), 'single');
    return;
end
Xm = X(mask);
lo = prctile(Xm, p1);
hi = prctile(Xm, p2);
Xn = (X - lo) ./ max(eps, (hi - lo));
Xn = min(max(Xn,0),1);
Xn(~mask) = 0;
end

function [tileMed, tileIqr] = local_tile_stats(X, mask, tile)
[h,w] = size(X);
ny = ceil(h/tile);
nx = ceil(w/tile);

tileMed = zeros(ny,nx,'single');
tileIqr = zeros(ny,nx,'single');

for yy=1:ny
    y0=(yy-1)*tile+1; y1=min(h, yy*tile);
    for xx=1:nx
        x0=(xx-1)*tile+1; x1=min(w, xx*tile);
        m = mask(y0:y1, x0:x1);
        if nnz(m) < 10
            tileMed(yy,xx) = 0; tileIqr(yy,xx) = 0; continue;
        end
        v = X(y0:y1, x0:x1);
        vm = v(m);
        tileMed(yy,xx) = single(median(vm,'omitnan'));
        tileIqr(yy,xx) = single(prctile(vm,95) - prctile(vm,5));
    end
end
end

function M = local_tile_mean(X, mask, tile)
[h,w] = size(X);
ny = ceil(h/tile);
nx = ceil(w/tile);

M = zeros(ny,nx,'single');
for yy=1:ny
    y0=(yy-1)*tile+1; y1=min(h, yy*tile);
    for xx=1:nx
        x0=(xx-1)*tile+1; x1=min(w, xx*tile);
        m = mask(y0:y1, x0:x1);
        if nnz(m) < 10, M(yy,xx)=0; continue; end
        v = X(y0:y1, x0:x1);
        vm = v(m);
        M(yy,xx) = single(mean(vm,'omitnan'));
    end
end
end

function V = local_local_var(gray)
s = stdfilt(gray, true(5));
V = s.^2;
V = mat2gray(V);
end

function T = local_stats_table_one(groupName, varNames, vecCell)
T = table(string(groupName), 'VariableNames', "group");
for i=1:numel(varNames)
    v = double(vecCell{i}(:));
    v = v(isfinite(v));
    if isempty(v), v = 0; end
    mu  = mean(v,'omitnan');
    sd  = std(v,0,'omitnan');
    p05 = prctile(v,5);
    p50 = prctile(v,50);
    p95 = prctile(v,95);
    ipr = p95 - p05;
    mn  = min(v);
    mx  = max(v);

    nm = string(varNames(i));
    T.(nm+"_mean") = mu;
    T.(nm+"_std")  = sd;
    T.(nm+"_p05")  = p05;
    T.(nm+"_p50")  = p50;
    T.(nm+"_p95")  = p95;
    T.(nm+"_ipr")  = ipr;
    T.(nm+"_min")  = mn;
    T.(nm+"_max")  = mx;
end
end

function Td = local_grade_fraction_table(D, mask, edges)
v = double(D(mask));
den = max(1, numel(v));
Td = table();
Td.n = den;
Td.edge1 = edges(1); Td.edge2 = edges(2); Td.edge3 = edges(3); Td.edge4 = edges(4);
for g=1:5
    Td.(sprintf("D%d_frac",g)) = nnz(v==g)/den;
end
end

function local_writetable_safe(T, fpath)
try
    writetable(T, fpath, "Encoding","UTF-8");
catch ME
    warning("CoreWeathering:writetableFail","%s", sprintf("writetable 실패: %s | %s", fpath, ME.message));
end
end

function T = local_tilegrid_summary_table(tileMedL,tileIqrL,tileMedb,tileIqrb,tileMedS,tileIqrS,tileMedV,tileIqrV,edgeDensity,localVar,tile)
T = table();
T.tile = tile;
T.medL_mean = mean(tileMedL(:),'omitnan'); T.iqrL_mean = mean(tileIqrL(:),'omitnan');
T.medb_mean = mean(tileMedb(:),'omitnan'); T.iqrb_mean = mean(tileIqrb(:),'omitnan');
T.medS_mean = mean(tileMedS(:),'omitnan'); T.iqrS_mean = mean(tileIqrS(:),'omitnan');
T.medV_mean = mean(tileMedV(:),'omitnan'); T.iqrV_mean = mean(tileIqrV(:),'omitnan');
T.edgeDensity_mean = mean(edgeDensity(:),'omitnan');
T.localVar_mean = mean(localVar(:),'omitnan');
end

function local_save_map(X, ttl, outFile, climRange)
try
    fig = figure("Visible","off","Position",[100 100 1600 480]);
    imagesc(X); axis image off; title(ttl); colorbar;
    if ~isempty(climRange), clim(climRange); end
    exportgraphics(fig, outFile, "Resolution", 260);
    close(fig);
catch ME
    warning("CoreWeathering:saveMapFail","%s", sprintf("맵 저장 실패: %s | %s", outFile, ME.message));
end
end

function local_save_overlay_scalar(I0, X, mask, outFile, climRange, ttl)
try
    fig = figure("Visible","off","Position",[100 100 1800 520]);
    imshow(I0); hold on;
    h = imagesc(X);
    set(h, "AlphaData", 0.35*double(mask));
    if ~isempty(climRange), clim(climRange); end
    colorbar; title(ttl);
    exportgraphics(fig, outFile, "Resolution", 260);
    close(fig);
catch ME
    warning("CoreWeathering:overlayFail","%s", sprintf("오버레이 저장 실패: %s | %s", outFile, ME.message));
end
end

%% -------------------- EXTRA TEXTURE PACK --------------------
function tex = local_texture_pack(Ln, bc, dEn, mask, OPT)
tex = struct();

% gradient magnitude (per-pixel)
try
    tex.gradL   = single(imgradient(Ln, OPT.GRAD_METHOD));
    tex.gradb   = single(imgradient(bc, OPT.GRAD_METHOD));
    tex.graddEn = single(imgradient(dEn, OPT.GRAD_METHOD));
catch
    tex.gradL   = single(local_gradmag_fdiff(Ln));
    tex.gradb   = single(local_gradmag_fdiff(bc));
    tex.graddEn = single(local_gradmag_fdiff(dEn));
end
tex.gradL(~mask) = 0; tex.gradb(~mask) = 0; tex.graddEn(~mask) = 0;

% entropy on Ln (0~1 -> uint8)
try
    Ln8 = uint8(255 * min(max(Ln,0),1));
    ent = entropyfilt(Ln8, true(9));
    tex.entL = local_norm_by_prc(single(ent), mask, 5, 95);
catch
    tex.entL = zeros(size(Ln),'single');
end
tex.entL(~mask) = 0;

% LoG (Laplacian of Gaussian) on Ln
try
    sig = double(OPT.LOG_SIGMA);
    if sig <= 0, sig = 1.0; end
    G = imgaussfilt(Ln, sig);
    Lo = abs(del2(G));
    tex.logL = local_norm_by_prc(single(Lo), mask, 5, 95);
catch
    tex.logL = zeros(size(Ln),'single');
end
tex.logL(~mask) = 0;

% multiscale std/range on Ln
for ww = OPT.TEXTURE_WINS
    w = round(double(ww));
    if w < 3, w = 3; end
    if mod(w,2)==0, w = w+1; end
    try
        s = stdfilt(Ln, true(w));
        r = rangefilt(Ln, true(w));
        tex.(sprintf("stdL_w%d",w)) = local_norm_by_prc(single(s), mask, 5, 95);
        tex.(sprintf("rngL_w%d",w)) = local_norm_by_prc(single(r), mask, 5, 95);
    catch
        tex.(sprintf("stdL_w%d",w)) = zeros(size(Ln),'single');
        tex.(sprintf("rngL_w%d",w)) = zeros(size(Ln),'single');
    end
    tex.(sprintf("stdL_w%d",w))(~mask) = 0;
    tex.(sprintf("rngL_w%d",w))(~mask) = 0;
end
end

function g = local_gradmag_fdiff(X)
X = single(X);
dx = [diff(X,1,2) zeros(size(X,1),1,'single')];
dy = [diff(X,1,1); zeros(1,size(X,2),'single')];
g = hypot(dx, dy);
end

function T = local_texture_tilegrid_summary_table(tileGradL,tileGradb,tileGradDE,tileEntL9,tileLoG,tileStdSmall,tileStdLarge,tileRngLarge,tile,wSmall,wLarge)
T = table();
T.tile = tile;
T.gradL_mean   = mean(tileGradL(:),'omitnan');
T.gradb_mean   = mean(tileGradb(:),'omitnan');
T.graddEn_mean = mean(tileGradDE(:),'omitnan');
T.entropyL_mean= mean(tileEntL9(:),'omitnan');
T.LoG_mean     = mean(tileLoG(:),'omitnan');

T.std_small_win = wSmall;
T.std_large_win = wLarge;
T.std_small_mean = mean(tileStdSmall(:),'omitnan');
T.std_large_mean = mean(tileStdLarge(:),'omitnan');
T.range_large_mean = mean(tileRngLarge(:),'omitnan');
end
