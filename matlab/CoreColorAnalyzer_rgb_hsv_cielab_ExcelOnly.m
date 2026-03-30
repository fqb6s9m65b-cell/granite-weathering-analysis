function runInfo = CoreColorAnalyzer_rgb_hsv_cielab_ExcelOnly(varargin)
% CoreColorAnalyzer_rgb_hsv_cielab_ExcelOnly (ONE-FILE, R2025a/b)
% -----------------------------------------------------------------------------
% 목적
%   - 코어 이미지(1장/다중/폴더)에서 RGB(0-255), HSV(0-1), CIELAB(L*:0-100,a*,b*)
%   - ROI(mask) 적용 후 채널별 통계(mean/std/p05/p50/p95/ipr/min/max)
%   - CSV + Excel 리포트 + (중요) per-image 분석 PNG 산출
%
% 출력 구조
%   OUT_BASEDIR\RUN_TAG_RUN_yyyymmdd_HHMMSS\
%     csv\   COLOR_STATS_PER_IMAGE_raw.csv
%     xlsx\  COLOR_STATS_EXCEL_REPORT.xlsx
%     png\   (각 이미지 요약 리포트 복사본)
%     per_image\<imageId>\png\ (원본/ROI/채널맵/히스토/ab공간/리포트)
%     per_image\<imageId>\csv\ (히스토/ab2Dhist)
%     log\   run_log.txt
%
% 사용
%   CoreColorAnalyzer_rgb_hsv_cielab_ExcelOnly
%   runInfo = CoreColorAnalyzer_rgb_hsv_cielab_ExcelOnly('MODE',"folder",'IMAGE_DIR',"D:\imgs");
%
% 옵션 (Name-Value)
%   'OUT_BASEDIR'  : 기본 "C:\Users\ROCKENG\Desktop\코랩 머신러닝\COLAB"
%   'RUN_TAG'      : 기본 "ColorStats"
%   'MODE'         : "select"(기본) | "single" | "folder"
%   'IMAGE_FILE'   : MODE="single"
%   'IMAGE_DIR'    : MODE="folder"
%   'RECURSIVE'    : false
%   'MAX_IMAGES'   : inf
%   'RESIZE_MAXW'  : 0 (미사용)  % 너무 큰 이미지 가로 제한
%
%   'ROI_MODE'     : "none"(기본) | "auto" | "manual"
%   'AUTO_BG_TH'   : 0.15 (HSV-V 기준)
%
%   'EXPORT_PNG'   : true (기본)
%   'REPORT_DPI'   : 220 (기본)
%   'MAX_SCATTER'  : 50000 (a*-b* scatter subsample)
% -----------------------------------------------------------------------------

OPT = wx_parse_opts(varargin{:});

tstamp  = string(datetime("now","Format","yyyyMMdd_HHmmss"));
runName = OPT.RUN_TAG + "_RUN_" + tstamp;

baseDir = string(OPT.OUT_BASEDIR);
if strlength(baseDir)==0, baseDir = string(pwd); end

outDir = string(fullfile(baseDir, runName));
csvDir = string(fullfile(outDir, "csv"));
xlsxDir = string(fullfile(outDir, "xlsx"));
pngDir = string(fullfile(outDir, "png"));
perDir = string(fullfile(outDir, "per_image"));
logDir = string(fullfile(outDir, "log"));

[outDir,csvDir,xlsxDir,pngDir,perDir,logDir] = wx_mkdir_or_fallback(outDir,csvDir,xlsxDir,pngDir,perDir,logDir);

try
    diary off;
    diary(char(fullfile(logDir, "run_log.txt")));
    diary on;
catch
end

fprintf("[INFO] Output root: %s\n", outDir);

% ---- collect images ----
files = wx_resolve_files(OPT, outDir);
files = string(files(:));
files(files=="") = [];
files = unique(files,"stable");
if isempty(files)
    error("분석할 이미지가 없습니다.");
end

rows = cell(numel(files),1);

for ii = 1:numel(files)
    f = files(ii);
    fprintf("\n[INFO] (%d/%d) %s\n", ii, numel(files), f);

    % read
    try
        I0 = imread(f);
    catch ME
        warning("ColorStats:imreadFail","%s", sprintf("읽기 실패 → 스킵: %s | %s", f, ME.message));
        continue;
    end
    I0 = wx_ensure_rgb_uint8(I0);

    % resize
    if OPT.RESIZE_MAXW > 0 && size(I0,2) > OPT.RESIZE_MAXW
        sc = double(OPT.RESIZE_MAXW) / double(size(I0,2));
        I0 = imresize(I0, sc, "bilinear");
    end

    Himg = size(I0,1);
    Wimg = size(I0,2);

    % ROI mask
    mask = wx_make_roi_mask(I0, OPT);
    nPix = nnz(mask);
    if nPix < 50
        warning("ColorStats:ROIFew","%s", sprintf("ROI 픽셀 수가 너무 적음(%d) → 전체 프레임으로 대체", nPix));
        mask = true(Himg,Wimg);
        nPix = nnz(mask);
    end

    % channels
    R = double(I0(:,:,1));  G = double(I0(:,:,2));  B = double(I0(:,:,3));

    I = im2single(I0);
    hsvImg = rgb2hsv(I);
    Hc = double(hsvImg(:,:,1));  Sc = double(hsvImg(:,:,2));  Vc = double(hsvImg(:,:,3));

    lab = rgb2lab(I);
    Lc = double(lab(:,:,1));  ac = double(lab(:,:,2));  bc = double(lab(:,:,3));

    % stats table row
    row = table();
    row.image   = string(f);
    row.H = Himg; row.W = Wimg;
    row.nPix = nPix;

    row = wx_add_stats(row,"R", R, mask);
    row = wx_add_stats(row,"G", G, mask);
    row = wx_add_stats(row,"B", B, mask);

    row = wx_add_stats(row,"H", Hc, mask);
    row = wx_add_stats(row,"S", Sc, mask);
    row = wx_add_stats(row,"V", Vc, mask);

    row = wx_add_stats(row,"L", Lc, mask);
    row = wx_add_stats(row,"a", ac, mask);
    row = wx_add_stats(row,"b", bc, mask);

    rows{ii} = row;

    % ---- per-image png/csv exports ----
    if OPT.EXPORT_PNG
        [~, stem, ext] = fileparts(f);
        imageId = string(sprintf("%06d_%s%s", ii, stem, ext));

        itemRoot = string(fullfile(perDir, imageId));
        itemPng  = string(fullfile(itemRoot, "png"));
        itemCsv  = string(fullfile(itemRoot, "csv"));
        wx_ensure_dirs({itemRoot,itemPng,itemCsv});

        % save original + ROI
        wx_write_im_safe(I0, fullfile(itemPng, "ORIGINAL.png"));
        wx_write_mask_safe(mask, fullfile(itemPng, "MASK.png"));
        wx_save_roi_overlay(I0, mask, fullfile(itemPng, "ROI_OVERLAY.png"), OPT.REPORT_DPI);

        % channel configs (맵/히스토 렌더용)
        cfg = wx_channel_configs(R,G,B,Hc,Sc,Vc,Lc,ac,bc,mask);

        % per-channel MAP + HIST(+CSV)
        for k=1:numel(cfg)
            c = cfg(k);
            wx_save_channel_map_png(c, mask, fullfile(itemPng, "MAP_"+c.key+".png"), OPT.REPORT_DPI);
            wx_save_channel_hist_png_csv(c, mask, fullfile(itemPng, "HIST_"+c.key+".png"), fullfile(itemCsv, "HIST_"+c.key+".csv"), OPT.REPORT_DPI);
        end

        % RGB overlay histogram (한 장)
        wx_save_rgb_overlay_hist(R,G,B,mask, fullfile(itemPng, "HIST_RGB_OVERLAY.png"), fullfile(itemCsv, "HIST_RGB_OVERLAY.csv"), OPT.REPORT_DPI);

        % Lab a*-b* space (scatter + 2D hist)
        wx_export_ab_space(ac, bc, mask, itemPng, itemCsv, OPT);

        % Rich report montage (원본+ROI+대표맵+대표히스토+통계 텍스트)
        outReport = fullfile(itemPng, "REPORT_COLOR_9CH_RICH.png");
        wx_plot_rich_report(I0, mask, row, cfg, outReport, OPT);

        % also copy to top-level png
        try
            copyfile(outReport, fullfile(pngDir, imageId + "_REPORT_COLOR_9CH_RICH.png"));
        catch
        end
    end
end

validRows = rows(~cellfun(@isempty, rows));
if isempty(validRows)
    T = table();
else
    T = vertcat(validRows{:});
end

% ---- save CSV ----
csvPath = fullfile(csvDir, "COLOR_STATS_PER_IMAGE_raw.csv");
wx_writetable_safe(T, csvPath);

% ---- save Excel ----
xlsxPath = fullfile(xlsxDir, "COLOR_STATS_EXCEL_REPORT.xlsx");
wx_writetable_xlsx_safe(T, xlsxPath, "PIXEL_STATS");

% index sheet
idxT = table();
idxT.image = T.image;
idxT.H = T.H; idxT.W = T.W; idxT.nPix = T.nPix;
wx_writetable_xlsx_safe(idxT, xlsxPath, "INDEX");

runInfo = struct();
runInfo.outDir = outDir;
runInfo.csvPath = string(csvPath);
runInfo.xlsxPath = string(xlsxPath);
runInfo.pngDir = pngDir;
runInfo.perImageDir = perDir;
runInfo.nFound = numel(files);
runInfo.nProcessed = height(T);

fprintf("\n[DONE] nProcessed=%d\n  CSV : %s\n  XLSX: %s\n  PNG : %s\n", runInfo.nProcessed, runInfo.csvPath, runInfo.xlsxPath, runInfo.pngDir);

try
    diary off;
catch
end
end

%% =============================================================================
% LOCAL FUNCTIONS
%% =============================================================================
function OPT = wx_parse_opts(varargin)
p = inputParser;
p.FunctionName = "CoreColorAnalyzer_rgb_hsv_cielab_ExcelOnly";

addParameter(p,"OUT_BASEDIR","C:\Users\ROCKENG\Desktop\코랩 머신러닝\COLAB");
addParameter(p,"RUN_TAG","ColorStats");

addParameter(p,"MODE","select");       % select | single | folder
addParameter(p,"IMAGE_FILE","");
addParameter(p,"IMAGE_DIR","");
addParameter(p,"RECURSIVE",false);
addParameter(p,"MAX_IMAGES",inf);

addParameter(p,"RESIZE_MAXW",0);

% ROI
addParameter(p,"ROI_MODE","none");     % none | auto | manual
addParameter(p,"AUTO_BG_TH",0.15);     % HSV-V threshold

% exports
addParameter(p,"EXPORT_PNG",true);
addParameter(p,"REPORT_DPI",220);
addParameter(p,"MAX_SCATTER",50000);

parse(p,varargin{:});
OPT = p.Results;

OPT.MODE = lower(string(OPT.MODE));
OPT.RUN_TAG = string(OPT.RUN_TAG);
OPT.OUT_BASEDIR = string(OPT.OUT_BASEDIR);

OPT.ROI_MODE = lower(string(OPT.ROI_MODE));
OPT.AUTO_BG_TH = max(0.0, min(1.0, double(OPT.AUTO_BG_TH)));

OPT.RESIZE_MAXW = max(0, round(double(OPT.RESIZE_MAXW)));

OPT.EXPORT_PNG = logical(OPT.EXPORT_PNG);
OPT.REPORT_DPI = max(72, round(double(OPT.REPORT_DPI)));
OPT.MAX_SCATTER = max(1000, round(double(OPT.MAX_SCATTER)));
end

function files = wx_resolve_files(OPT, outDir)
mode = lower(string(OPT.MODE));
startPath = char(string(outDir));
if isempty(startPath) || ~isfolder(startPath), startPath = pwd; end

switch mode
    case "select"
        [fn, fp] = uigetfile( ...
            {'*.jpg;*.jpeg;*.png;*.tif;*.tiff;*.bmp','Image files';'*.*','All'}, ...
            '코어 이미지 선택(다중 선택 가능)', startPath, ...
            "MultiSelect","on");
        if isequal(fn,0)
            d = uigetdir(pwd, "코어 이미지 폴더 선택");
            if isequal(d,0), files = strings(0,1); return; end
            files = wx_list_files(string(d), logical(OPT.RECURSIVE));
        else
            if iscell(fn)
                tmp = strings(numel(fn),1);
                for i=1:numel(fn), tmp(i) = string(fullfile(fp, fn{i})); end
                files = tmp;
            else
                files = string(fullfile(fp, fn));
            end
        end

    case "single"
        f = string(OPT.IMAGE_FILE);
        if strlength(f)==0
            [fn, fp] = uigetfile( ...
                {'*.jpg;*.jpeg;*.png;*.tif;*.tiff;*.bmp','Image files';'*.*','All'}, ...
                '코어 이미지 1장 선택', startPath);
            if isequal(fn,0), files = strings(0,1); return; end
            f = string(fullfile(fp, fn));
        end
        if ~isfile(f), error("IMAGE_FILE이 유효하지 않습니다: %s", f); end
        files = f;

    case "folder"
        d = string(OPT.IMAGE_DIR);
        if strlength(d)==0
            dd = uigetdir(pwd, "코어 이미지 폴더 선택");
            if isequal(dd,0), files = strings(0,1); return; end
            d = string(dd);
        end
        if ~isfolder(d), error("IMAGE_DIR 폴더가 없습니다: %s", d); end
        files = wx_list_files(d, logical(OPT.RECURSIVE));

    otherwise
        error("MODE는 select|single|folder 중 하나여야 합니다.");
end

files = string(files(:));
files(files=="") = [];
files = unique(files,"stable");

if isfinite(OPT.MAX_IMAGES) && OPT.MAX_IMAGES > 0 && OPT.MAX_IMAGES < numel(files)
    rng(0);
    idx = randperm(numel(files), OPT.MAX_IMAGES);
    files = files(idx);
end
end

function files = wx_list_files(d, recursive)
patterns = ["*.jpg","*.jpeg","*.png","*.tif","*.tiff","*.bmp"];
files = strings(0,1);
for p = 1:numel(patterns)
    if recursive
        S = dir(fullfile(char(d), "**", char(patterns(p))));
    else
        S = dir(fullfile(char(d), char(patterns(p))));
    end
    S = S(~[S.isdir]);
    tmp = strings(numel(S),1);
    for k=1:numel(S)
        tmp(k) = string(fullfile(S(k).folder, S(k).name));
    end
    files = [files; tmp]; %#ok<AGROW>
end
files = unique(files,"stable");
end

function [outDir,csvDir,xlsxDir,pngDir,perDir,logDir] = wx_mkdir_or_fallback(outDir,csvDir,xlsxDir,pngDir,perDir,logDir)
paths = {outDir,csvDir,xlsxDir,pngDir,perDir,logDir};
okAll = true;
for i=1:numel(paths)
    d = string(paths{i});
    if ~isfolder(d)
        [ok,msg] = mkdir(d);
        if ~ok
            okAll = false;
            warning("ColorStats:mkdirFail","%s", sprintf("폴더 생성 실패: %s | %s", d, msg));
        end
    end
end
if okAll, return; end

td = string(fullfile(tempdir, "ColorStats_" + string(datetime("now","Format","yyyyMMdd_HHmmss"))));
outDir = td;
csvDir = fullfile(outDir,"csv");
xlsxDir = fullfile(outDir,"xlsx");
pngDir = fullfile(outDir,"png");
perDir = fullfile(outDir,"per_image");
logDir = fullfile(outDir,"log");
wx_ensure_dirs({outDir,csvDir,xlsxDir,pngDir,perDir,logDir});
warning("ColorStats:mkdirFallback","%s", sprintf("출력 폴더를 tempdir로 변경: %s", outDir));
end

function wx_ensure_dirs(C)
for i=1:numel(C)
    d = string(C{i});
    if ~isfolder(d), mkdir(d); end
end
end

function I = wx_ensure_rgb_uint8(I0)
if ismatrix(I0) || size(I0,3)==1
    I = repmat(I0,1,1,3);
else
    I = I0(:,:,1:3);
end
if ~isa(I,"uint8"), I = im2uint8(I); end
end

function mask = wx_make_roi_mask(I0, OPT)
H = size(I0,1); W = size(I0,2);
mode = lower(string(OPT.ROI_MODE));
switch mode
    case "none"
        mask = true(H,W);

    case "manual"
        try
            figure("Visible","on"); imshow(I0); title("ROI 선택: 드래그 후 더블클릭/Enter");
            h = drawfreehand("Color","y");
            wait(h);
            mask = createMask(h);
            close(gcf);
        catch ME
            warning("ColorStats:ROIFail","%s", sprintf("manual ROI 실패 → 전체 프레임 사용: %s", ME.message));
            mask = true(H,W);
        end

    case "auto"
        I = im2single(I0);
        hsvI = rgb2hsv(I);
        V = hsvI(:,:,3);
        th = double(OPT.AUTO_BG_TH);
        mask = V > th;

        if nnz(mask) < 0.1*numel(mask)
            warning("ColorStats:AutoROIWeak","%s","auto ROI가 너무 작음 → 전체 프레임 사용");
            mask = true(H,W);
        end

    otherwise
        warning("ColorStats:ROIUnknown","%s","ROI_MODE 알 수 없음 → 전체 프레임 사용");
        mask = true(H,W);
end
end

function row = wx_add_stats(row, key, X, mask)
v = double(X(mask));
v = v(isfinite(v));
if isempty(v), v = 0; end

[p05,p50,p95] = wx_prct(v);

row.(key+"_mean") = mean(v,"omitnan");
row.(key+"_std")  = std(v,0,"omitnan");
row.(key+"_p05")  = p05;
row.(key+"_p50")  = p50;
row.(key+"_p95")  = p95;
row.(key+"_ipr")  = p95 - p05;
row.(key+"_min")  = min(v);
row.(key+"_max")  = max(v);
end

function [p05,p50,p95] = wx_prct(v)
try
    p = prctile(v,[5 50 95]);
    p05=p(1); p50=p(2); p95=p(3);
catch
    vv = sort(v(:));
    n = numel(vv);
    p05 = vv(max(1, round(0.05*n)));
    p50 = vv(max(1, round(0.50*n)));
    p95 = vv(max(1, round(0.95*n)));
end
end

function wx_writetable_safe(T, fpath)
try
    writetable(T, fpath, "Encoding","UTF-8");
catch ME
    warning("ColorStats:writetableFail","%s", sprintf("writetable 실패: %s | %s", fpath, ME.message));
end
end

function wx_writetable_xlsx_safe(T, xlsxPath, sheetName)
try
    writetable(T, xlsxPath, "Sheet", sheetName);
catch ME
    warning("ColorStats:xlsxFail","%s", sprintf("Excel 저장 실패(%s): %s", sheetName, ME.message));
end
end

function wx_write_im_safe(I, fpath)
try
    imwrite(I, fpath);
catch ME
    warning("ColorStats:imwriteFail","%s", sprintf("imwrite 실패: %s | %s", fpath, ME.message));
end
end

function wx_write_mask_safe(mask, fpath)
try
    imwrite(uint8(mask)*255, fpath);
catch ME
    warning("ColorStats:maskWriteFail","%s", sprintf("mask 저장 실패: %s | %s", fpath, ME.message));
end
end

function wx_export_fig(fig, outFile, reso)
try
    exportgraphics(fig, outFile, "Resolution", reso);
catch ME
    warning("ColorStats:exportFail","%s", sprintf("그림 저장 실패: %s", ME.message));
    try
        print(fig, outFile, "-dpng", sprintf("-r%d", reso));
    catch
    end
end
end

%% ------------------ Channel Configs & Render ------------------
function cfg = wx_channel_configs(R,G,B,Hc,Sc,Vc,Lc,ac,bc,~)
tpl = struct("key","", "title","", "units","", "data",[], "vmin",nan, "vmax",nan, "render","scalar", "hue",nan, "diverging",false);
cfg = repmat(tpl,0,1);

cfg(end+1) = wx_ch(tpl,"R","R","0-255",R,0,255,"huesat",0.00,false);
cfg(end+1) = wx_ch(tpl,"G","G","0-255",G,0,255,"huesat",1/3,false);
cfg(end+1) = wx_ch(tpl,"B","B","0-255",B,0,255,"huesat",2/3,false);

cfg(end+1) = wx_ch(tpl,"H","H","0-1",Hc,0,1,"hueonly",nan,false);
cfg(end+1) = wx_ch(tpl,"S","S","0-1",Sc,0,1,"scalar",nan,false);
cfg(end+1) = wx_ch(tpl,"V","V","0-1",Vc,0,1,"scalar",nan,false);

cfg(end+1) = wx_ch(tpl,"L","L*","0-100",Lc,0,100,"scalar",nan,false);
cfg(end+1) = wx_ch(tpl,"a","a*","-128~127",ac,-128,127,"scalar",nan,true);
cfg(end+1) = wx_ch(tpl,"b","b*","-128~127",bc,-128,127,"scalar",nan,true);

% mask는 render에서 사용(별도 전달), cfg에는 저장 안 함
end

function c = wx_ch(tpl,key,titleTxt,units,X,vmin,vmax,renderMode,hue,isDiv)
c = tpl;
c.key = string(key);
c.title = string(titleTxt);
c.units = string(units);
c.data = double(X);
c.vmin = double(vmin);
c.vmax = double(vmax);
c.render = string(renderMode);
c.hue = double(hue);
c.diverging = logical(isDiv);
end

function wx_save_roi_overlay(I0, mask, outPng, dpi)
fig = figure("Visible","off","Position",[100 100 1100 850]);
imshow(I0); hold on;
ov = cat(3, ones(size(mask))*1, zeros(size(mask)), zeros(size(mask))); % red overlay
h = imshow(ov);
set(h,"AlphaData", 0.20*double(mask));
title("ROI overlay");
wx_export_fig(fig, outPng, dpi);
close(fig);
end

function wx_save_channel_map_png(c, mask, outPng, dpi)
X = c.data;
X(~mask) = NaN;

fig = figure("Visible","off","Position",[100 100 950 760]);
ax = axes(fig);

switch lower(c.render)
    case "huesat"
        xn = (X - c.vmin) ./ max(eps, (c.vmax - c.vmin));
        xn = min(max(xn,0),1);
        H = ones(size(xn)) * c.hue;
        S = xn;
        V = ones(size(xn));
        rgb = hsv2rgb(cat(3,H,S,V));
        imshow(rgb, "Parent", ax);
        axis(ax,"image"); axis(ax,"off");

    case "hueonly"
        H = X; H = min(max(H,0),1);
        S = ones(size(H));
        V = ones(size(H));
        rgb = hsv2rgb(cat(3,H,S,V));
        imshow(rgb, "Parent", ax);
        axis(ax,"image"); axis(ax,"off");

    otherwise
        imagesc(ax, X);
        axis(ax,"image"); axis(ax,"off");
        if c.diverging
            colormap(ax, wx_diverging_map(256)); clim(ax,[c.vmin c.vmax]);
        else
            colormap(ax, parula(256)); clim(ax,[c.vmin c.vmax]);
        end
        colorbar(ax);
end

title(ax, sprintf("MAP %s (%s)", c.title, c.units), "Interpreter","none");
wx_export_fig(fig, outPng, dpi);
close(fig);
end

function wx_save_channel_hist_png_csv(c, mask, outPng, outCsv, dpi)
v = c.data(mask);
v = v(isfinite(v));
if isempty(v), v = 0; end

edges = wx_edges_for_channel(c.key);
centers = (edges(1:end-1)+edges(2:end))/2;
p = histcounts(v, edges, "Normalization","probability");

[p05,p50,p95] = wx_prct(v);

T = table();
T.center = centers(:);
T.prob = p(:);
T.p05 = repmat(p05,numel(centers),1);
T.p50 = repmat(p50,numel(centers),1);
T.p95 = repmat(p95,numel(centers),1);
wx_writetable_safe(T, outCsv);

fig = figure("Visible","off","Position",[100 100 1000 720]);
ax = axes(fig);
bar(ax, centers, p, 1.0, "EdgeColor","none");
grid(ax,"on");
xlabel(ax,"value"); ylabel(ax,"probability");
title(ax, sprintf("HIST %s (%s)", c.title, c.units), "Interpreter","none");
xline(ax, p05, "--", "LineWidth", 1.0);
xline(ax, p50, "-",  "LineWidth", 1.2);
xline(ax, p95, "--", "LineWidth", 1.0);
wx_export_fig(fig, outPng, dpi);
close(fig);
end

function wx_save_rgb_overlay_hist(R,G,B,mask,outPng,outCsv,dpi)
rv = double(R(mask)); gv = double(G(mask)); bv = double(B(mask));
rv = rv(isfinite(rv)); gv = gv(isfinite(gv)); bv = bv(isfinite(bv));
if isempty(rv), rv=0; end
if isempty(gv), gv=0; end
if isempty(bv), bv=0; end

edges = linspace(0,255,257);
centers = (edges(1:end-1)+edges(2:end))/2;

pr = histcounts(rv, edges, "Normalization","probability");
pg = histcounts(gv, edges, "Normalization","probability");
pb = histcounts(bv, edges, "Normalization","probability");

T = table(centers(:), pr(:), pg(:), pb(:), "VariableNames",["center","R","G","B"]);
wx_writetable_safe(T, outCsv);

fig = figure("Visible","off","Position",[100 100 1100 720]);
ax = axes(fig); hold(ax,"on");
plot(ax, centers, pr, "LineWidth",1.3);
plot(ax, centers, pg, "LineWidth",1.3);
plot(ax, centers, pb, "LineWidth",1.3);
grid(ax,"on");
xlabel(ax,"value"); ylabel(ax,"probability");
title(ax,"RGB distribution overlay (ROI masked)","Interpreter","none");
legend(ax,["R","G","B"],"Location","northeast");
wx_export_fig(fig, outPng, dpi);
close(fig);
end

function wx_export_ab_space(ac, bc, mask, itemPng, itemCsv, OPT)
a = double(ac(mask)); b = double(bc(mask));
ok = isfinite(a) & isfinite(b);
a = a(ok); b = b(ok);
if isempty(a), a=0; b=0; end

ae = linspace(-128,127,129);
be = linspace(-128,127,129);
N = histcounts2(a,b,ae,be,"Normalization","probability");

% CSV 2D hist
wx_writetable_safe(array2table(N), fullfile(itemCsv, "LAB_AB_HIST2_PROB.csv"));

% PNG
fig = figure("Visible","off","Position",[100 100 1200 520]);
tiledlayout(fig,1,2,"Padding","compact","TileSpacing","compact");

nexttile;
imagesc((be(1:end-1)+be(2:end))/2, (ae(1:end-1)+ae(2:end))/2, N);
axis image; set(gca,"YDir","normal");
xlabel("b*"); ylabel("a*"); title("Lab a*-b* 2D hist (prob)");
colormap(parula(256)); colorbar;

nexttile;
ns = min(OPT.MAX_SCATTER, numel(a));
rng(0); idx = randperm(numel(a), ns);
scatter(a(idx), b(idx), 3, '.', "MarkerEdgeAlpha",0.15);
axis equal; grid on;
xlim([-128 127]); ylim([-128 127]);
xlabel("a*"); ylabel("b*"); title(sprintf("Lab a*-b* scatter (n=%d)", ns));

wx_export_fig(fig, fullfile(itemPng, "LAB_AB_SPACE.png"), OPT.REPORT_DPI);
close(fig);
end

function wx_plot_rich_report(I0, mask, row, cfg, outPng, OPT)
% 논문용: 한 장에 원본+ROI+대표맵+대표히스토+통계 텍스트
fig = figure("Visible","off","Position",[60 60 2600 1900]);
tiledlayout(fig,3,4,"Padding","compact","TileSpacing","compact");

% (1) original
nexttile; imshow(I0); title("Original","Interpreter","none");

% (2) ROI overlay
nexttile;
imshow(I0); hold on;
ov = cat(3, ones(size(mask))*1, zeros(size(mask)), zeros(size(mask)));
h = imshow(ov); set(h,"AlphaData",0.22*double(mask));
title(sprintf("ROI overlay | nPix=%d", row.nPix), "Interpreter","none");
axis image off;

% (3) L* map
nexttile; wx_quick_map(cfg, "L", mask); title("L* map","Interpreter","none");
% (4) a* map
nexttile; wx_quick_map(cfg, "a", mask); title("a* map","Interpreter","none");

% (5) b* map
nexttile; wx_quick_map(cfg, "b", mask); title("b* map","Interpreter","none");
% (6) V map
nexttile; wx_quick_map(cfg, "V", mask); title("V map","Interpreter","none");

% (7) RGB overlay hist
nexttile;
axis tight; grid on;
% 간단히 row 기반으로 텍스트(히스토는 별도 파일에서 확인)
text(0,1,"See HIST_RGB_OVERLAY.png","VerticalAlignment","top","FontName","Consolas","FontSize",12);
axis off; title("RGB dist overlay","Interpreter","none");

% (8) stats text
nexttile([1 1]);
axis off;
wx_draw_stats_text(row);

% (9~12) R/G/B/H maps
nexttile; wx_quick_map(cfg, "R", mask); title("R map","Interpreter","none");
nexttile; wx_quick_map(cfg, "G", mask); title("G map","Interpreter","none");
nexttile; wx_quick_map(cfg, "B", mask); title("B map","Interpreter","none");
nexttile; wx_quick_map(cfg, "H", mask); title("H map","Interpreter","none");

wx_export_fig(fig, outPng, OPT.REPORT_DPI);
close(fig);
end

function wx_draw_stats_text(row)
keys = ["R","G","B","H","S","V","L","a","b"];
lines = strings(0,1);
lines(end+1) = "Channel stats (ROI-masked):";
for k=keys
    p50 = row.(k+"_p50");
    ipr = row.(k+"_ipr");
    mu  = row.(k+"_mean");
    sd  = row.(k+"_std");
    lines(end+1) = sprintf("%s: p50=%.4g | IPR=%.4g | mean=%.4g ± %.4g", k, p50, ipr, mu, sd); %#ok<AGROW>
end
text(0,1,strjoin(lines,newline),"VerticalAlignment","top","FontName","Consolas","FontSize",11);
end

function wx_quick_map(cfg, key, mask)
k = string(key);
idx = find([cfg.key]==k,1);
if isempty(idx)
    imagesc(zeros(size(mask))); axis image off; return;
end
c = cfg(idx);
X = c.data; X(~mask)=NaN;

switch lower(c.render)
    case "huesat"
        xn = (X - c.vmin) ./ max(eps, (c.vmax - c.vmin));
        xn = min(max(xn,0),1);
        H = ones(size(xn)) * c.hue;
        S = xn;
        V = ones(size(xn));
        rgb = hsv2rgb(cat(3,H,S,V));
        imshow(rgb); axis image off;

    case "hueonly"
        H = X; H = min(max(H,0),1);
        S = ones(size(H));
        V = ones(size(H));
        rgb = hsv2rgb(cat(3,H,S,V));
        imshow(rgb); axis image off;

    otherwise
        imagesc(X); axis image off;
        if c.diverging
            colormap(wx_diverging_map(256)); clim([c.vmin c.vmax]);
        else
            colormap(parula(256)); clim([c.vmin c.vmax]);
        end
        colorbar;
end
end

function edges = wx_edges_for_channel(key)
k = string(key);
switch k
    case {"R","G","B"}
        edges = linspace(0,255,257);
    case {"H","S","V"}
        edges = linspace(0,1,101);
    case {"L"}
        edges = linspace(0,100,101);
    case {"a","b"}
        edges = linspace(-128,127,129);
    otherwise
        edges = linspace(0,1,101);
end
end

function cmap = wx_diverging_map(n)
n = double(n);
x = linspace(0,1,n)';
cmap = zeros(n,3);

i1 = x<=0.5;
t1 = x(i1)/0.5;
cmap(i1,1) = t1;
cmap(i1,2) = t1;
cmap(i1,3) = 1;

i2 = x>0.5;
t2 = (x(i2)-0.5)/0.5;
cmap(i2,1) = 1;
cmap(i2,2) = 1 - t2;
cmap(i2,3) = 1 - t2;
end
