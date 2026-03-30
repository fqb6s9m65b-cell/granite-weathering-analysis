function CoreColorStats_MatchImageJson_ToExcel_OneFile()
% CoreColorStats_Final — 화성암 코어 이미지 색상 픽셀 완전 분석 (Excel 전용)
% =========================================================================
% JSON 추출: rw (풍화등급), lux (조도), surface (표면상태)
%
% 분석 채널 (11채널):
%   RGB: R G B (0-255)  |  HSV: H S V (0-1)
%   CIELAB: L*(0-100), a*, b*, C*(Chroma), h_ab(0-360)
%
% 통계 (채널당 14개):
%   mean std min max p05 p25 p50 p75 p95 iqr ipr skew kurt entropy
%
% 색상 지수 21개:
%   delta_E_D1 RI CI YI CWI SAI GRI WHI SVI RBI
%   R/G/B_norm RB_ratio RG_ratio dark/mid/bright_pct munsell_V/C/H
%
% 출력 (Excel 전용):
%   IMAGE_SUMMARY.xlsx/.csv
%   GRADE_SUMMARY.xlsx  SURFACE_SUMMARY.xlsx  LUX_SUMMARY.xlsx
%   CROSS_GRADE_SURFACE.xlsx
%
% MATLAB R2025a/b  |  GPU + parfor 병렬
% =========================================================================
clc;

%% [0] GPU
useGPU = false;
try
    try parallel.gpu.enableCUDAForwardCompatibility(true); catch; end
    if canUseGPU
        g = gpuDevice(1);
        fprintf('[GPU] %s (%.1f GB)\n', g.Name, g.TotalMemory/1e9);
        useGPU = true;
    end
catch
    fprintf('[GPU] 비활성 -> CPU\n');
end

%% [0b] CPU parfor
nWorkers = 0;
try
    if license('test','Distrib_Computing_Toolbox')
        pp = gcp('nocreate');
        if isempty(pp), pp = parpool('Processes'); end
        nWorkers = pp.NumWorkers;
        fprintf('[CPU] workers: %d\n', nWorkers);
    end
catch
    fprintf('[CPU] Parallel Toolbox 없음\n');
end

%% [1] 설정
cfg_imageRoots = {
    'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\images_igneous_rock\TS_1, 기반암 암종 분류 데이터_1. 화성암_1'
    'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\images_igneous_rock\TS_1, 기반암 암종 분류 데이터_1. 화성암_2'
    'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\images_igneous_rock\TS_2. 기반암 절리 탐지 데이터_1. 화성암'
    'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\images_igneous_rock\VS_1. 기반암 암종 분류 데이터_1. 화성암'
    'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\images_igneous_rock\VS_2. 기반암 절리 탐지 데이터_1. 화성암'
};
cfg_jsonRoots = {
    'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\json_igneous_rock\TL_1. 기반암 암종 분류 데이터_1. 화성암'
    'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\json_igneous_rock\TL_2. 기반암 절리 탐지 데이터_1. 화성암'
    'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\json_igneous_rock\VL_1. 기반암 암종 분류 데이터_1. 화성암'
    'C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\json_igneous_rock\VL_2. 기반암 절리 탐지 데이터_1. 화성암'
};
cfg_outRoot      = fullfile('C:\Users\ROCKENG\Desktop\코랩 머신러닝\results', ...
    char("ColorStats_RUN_" + string(datetime('now','Format','yyyyMMdd_HHmmss'))));
cfg_imageExts    = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'};
cfg_maxExcelRows = 1000000;
cfg_GPU_BATCH    = 128;
cfg_CKPT_EVERY   = 5000;
cfg_minPx        = 50;
cfg_REF_D1_L     = 48.12;
cfg_REF_D1_a     = -1.77;
cfg_REF_D1_b     = -1.29;

%% [2] 출력 폴더
t0 = tic;
mkdirSafe(cfg_outRoot);
logDir = fullfile(cfg_outRoot,'log');
mkdirSafe(logDir);
try diary(fullfile(logDir,'run_log.txt')); diary on; catch; end

fprintf('\n%s\n  CoreColorStats_Final\n  출력: %s\n%s\n\n', ...
    repmat('=',1,76), cfg_outRoot, repmat('=',1,76));

%% [3] 이미지 수집
fprintf('[1] 이미지 수집\n');
imageFiles = collectImages(cfg_imageRoots, cfg_imageExts);
nTotal = numel(imageFiles);
if nTotal==0, error('CoreColor:NoImages','이미지 없음'); end
fprintf('  합계: %s장\n\n', fmtN(nTotal));

%% [4] JSON 인덱싱
fprintf('[2] JSON 인덱싱\n');
jsonMap = buildJsonMap(cfg_jsonRoots);
fprintf('  JSON: %s개\n', fmtN(jsonMap.Count));

matchedFiles = {}; unmatchedFiles = {};
for fi = 1:nTotal
    [~,st,~] = fileparts(imageFiles{fi});
    if jsonMap.isKey(lower(st))
        matchedFiles{end+1}   = imageFiles{fi}; %#ok<AGROW>
    else
        unmatchedFiles{end+1} = imageFiles{fi}; %#ok<AGROW>
    end
end
nMatched   = numel(matchedFiles);
nUnmatched = numel(unmatchedFiles);
fprintf('  매칭: %s장 | 미매칭(스킵): %s장\n\n', fmtN(nMatched), fmtN(nUnmatched));

if nUnmatched > 0
    try
        writetable(table(string(unmatchedFiles(:)),'VariableNames',{'file'}), ...
            fullfile(logDir,'UNMATCHED.csv'));
    catch
    end
end
if nMatched==0, error('CoreColor:NoMatch','JSON 매칭 이미지 없음'); end

cellFiles = matchedFiles;
nImg      = nMatched;

% containers.Map은 parfor 불가 -> 개별 1D 셀 배열로 사전 변환
% (2D 셀 = broadcast 경고 -> 컬럼별 분리)
arr_rw      = cell(nImg,1);
arr_lux     = cell(nImg,1);
arr_surface = cell(nImg,1);
arr_lux_bin = cell(nImg,1);
for fi = 1:nImg
    [~,st,~] = fileparts(cellFiles{fi});
    m = jsonMap(lower(st));
    arr_rw{fi}      = m.rw;
    arr_lux{fi}     = m.lux;
    arr_surface{fi} = m.surface;
    arr_lux_bin{fi} = m.lux_bin;
end

%% [5] 요약 컬럼
varNames = getSummaryVarNames();
allRows  = cell(nImg,1);

%% [6] 체크포인트
ckptPath = fullfile(cfg_outRoot,'CKPT.mat');
startIdx = 1;
if exist(ckptPath,'file')
    try
        ck = load(ckptPath);
        if isfield(ck,'lastIdx') && ck.lastIdx < nImg
            startIdx = ck.lastIdx + 1;
            allRows  = ck.allRows;
            fprintf('[CKPT] %s/%s에서 재개\n\n', fmtN(ck.lastIdx), fmtN(nImg));
        end
    catch
        warning('CoreColor:Ckpt','%s','체크포인트 로드 실패');
    end
end

%% [7] 메인 루프
fprintf('[3] 분석 시작 (GPU=%s | Workers=%d | Batch=%d)\n', ...
    yn(useGPU), nWorkers, cfg_GPU_BATCH);

BATCH   = cfg_GPU_BATCH;
todoIdx = startIdx:nImg;
nBatch  = ceil(numel(todoIdx)/BATCH);
skipLog = {};

refL  = cfg_REF_D1_L;
refa  = cfg_REF_D1_a;
refb  = cfg_REF_D1_b;
minPx = cfg_minPx;

for bi = 1:nBatch

    bs   = (bi-1)*BATCH + 1;
    be   = min(bi*BATCH, numel(todoIdx));
    bIdx = todoIdx(bs:be);
    nB   = numel(bIdx);

    % 1D 셀 배열 슬라이스 -> parfor sliced variable (broadcast 없음)
    bFiles   = cellFiles(bIdx);
    bRW      = arr_rw(bIdx);
    bLux     = arr_lux(bIdx);
    bSurface = arr_surface(bIdx);
    bLuxBin  = arr_lux_bin(bIdx);

    %% 이미지 읽기 (parfor)
    imgs    = cell(nB,1);
    stems   = cell(nB,1);
    readOK  = true(nB,1);
    readErr = cell(nB,1);

    parfor ii = 1:nB
        f = bFiles{ii};
        try
            I = ensureRGB(imread(f));
            imgs{ii} = I;
            [~,st,~] = fileparts(f);
            stems{ii} = st;
        catch ME
            readOK(ii)  = false;
            readErr{ii} = ME.message;
        end
    end

    %% GPU 색공간 변환 (순차)
    hsvC = cell(nB,1);
    labC = cell(nB,1);
    for ii = 1:nB
        if ~readOK(ii) || isempty(imgs{ii}), continue; end
        try
            if useGPU
                Ig       = gpuArray(single(imgs{ii}))/255;
                hsvC{ii} = double(gather(rgb2hsv(Ig)));
                labC{ii} = double(gather(rgb2lab(Ig)));
            else
                Iflt     = im2single(imgs{ii});
                hsvC{ii} = double(rgb2hsv(Iflt));
                labC{ii} = double(rgb2lab(Iflt));
            end
        catch MEg
            if useGPU && contains(lower(MEg.message),'out of memory')
                try reset(gpuDevice()); catch; end
                useGPU = false;
                warning('CoreColor:OOM','%s','GPU OOM -> CPU 폴백');
            end
            try
                Iflt     = im2single(imgs{ii});
                hsvC{ii} = double(rgb2hsv(Iflt));
                labC{ii} = double(rgb2lab(Iflt));
            catch
                readOK(ii) = false;
            end
        end
    end

    %% 통계 계산 (parfor)
    bRows = cell(nB,1);

    parfor ii = 1:nB
        if ~readOK(ii) || isempty(hsvC{ii}), continue; end
        try
            I0  = imgs{ii};
            Rp  = double(reshape(I0(:,:,1),[],1));
            Gp  = double(reshape(I0(:,:,2),[],1));
            Bp  = double(reshape(I0(:,:,3),[],1));
            Hp  = reshape(hsvC{ii}(:,:,1),[],1);
            Sp  = reshape(hsvC{ii}(:,:,2),[],1);
            Vp  = reshape(hsvC{ii}(:,:,3),[],1);
            Lp  = reshape(labC{ii}(:,:,1),[],1);
            ap  = reshape(labC{ii}(:,:,2),[],1);
            bp2 = reshape(labC{ii}(:,:,3),[],1);

            if numel(Rp) < minPx, continue; end

            Cp   = sqrt(ap.^2 + bp2.^2);
            habp = atan2d(bp2, ap);
            habp(habp<0) = habp(habp<0) + 360;

            [Hpx,Wpx,~] = size(I0);

            bRows{ii} = buildRow( ...
                bFiles{ii}, stems{ii}, ...
                bRW{ii}, bLux{ii}, bSurface{ii}, bLuxBin{ii}, ...
                Hpx, Wpx, ...
                Rp,Gp,Bp,Hp,Sp,Vp,Lp,ap,bp2,Cp,habp, ...
                refL, refa, refb);
        catch
        end
    end

    %% 결과 수집
    for ii = 1:nB
        gi2 = bIdx(ii);
        if ~readOK(ii)
            skipLog{end+1} = sprintf('[읽기실패] %s | %s', bFiles{ii}, readErr{ii}); %#ok<AGROW>
        elseif isempty(bRows{ii})
            skipLog{end+1} = sprintf('[통계실패] %s', bFiles{ii}); %#ok<AGROW>
        else
            allRows{gi2} = bRows{ii};
        end
    end

    curIdx = bIdx(end);
    el     = toc(t0);
    speed  = curIdx / max(el,1e-6);
    eta    = (nImg-curIdx) / max(speed,0.01);
    fprintf('  [%s/%s] %.1f img/s  ETA %.0fs (%.1f min)  스킵 %d\n', ...
        fmtN(curIdx), fmtN(nImg), speed, eta, eta/60, numel(skipLog));

    if useGPU && mod(bi,20)==0, try gpuDevice; catch; end; end

    if mod(curIdx,cfg_CKPT_EVERY)<BATCH || bi==nBatch
        lastIdx = curIdx; 
        save(ckptPath,'lastIdx','allRows','-v7.3');
    end

    imgs=[]; hsvC=[]; labC=[]; %#ok<NASGU>

end % for bi

%% [8] 스킵 로그
if ~isempty(skipLog)
    try
        writetable(table(string(skipLog(:)),'VariableNames',{'reason'}), ...
            fullfile(logDir,'SKIP_LOG.csv'));
    catch
    end
    fprintf('\n  스킵: %d건\n', numel(skipLog));
end

%% [9] 테이블 조립
fprintf('\n[4] 테이블 조립\n');
validRows = allRows(~cellfun(@isempty,allRows));
nValid    = numel(validRows);
fprintf('  성공: %s / 매칭: %s / 전체: %s\n', fmtN(nValid), fmtN(nMatched), fmtN(nTotal));
if nValid==0, error('CoreColor:NoValid','분석된 이미지 없음'); end

Tsum = cell2table(vertcat(validRows{:}), 'VariableNames', varNames);

printDist(Tsum,'rw_grade','풍화등급');
printDist(Tsum,'surface', '표면상태');
printDist(Tsum,'lux_bin', '조도구간');

%% [10] IMAGE_SUMMARY
fprintf('\n[5] IMAGE_SUMMARY 저장\n');
csvPath = fullfile(cfg_outRoot,'IMAGE_SUMMARY.csv');
try
    writetable(Tsum, csvPath, 'Encoding','UTF-8');
    fprintf('  CSV: %s\n', csvPath);
catch ME
    warning(ME.identifier,'%s',sprintf('CSV 실패: %s',ME.message));
end
writeExcelSplit(Tsum, fullfile(cfg_outRoot,'IMAGE_SUMMARY'), 'IMAGE_SUMMARY', cfg_maxExcelRows);

%% [11] 집계
fprintf('\n[6] 집계 저장\n');
numCols = getNumCols(Tsum);
writeAggExcel(Tsum,'rw_grade',numCols,fullfile(cfg_outRoot,'GRADE_SUMMARY.xlsx'),  'GRADE_STATS');
writeAggExcel(Tsum,'surface', numCols,fullfile(cfg_outRoot,'SURFACE_SUMMARY.xlsx'),'SURFACE_STATS');
writeAggExcel(Tsum,'lux_bin', numCols,fullfile(cfg_outRoot,'LUX_SUMMARY.xlsx'),    'LUX_STATS');
writeCrossTab(Tsum,'rw_grade','surface',fullfile(cfg_outRoot,'CROSS_GRADE_SURFACE.xlsx'),'CROSS');

%% [12] 완료
if exist(ckptPath,'file'), delete(ckptPath); end
elapsed = toc(t0);
fprintf('\n%s\n  DONE | %.1f min | 분석 %s / 전체 %s | 스킵 %d\n', ...
    repmat('=',1,76), elapsed/60, fmtN(nValid), fmtN(nTotal), numel(skipLog));
fprintf('  GPU=%s | Workers=%d\n  출력: %s\n%s\n\n', ...
    yn(useGPU), nWorkers, cfg_outRoot, repmat('=',1,76));
try diary off; catch; end

end % function CoreColorStats_Final


%%=========================================================================
function vn = getSummaryVarNames()
meta = {'img_file','img_stem','rw_grade','lux','lux_bin','surface','nPixels','img_H','img_W'};

chans = {'R','G','B','H','S','V','L','a','b','C','hab'};
suf   = {'_mean','_std','_min','_max', ...
         '_p05','_p25','_p50','_p75','_p95', ...
         '_iqr','_ipr','_skew','_kurt','_entropy'};
chCols = {};
for c = 1:numel(chans)
    for s = 1:numel(suf)
        chCols{end+1} = [chans{c} suf{s}]; %#ok<AGROW>
    end
end

idx = {'delta_E_D1','RI','CI','YI','CWI','SAI','GRI','WHI','SVI','RBI', ...
    'R_norm','G_norm','B_norm','RB_ratio','RG_ratio', ...
    'dark_pct','mid_pct','bright_pct','munsell_V','munsell_C','munsell_H_name'};

vn = [meta, chCols, idx];
end


%%=========================================================================
function row = buildRow(fpath, stem, rw, lux, surface, lux_bin, ...
        Hpx, Wpx, R,G,B,H,S,V,L,a,b,C,hab, refL,refa,refb)

meta = {fpath, stem, rw, lux, lux_bin, surface, numel(R), Hpx, Wpx};

chList = {R,G,B,H,S,V,L,a,b,C,hab};
chCols = {};
for ci = 1:numel(chList)
    chCols = [chCols, chanStats(chList{ci})]; %#ok<AGROW>
end

Rm  = mean(R);  Gm  = mean(G);  Bm  = mean(B);
Lm  = mean(L);  am  = mean(a);  bm  = mean(b);
Cm_ = mean(C);  Sm  = mean(S);  Vm  = mean(V);
Lp50 = prctile(L,50); bp50 = prctile(b,50);
sRGB = Rm+Gm+Bm+1e-6; eps1 = 1e-6;

dE  = sqrt((Lm-refL)^2+(am-refa)^2+(bm-refb)^2);
RI  = (Rm^2)/max(Bm*sRGB,eps1);
CI  = (Rm-Bm)/max(Rm+Bm,eps1);
YI  = bm/max(Lm,eps1)*100;
CWI = (bp50+10)/max(Lp50/10,eps1);
SAI = Cm_/max(Lm,eps1)*100;
GRI = 1-Cm_/30;
WHI = bm/(abs(am)+1);
SVI = Sm*(1-Vm)*100;
RBI = (Rm-Bm)/255;
Rn=Rm/sRGB; Gn=Gm/sRGB; Bn=Bm/sRGB;
RBr=Rm/max(Bm,1); RGr=Rm/max(Gm,1);
dkP=mean(V<0.20)*100;
mdP=mean(V>=0.20&V<0.80)*100;
brP=mean(V>=0.80)*100;
[mV,mC,~,mHn] = lab2munsell(Lm,am,bm);

idxC = {dE,RI,CI,YI,CWI,SAI,GRI,WHI,SVI,RBI, ...
    Rn,Gn,Bn,RBr,RGr,dkP,mdP,brP,mV,mC,mHn};

row = [meta, chCols, idxC];
end


%%=========================================================================
function s = chanStats(x)
if isempty(x), x=0; end
v = x(isfinite(x));
if isempty(v), v=0; end

mu=mean(v); sg=std(v,0); mn=min(v); mx=max(v);
pq=prctile(v,[5 25 50 75 95]);

if numel(v)>=4 && sg>1e-10
    zv=(v-mu)/sg; sk=mean(zv.^3); kt=mean(zv.^4)-3;
else
    sk=NaN; kt=NaN;
end

if mn<mx, e64=linspace(mn,mx+1e-9,65); else, e64=[mn-0.5,mn+0.5]; end
cnt=histcounts(v,e64); pp=cnt/max(sum(cnt),1); pp=pp(pp>0);
if isempty(pp), en=NaN; else, en=-sum(pp.*log2(pp)); end

s = {mu,sg,mn,mx,pq(1),pq(2),pq(3),pq(4),pq(5), ...
     pq(4)-pq(2),pq(5)-pq(1),sk,kt,en};
end


%%=========================================================================
function jMap = buildJsonMap(roots)
jMap = containers.Map('KeyType','char','ValueType','any');
for r = 1:numel(roots)
    root = roots{r};
    if ~exist(root,'dir')
        warning('CoreColor:JsonDir','%s',sprintf('JSON 폴더 없음: %s',root));
        continue;
    end
    ff=dir(fullfile(root,'**','*.json'));
    nAdd=0;
    for fi = 1:numel(ff)
        fp=[ff(fi).folder filesep ff(fi).name];
        [~,st]=fileparts(ff(fi).name);
        key=lower(st);
        if jMap.isKey(key), continue; end
        try
            J=jsondecode(fileread(fp));

            rw='';
            for fn={'rw','weathering_grade','weathering','grade'}
                if isfield(J,fn{1})&&~isempty(J.(fn{1}))
                    rw=char(string(J.(fn{1}))); break;
                end
            end

            lux=NaN;
            for fn={'lux','illuminance','lux_value'}
                if isfield(J,fn{1})
                    v=J.(fn{1});
                    if isnumeric(v)&&isscalar(v), lux=double(v);
                    elseif ischar(v)||isstring(v), lux=str2double(v); end
                    if isfinite(lux), break; end
                end
            end

            surface='UNKNOWN';
            for fn={'humidity','wetdry','wet_dry','surface','condition'}
                if isfield(J,fn{1})&&~isempty(J.(fn{1}))
                    raw=upper(strtrim(char(string(J.(fn{1})))));
                    if contains(raw,{'WET','습윤'})||strcmp(raw,'W')
                        surface='W'; break;
                    end
                    if contains(raw,{'DRY','건조'})||strcmp(raw,'D')
                        surface='D'; break;
                    end
                end
            end

            m.rw=normalizeRW(rw); m.lux=lux;
            m.surface=surface; m.lux_bin=luxBin(lux);
            jMap(key)=m; nAdd=nAdd+1;
        catch
        end
    end
    fprintf('  %s -> %s개\n', root, fmtN(nAdd));
end
end


function rw = normalizeRW(raw)
rw='UNKNOWN';
if isempty(raw), return; end
s=upper(strtrim(char(string(raw))));
t=regexp(s,'D([1-5])~D[1-5]','tokens','once');
if ~isempty(t), rw=sprintf('D%s',t{1}); return; end
t=regexp(s,'D([1-5])','tokens','once');
if ~isempty(t), rw=sprintf('D%s',t{1}); return; end
end


function b = luxBin(lux)
if ~isfinite(lux), b='UNKNOWN';
elseif lux<300,    b='LOW';
elseif lux<1000,   b='MID';
else,               b='HIGH';
end
end


%%=========================================================================
function [V,C,H_code,H_name] = lab2munsell(L,a,b)
V=max(0,min(10,L/10)); C=max(0,hypot(a,b)/5.5);
if hypot(a,b)<4.0, H_code=-1; H_name='N'; return; end
h=mod(atan2d(b,a),360);
hMap=[0 27;27 45;45 63;63 80;80 99;99 116;116 207;207 261;261 297;297 360];
hNames={'5R','5YR','7.5Y','10YR','2.5Y','5Y','5GY','5B','5PB','5P'};
H_code=2.5; H_name='5YR';
for ti=1:size(hMap,1)
    if h>=hMap(ti,1)&&h<hMap(ti,2), H_code=ti; H_name=hNames{ti}; return; end
end
end


%%=========================================================================
function numCols = getNumCols(T)
skip={'img_file','img_stem','rw_grade','lux_bin','surface','munsell_H_name'};
numCols={};
for ci=1:width(T)
    col=T.Properties.VariableNames{ci};
    if any(strcmp(col,skip)), continue; end
    if isnumeric(T.(col)), numCols{end+1}=col; end %#ok<AGROW>
end
end


%%=========================================================================
function writeAggExcel(T,grpCol,numCols,xlPath,sheetName)
if ~any(strcmp(T.Properties.VariableNames,grpCol)), return; end
grps=sort(unique(string(T.(grpCol))));
grps=grps(strlength(strtrim(grps))>0);
rows={};
for gi=1:numel(grps)
    g=grps(gi); mk=string(T.(grpCol))==g; n=nnz(mk); sub=T(mk,:);
    for ci=1:numel(numCols)
        col=numCols{ci};
        if ~any(strcmp(sub.Properties.VariableNames,col)), continue; end
        vals=sub.(col);
        if iscell(vals), vals=cell2mat(vals(cellfun(@isnumeric,vals))); end
        vals=vals(isfinite(vals));
        if isempty(vals), continue; end
        pq=prctile(vals,[5 25 50 75 95]);
        rows{end+1,1}=char(g); %#ok<AGROW>
        rows{end,2}=n; rows{end,3}=col;
        rows{end,4}=mean(vals); rows{end,5}=std(vals,0);
        rows{end,6}=min(vals); rows{end,7}=pq(1); rows{end,8}=pq(2);
        rows{end,9}=pq(3); rows{end,10}=pq(4); rows{end,11}=pq(5);
        rows{end,12}=max(vals);
    end
end
if ~isempty(rows)
    tbl=cell2table(rows,'VariableNames', ...
        {grpCol,'n','parameter','mean','std','min','p05','p25','p50','p75','p95','max'});
    safeWriteTable(tbl,xlPath,sheetName);
    fprintf('  %s -> %s\n',sheetName,xlPath);
end
end


%%=========================================================================
function writeCrossTab(T,rv,cv,xlPath,sheetName)
if ~any(strcmp(T.Properties.VariableNames,rv))||...
   ~any(strcmp(T.Properties.VariableNames,cv)), return; end
rV=sort(unique(string(T.(rv)))); rV=rV(strlength(strtrim(rV))>0);
cV=sort(unique(string(T.(cv)))); cV=cV(strlength(strtrim(cV))>0);
data=zeros(numel(rV),numel(cV));
for ri=1:numel(rV)
    for ci=1:numel(cV)
        data(ri,ci)=nnz(string(T.(rv))==rV(ri)&string(T.(cv))==cV(ci));
    end
end
Tc=array2table(data,'RowNames',cellstr(rV), ...
    'VariableNames',matlab.lang.makeValidName(cellstr(cV)));
Tc=[table(rV','VariableNames',{rv}),Tc]; Tc.total=sum(data,2);
safeWriteTable(Tc,xlPath,sheetName);
fprintf('  %s -> %s\n',sheetName,xlPath);
end


%%=========================================================================
function writeExcelSplit(T,basePath,sheetName,maxRows)
nRow=height(T); nParts=ceil(nRow/maxRows);
for pp=1:nParts
    r1=(pp-1)*maxRows+1; r2=min(pp*maxRows,nRow);
    if nParts==1, fp=[basePath '.xlsx'];
    else, fp=sprintf('%s_part%03d.xlsx',basePath,pp); end
    try
        writetable(T(r1:r2,:),fp,'Sheet',sheetName);
        fprintf('  XLSX: %s (%s행)\n',fp,fmtN(r2-r1+1));
    catch ME
        warning(ME.identifier,'%s',sprintf('Excel 저장 실패: %s',ME.message));
    end
end
end


%%=========================================================================
function printDist(T,col,label)
if ~any(strcmp(T.Properties.VariableNames,col)), return; end
fprintf('\n  %s 분포:\n',label);
[u,~,idx]=unique(string(T.(col))); nV=height(T);
for ri=1:numel(u)
    fprintf('    %-10s: %s장 (%.1f%%)\n',u(ri),fmtN(nnz(idx==ri)),100*nnz(idx==ri)/nV);
end
end

function imageFiles = collectImages(roots,exts)
imageFiles={};
for r=1:numel(roots)
    root=roots{r}; if ~exist(root,'dir'), continue; end
    dd=dir(fullfile(root,'**','*')); nAdd=0;
    for k=1:numel(dd)
        if dd(k).isdir, continue; end
        [~,~,e]=fileparts(dd(k).name);
        if any(strcmpi(e,exts))
            imageFiles{end+1,1}=fullfile(dd(k).folder,dd(k).name); %#ok<AGROW>
            nAdd=nAdd+1;
        end
    end
    fprintf('  %s -> %s장\n',root,fmtN(nAdd));
end
imageFiles=unique(imageFiles,'stable');
end

function I = ensureRGB(I0)
if isempty(I0), error('CoreColor:Empty','빈 이미지'); end
if ismatrix(I0),     I0=repmat(I0,1,1,3); end
if size(I0,3)>3,     I0=I0(:,:,1:3); end
if ~isa(I0,'uint8'), I0=im2uint8(I0); end
I=I0;
end

function safeWriteTable(tbl,xlPath,sheetName)
for k=1:3
    try writetable(tbl,xlPath,'Sheet',sheetName); return; catch, pause(1); end
end
warning('CoreColor:WriteFail','%s',sprintf('저장 실패: %s/%s',xlPath,sheetName));
end

function mkdirSafe(d)
if ~exist(d,'dir'), mkdir(d); end
end

function s = fmtN(n)
s=char(string(n)); idx=strfind(s,'.');
if isempty(idx), idx=numel(s)+1; end
for i=idx-4:-3:1, s=[s(1:i),',',s(i+1:end)]; end
end

function s = yn(v)
if v, s='YES'; else, s='NO'; end
end