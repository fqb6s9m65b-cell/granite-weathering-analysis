%% RUN_TunnelFaceROI_Weathering_Max2026_ONEFILE
%  1) 사용자: 막장면 이미지 1장 선택
%  2) ROI(바닥 직선 + 상부 아치/반원) 윤곽 추출 + 마스크/오버레이/디버그 저장
%  3) ROI 내부만으로 COLOR_STATS_PER_IMAGE_raw.csv 생성
%  4) 내장 wx_CoreWeatheringAnalyzer_Max2026 실행 → XLSX 출력
%
% 저장 폴더(고정):
%   C:\Users\ROCKENG\Desktop\코랩 머신러닝\COLAB_OUT
%
% 요구사항 반영:
% - 한 파일로 통합(외부 .m 필요 없음)
% - 이미지 한 장 선택
% - table VariableNames 오류 수정(문자형으로 강제)
% - 바닥 직선 + 아치/반원(완전 원 필요 없음)
% - 실패 시 threshold 기반 fallback 포함

clc; clear; close all;

%% ===== 결과 저장 폴더(고정) =====
outDir = "C:\Users\ROCKENG\Desktop\코랩 머신러닝\COLAB_OUT";
if ~exist(outDir,'dir'), mkdir(outDir); end

runTag = string(datetime('now','Format','yyyyMMdd_HHmmss'));
runDir = fullfile(outDir, "RUN_" + runTag);
roiDir = fullfile(runDir, "ROI");
mskDir = fullfile(runDir, "MASK");
dbgDir = fullfile(runDir, "DEBUG");
mkdir(runDir); mkdir(roiDir); mkdir(mskDir); mkdir(dbgDir);

%% ===== 입력 이미지 1장 선택(사용자) =====
imgPath = wx_pick_one_image();
if strlength(imgPath)==0
    disp("취소됨");
    return;
end
fprintf("[Info] Selected image: %s\n", imgPath);

[fp, bn, ex] = fileparts(char(imgPath));
bn = string(bn); ex = string(ex);

%% ===== 파라미터(안정성 우선 / 시간 상관없음) =====
P = struct();

% 전처리
P.sigma       = 1.3;
P.doIllumCorr = true;
P.bgSigma     = 110;        % 조도 편차 심하면 90~150
P.useCLAHE    = true;
P.clipLoPct   = 1;
P.clipHiPct   = 99;

% 엣지
P.minEdgeArea = 80;
P.gradPct     = 90;         % gradient 보강 percentile(낮추면 엣지 증가)

% 바닥선 탐색(강하게)
P.baseRoiFracs = [0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85];
P.angleTolDeg  = 35;
P.maxLines     = 40;
P.fillGaps     = [60 90 120 160 220];
P.minLenFracs  = [0.35 0.25 0.18 0.12 0.08 0.06];

P.useLineRansac     = true;
P.lineRansacIters   = 4000;
P.lineInlierTolPx   = 3.0;
P.lineMinInlierFrac = 0.02;

P.endpointBandPx = 5;
P.endpointYPadPx = 35;

% 아치/반원 스캔(완전 원 필요 없음)
P.tScanN      = 240;
P.tMaxFactor  = 5.0;
P.arcSamples  = 900;
P.arcEdgeTol  = 5.0;
P.scoreW_grad = 1.0;
P.scoreW_inl  = 2.0;

% 마스크/정밀화
P.initDilate   = 6;
P.smoothCloseR = 6;
P.doRefine     = true;
P.refineIter   = 260;       % 120~400
P.outlineWidth = 3;

% Fallback(아치 스캔 실패 시)
P.useThreshFallback = true;
P.adaptSens         = 0.55; % 0.45~0.7
P.thCloseR          = 12;
P.thMinAreaFrac     = 0.08;

% (선택) GPU 일부 가속(가능한 경우만)
P.tryGPU = true;

%% ===== 디버그/출력 경로 =====
dbgIpre = fullfile(dbgDir, bn + "_dbg_Ipre.png");
dbgEdge = fullfile(dbgDir, bn + "_dbg_edges.png");
dbgLine = fullfile(dbgDir, bn + "_dbg_bottomLine.png");
dbgArc  = fullfile(dbgDir, bn + "_dbg_arc.png");
dbgInit = fullfile(dbgDir, bn + "_dbg_initMask.png");
dbgRef  = fullfile(dbgDir, bn + "_dbg_refinedMask.png");

outMask    = fullfile(mskDir, bn + "_face_mask.png");
outOutline = fullfile(mskDir, bn + "_face_outline.png");
outOverlay = fullfile(mskDir, bn + "_face_overlay.png");
outROI     = fullfile(roiDir, bn + "_ROI" + ex);

%% ===== 1) ROI 추출 + 저장 =====
I = imread(imgPath);
if size(I,3)==4, I = I(:,:,1:3); end

[BWface, BWoutline, Iov, Iface] = wx_extract_face_and_outputs(I, P, ...
    dbgIpre, dbgEdge, dbgLine, dbgArc, dbgInit, dbgRef);

imwrite(BWface, outMask);
imwrite(BWoutline, outOutline);
imwrite(im2uint8(Iov), outOverlay);
imwrite(im2uint8(Iface), outROI);

fprintf("[OK] ROI outputs saved.\n");

%% ===== 2) ROI 내부 통계 → CSV =====
stats = wx_stats_from_mask(I, BWface);

T = table( ...
    string(imgPath), bn + ex, "", ...
    stats.R_mean, stats.G_mean, stats.B_mean, stats.GrayY_mean, ...
    stats.H_mean, stats.S_mean, stats.V_mean, ...
    stats.L_mean, stats.a_mean, stats.b_mean, ...
    'VariableNames', {'file','filename','rw_grade', ...
                      'R_mean','G_mean','B_mean','GrayY_mean', ...
                      'H_mean','S_mean','V_mean', ...
                      'L_mean','a_mean','b_mean'} );

statsCsv = fullfile(runDir, "COLOR_STATS_PER_IMAGE_raw.csv");
writetable(T, statsCsv);
fprintf("[OK] CSV saved: %s\n", statsCsv);

%% ===== 3) Max2026(내장) 실행 → XLSX =====
outXlsx = fullfile(runDir, "WEATHERING_ANALYSIS_Max2026.xlsx");
wx_CoreWeatheringAnalyzer_Max2026(statsCsv, outXlsx, ...
    'BIN_VERSION', 2, ...
    'USE_IGNEOUS_PRESET', true, ...
    'PRIMARY_VAR', "L_mean", ...
    'PRIMARY_DIRECTION', "auto");

fprintf("\n=== DONE ===\n");
fprintf("RUN DIR : %s\n", runDir);
fprintf("CSV     : %s\n", statsCsv);
fprintf("XLSX    : %s\n", outXlsx);
fprintf("MASK    : %s\n", outMask);
fprintf("OVERLAY : %s\n", outOverlay);

%% ===================== Local Functions =====================

function imgPath = wx_pick_one_image()
    imgPath = "";
    [fn, fp] = uigetfile({'*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp','Image Files'}, ...
        "막장면 이미지 1장 선택");
    if isequal(fn,0), return; end
    imgPath = string(fullfile(fp, fn));
end

function stats = wx_stats_from_mask(I, BWface)
    Irgb = im2double(I);
    if size(Irgb,3)==1, Irgb = repmat(Irgb,[1 1 3]); end
    M = BWface;

    if nnz(M) < 200
        error("ROI(mask) 픽셀이 너무 적음");
    end

    R = Irgb(:,:,1); G = Irgb(:,:,2); B = Irgb(:,:,3);
    stats.R_mean = mean(255*R(M));
    stats.G_mean = mean(255*G(M));
    stats.B_mean = mean(255*B(M));

    Gy = rgb2gray(Irgb);
    stats.GrayY_mean = mean(255*Gy(M));

    hsv = rgb2hsv(Irgb);
    Hh = hsv(:,:,1); Ss = hsv(:,:,2); Vv = hsv(:,:,3);
    stats.H_mean = mean(180*Hh(M));   % 0~180
    stats.S_mean = mean(255*Ss(M));   % 0~255
    stats.V_mean = mean(255*Vv(M));   % 0~255

    if exist('rgb2lab','file') == 2
        lab = rgb2lab(Irgb);
        Lc = lab(:,:,1); ac = lab(:,:,2); bc = lab(:,:,3);
        stats.L_mean = mean(Lc(M));
        stats.a_mean = mean(ac(M));
        stats.b_mean = mean(bc(M));
    else
        stats.L_mean = nan; stats.a_mean = nan; stats.b_mean = nan;
    end
end

function [BWface, BWoutline, Iov, Iface] = wx_extract_face_and_outputs(I, P, dbgIpre, dbgEdge, dbgLine, dbgArc, dbgInit, dbgRef)

    % ---- gray ----
    if size(I,3)==3
        Ig = rgb2gray(I);
    else
        Ig = I;
    end
    Ig = im2double(Ig);
    [H,W] = size(Ig);

    % ---- optional GPU (부분만) ----
    useGPU = false;
    if isfield(P,'tryGPU') && P.tryGPU
        try
            gd = gpuDevice;
            if ~isempty(gd), useGPU = true; end %#ok<NASGU>
        catch
            useGPU = false;
        end
    end

    % ---- preprocess (Ipre) ----
    if useGPU
        try
            IgG = gpuArray(Ig);
            Ig_sG = imgaussfilt(IgG, P.sigma);
            Ig_s = gather(Ig_sG);
        catch
            Ig_s = imgaussfilt(Ig, P.sigma);
        end
    else
        Ig_s = imgaussfilt(Ig, P.sigma);
    end

    Icorr = Ig_s;
    if P.doIllumCorr
        if useGPU
            try
                bgG = imgaussfilt(gpuArray(Ig_s), P.bgSigma);
                bg  = gather(bgG);
            catch
                bg = imgaussfilt(Ig_s, P.bgSigma);
            end
        else
            bg = imgaussfilt(Ig_s, P.bgSigma);
        end

        vbg = sort(bg(:));
        bgFloor = vbg(max(1, round(0.02*numel(vbg))));
        bg = max(bg, bgFloor);

        ratio = Ig_s ./ (bg + eps);

        medr = median(ratio(:));
        if medr > 0, ratio = ratio ./ medr; end

        vr = sort(ratio(:)); nvr = numel(vr);
        lo = vr(max(1, round((P.clipLoPct/100)*nvr)));
        hi = vr(min(nvr, round((P.clipHiPct/100)*nvr)));
        ratio = min(max(ratio, lo), hi);

        Icorr = mat2gray(ratio);
    end

    Ipre = Icorr;
    if P.useCLAHE && exist('adapthisteq','file')==2
        Ipre = adapthisteq(Icorr, 'ClipLimit', 0.01, 'Distribution', 'rayleigh');
    end
    imwrite(im2uint8(Ipre), dbgIpre);

    % ---- edges ----
    BW = edge(Ipre,'Canny');
    BW = bwareaopen(BW, P.minEdgeArea);

    Gm = imgradient(Ipre,'sobel');
    Gm = mat2gray(Gm);
    BW = BW | (Gm > prctile(Gm(:), P.gradPct));
    BW = bwareaopen(BW, P.minEdgeArea);

    imwrite(im2uint8(BW), dbgEdge);

    % ---- bottom line ----
    [A,Bp] = wx_find_bottom_line(BW, P);
    if isempty(A)
        % 최후 fallback: 하단 10% 위치 수평선
        y = round(0.90*H);
        A = [1, y]; Bp = [W, y];
    end

    [A,Bp] = wx_refine_line_endpoints(BW, A, Bp, P.endpointBandPx, P.endpointYPadPx);

    Iline = wx_draw_line_color(I, A, Bp, 4, [1 1 0]); % yellow
    imwrite(im2uint8(Iline), dbgLine);

    % ---- arc scan(반원/아치) ----
    sTop = wx_signed_side(A, Bp, [W/2, 1]);
    [yy,xx] = find(BW);
    Pall = [double(xx), double(yy)];
    isAbove = (wx_signed_side(A,Bp,Pall) * sTop) > 0;
    Parc = Pall(isAbove,:);

    arcPts = [];
    if size(Parc,1) >= 250
        arcPts = wx_best_arc_by_scan(A, Bp, sTop, Gm, Parc, P);
    end

    % ---- init mask ----
    if ~isempty(arcPts)
        Iarc = wx_draw_polyline_color(I, arcPts, 3, [0 1 0]); % green
        imwrite(im2uint8(Iarc), dbgArc);

        polyX = [arcPts(:,1); Bp(1); A(1)];
        polyY = [arcPts(:,2); Bp(2); A(2)];
        BWinit = poly2mask(polyX, polyY, H, W);
        BWinit = imfill(BWinit,'holes');

        if P.initDilate > 0
            BWinit = imdilate(BWinit, strel('disk', P.initDilate));
        end
        BWinit = imclose(BWinit, strel('disk', P.smoothCloseR));
        BWinit = imfill(BWinit,'holes');
        BWinit = bwareafilt(BWinit,1);

    elseif P.useThreshFallback
        BWinit = wx_initmask_by_threshold(Ipre, A, Bp, sTop, P);
    else
        error("원호/threshold 초기마스크 모두 실패");
    end

    imwrite(BWinit, dbgInit);

    % ---- refine ----
    BWface = BWinit;
    if P.doRefine && exist('activecontour','file')==2
        BWface = activecontour(Ipre, BWinit, P.refineIter, 'edge');
        BWface = imclose(BWface, strel('disk', P.smoothCloseR));
        BWface = imfill(BWface,'holes');
        BWface = bwareafilt(BWface,1);
    else
        BWface = imclose(BWface, strel('disk', P.smoothCloseR));
        BWface = imfill(BWface,'holes');
        BWface = bwareafilt(BWface,1);
    end
    imwrite(BWface, dbgRef);

    % ---- outline ----
    BWoutline = bwperim(BWface);
    if P.outlineWidth > 0
        BWoutline = imdilate(BWoutline, strel('disk', P.outlineWidth));
    end

    % ---- overlay / ROI image ----
    Irgb = im2double(I);
    if size(Irgb,3)==1, Irgb = repmat(Irgb,[1 1 3]); end

    Iov = Irgb;
    Iov(:,:,1) = Iov(:,:,1).*(~BWoutline) + 1.0.*BWoutline;
    Iov(:,:,2) = Iov(:,:,2).*(~BWoutline);
    Iov(:,:,3) = Iov(:,:,3).*(~BWoutline);

    Iface = Irgb;
    for k=1:3
        tmp = Iface(:,:,k);
        tmp(~BWface) = 0;
        Iface(:,:,k) = tmp;
    end
end

function BWinit = wx_initmask_by_threshold(Ipre, A, B, sTop, P)
    [H,W] = size(Ipre);

    T = adaptthresh(Ipre, P.adaptSens);
    BW1 = imbinarize(Ipre, T);
    BW2 = ~BW1;

    cand = {BW1, BW2};
    bestScore = -inf;
    BWinit = [];

    [yy,xx] = ndgrid(1:H, 1:W);
    PxyAll = [double(xx(:)), double(yy(:))];
    above = (wx_signed_side(A,B,PxyAll) * sTop) > 0;
    above = reshape(above, H, W);

    for k=1:2
        BW = cand{k} & above;

        BW = imclose(BW, strel('disk', P.thCloseR));
        BW = imfill(BW,'holes');
        BW = bwareafilt(BW, 3);
        if nnz(BW)==0, continue; end

        CC = bwconncomp(BW);
        if CC.NumObjects==0, continue; end

        areas = cellfun(@numel, CC.PixelIdxList);
        for j=1:CC.NumObjects
            areaFrac = areas(j)/(H*W);
            if areaFrac < P.thMinAreaFrac || areaFrac > 0.95
                continue;
            end
            tmp = false(H,W);
            tmp(CC.PixelIdxList{j}) = true;

            cx = round(W/2); cy = round(H/2);
            centerHit = tmp(cy,cx);

            score = areas(j) + 0.15*(H*W)*double(centerHit);
            if score > bestScore
                bestScore = score;
                BWinit = tmp;
            end
        end
    end

    if isempty(BWinit)
        error("threshold 기반 초기마스크 실패: bgSigma↑ 또는 adaptSens 조정 필요");
    end

    BWinit = bwareafilt(BWinit,1);
end

function s = wx_signed_side(A,B,P)
    ax=A(1); ay=A(2); bx=B(1); by=B(2);
    px=P(:,1); py=P(:,2);
    s = (bx-ax).*(py-ay) - (by-ay).*(px-ax);
end

function [A,B] = wx_find_bottom_line(BW, P)
    [H,W] = size(BW);
    A=[]; B=[];
    bestScore = -inf;

    % 1) Hough sweep
    for f = P.baseRoiFracs(:).'
        y0 = max(1, floor(f*H));
        BWroi = BW(y0:end, :);
        if nnz(BWroi) < 200, continue; end

        [Hh,Th,Rh] = hough(BWroi);
        Pk = houghpeaks(Hh, P.maxLines);

        for fg = P.fillGaps(:).'
            for mlf = P.minLenFracs(:).'
                minLen = max(20, round(mlf*W));
                lines = houghlines(BWroi, Th, Rh, Pk, 'FillGap', fg, 'MinLength', minLen);
                if isempty(lines), continue; end

                for k = 1:numel(lines)
                    p1 = double(lines(k).point1);
                    p2 = double(lines(k).point2);
                    dx = p2(1)-p1(1); dy = p2(2)-p1(2);

                    ang = atan2d(dy, dx);
                    ang = mod(ang+180, 360)-180;

                    if ~(abs(ang) <= P.angleTolDeg || abs(abs(ang)-180) <= P.angleTolDeg)
                        continue;
                    end

                    L = hypot(dx,dy);
                    yMean = (p1(2)+p2(2))/2;
                    yPref = yMean / max(1,size(BWroi,1));
                    score = L + 0.35*W*yPref - 2.0*abs(ang);

                    if score > bestScore
                        bestScore = score;
                        A0=p1; B0=p2;
                        A0(2)=A0(2)+(y0-1);
                        B0(2)=B0(2)+(y0-1);
                        A=A0; B=B0;
                    end
                end
            end
        end
    end
    if ~isempty(A), return; end

    % 2) RANSAC backup
    if P.useLineRansac
        for f = P.baseRoiFracs(:).'
            y0 = max(1, floor(f*H));
            BWroi = BW(y0:end,:);
            [yy,xx] = find(BWroi);
            if numel(xx) < 300, continue; end
            Pxy = [double(xx), double(yy + (y0-1))];

            [A1,B1,inl] = wx_ransac_horizontal_line(Pxy, W, P.lineRansacIters, P.lineInlierTolPx, P.angleTolDeg);
            if isempty(inl), continue; end
            if numel(inl) < round(P.lineMinInlierFrac * size(Pxy,1)), continue; end

            score = numel(inl);
            if score > bestScore
                bestScore = score;
                A=A1; B=B1;
            end
        end
        if ~isempty(A), return; end
    end

    % 3) 최후 backup: 하부 ROI에서 엣지 픽셀 최다 행
    y0 = max(1, floor(0.70*H));
    BWroi = BW(y0:end,:);
    rowCnt = sum(BWroi,2);
    [mx, iy] = max(rowCnt);
    if mx < 50, return; end
    y = (y0-1) + iy;

    band = 3;
    ys = max(1,y-band):min(H,y+band);
    cols = find(any(BW(ys,:),1));
    if numel(cols) >= 2
        A = [cols(1), y];
        B = [cols(end), y];
    else
        A = [1, y];
        B = [W, y];
    end
end

function [A,B,inliersBest] = wx_ransac_horizontal_line(Pxy, W, iters, tolPx, angleTolDeg)
    inliersBest = [];
    A=[]; B=[];
    N = size(Pxy,1);
    best = -inf;

    for t = 1:iters
        id = randperm(N,2);
        p1 = Pxy(id(1),:); p2 = Pxy(id(2),:);

        dx = p2(1)-p1(1); dy = p2(2)-p1(2);
        if abs(dx) < 1, continue; end

        ang = atan2d(dy, dx);
        ang = mod(ang+180, 360)-180;

        if ~(abs(ang) <= angleTolDeg || abs(abs(ang)-180) <= angleTolDeg)
            continue;
        end

        a = dy; b = -dx; c = -(a*p1(1) + b*p1(2));
        den = hypot(a,b);

        d = abs(a*Pxy(:,1) + b*Pxy(:,2) + c) / (den + eps);
        inl = find(d <= tolPx);

        score = numel(inl) - 0.5*abs(ang);
        if score > best
            best = score;
            inliersBest = inl;

            m = dy/(dx+eps);
            y_at_1 = p1(2) + m*(1 - p1(1));
            y_at_W = p1(2) + m*(W - p1(1));
            A = [1, y_at_1];
            B = [W, y_at_W];
        end
    end
end

function [A2,B2] = wx_refine_line_endpoints(BW, A, B, bandPx, yPadPx)
    [H,W] = size(BW);
    dx = B(1)-A(1); dy = B(2)-A(2);
    a = dy; b = -dx; c = -(a*A(1) + b*A(2));
    den = hypot(a,b);

    [yy,xx] = find(BW);
    Pxy = [double(xx), double(yy)];

    dist = abs(a*Pxy(:,1) + b*Pxy(:,2) + c) / (den + eps);
    yMin = min(A(2),B(2)) - yPadPx;
    yMax = max(A(2),B(2)) + yPadPx;
    ok = (dist <= bandPx) & (Pxy(:,2) >= yMin) & (Pxy(:,2) <= yMax);

    Pn = Pxy(ok,:);
    if size(Pn,1) < 30
        A2=A; B2=B; return;
    end

    [~,iL] = min(Pn(:,1));
    [~,iR] = max(Pn(:,1));
    QL = Pn(iL,:); QR = Pn(iR,:);

    dvec = [dx,dy];
    tL = dot(QL - A, dvec) / (dot(dvec,dvec)+eps);
    tR = dot(QR - A, dvec) / (dot(dvec,dvec)+eps);

    A2 = A + tL*dvec;
    B2 = A + tR*dvec;

    if A2(1) > B2(1)
        tmp=A2; A2=B2; B2=tmp;
    end

    A2(1)=min(max(A2(1),1),W); A2(2)=min(max(A2(2),1),H);
    B2(1)=min(max(B2(1),1),W); B2(2)=min(max(B2(2),1),H);
end

function arcPts = wx_best_arc_by_scan(A, B, sTop, Gm, Parc, P)
    arcPts = [];

    d = B - A;
    L = hypot(d(1), d(2));
    if L < 10, return; end
    du = d / L;

    nu = [-du(2), du(1)];
    if nu(2) > 0, nu = -nu; end  % 위쪽(y 감소)

    M = 0.5*(A + B);
    R0 = 0.5*L;

    tMax  = P.tMaxFactor * R0;
    tList = linspace(0, tMax, P.tScanN);

    bestScore = -inf;
    bestArc = [];

    [H,W] = size(Gm);
    Pxy = Parc;

    for t = tList
        C = M + t*nu;
        rr = hypot(A(1)-C(1), A(2)-C(2));
        if rr < R0*0.8 || rr > R0*20, continue; end

        angA = atan2(A(2)-C(2), A(1)-C(1));
        angB = atan2(B(2)-C(2), B(1)-C(1));

        arc1 = wx_sample_arc(C, rr, angA, angB, P.arcSamples, +1);
        arc2 = wx_sample_arc(C, rr, angA, angB, P.arcSamples, -1);

        sc1 = sum((wx_signed_side(A,B,arc1) * sTop) > 0);
        sc2 = sum((wx_signed_side(A,B,arc2) * sTop) > 0);
        arc = arc1; if sc2 > sc1, arc = arc2; end

        ok = arc(:,1)>=1 & arc(:,1)<=W & arc(:,2)>=1 & arc(:,2)<=H;
        arc = arc(ok,:);
        if size(arc,1) < 120, continue; end

        gvals = interp2(Gm, arc(:,1), arc(:,2), 'linear', 0);
        scoreGrad = mean(gvals);

        dist = abs(hypot(Pxy(:,1)-C(1), Pxy(:,2)-C(2)) - rr);
        inlFrac = mean(dist <= P.arcEdgeTol);

        score = P.scoreW_grad*scoreGrad + P.scoreW_inl*inlFrac;
        if score > bestScore
            bestScore = score;
            bestArc = arc;
        end
    end

    arcPts = bestArc;
end

function arc = wx_sample_arc(C, r, angA, angB, n, dirSign)
    angA = wx_wrapToPi(angA);
    angB = wx_wrapToPi(angB);

    if dirSign > 0
        da = wx_wrapTo2Pi(angB - angA);
        ang = angA + linspace(0, da, n).';
    else
        da = wx_wrapTo2Pi(angA - angB);
        ang = angA - linspace(0, da, n).';
    end

    x = C(1) + r*cos(ang);
    y = C(2) + r*sin(ang);
    arc = [x y];
end

function a = wx_wrapToPi(a)
    a = mod(a + pi, 2*pi) - pi;
end

function a = wx_wrapTo2Pi(a)
    a = mod(a, 2*pi);
end

function I2 = wx_draw_line_color(I, A, B, thick, rgb)
    I2 = im2double(I);
    if size(I2,3)==1, I2 = repmat(I2,[1 1 3]); end
    H = size(I2,1); W = size(I2,2);

    x1=A(1); y1=A(2); x2=B(1); y2=B(2);
    n = max(200, round(hypot(x2-x1,y2-y1)));
    xs = linspace(x1,x2,n);
    ys = linspace(y1,y2,n);

    for i=1:n
        x = round(xs(i)); y = round(ys(i));
        if x<1||x>W||y<1||y>H, continue; end
        xlo=max(1,x-thick); xhi=min(W,x+thick);
        ylo=max(1,y-thick); yhi=min(H,y+thick);
        I2(ylo:yhi,xlo:xhi,1)=rgb(1);
        I2(ylo:yhi,xlo:xhi,2)=rgb(2);
        I2(ylo:yhi,xlo:xhi,3)=rgb(3);
    end
end

function I2 = wx_draw_polyline_color(I, Pts, thick, rgb)
    I2 = im2double(I);
    if size(I2,3)==1, I2 = repmat(I2,[1 1 3]); end
    H = size(I2,1); W = size(I2,2);

    for i = 1:size(Pts,1)
        x = round(Pts(i,1)); y = round(Pts(i,2));
        if x<1||x>W||y<1||y>H, continue; end
        xlo=max(1,x-thick); xhi=min(W,x+thick);
        ylo=max(1,y-thick); yhi=min(H,y+thick);
        I2(ylo:yhi,xlo:xhi,1)=rgb(1);
        I2(ylo:yhi,xlo:xhi,2)=rgb(2);
        I2(ylo:yhi,xlo:xhi,3)=rgb(3);
    end
end

%% ======================================================================
%%                 wx_CoreWeatheringAnalyzer_Max2026 (내장)
%% ======================================================================
function wx_CoreWeatheringAnalyzer_Max2026(statsCsv, outXlsx, varargin)
% wx_CoreWeatheringAnalyzer_Max2026 (ONE-FILE, igneous preset thresholds)
% - 입력 CSV에서 PRIMARY_VAR 기준으로 D1~D5 예측
% - BIN_EDGES, IMAGE_PRED, SUMMARY_BY_PRED, (optional) CONFUSION, INDEX 출력
%
% (중요 수정) table VariableNames는 모두 char(cellstr)로 강제하여 오류 방지.

p = inputParser;
p.addRequired('statsCsv', @(s)ischar(s)||isstring(s));
p.addRequired('outXlsx',  @(s)ischar(s)||isstring(s)||isempty(s));

p.addParameter('BIN_VERSION', 2, @(x)isscalar(x)&&any(x==[1 2 3]));
p.addParameter('USE_IGNEOUS_PRESET', true, @(x)islogical(x)&&isscalar(x));

p.addParameter('PRIMARY_VAR', "L_mean", @(s)ischar(s)||isstring(s));
p.addParameter('PRIMARY_DIRECTION', "auto", @(s)any(strcmpi(string(s),["auto","high_is_fresh","high_is_weathered"])) );

p.addParameter('VAR_LIST', ["R_mean","G_mean","B_mean","GrayY_mean","H_mean","S_mean","V_mean","L_mean","a_mean","b_mean"], ...
    @(x)isstring(x)||iscellstr(x));

p.parse(statsCsv, outXlsx, varargin{:});
OPT = p.Results;

statsCsv = string(statsCsv);
if ~isfile(statsCsv)
    error("statsCsv가 없습니다: %s", statsCsv);
end

if nargin < 2 || isempty(outXlsx)
    [sd, sb, ~] = fileparts(statsCsv);
    outXlsx = fullfile(sd, sb + "_WEATHERING_ANALYSIS.xlsx");
end
outXlsx = char(outXlsx);

% 출력 파일은 매번 새로(이전 시트 잔상 방지)
if isfile(outXlsx)
    try delete(outXlsx); catch, end
end

% ---- Read ----
opts = detectImportOptions(statsCsv, 'NumHeaderLines', 0);
T = readtable(statsCsv, opts);

if ~any(string(T.Properties.VariableNames)=="file")
    error("CSV에 'file' 컬럼이 없습니다.");
end
if ~any(string(T.Properties.VariableNames)=="filename")
    % filename 없으면 생성
    T.filename = strings(height(T),1);
    for i=1:height(T)
        [~,b,e] = fileparts(char(string(T.file(i))));
        T.filename(i) = string([b e]);
    end
end
if ~any(string(T.Properties.VariableNames)=="rw_grade")
    T.rw_grade = strings(height(T),1);
end

T.file     = string(T.file);
T.filename = string(T.filename);
T.rw_grade = wx_normalize_grade_vec(T.rw_grade);

primary = string(OPT.PRIMARY_VAR);
if ~any(string(T.Properties.VariableNames)==primary)
    error("CSV에 PRIMARY_VAR(%s) 컬럼이 없습니다.", primary);
end

varList = string(OPT.VAR_LIST(:))';
varList = varList(ismember(varList, string(T.Properties.VariableNames)));

% ---- edgesMap ----
edgesMap = containers.Map('KeyType','char','ValueType','any');
presetMap = containers.Map('KeyType','char','ValueType','any');
if OPT.USE_IGNEOUS_PRESET
    presetMap = wx_get_igneous_preset_edges(OPT.BIN_VERSION);
end

for v = varList
    if OPT.USE_IGNEOUS_PRESET && isKey(presetMap, char(v))
        edges6 = presetMap(char(v));
    else
        x = double(T.(v));
        edges6 = wx_compute_edges6(x, OPT.BIN_VERSION, 200000, 42);
    end
    edgesMap(char(v)) = edges6(:).';
end

% ---- edges table ----
edgesNames = {'variable','edge1','edge2','edge3','edge4','edge5','edge6','bin_version','source'};
E = table('Size',[numel(varList) numel(edgesNames)], ...
    'VariableTypes',["string","double","double","double","double","double","double","double","string"], ...
    'VariableNames',edgesNames);

for i=1:numel(varList)
    v = varList(i);
    e6 = edgesMap(char(v));
    E.variable(i) = v;
    E.edge1(i)=e6(1); E.edge2(i)=e6(2); E.edge3(i)=e6(3);
    E.edge4(i)=e6(4); E.edge5(i)=e6(5); E.edge6(i)=e6(6);
    E.bin_version(i)=OPT.BIN_VERSION;

    if OPT.USE_IGNEOUS_PRESET && isKey(presetMap, char(v))
        E.source(i) = "IGNEOUS_PRESET";
    else
        E.source(i) = "DATA_FALLBACK";
    end
end

% ---- direction ----
dir = string(OPT.PRIMARY_DIRECTION);
if strcmpi(dir,"auto")
    dir = wx_default_direction_for_primary(primary);
end

% ---- assign pred_grade ----
primaryEdges = edgesMap(char(primary));
xP = double(T.(primary));
T.pred_grade = wx_bin5_labels_vector(xP, primaryEdges, dir);

% ---- summaries ----
sumPred = wx_group_summary(T, "pred_grade", primary);

% ---- confusion (있으면) ----
confT = table;
hasGT = any(strlength(T.rw_grade)>0);
if hasGT
    gt = categorical(T.rw_grade, ["D1","D2","D3","D4","D5"]);
    pr = categorical(T.pred_grade, ["D1","D2","D3","D4","D5"]);
    [C,gtCats,prCats] = crosstab(gt, pr);
    confT = array2table(C, 'VariableNames', cellstr("Pred_" + string(prCats)));
    confT = addvars(confT, string(gtCats), 'Before', 1, 'NewVariableNames',"GT");
end

% ---- index ----
Index = table( ...
    "wx_CoreWeatheringAnalyzer_Max2026", ...
    string(datetime('now')), ...
    string(statsCsv), ...
    string(outXlsx), ...
    string(OPT.BIN_VERSION), ...
    string(OPT.USE_IGNEOUS_PRESET), ...
    primary, ...
    dir, ...
    strjoin(varList, ", "), ...
    'VariableNames',{'tool','timestamp','statsCsv','outXlsx','bin_version','use_igneous_preset','primary_var','primary_direction','var_list'} ...
);

% ---- export table ----
Texport = T(:, intersect(["file","filename","rw_grade","pred_grade",primary], string(T.Properties.VariableNames), 'stable'));

% ---- write xlsx ----
writetable(E,       outXlsx, 'Sheet','BIN_EDGES',       'UseExcel',false);
writetable(Texport, outXlsx, 'Sheet','IMAGE_PRED',      'UseExcel',false);
writetable(sumPred, outXlsx, 'Sheet','SUMMARY_BY_PRED', 'UseExcel',false);
writetable(Index,   outXlsx, 'Sheet','INDEX',           'UseExcel',false);

if ~isempty(confT) && height(confT)>0
    writetable(confT, outXlsx, 'Sheet','CONFUSION', 'UseExcel',false);
end

fprintf("=== wx_CoreWeatheringAnalyzer_Max2026 done ===\n");
fprintf("Input CSV : %s\n", statsCsv);
fprintf("Output XLSX: %s\n", outXlsx);
fprintf("PRIMARY_VAR: %s (%s)\n", primary, dir);

end

%% --------- Analyzer helpers ---------
function presetMap = wx_get_igneous_preset_edges(binVersion)
presetMap = containers.Map('KeyType','char','ValueType','any');
switch binVersion
    case 1
        presetMap('R_mean')     = [13.74 52.99 92.24 131.49 170.74 209.99];
        presetMap('G_mean')     = [13.84 54.01 94.19 134.36 174.54 214.71];
        presetMap('B_mean')     = [14.52 55.19 95.85 136.51 177.17 217.83];
        presetMap('GrayY_mean') = [13.87 53.88 93.89 133.91 173.92 213.93];
        presetMap('H_mean')     = [9.36 41.04 72.71 104.39 136.07 167.74];
        presetMap('S_mean')     = [1.79 31.92 62.04 92.16 122.29 152.41];
        presetMap('V_mean')     = [14.53 55.21 95.90 136.59 177.28 217.96];
        presetMap('L_mean')     = [4.23 20.49 36.75 53.01 69.27 85.53];
        presetMap('a_mean')     = [-6.37 -2.03 2.31 6.64 10.98 15.32];
        presetMap('b_mean')     = [-9.35 -1.96 5.42 12.81 20.20 27.58];
    case 2
        presetMap('R_mean')     = [60.41 82.48 104.56 126.63 148.71 170.78];
        presetMap('G_mean')     = [63.96 85.98 108.00 130.02 152.04 174.06];
        presetMap('B_mean')     = [61.42 84.25 107.08 129.91 152.74 175.58];
        presetMap('GrayY_mean') = [63.29 85.28 107.27 129.25 151.24 173.23];
        presetMap('H_mean')     = [25.87 49.58 73.30 97.01 120.73 144.44];
        presetMap('S_mean')     = [8.69 17.32 25.96 34.59 43.22 51.86];
        presetMap('V_mean')     = [66.34 88.58 110.82 133.06 155.30 177.54];
        presetMap('L_mean')     = [26.51 35.33 44.15 52.97 61.79 70.61];
        presetMap('a_mean')     = [-3.25 -2.19 -1.13 -0.07 0.99 2.05];
        presetMap('b_mean')     = [-4.77 -2.22 0.33 2.88 5.42 7.97];
    case 3
        presetMap('R_mean')     = [13.74 85.60 106.28 125.73 146.91 209.99];
        presetMap('G_mean')     = [13.84 88.67 109.09 128.99 151.33 214.71];
        presetMap('B_mean')     = [14.52 85.90 107.67 128.50 152.53 217.83];
        presetMap('GrayY_mean') = [13.87 88.08 108.37 128.25 150.24 213.93];
        presetMap('H_mean')     = [9.36 53.48 103.14 127.81 138.60 167.74];
        presetMap('S_mean')     = [1.79 14.02 19.89 25.92 33.78 152.41];
        presetMap('V_mean')     = [14.53 91.88 113.05 132.79 155.48 217.96];
        presetMap('L_mean')     = [4.23 37.07 45.46 53.44 61.97 85.53];
        presetMap('a_mean')     = [-6.37 -2.37 -1.81 -1.27 -0.43 15.32];
        presetMap('b_mean')     = [-9.35 -3.00 -1.58 -0.06 2.64 27.58];
end
end

function dir = wx_default_direction_for_primary(primaryVar)
v = string(primaryVar);
if any(v == ["a_mean","b_mean"])
    dir = "high_is_weathered";
else
    dir = "high_is_fresh";
end
end

function edges6 = wx_compute_edges6(x, binVersion, maxSample, rngSeed)
x = x(:);
x = x(isfinite(x));
if isempty(x)
    edges6 = linspace(0,1,6);
    return;
end

n = numel(x);
if n > maxSample
    rng(rngSeed);
    idx = randperm(n, maxSample);
    xs = x(idx);
else
    xs = x;
end

xs = sort(double(xs));
xmin = min(xs);
xmax = max(xs);
if xmin == xmax
    epsv = max(1e-6, abs(xmin)*1e-6);
    xmin = xmin - epsv;
    xmax = xmax + epsv;
end

switch binVersion
    case 1
        edges6 = linspace(xmin, xmax, 6);
    case 2
        p5  = wx_quantile_sorted(xs, 0.05);
        p95 = wx_quantile_sorted(xs, 0.95);
        if p5 == p95
            edges6 = linspace(xmin, xmax, 6);
        else
            edges6 = linspace(p5, p95, 6);
            edges6 = wx_make_edges_strict_inc(edges6);
        end
    case 3
        q0   = wx_quantile_sorted(xs, 0.00);
        q20  = wx_quantile_sorted(xs, 0.20);
        q40  = wx_quantile_sorted(xs, 0.40);
        q60  = wx_quantile_sorted(xs, 0.60);
        q80  = wx_quantile_sorted(xs, 0.80);
        q100 = wx_quantile_sorted(xs, 1.00);
        edges6 = [q0 q20 q40 q60 q80 q100];
        edges6 = wx_make_edges_strict_inc(edges6);
end
end

function edges6 = wx_make_edges_strict_inc(edges6)
edges6 = double(edges6(:))';
for i=2:numel(edges6)
    if edges6(i) <= edges6(i-1)
        edges6(i) = edges6(i-1) + max(1e-6, abs(edges6(i-1))*1e-6);
    end
end
end

function q = wx_quantile_sorted(xs, p)
xs = double(xs(:));
n = numel(xs);
if n == 1, q = xs(1); return; end
pos = 1 + (n-1)*p;
lo = floor(pos); hi = ceil(pos);
lo = max(1, min(n, lo));
hi = max(1, min(n, hi));
if lo == hi
    q = xs(lo);
else
    q = xs(lo) + (pos-lo) * (xs(hi) - xs(lo));
end
end

function lab = wx_bin5_labels_vector(x, edges6, direction)
x = double(x(:));
e = double(edges6(:))';

b = discretize(x, e);     % 1..5 inside, NaN outside
b(x < e(1))    = 1;
b(x >= e(end)) = 5;
b(isnan(b))    = 3;

direction = lower(string(direction));
switch direction
    case "high_is_fresh"
        bh = 6 - b; % high -> D1
    case "high_is_weathered"
        bh = b;     % high -> D5
    otherwise
        bh = 6 - b;
end

lab = "D" + string(bh);
end

function g = wx_normalize_grade_vec(gin)
g = string(gin);
g = strtrim(upper(g));
g = replace(g," ","");
g = erase(g,'"'); g = erase(g,"'");

wd = startsWith(g,"WD");
g(wd) = "D" + extractAfter(g(wd), 2);

for i=1:numel(g)
    if strlength(g(i))==0, continue; end
    tok = regexp(char(g(i)), '(D|W)[-_]?(\d+)', 'tokens','once');
    if ~isempty(tok)
        g(i) = string(tok{1}{1}) + string(tok{1}{2});
    else
        tok2 = regexp(char(g(i)), '\d+', 'match','once');
        if ~isempty(tok2)
            g(i) = "D" + string(tok2);
        end
    end
end
end

function S = wx_group_summary(T, groupVar, primaryVar)
% (중요) VariableNames를 char 셀 배열로 강제하여 table 오류 방지
groupVar   = string(groupVar);
primaryVar = string(primaryVar);

g = string(T.(groupVar));
[G, cats] = findgroups(g);

n = splitapply(@numel, g, G);

x = double(T.(primaryVar));
pv_mean = splitapply(@(z) mean(z,'omitnan'),   x, G);
pv_std  = splitapply(@(z) std(z,'omitnan'),    x, G);
pv_p50  = splitapply(@(z) median(z,'omitnan'), x, G);

vn = { ...
    char(groupVar), ...
    'n', ...
    char(primaryVar + "_mean"), ...
    char(primaryVar + "_std"), ...
    char(primaryVar + "_median") ...
};

S = table(cats, n, pv_mean, pv_std, pv_p50, 'VariableNames', vn);
end
