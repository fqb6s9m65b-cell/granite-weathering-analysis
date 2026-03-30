%% Tunnel Face Outline: Flat Bottom + Arc (NO Arc-RANSAC, Robust Scan)
clc; clear; close all;

%% ===== 결과 저장 폴더(고정) =====
outDir = 'C:\Users\ROCKENG\Desktop\코랩 머신러닝\COLAB_OUT';
if ~exist(outDir,'dir'), mkdir(outDir); end

%% ===== 입력 이미지 선택(사용자) =====
[fn, fp] = uigetfile({'*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp','Image Files'}, ...
                     '막장면 이미지를 선택하세요');
if isequal(fn,0), disp('취소됨'); return; end
imgPath = fullfile(fp, fn);
[~, baseName, ~] = fileparts(fn);

outMask    = fullfile(outDir, baseName + "_face_mask.png");
outOutline = fullfile(outDir, baseName + "_face_outline.png");
outOverlay = fullfile(outDir, baseName + "_face_overlay.png");

dbgIpre    = fullfile(outDir, baseName + "_dbg_Ipre.png");
dbgEdge    = fullfile(outDir, baseName + "_dbg_edges.png");
dbgLine    = fullfile(outDir, baseName + "_dbg_bottomLine.png");
dbgArc     = fullfile(outDir, baseName + "_dbg_arcCandidate.png");
dbgInit    = fullfile(outDir, baseName + "_dbg_initMask.png");
dbgRef     = fullfile(outDir, baseName + "_dbg_refinedMask.png");

%% ===== 파라미터 =====
% 전처리
sigma       = 1.2;
doIllumCorr = true;
bgSigma     = 75;      % (중요) 조도 편차 심하면 90~120까지 올려도 됨
useCLAHE    = true;
clipLoPct   = 1;
clipHiPct   = 99;

% 엣지
minEdgeArea = 80;

% 바닥선(자동 스윕 + 백업)
baseRoiFracs = [0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80];
angleTolDeg  = 25;
maxLines     = 30;
fillGaps     = [30 60 90 120 160];
minLenFracs  = [0.35 0.25 0.18 0.12 0.08];

useLineRansac = true;
lineRansacIters = 2500;
lineInlierTolPx = 2.5;
lineMinInlierFrac = 0.03;

% 바닥선 끝점 보정(바닥선 주변 엣지로 좌우 끝 다시 잡기)
endpointBandPx = 4;     % 바닥선으로부터 거리(픽셀)
endpointYPadPx = 25;    % 바닥선 주변 y 범위

% 원호(스캔 기반)
tScanN      = 120;     % 스캔 분해능(시간↑면 ↑)
tMaxFactor  = 3.0;     % 최대 이동 = tMaxFactor * (L/2)
arcSamples  = 700;
arcEdgeTol  = 3.0;     % 원호 점수 계산 시 엣지 근접 허용
scoreW_grad = 1.0;     % 그라디언트 점수 가중
scoreW_inl  = 2.0;     % 엣지-원호 근접(inlier) 점수 가중

% 마스크/정밀화
initDilate  = 6;       % 초기 마스크 약간 키우기
smoothCloseR= 6;
doRefine    = true;
refineIter  = 220;     % 120~400
outlineWidth= 3;

showFigures = true;

%% ===== 1) 읽기/그레이 =====
I = imread(imgPath);
if size(I,3) == 4, I = I(:,:,1:3); end

if size(I,3) == 3
    Ig = rgb2gray(I);
else
    Ig = I;
end
Ig = im2double(Ig);
[H,W] = size(Ig);

%% ===== 2) 전처리(Ipre) =====
Ig_s = imgaussfilt(Ig, sigma);

Icorr = Ig_s;
if doIllumCorr
    bg = imgaussfilt(Ig_s, bgSigma);

    vbg = sort(bg(:));
    bgFloor = vbg(max(1, round(0.02*numel(vbg))));
    bg = max(bg, bgFloor);

    ratio = Ig_s ./ (bg + eps);

    medr = median(ratio(:));
    if medr > 0, ratio = ratio ./ medr; end

    vr  = sort(ratio(:)); nvr = numel(vr);
    lo  = vr(max(1, round((clipLoPct/100)*nvr)));
    hi  = vr(min(nvr, round((clipHiPct/100)*nvr)));
    ratio = min(max(ratio, lo), hi);

    Icorr = mat2gray(ratio);
end

Ipre = Icorr;
if useCLAHE && exist('adapthisteq','file') == 2
    Ipre = adapthisteq(Icorr, 'ClipLimit', 0.01, 'Distribution', 'rayleigh');
end
imwrite(im2uint8(Ipre), dbgIpre);

%% ===== 3) 엣지(바닥선+원호 둘 다 쓰임) =====
BW = edge(Ipre, 'Canny');
BW = bwareaopen(BW, minEdgeArea);

% 약한 외곽을 위해 gradient 엣지를 조금 보강
Gm = imgradient(Ipre, 'sobel');
Gm = mat2gray(Gm);
BW = BW | (Gm > prctile(Gm(:), 92));
BW = bwareaopen(BW, minEdgeArea);

imwrite(im2uint8(BW), dbgEdge);

%% ===== 4) 바닥 직선 검출 =====
[A, B] = local_find_bottom_line_robust(BW, baseRoiFracs, angleTolDeg, maxLines, fillGaps, minLenFracs, ...
    useLineRansac, lineRansacIters, lineInlierTolPx, lineMinInlierFrac);

if isempty(A)
    error('바닥선 검출 실패. (1) baseRoiFracs에 0.85 추가 (2) bgSigma↑ (3) minLenFracs 더 낮추기');
end

%% ===== 4-1) 바닥선 끝점 보정(바닥선 주변 엣지로 좌/우 끝 재설정) =====
[A, B] = local_refine_line_endpoints(BW, A, B, endpointBandPx, endpointYPadPx);

% 디버그: 바닥선 표시
Iline = local_draw_line_yellow(I, A, B, 4);
imwrite(im2uint8(Iline), dbgLine);

%% ===== 5) 원호: RANSAC 대신 "수직이등분선 방향 스캔"으로 최적 원 선택 =====
% 바닥선 위쪽을 정의(이미지 상단이 위)
sTop = local_signed_side(A, B, [W/2, 1]);

% 원호 점수 계산용 엣지 포인트(바닥선 위쪽만)
[yy,xx] = find(BW);
Pall = [double(xx), double(yy)];
isAbove = (local_signed_side(A,B,Pall) * sTop) > 0;
Parc = Pall(isAbove,:);

if size(Parc,1) < 400
    error('바닥선 위쪽 엣지 포인트가 너무 적음: bgSigma↑ 또는 percentile(92→90)로 엣지 보강');
end

[cx, cy, r, arcPts] = local_best_arc_by_scan(A, B, sTop, Gm, Parc, ...
    tScanN, tMaxFactor, arcSamples, arcEdgeTol, scoreW_grad, scoreW_inl);

if isempty(arcPts)
    error('원호 스캔 실패: tMaxFactor↑(3→5) 또는 bgSigma↑ 또는 arcEdgeTol↑(3→5)');
end

% 디버그: 후보 원호 표시(초록)
Iarc = local_draw_polyline_green(I, arcPts, 3);
imwrite(im2uint8(Iarc), dbgArc);

%% ===== 6) 폴리곤(원호 + 바닥선) -> 초기 마스크 =====
polyX = [arcPts(:,1); B(1); A(1)];
polyY = [arcPts(:,2); B(2); A(2)];

BWinit = poly2mask(polyX, polyY, H, W);
BWinit = imfill(BWinit, 'holes');
if initDilate > 0
    BWinit = imdilate(BWinit, strel('disk', initDilate));
end
BWinit = imclose(BWinit, strel('disk', smoothCloseR));
BWinit = imfill(BWinit, 'holes');
BWinit = bwareafilt(BWinit, 1);

imwrite(BWinit, dbgInit);

%% ===== 7) (선택) activecontour로 경계에 붙이기 =====
BWface = BWinit;
if doRefine && exist('activecontour','file') == 2
    BWface = activecontour(Ipre, BWinit, refineIter, 'edge');
    BWface = imclose(BWface, strel('disk', smoothCloseR));
    BWface = imfill(BWface,'holes');
    BWface = bwareafilt(BWface, 1);
end
imwrite(BWface, dbgRef);

%% ===== 8) 윤곽선만 + 오버레이 =====
BWoutline = bwperim(BWface);
if outlineWidth > 0
    BWoutline = imdilate(BWoutline, strel('disk', outlineWidth));
end

Irgb = im2double(I);
if size(Irgb,3)==1, Irgb = repmat(Irgb,[1 1 3]); end

Iov = Irgb;
Iov(:,:,1) = Iov(:,:,1).*(~BWoutline) + 1.0.*BWoutline;
Iov(:,:,2) = Iov(:,:,2).*(~BWoutline);
Iov(:,:,3) = Iov(:,:,3).*(~BWoutline);

imwrite(BWface, outMask);
imwrite(BWoutline, outOutline);
imwrite(im2uint8(Iov), outOverlay);

disp("저장 완료:");
disp(" - " + outMask);
disp(" - " + outOutline);
disp(" - " + outOverlay);
disp(" - " + dbgIpre);
disp(" - " + dbgEdge);
disp(" - " + dbgLine);
disp(" - " + dbgArc);
disp(" - " + dbgInit);
disp(" - " + dbgRef);

%% ===== 표시 =====
if showFigures
    figure('Color','w','Name','Tunnel Face: Flat Bottom + Arc (Scan Prior)');
    subplot(2,3,1); imshow(I);           title('Original');
    subplot(2,3,2); imshow(Ipre,[]);     title('Ipre');
    subplot(2,3,3); imshow(BW);          title('Edges');
    subplot(2,3,4); imshow(BWinit);      title('Init mask (arc+line)');
    subplot(2,3,5); imshow(BWoutline);   title('Outline only');
    subplot(2,3,6); imshow(Iov);         title('Overlay (Red)');
end

%% ===================== Local Functions =====================
function s = local_signed_side(A,B,P)
    ax=A(1); ay=A(2); bx=B(1); by=B(2);
    px=P(:,1); py=P(:,2);
    s = (bx-ax).*(py-ay) - (by-ay).*(px-ax);
end

function [A, B] = local_find_bottom_line_robust(BW, baseRoiFracs, angleTolDeg, maxLines, fillGaps, minLenFracs, ...
        useRansac, iters, tolPx, minInlierFrac)

    [H,W] = size(BW);
    A=[]; B=[];
    bestScore = -inf;

    % Hough 스윕
    for f = baseRoiFracs(:).'
        y0 = max(1, floor(f*H));
        BWroi = BW(y0:end, :);
        if nnz(BWroi) < 200, continue; end

        [Hh,Th,Rh] = hough(BWroi);
        Pk = houghpeaks(Hh, maxLines);

        for fg = fillGaps(:).'
            for mlf = minLenFracs(:).'
                minLen = round(mlf*W);
                lines = houghlines(BWroi, Th, Rh, Pk, 'FillGap', fg, 'MinLength', minLen);
                if isempty(lines), continue; end

                for k = 1:numel(lines)
                    p1 = double(lines(k).point1);
                    p2 = double(lines(k).point2);

                    dx = p2(1)-p1(1); dy = p2(2)-p1(2);
                    ang = atan2d(dy, dx);
                    ang = mod(ang+180, 360)-180;

                    if ~(abs(ang) <= angleTolDeg || abs(abs(ang)-180) <= angleTolDeg)
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

    % 백업: RANSAC 수평선
    if ~useRansac, return; end

    for f = baseRoiFracs(:).'
        y0 = max(1, floor(f*H));
        BWroi = BW(y0:end,:);
        [yy,xx] = find(BWroi);
        if numel(xx) < 300, continue; end
        P = [double(xx), double(yy + (y0-1))];

        [A1,B1,inliers] = local_ransac_horizontal_line(P, W, iters, tolPx, angleTolDeg);
        if isempty(inliers), continue; end
        if numel(inliers) < round(minInlierFrac * size(P,1)), continue; end

        score = numel(inliers);
        if score > bestScore
            bestScore = score;
            A=A1; B=B1;
        end
    end
end

function [A,B,inliersBest] = local_ransac_horizontal_line(P, W, iters, tolPx, angleTolDeg)
    inliersBest = [];
    A=[]; B=[];
    N = size(P,1);
    best = -inf;

    for t = 1:iters
        id = randperm(N,2);
        p1 = P(id(1),:); p2 = P(id(2),:);

        dx = p2(1)-p1(1); dy = p2(2)-p1(2);
        if abs(dx) < 1, continue; end

        ang = atan2d(dy, dx);
        ang = mod(ang+180, 360)-180;

        if ~(abs(ang) <= angleTolDeg || abs(abs(ang)-180) <= angleTolDeg)
            continue;
        end

        a = dy; b = -dx; c = -(a*p1(1) + b*p1(2));
        den = hypot(a,b);

        d = abs(a*P(:,1) + b*P(:,2) + c) / (den + eps);
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

function [A2,B2] = local_refine_line_endpoints(BW, A, B, bandPx, yPadPx)
    % 바닥선 가까운 엣지 픽셀 중 좌/우 극값으로 끝점 재설정
    [H,W] = size(BW);
    dx = B(1)-A(1); dy = B(2)-A(2);
    a = dy; b = -dx; c = -(a*A(1) + b*A(2));
    den = hypot(a,b);

    [yy,xx] = find(BW);
    P = [double(xx), double(yy)];

    % 바닥선 근처(거리 band) + y 범위 제한
    dist = abs(a*P(:,1) + b*P(:,2) + c) / (den + eps);
    yMin = min(A(2),B(2)) - yPadPx;
    yMax = max(A(2),B(2)) + yPadPx;
    ok = (dist <= bandPx) & (P(:,2) >= yMin) & (P(:,2) <= yMax);

    Pn = P(ok,:);
    if size(Pn,1) < 30
        A2=A; B2=B; return;
    end

    [~,iL] = min(Pn(:,1));
    [~,iR] = max(Pn(:,1));
    QL = Pn(iL,:); QR = Pn(iR,:);

    % QL/QR를 직선 AB에 직교투영
    d = [dx,dy];
    tL = dot(QL - A, d) / (dot(d,d)+eps);
    tR = dot(QR - A, d) / (dot(d,d)+eps);

    A2 = A + tL*d;
    B2 = A + tR*d;

    % 정렬(왼쪽이 A가 되도록)
    if A2(1) > B2(1)
        tmp=A2; A2=B2; B2=tmp;
    end

    % 범위 클램프
    A2(1)=min(max(A2(1),1),W); A2(2)=min(max(A2(2),1),H);
    B2(1)=min(max(B2(1),1),W); B2(2)=min(max(B2(2),1),H);
end

function [cx, cy, r, arcPts] = local_best_arc_by_scan(A, B, sTop, Gm, Parc, ...
    tScanN, tMaxFactor, arcSamples, arcEdgeTol, wGrad, wInl)

    arcPts = [];
    cx=[]; cy=[]; r=[];

    % 선 방향/법선(위쪽으로)
    d = B - A;
    L = hypot(d(1), d(2));
    if L < 10, return; end
    du = d / L;

    nu = [-du(2), du(1)];        % 법선
    if nu(2) > 0, nu = -nu; end  % 위쪽(y 감소)으로

    M = 0.5*(A + B);
    R0 = 0.5*L;

    tMax = tMaxFactor * R0;
    tList = linspace(0, tMax, tScanN);

    bestScore = -inf;
    bestC = [];
    bestArc = [];

    % 점수 계산 준비(gradient는 interp2로 샘플)
    [H,W] = size(Gm);

    % Parc는 바닥선 위쪽 엣지들
    P = Parc;

    for t = tList
        C = M + t*nu;
        rr = hypot(A(1)-C(1), A(2)-C(2));

        if rr < R0*0.8 || rr > R0*20
            continue;
        end

        % A,B 각도
        angA = atan2(A(2)-C(2), A(1)-C(1));
        angB = atan2(B(2)-C(2), B(1)-C(1));

        % 두 방향 arc 생성 후 위쪽 점 많은 쪽 선택
        arc1 = local_sample_arc(C, rr, angA, angB, arcSamples, +1);
        arc2 = local_sample_arc(C, rr, angA, angB, arcSamples, -1);

        sc1 = sum((local_signed_side(A,B,arc1) * sTop) > 0);
        sc2 = sum((local_signed_side(A,B,arc2) * sTop) > 0);
        if sc2 > sc1
            arc = arc2;
        else
            arc = arc1;
        end

        % arc 점들이 화면 안에 들어오는 것만
        ok = arc(:,1)>=1 & arc(:,1)<=W & arc(:,2)>=1 & arc(:,2)<=H;
        arc = arc(ok,:);
        if size(arc,1) < 100, continue; end

        % (1) gradient 점수: 원호 위 평균 그라디언트
        gvals = interp2(Gm, arc(:,1), arc(:,2), 'linear', 0);
        scoreGrad = mean(gvals);

        % (2) inlier 점수: 엣지 포인트가 원호 근처에 얼마나 붙는지
        dist = abs(hypot(P(:,1)-C(1), P(:,2)-C(2)) - rr);
        inlFrac = mean(dist <= arcEdgeTol);

        score = wGrad*scoreGrad + wInl*inlFrac;

        if score > bestScore
            bestScore = score;
            bestC = C;
            bestArc = arc;
            bestR = rr;
        end
    end

    if isempty(bestArc)
        return;
    end

    cx = bestC(1); cy = bestC(2); r = bestR;
    arcPts = bestArc;
end

function arc = local_sample_arc(C, r, angA, angB, n, dirSign)
    angA = local_wrapToPi(angA);
    angB = local_wrapToPi(angB);

    if dirSign > 0
        da = local_wrapTo2Pi(angB - angA);
        ang = angA + linspace(0, da, n).';
    else
        da = local_wrapTo2Pi(angA - angB);
        ang = angA - linspace(0, da, n).';
    end

    x = C(1) + r*cos(ang);
    y = C(2) + r*sin(ang);
    arc = [x y];
end

function a = local_wrapToPi(a)
    a = mod(a + pi, 2*pi) - pi;
end

function a = local_wrapTo2Pi(a)
    a = mod(a, 2*pi);
end

function I2 = local_draw_line_yellow(I, A, B, thick)
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
        I2(ylo:yhi,xlo:xhi,1)=1; I2(ylo:yhi,xlo:xhi,2)=1; I2(ylo:yhi,xlo:xhi,3)=0;
    end
end

function I2 = local_draw_polyline_green(I, P, thick)
    I2 = im2double(I);
    if size(I2,3)==1, I2 = repmat(I2,[1 1 3]); end
    H = size(I2,1); W = size(I2,2);

    for i = 1:size(P,1)
        x = round(P(i,1)); y = round(P(i,2));
        if x<1||x>W||y<1||y>H, continue; end
        xlo=max(1,x-thick); xhi=min(W,x+thick);
        ylo=max(1,y-thick); yhi=min(H,y+thick);
        I2(ylo:yhi,xlo:xhi,1)=0; I2(ylo:yhi,xlo:xhi,2)=1; I2(ylo:yhi,xlo:xhi,3)=0;
    end
end
