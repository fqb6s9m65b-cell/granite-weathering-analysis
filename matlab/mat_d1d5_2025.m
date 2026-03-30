function mat_d1d5_2025()
% core_wd5_resnet18_patch_analyzer (세밀 타일 + 신뢰도/스칼라 지표 버전)
%  - net_resnet18_d1d5.mat (입력 224x224) 기반 WD-5 패치 분류
%  - 큰 코어 이미지를 작은 타일로 나눠 WD-5 패치맵/오버레이 생성
%  - (PyTorch-style) ImageNet 전처리 ((I - mu) ./ sig) 적용
%  - ROI 마스크 없이, 이미지 전체를 겹치는 타일로 분석
%  - 주요 기능:
%       · 패치별 확률(scores), 좌표, 반경 정보 저장
%       · WD-5 분포, radial 분포, 패치맵, 오버레이 시각화
%       · summary + patch detail + scalar summary 엑셀 저장
%       · MAT에 analysisMeta 구조체로 메타데이터/지표 기록
%       · 신뢰도 기반 평균 등급, 가중 평균 등급, D4–D5 비율 등 스칼라 지표 제공
%
%  * 타일 크기(원본 위 공간 해상도)는 tileScale 로 조절:
%      - tileScale = 1.0 → 224x224 (네트 입력 크기와 동일)
%      - tileScale = 0.5 → 112x112 (2배 세밀)
%      - tileScale = 0.25 → 56x56 (4배 세밀)

    %% 1. 네트워크 로드
    matDefault = 'net_resnet18_d1d5.mat';
    if exist(matDefault,'file')
        matPath = matDefault;
        fprintf('[Info] MAT 파일 사용: %s\n', matPath);
    else
        [fnM,fpM] = uigetfile('*.mat','훈련된 네트워크 MAT 파일 선택');
        if isequal(fnM,0)
            disp('취소됨.'); return;
        end
        matPath = fullfile(fpM,fnM);
    end

    S = load(matPath);
    if isfield(S,'netTrained')
        net = S.netTrained;
        fprintf('[Info] 변수 netTrained 사용.\n');
    elseif isfield(S,'net')
        net = S.net;
        fprintf('[Info] 변수 net 사용.\n');
    else
        error('MAT 파일 안에 ''net'' 또는 ''netTrained'' 변수가 없습니다.');
    end

    % 네트워크 입력 크기 확인 (예: [224 224 3])
    inputSize = net.Layers(1).InputSize;
    if numel(inputSize) ~= 3
        error('이 코드는 3채널 2D 입력 네트워크만 지원합니다. (InputSize = [H W 3])');
    end
    netH = inputSize(1);   % 네트 입력 높이
    netW = inputSize(2);   % 네트 입력 너비

    % ====== 실제 잘라오는 타일 크기 설정 (분석 해상도 조절용) ======
    % 더 세밀하게 보고 싶으면 tileScale 을 0.5, 0.25 등으로 줄이면 됨.
    tileScale = 0.5;                 % 예: 0.5 → 112x112 타일
    tileH = max(16, round(netH*tileScale));
    tileW = max(16, round(netW*tileScale));

    fprintf('[Info] 네트 입력 크기: %d x %d, 타일 크기(원본 상): %d x %d\n', ...
        netH, netW, tileH, tileW);

    % WD-5 클래스 이름 (네트워크에서 읽음 - 옵션)
    try
        classesNet  = net.Layers(end).Classes;
        classesCell = cellstr(classesNet(:));
        fprintf('[Info] 네트워크 클래스(%d): %s\n', ...
            numel(classesCell), strjoin(classesCell, ', '));
    catch
        classesCell = {};
        warning('마지막 레이어에서 Classes 정보를 읽지 못했습니다.');
    end

    %% 2. 코어 이미지 선택
    [fnI,fpI] = uigetfile( ...
        {'*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff','Image files'}, ...
        '분석할 코어 사진 선택');
    if isequal(fnI,0)
        disp('취소됨.'); return;
    end
    imgPath = fullfile(fpI,fnI);
    fprintf('[Info] 선택된 이미지: %s\n', imgPath);

    Iorig = imread(imgPath);
    Iorig = ensure_rgb_local(Iorig);
    [H,W,~] = size(Iorig);
    fprintf('[Info] 원본 크기: %d x %d px\n', H, W);

    %% 3. 코어 중심/반경: 이미지 전체 기준 (ROI 없이)
    coreCenter = [W/2, H/2];   % [cx cy]
    coreRadius = sqrt( (W/2)^2 + (H/2)^2 );   % 중심-모서리 거리 ~ 최대 반경
    fprintf('[Info] 이미지 기준 중심 (cx, cy) = (%.1f, %.1f), 반경 R ≈ %.1f px\n', ...
        coreCenter(1), coreCenter(2), coreRadius);

    %% 4. 패치 그리드 생성 (반 겹치기) + 전처리/사전할당
    %  - 그리드는 "타일 크기" 기준으로 생성
    strideY = round(tileH * 0.5);   % 세로 50% overlap
    strideX = round(tileW * 0.5);   % 가로 50% overlap

    ys = 1:strideY:max(1,H-tileH+1);
    xs = 1:strideX:max(1,W-tileW+1);

    % 최대 패치 개수 기준 사전할당
    maxPatches      = numel(ys) * numel(xs);          % 전체 그리드 수
    % I4는 "네트 입력 크기" 기준 (전처리 후 netH x netW 로 저장)
    I4              = zeros(netH, netW, 3, maxPatches, 'single');  % [H W 3 N]
    gridPos         = zeros(maxPatches, 2);           % [iy ix] (그리드 인덱스)
    gridIdx         = nan(numel(ys), numel(xs));      % 그리드 인덱스 맵
    patchStartYX    = zeros(maxPatches, 2);           % [y x] 좌상단 좌표 (원본 기준)
    patchCenterYX   = zeros(maxPatches, 2);           % [cy cx] 중심 좌표 (원본 기준)

    idxPatch = 0;
    for iy = 1:numel(ys)
        y  = ys(iy);
        y2 = y + tileH - 1;
        if y2 > H, continue; end

        for ix = 1:numel(xs)
            x  = xs(ix);
            x2 = x + tileW - 1;
            if x2 > W, continue; end

            % (ROI 없이) 모든 타일을 사용
            cy = min(H, max(1, round(y + tileH/2)));
            cx = min(W, max(1, round(x + tileW/2)));

            idxPatch = idxPatch + 1;

            % 타일 추출 (원본 상 tileH x tileW) + 전처리(→ netH x netW)
            patch = Iorig(y:y2, x:x2, :);
            I4(:,:,:,idxPatch) = imagenet_preprocess_local(patch, [netH netW]);

            % 그리드 좌표/인덱스/좌표 저장
            gridPos(idxPatch,:)       = [iy ix];
            gridIdx(iy,ix)            = idxPatch;
            patchStartYX(idxPatch,:)  = [y x];
            patchCenterYX(idxPatch,:) = [cy cx];
        end
    end

    if idxPatch == 0
        error('추출된 타일이 없습니다. 이미지 크기/타일 크기를 확인하세요.');
    end

    % 실제 사용되는 부분만 잘라서 사용
    I4            = I4(:,:,:,1:idxPatch);      % [netH netW 3 N]
    gridPos       = gridPos(1:idxPatch,:);
    patchStartYX  = patchStartYX(1:idxPatch,:);
    patchCenterYX = patchCenterYX(1:idxPatch,:);

    fprintf('[Info] 추출된 타일(패치) 개수: %d\n', idxPatch);

    %% 5. 패치 분류 (훈련과 동일 전처리 → 같은 세계)
    if canUseGPU
        execEnv = 'gpu';
    else
        execEnv = 'cpu';
    end

    miniBatchSize = 32;   % inference용
    [YPatch, scoresPatch] = classify(net, I4, ...
        'MiniBatchSize', miniBatchSize, ...
        'ExecutionEnvironment', execEnv); 

    % 패치 예측 클래스 분포 출력
    if isa(YPatch,'categorical')
        cats = categories(YPatch);
        cnts = countcats(YPatch);
        fprintf('\n[Info] 패치 예측 클래스 분포:\n');
        for i = 1:numel(cats)
            fprintf('  %s : %d\n', cats{i}, cnts(i));
        end
    end

    %% 6. WD-5 숫자 등급으로 변환 (D1~D5 → 1~5)
    gradePatch = labels_to_grade_local(YPatch);

    % 유효 grade만 사용
    validMask  = ~isnan(gradePatch);
    gradeValid = gradePatch(validMask);
    nValid     = numel(gradeValid);

    if nValid == 0
        error('예측된 WD-5 등급이 모두 NaN입니다. 클래스 이름 형식(D1~D5) 확인 필요.');
    end

    % 히스토그램 (1~5)
    K     = 5;
    edges = 0.5:1:5.5;
    cnt   = histcounts(gradeValid, edges);
    pct   = cnt / nValid * 100;

    WD = strcat("D", string(1:K)).';
    Tsummary = table(WD, cnt.', pct.', ...
        'VariableNames', {'WD','patch_count','patch_percent'});

    meanGrade = mean(gradeValid,'omitnan');
    stdGrade  = std(gradeValid,'omitnan');

    % D4–D5 비율 (국소 고풍화 비율 지표)
    ratioD45 = mean(gradeValid >= 4);

    fprintf('\n==== WD-5 패치 요약 ====\n');
    disp(Tsummary);
    fprintf('패치 등급 평균: %.2f (표준편차: %.2f)\n', meanGrade, stdGrade);
    fprintf('D4–D5 비율 (패치 기준): %.3f\n', ratioD45);

    %% 6-1. 패치 단위 상세 정보(학회용 분석 확대)
    % 최대 신뢰도, 정규화 반경 r/R 계산
    [maxScore, ~] = max(scoresPatch, [], 2);   % 각 패치에서 최대 확률

    cx = patchCenterYX(:,2);
    cy = patchCenterYX(:,1);
    r  = sqrt( (cx - coreCenter(1)).^2 + (cy - coreCenter(2)).^2 );
    rNorm = r / coreRadius;

    patchID   = (1:idxPatch).';
    predLabel = YPatch(:);
    predGrade = gradePatch(:);

    patchInfo = table(patchID, ...
        patchStartYX(:,1), patchStartYX(:,2), ...
        patchCenterYX(:,1), patchCenterYX(:,2), ...
        r, rNorm, ...
        predLabel, predGrade, maxScore, ...
        'VariableNames', { ...
            'patchID', ...
            'y_start', 'x_start', ...
            'y_center','x_center', ...
            'radius_px','radius_norm', ...
            'predLabel','predGrade','maxScore'});

    %% 6-2. [추가] 신뢰도 기반 통계 (4번)
    %  - 특정 threshold 이상인 패치만 별도로 평가
    %  - maxScore로 가중 평균 등급 계산
    confThreshold = 0.6;      % 필요하면 나중에 파라미터화
    maskConf      = (maxScore >= confThreshold) & ~isnan(predGrade);

    nConf   = sum(maskConf);
    fracConf = nConf / idxPatch;   % 전체 패치 대비 고신뢰 패치 비율

    if nConf > 0
        grade_conf      = predGrade(maskConf);
        meanGrade_conf  = mean(grade_conf, 'omitnan');
    else
        meanGrade_conf  = NaN;
    end

    maskValidForW = ~isnan(predGrade) & ~isnan(maxScore);
    grade_w       = predGrade(maskValidForW);
    score_w       = maxScore(maskValidForW);

    if ~isempty(score_w) && sum(score_w) > 0
        wMeanGrade = sum(double(grade_w) .* double(score_w)) / sum(double(score_w));
    else
        wMeanGrade = NaN;
    end

    fprintf('\n==== 신뢰도 기반 통계 (4번) ====\n');
    fprintf('고신뢰 패치 (maxScore >= %.2f): %d개 (전체의 %.1f%%)\n', ...
        confThreshold, nConf, fracConf*100);
    fprintf('고신뢰 패치 평균 등급: %.2f\n', meanGrade_conf);
    fprintf('가중 평균 등급 (maxScore 가중): %.2f\n', wMeanGrade);

    %% 6-3. WD-5 분포 막대그래프 / Radial scatter plot
    % 분포 막대그래프
    figure('Name','WD-5 Distribution (Patch Histogram)','NumberTitle','off');
    bar(1:K, pct);
    set(gca,'XTick',1:K,'XTickLabel',cellstr(WD));
    xlabel('Weathering Grade (WD-5)');
    ylabel('Patch Ratio (%)');
    title(sprintf('WD-5 Patch Distribution (mean grade = %.2f)', meanGrade));
    grid on;

    % Radial scatter: r/R vs grade
    figure('Name','Radial Distribution of WD-5','NumberTitle','off');
    scatter(rNorm, predGrade, 20, predGrade, 'filled');
    colormap(jet(5));
    colorbar;
    xlabel('Normalized Radius r/R (0=center, 1=edge~corner)');
    ylabel('WD-5 Grade');
    ylim([0.5 5.5]);
    yticks(1:5);
    title('Radial Distribution of WD-5 (patch-wise)');
    grid on;

    %% 7. 그리드 상에 등급 맵 구성
    gradeMap = nan(size(gridIdx));  % (numel(ys) x numel(xs))
    for k = 1:idxPatch
        iy = gridPos(k,1);
        ix = gridPos(k,2);
        gradeMap(iy,ix) = gradePatch(k);
    end

    % 패치 그리드 레벨 히트맵
    figure('Name','Patch-wise WD-5 Grade Map','NumberTitle','off');
    imagesc(gradeMap,[1 5]);
    axis image; colorbar;
    colormap(jet(5));
    set(gca,'YDir','normal');
    xticks(1:numel(xs)); yticks(1:numel(ys));
    xlabel('Grid X'); ylabel('Grid Y');
    title(sprintf('Patch-wise WD-5 map (mean grade = %.2f)', meanGrade));

    %% 8. 원본 해상도로 업샘플 + 오버레이
    gradeMap_big = imresize(gradeMap,[H W],'nearest');   % 이미지 크기에 맞게 확대
    alpha = 0.5;                                        % 오버레이 투명도

    figure('Name','WD-5 Overlay','NumberTitle','off');
    imshow(Iorig); hold on;
    hOverlay = imagesc(gradeMap_big,[1 5]);
    colormap(jet(5));
    set(hOverlay, 'AlphaData', alpha * ~isnan(gradeMap_big));
    colorbar;
    title(sprintf('WD-5 overlay (mean grade = %.2f)', meanGrade));

    %% 9. 결과 저장 (이미지 폴더 안에 OUT 폴더 생성)
    [~,baseName,~] = fileparts(fnI);
    outDir = fullfile(fpI, sprintf('%s_WD5_RESNET18_OUT', baseName));
    if ~exist(outDir,'dir'), mkdir(outDir); end

    % 타임스탬프 (파일명 suffix로 사용 가능)
    timeTag = char(datetime("now","Format","yyyyMMdd_HHmmss"));

    % ----- (6번) 여러 코어 비교용 스칼라 지표 정리 -----
    %  - 하나의 코어를 한 행으로 쓰기 좋은 summary_scalar 테이블
    Tscalar = table( ...
        meanGrade, stdGrade, ...
        wMeanGrade, meanGrade_conf, ...
        ratioD45, ...
        idxPatch, nValid, ...
        nConf, fracConf, confThreshold, ...
        'VariableNames', { ...
            'meanGrade', 'stdGrade', ...
            'weightedMeanGrade', 'meanGrade_conf', ...
            'ratioD45', ...
            'numPatches_total', 'numPatches_valid', ...
            'numPatches_conf', 'fracPatches_conf', 'confThreshold'});

    % WD-5 패치 요약 + 패치 상세 + 스칼라 요약 엑셀
    xlsxPath = fullfile(outDir, 'wd5_patch_summary.xlsx');
    writetable(Tsummary, xlsxPath, ...
        'Sheet','summary','WriteMode','overwritesheet','UseExcel',false);
    writetable(patchInfo, xlsxPath, ...
        'Sheet','patch_detail','WriteMode','overwritesheet','UseExcel',false);
    writetable(Tscalar, xlsxPath, ...
        'Sheet','summary_scalar','WriteMode','overwritesheet','UseExcel',false);

    % overlay figure 저장
    figOverlay = findobj('Type','figure','Name','WD-5 Overlay');
    if ~isempty(figOverlay)
        figOverlay = figOverlay(1);
        overlayPath = fullfile(outDir, sprintf('wd5_overlay_%s.png', timeTag));
        exportgraphics(figOverlay, overlayPath, 'Resolution', 300);
        fprintf('[Save] 오버레이: %s\n', overlayPath);
    end

    % 패치 맵 figure 저장
    figGrid = findobj('Type','figure','Name','Patch-wise WD-5 Grade Map');
    if ~isempty(figGrid)
        figGrid = figGrid(1);
        heatmapPath = fullfile(outDir, sprintf('wd5_patch_map_%s.png', timeTag));
        exportgraphics(figGrid, heatmapPath, 'Resolution', 300);
        fprintf('[Save] 패치 맵: %s\n', heatmapPath);
    end

    % WD-5 분포 막대그래프 저장
    figHist = findobj('Type','figure','Name','WD-5 Distribution (Patch Histogram)');
    if ~isempty(figHist)
        figHist = figHist(1);
        histPath = fullfile(outDir, sprintf('wd5_hist_%s.png', timeTag));
        exportgraphics(figHist, histPath, 'Resolution', 300);
        fprintf('[Save] WD-5 히스토그램: %s\n', histPath);
    end

    % Radial scatter plot 저장
    figRadial = findobj('Type','figure','Name','Radial Distribution of WD-5');
    if ~isempty(figRadial)
        figRadial = figRadial(1);
        radialPath = fullfile(outDir, sprintf('wd5_radial_%s.png', timeTag));
        exportgraphics(figRadial, radialPath, 'Resolution', 300);
        fprintf('[Save] WD-5 radial 분포: %s\n', radialPath);
    end

    %% 10. 분석 메타 정보 구조체 (analysisMeta, 4번/6번 지표 포함)
    analysisMeta = struct();
    analysisMeta.imagePath     = imgPath;
    analysisMeta.netMatPath    = matPath;
    analysisMeta.analysisTime  = datetime("now");
    analysisMeta.inputSize     = inputSize;          % 네트 입력 크기 (예: 224x224x3)
    analysisMeta.tileSize      = [tileH tileW];      % 원본 상 타일 크기
    analysisMeta.stride        = [strideY strideX];
    analysisMeta.numPatches    = idxPatch;
    analysisMeta.numValid      = nValid;
    analysisMeta.classesNet    = classesCell;
    analysisMeta.meanGrade     = meanGrade;
    analysisMeta.stdGrade      = stdGrade;
    analysisMeta.WD_counts     = cnt;
    analysisMeta.WD_percent    = pct;
    analysisMeta.ratioD45      = ratioD45;
    analysisMeta.coreCenter    = coreCenter;         % [cx cy]
    analysisMeta.coreRadius    = coreRadius;
    analysisMeta.ys            = ys;
    analysisMeta.xs            = xs;

    % 신뢰도 기반 지표 (4번)
    analysisMeta.confThreshold   = confThreshold;
    analysisMeta.numConfPatches  = nConf;
    analysisMeta.fracConfPatches = fracConf;
    analysisMeta.meanGrade_conf  = meanGrade_conf;
    analysisMeta.weightedMeanGrade = wMeanGrade;

    % MAT 저장 (패치 예측 결과, 맵 등 + 메타데이터)
    matOut = fullfile(outDir, sprintf('wd5_patch_result_%s.mat', timeTag));
    save(matOut, ...
        'YPatch','scoresPatch','gradePatch', ...
        'gridPos','gridIdx','gradeMap','gradeMap_big', ...
        'patchStartYX','patchCenterYX','patchInfo', ...
        'Tsummary','Tscalar', ...
        'meanGrade','stdGrade','classesCell', ...
        'analysisMeta', ...
        '-v7.3');

    fprintf('\n[Save] 요약/상세/스칼라 엑셀: %s\n', xlsxPath);
    fprintf('[Save] 세부 결과 MAT: %s\n', matOut);
    fprintf('완료.\n');
end

%% =============== Local helpers ===============

function Iout = ensure_rgb_local(I)
% grayscale / 1채널 / 4채널(RGBA) → 3채널 RGB
    if ismatrix(I) || size(I,3) == 1
        Iout = repmat(I,[1 1 3]);
    elseif size(I,3) >= 3
        Iout = I(:,:,1:3);
    else
        Iout = I;
    end
end

function grade = labels_to_grade_local(Y)
% categorical 라벨 (예: 'D1','D2',...) → 숫자 등급 (1,2,...)
    Ycell = cellstr(Y);
    n     = numel(Ycell);
    grade = nan(n,1);

    for i = 1:n
        lab   = Ycell{i};
        dmask = isstrprop(lab, 'digit');
        if any(dmask)
            grade(i) = str2double(lab(dmask));
        else
            grade(i) = NaN;
        end
    end
end

function Iout = imagenet_preprocess_local(I, targetSize)
% (PyTorch/Imagenet-style) 전처리:
%  - resize → im2single → (I - mu) ./ sig
%    (mu, sig 은 [0,1] 스케일 RGB 평균/표준편차)
    if size(I,3) == 1
        I = repmat(I,1,1,3);
    end
    I = im2single(imresize(I, targetSize));
    mu  = reshape([0.485 0.456 0.406],1,1,3);
    sig = reshape([0.229 0.224 0.225],1,1,3);
    Iout = (I - mu) ./ sig;
end
