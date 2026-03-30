function color_svm_analyzer()
% COLOR_SVM_ANALYZER  (color_svm_model.mat 전용)
% ---------------------------------------------------------------
% • color_svm_model.mat에 저장된 분류기(mdlColor)를 불러와
%   단일 코어 이미지의 WD-5 등급(D1~D5)을 예측한다.
% • score(음수 loss/margin)를 softmax 로 0~1 범위로 정규화해
%   pseudo-probability로 출력한다.
% • 추가 산출물:
%     - 원본 이미지 + 예측 결과 타이틀 figure
%     - WD-5별 pseudo-probability 막대그래프
%     - WD-5별 내림차순 확률 그래프(Confidence margin 표기)
%     - CIELAB L* 히스토그램
%     - CIELAB a*–b* 산점도
%     - 엑셀(xlsx): summary, svm_score, prob_detail, features, L_hist
%     - MAT: analysisMeta 구조체 (모든 수치/메타데이터 저장)
% ---------------------------------------------------------------
% 2025-12-03  준엽 프로젝트 – PKNU PTCT Lab
% ---------------------------------------------------------------

    %% 1. MAT 파일 로드
    matDefault = 'color_svm_model.mat';
    if exist(matDefault, 'file')
        matPath = matDefault;
        fprintf('[Info] MAT 파일 사용: %s\n', matPath);
    else
        [fnM, fpM] = uigetfile('*.mat', 'color_svm_model.mat 선택');
        if isequal(fnM, 0)
            disp('취소됨.');
            return;
        end
        matPath = fullfile(fpM, fnM);
    end

    S = load(matPath);

    % 1-1. 분류기 객체 찾기 (우선 mdlColor)
    mdl = [];
    if isfield(S, 'mdlColor') && isvalid_classifier_object(S.mdlColor)
        mdl = S.mdlColor;
        fprintf('[Info] 분류기 변수 사용: mdlColor\n');
    else
        fn = fieldnames(S);
        for k = 1:numel(fn)
            obj = S.(fn{k});
            if isvalid_classifier_object(obj)
                mdl = obj;
                fprintf('[Info] 분류기 변수(검색): %s\n', fn{k});
                break;
            end
        end
    end

    if isempty(mdl)
        error('color_svm_model.mat 안에서 predict 가능한 분류기 객체를 찾지 못했습니다.');
    end

    % 1-2. 보조 변수(muF, sdF, refLAB) 로드 (있으면 사용)
    if isfield(S, 'muF'),    muF    = S.muF(:).';     else, muF    = []; end
    if isfield(S, 'sdF'),    sdF    = S.sdF(:).';     else, sdF    = []; end
    if isfield(S, 'refLAB'), refLAB = S.refLAB(:).';  else, refLAB = []; end

    %% 2. Predictor 이름 / 차원 확인
    if ~isprop(mdl, 'PredictorNames')
        error('분류기 객체에 PredictorNames 속성이 없습니다.');
    end

    predictorNames = cellstr(mdl.PredictorNames);
    nFeat = numel(predictorNames);

    fprintf('[Info] 모델 predictor 개수: %d\n', nFeat);
    fprintf('[Info] PredictorNames: %s\n', strjoin(predictorNames, ', '));

    if ~isempty(muF) && numel(muF) ~= nFeat
        warning('muF 길이(%d)와 predictor 개수(%d)가 달라 정규화를 생략합니다.', ...
                numel(muF), nFeat);
        muF = [];
        sdF = [];
    end

    %% 3. 이미지 선택
    [fnI, fpI] = uigetfile( ...
        {'*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff', 'Image files'}, ...
        '분석할 코어 사진 선택');
    if isequal(fnI,0)
        disp('취소됨.');
        return;
    end
    imgPath = fullfile(fpI, fnI);
    fprintf('[Info] 선택된 이미지: %s\n', imgPath);

    I = imread(imgPath);

    %% 4. 색상 특징 추출 (원시 특징 → 정규화 → x1..xN)
    featRowRaw = extract_color_features_local(I, nFeat, refLAB);   % 1×nFeat

    if ~isempty(muF) && ~isempty(sdF)
        muFrow = reshape(muF, 1, []);   % 1×nFeat
        sdFrow = reshape(sdF, 1, []);   % 1×nFeat
        featRow = (featRowRaw - muFrow) ./ sdFrow;
        fprintf('[Info] muF/sdF 기반 표준화 적용 완료.\n');
    else
        featRow = featRowRaw;
        fprintf('[Info] muF/sdF 정보가 없어 정규화를 생략합니다.\n');
    end

    % predictor 이름(x1..xN)에 맞춰 테이블 생성
    Tfeat = array2table(featRow, 'VariableNames', predictorNames);

    %% 5. 예측
    [YPred, score] = predict(mdl, Tfeat);       % score: 1×K

    % score → softmax 기반 pseudo-probability (0~1, 합=1)
    scoreVec = double(score(1,:));
    validMask = ~isnan(scoreVec);
    prob = nan(size(scoreVec));

    if any(validMask)
        s = scoreVec(validMask);
        sShift = s - max(s);          % 수치 안정성
        expS = exp(sShift);
        prob(validMask) = expS ./ sum(expS);
    end

  %% 6. 결과 출력 (score + pseudo-prob + 신뢰도 분석)

    % 클래스 라벨
    try
        classLabels = mdl.ClassNames;
    catch
        classLabels = [];
    end
    if isempty(classLabels) && iscategorical(YPred)
        classLabels = categories(YPred);
    end
    if isempty(classLabels)
        classLabels = "Cls" + string(1:size(score,2));
    end

    % 내림차순 정렬 (확률 기준)
    [probSorted, idxSorted] = sort(prob, 'descend');
    classSorted = classLabels(idxSorted);
    scoreSorted = scoreVec(idxSorted);   % → 아래에서 사용

    % 상위 1–2위 확률 및 margin
    p1 = probSorted(1);
    if numel(probSorted) >= 2
        p2 = probSorted(2);
        marginProb = p1 - p2;
    else
        p2 = NaN;
        marginProb = NaN;
    end

    % 신뢰도 레벨
    if isnan(marginProb)
        confidenceLevel = "Unknown";
    elseif marginProb >= 0.20
        confidenceLevel = "High";
    elseif marginProb >= 0.10
        confidenceLevel = "Medium";
    else
        confidenceLevel = "Low";
    end

    % --- 원래 클래스 순서 기준 출력 ---
    fprintf('\n===== 예측 결과 (score + pseudo-prob) =====\n');
    for i = 1:numel(classLabels)
        fprintf('  %-3s : score = %8.4f   p = %6.3f\n', ...
            string(classLabels(i)), scoreVec(i), prob(i));
    end

    % WD-5 등급 텍스트
    [gradeNum, gradeStr] = label_to_wd5_grade_local(YPred);
    if ~isnan(gradeNum)
        fprintf('\n>> 최종 WD-5 등급 : %s (숫자 %.0f)\n', gradeStr, gradeNum);
    else
        fprintf('\n>> WD-5 형식(D1~D5, 1~5) 라벨을 인식하지 못했습니다.\n');
    end

    % 상위 1–2위 정보 + margin
    fprintf('   - 최고 확률 클래스: %s (p1 = %.3f)\n', ...
            string(classSorted(1)), p1);
    fprintf('   - 2위 클래스:       %s (p2 = %.3f)\n', ...
            string(classSorted(min(2,end))), p2);
    fprintf('   - 확률 차이 margin = p1 - p2 = %.3f  → Confidence: %s\n', ...
            marginProb, confidenceLevel);

    % --- 추가: 상위 3개 클래스 정렬 출력 (scoreSorted 사용) ---
    fprintf('\n   [Top classes by probability]\n');
    nShow = min(3, numel(classSorted));
    for k = 1:nShow
        fprintf('     #%d: %s  score = %8.4f   p = %6.3f\n', ...
            k, string(classSorted(k)), scoreSorted(k), probSorted(k));
    end


    %% 7. 색상 분포 시각화용 통계 (L* 히스토그램, a*b* 샘플)
    colorStats = compute_color_stats_for_plot(I);

    %% 8. 시각화 (이미지 + 확률 + 색상 분포)

    % 8-1) 원본 이미지 + 예측 결과
    figImg = figure('Name','WD-5 SVM Result','NumberTitle','off');
    imshow(I);
    if ~isnan(gradeNum)
        title(sprintf('SVM WD-5 = %s (p = %.3f, %s confidence)', ...
            string(gradeStr), p1, confidenceLevel));
    else
        title(sprintf('SVM prediction = %s (p = %.3f, %s confidence)', ...
            string(YPred), p1, confidenceLevel));
    end

    % 8-2) WD-5별 pseudo-probability 막대그래프 (원래 순서)
    figProb = figure('Name','WD-5 SVM Pseudo-Probability','NumberTitle','off');
    bar(prob);
    set(gca,'XTick',1:numel(classLabels), ...
            'XTickLabel',cellstr(string(classLabels)));
    xlabel('Weathering Grade (WD-5)');
    ylabel('Pseudo-probability');
    title('SVM pseudo-probabilities (original class order)');
    grid on;

    % 8-3) 내림차순 확률 그래프 (margin 강조)
    figProbSorted = figure('Name','WD-5 Probabilities (sorted)','NumberTitle','off');
    bar(probSorted);
    set(gca,'XTick',1:numel(classSorted), ...
            'XTickLabel',cellstr(string(classSorted)));
    xlabel('Weathering Grade (sorted by p)');
    ylabel('Pseudo-probability');
    title(sprintf('Sorted probabilities (margin p1-p2 = %.3f, %s)', ...
        marginProb, confidenceLevel));
    grid on;

    % 8-4) L* 히스토그램
    figLhist = figure('Name','L* Histogram','NumberTitle','off');
    bar(colorStats.L_centers, colorStats.L_histNorm);
    xlabel('L* (lightness)');
    ylabel('Frequency (normalized)');
    title('CIELAB L* histogram (0–100)');
    xlim([0 100]);
    grid on;

    % 8-5) a*-b* 산점도 (샘플링)
    figAB = figure('Name','a*-b* Scatter','NumberTitle','off');
    scatter(colorStats.a_sample, colorStats.b_sample, 8, colorStats.L_sample, 'filled');
    xlabel('a*');
    ylabel('b*');
    title('CIELAB a*-b* scatter (colored by L*)');
    colorbar;
    grid on;

    %% 9. 결과 저장 (엑셀 + PNG + MAT)
    [~, baseName, ~] = fileparts(fnI);
    outDir = fullfile(fpI, sprintf('%s_SVM_OUT', baseName));
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    timeTag = char(datetime("now","Format","yyyyMMdd_HHmmss"));

    % 9-1) 엑셀: summary
    summaryTable = table( ...
        string(YPred), string(gradeStr), gradeNum, ...
        p1, p2, marginProb, string(confidenceLevel), ...
        'VariableNames', { ...
            'PredictedLabel','PredictedWD5','WD5_numeric', ...
            'p1_top','p2_second','margin_p1_minus_p2','ConfidenceLevel'});

    WD      = string(classLabels(:));
    scoreCol = scoreVec(:);
    probCol  = prob(:);
    Tsvm = table(WD, scoreCol, probCol, ...
        'VariableNames', {'WD','Score_raw','Prob_softmax'});

    WD_sorted   = string(classSorted(:));
    prob_sorted = probSorted(:);
    cumProb     = cumsum(prob_sorted);
    TprobDetail = table(WD_sorted, prob_sorted, cumProb, ...
        'VariableNames', {'WD_sorted','Prob_sorted','CumProb'});

    TfeatSummary = table( ...
        predictorNames(:), ...
        featRowRaw(:), ...
        featRow(:), ...
        'VariableNames', {'Predictor','Feature_raw','Feature_norm'});

    TLhist = table(colorStats.L_centers(:), colorStats.L_histNorm(:), ...
        'VariableNames', {'L_center','Freq_norm'});

    xlsxPath = fullfile(outDir, sprintf('svm_result_%s.xlsx', timeTag));
    writetable(summaryTable,  xlsxPath, 'Sheet','summary',    ...
               'WriteMode','overwritesheet','UseExcel',false);
    writetable(Tsvm,          xlsxPath, 'Sheet','svm_score',  ...
               'WriteMode','overwritesheet','UseExcel',false);
    writetable(TprobDetail,   xlsxPath, 'Sheet','prob_detail', ...
               'WriteMode','overwritesheet','UseExcel',false);
    writetable(TfeatSummary,  xlsxPath, 'Sheet','features',   ...
               'WriteMode','overwritesheet','UseExcel',false);
    writetable(TLhist,        xlsxPath, 'Sheet','L_hist',     ...
               'WriteMode','overwritesheet','UseExcel',false);

    % 9-2) figure 저장
    if ~isempty(figImg) && ishghandle(figImg)
        imgOut = fullfile(outDir, sprintf('svm_result_image_%s.png', timeTag));
        exportgraphics(figImg, imgOut, 'Resolution', 300);
        fprintf('[Save] 결과 이미지: %s\n', imgOut);
    end

    if ~isempty(figProb) && ishghandle(figProb)
        probOut = fullfile(outDir, sprintf('svm_prob_%s.png', timeTag));
        exportgraphics(figProb, probOut, 'Resolution', 300);
        fprintf('[Save] 확률 막대그래프: %s\n', probOut);
    end

    if ~isempty(figProbSorted) && ishghandle(figProbSorted)
        probSortedOut = fullfile(outDir, sprintf('svm_prob_sorted_%s.png', timeTag));
        exportgraphics(figProbSorted, probSortedOut, 'Resolution', 300);
        fprintf('[Save] 정렬 확률 그래프: %s\n', probSortedOut);
    end

    if ~isempty(figLhist) && ishghandle(figLhist)
        LhistOut = fullfile(outDir, sprintf('svm_Lhist_%s.png', timeTag));
        exportgraphics(figLhist, LhistOut, 'Resolution', 300);
        fprintf('[Save] L* 히스토그램: %s\n', LhistOut);
    end

    if ~isempty(figAB) && ishghandle(figAB)
        abOut = fullfile(outDir, sprintf('svm_ab_scatter_%s.png', timeTag));
        exportgraphics(figAB, abOut, 'Resolution', 300);
        fprintf('[Save] a*-b* 산점도: %s\n', abOut);
    end

    % 9-3) MAT: 메타데이터 저장
    analysisMeta = struct();
    analysisMeta.imagePath        = imgPath;
    analysisMeta.modelMatPath     = matPath;
    analysisMeta.analysisTime     = datetime("now");
    analysisMeta.predictorNames   = predictorNames;
    analysisMeta.featRowRaw       = featRowRaw;
    analysisMeta.featRowNorm      = featRow;
    analysisMeta.classLabels      = classLabels;
    analysisMeta.scoreRaw         = scoreVec;
    analysisMeta.probSoftmax      = prob;
    analysisMeta.YPred            = YPred;
    analysisMeta.gradeNum         = gradeNum;
    analysisMeta.gradeStr         = gradeStr;
    analysisMeta.probSorted       = probSorted;
    analysisMeta.classSorted      = classSorted;
    analysisMeta.marginProb       = marginProb;
    analysisMeta.confidenceLevel  = confidenceLevel;
    analysisMeta.colorStats       = colorStats;

    matOut = fullfile(outDir, sprintf('svm_result_%s.mat', timeTag));
    save(matOut, 'analysisMeta', '-v7.3');
    fprintf('[Save] 분석 메타데이터 MAT: %s\n', matOut);

    fprintf('\n[Done] SVM 분석/시각화/저장 완료.\n');

end % ===== end of main =====


%% =====================================================================
% Helper 1: 분류기 유효성 검사 (predict 메서드 보유 여부)
function tf = isvalid_classifier_object(obj)
    tf = isobject(obj) && ismethod(obj, 'predict');
end


%% =====================================================================
% Helper 2: grayscale/1채널/4채널 → 3채널 RGB
function Iout = ensure_rgb_local(I)
    if ismatrix(I) || size(I,3) == 1
        Iout = repmat(I, [1 1 3]);      % 1채널 → 3채널 복제
    elseif size(I,3) >= 3
        Iout = I(:,:,1:3);              % RGBA 등 → 앞의 3채널
    else
        Iout = repmat(I, [1 1 3]);      % 방어적 처리
    end
end


%% =====================================================================
% Helper 3: 색상 특징 추출 (원시 특징 F(1:nFeat))
function featRow = extract_color_features_local(I, nFeat, refLAB)
% ---------------------------------------------------------------
% • 입력:
%     I      : 원본 이미지
%     nFeat  : 모델이 요구하는 특징 차원 (예: 23)
%     refLAB : 기준 Lab 값 (1×3, [L a b]) – 없으면 빈 배열
% • 출력:
%     featRow : 1×nFeat 원시 특징 벡터
% ---------------------------------------------------------------

    I = ensure_rgb_local(I);

    % double [0,1] 스케일
    if isinteger(I)
        Id = im2double(I);
    else
        Id = I;
        if max(Id(:)) > 1
            Id = im2double(Id);
        end
    end

    R = Id(:,:,1);
    G = Id(:,:,2);
    B = Id(:,:,3);

    % GrayY (luma)
    GrayY = 0.2126*R + 0.7152*G + 0.0722*B;

    % Lab
    Lab = rgb2lab(Id);
    L   = Lab(:,:,1);
    a   = Lab(:,:,2);
    b   = Lab(:,:,3);

    % refLAB과의 차이
    if ~isempty(refLAB) && numel(refLAB) == 3
        dL = mean(L(:),'omitnan') - refLAB(1);
        da = mean(a(:),'omitnan') - refLAB(2);
        db = mean(b(:),'omitnan') - refLAB(3);
    else
        dL = 0; da = 0; db = 0;
    end

    % L* 구간별 비율 (0–25/25–50/50–75/75–100)
    edges = [0 25 50 75 100.001];
    [cntL, ~] = histcounts(L(:), edges);
    totalL = sum(cntL);
    if totalL == 0
        ratioL = zeros(1,4);
    else
        ratioL = cntL ./ totalL;
    end

    % ---- 고정 길이 원시 특징 벡터 F(1:23) 채우기 ----
    F = zeros(1,23);

    % 1–4: Lab/Gray 평균
    F(1)  = mean(L(:),    'omitnan');   % L_mean
    F(2)  = mean(a(:),    'omitnan');   % a_mean
    F(3)  = mean(b(:),    'omitnan');   % b_mean
    F(4)  = mean(GrayY(:),'omitnan');   % Gray_mean

    % 5–8: Lab/Gray 표준편차
    F(5)  = std(L(:),    0,'omitnan');  % L_std
    F(6)  = std(a(:),    0,'omitnan');  % a_std
    F(7)  = std(b(:),    0,'omitnan');  % b_std
    F(8)  = std(GrayY(:),0,'omitnan');  % Gray_std

    % 9–14: RGB 평균/표준편차
    F(9)  = mean(R(:), 'omitnan');      % R_mean
    F(10) = mean(G(:), 'omitnan');      % G_mean
    F(11) = mean(B(:), 'omitnan');      % B_mean
    F(12) = std(R(:), 0,'omitnan');     % R_std
    F(13) = std(G(:), 0,'omitnan');     % G_std
    F(14) = std(B(:), 0,'omitnan');     % B_std

    % 15–18: L* 히스토그램 비율
    F(15) = ratioL(1);                  % L_bin1_ratio (0–25)
    F(16) = ratioL(2);                  % L_bin2_ratio (25–50)
    F(17) = ratioL(3);                  % L_bin3_ratio (50–75)
    F(18) = ratioL(4);                  % L_bin4_ratio (75–100)

    % 19–21: refLAB과의 차이
    F(19) = dL;
    F(20) = da;
    F(21) = db;

    % 22–23: L 최소/최대값
    F(22) = min(L(:));
    F(23) = max(L(:));

    % ---- 모델이 요구하는 차원(nFeat)에 맞춰 앞부분만 사용 ----
    k = min(nFeat, numel(F));
    featRow = zeros(1, nFeat);
    featRow(1:k) = F(1:k);

end


%% =====================================================================
% Helper 4: 색상 분포 시각화용 통계 (L* 히스토그램, a*b* 샘플)
function stats = compute_color_stats_for_plot(I)

    I = ensure_rgb_local(I);

    if isinteger(I)
        Id = im2double(I);
    else
        Id = I;
        if max(Id(:)) > 1
            Id = im2double(Id);
        end
    end

    Lab = rgb2lab(Id);
    L   = Lab(:,:,1);
    a   = Lab(:,:,2);
    b   = Lab(:,:,3);

    % L* 히스토그램 (5단계 구간, 0~100)
    edgesL = 0:5:100;              % 5 단위
    [cntL, edgesL] = histcounts(L(:), edgesL);
    totalL = sum(cntL);
    if totalL == 0
        histNorm = zeros(size(cntL));
    else
        histNorm = cntL ./ totalL;
    end
    centersL = (edgesL(1:end-1) + edgesL(2:end)) / 2;

    % a*b* 산점도 샘플 (최대 5000 픽셀)
    N = numel(L);
    maxSample = 5000;
    if N > maxSample
        idxSample = randperm(N, maxSample);
    else
        idxSample = 1:N;
    end
    a_sample = a(idxSample);
    b_sample = b(idxSample);
    L_sample = L(idxSample);

    stats = struct();
    stats.L_edges     = edgesL;
    stats.L_centers   = centersL;
    stats.L_histNorm  = histNorm;
    stats.a_sample    = a_sample;
    stats.b_sample    = b_sample;
    stats.L_sample    = L_sample;

end


%% =====================================================================
% Helper 5: 라벨 → WD-5 숫자/문자 등급 변환
function [gradeNum, gradeStr] = label_to_wd5_grade_local(Y)

    s = string(Y);
    s = strtrim(s);

    token = regexp(s, '\d+', 'match', 'once');  % 숫자만 추출 (예: 'D3' → '3')

    if isempty(token)
        gradeNum = NaN;
        gradeStr = "";
        return;
    end

    gradeNum = str2double(token);
    gradeStr = "D" + string(gradeNum);

end
