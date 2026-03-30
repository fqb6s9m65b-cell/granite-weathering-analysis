%% Granite_WD_SVM_simple.m
% 라벨링된 JSON + 이미지 폴더 기반 SVM 분류 (HOG+LBP, 데이터 증강만 사용)
%  - JSON → (imageRef, labelRaw) 수집
%  - 이미지 이름 fuzzy match
%  - HOG + LBP 특징만 사용 (색상 통계 X)
%  - 클래스 불균형 → 데이터 증강(회전/반전)으로 맞춤
%  - ECOC SVM (RBF) 학습/평가 + 최종 모델/CSV 저장
%
% 필요 Toolbox:
%  - Statistics and Machine Learning Toolbox
%  - Computer Vision Toolbox

clear; clc; close all;

%% ====== PATHS (여기만 수정해서 사용) ======
dataRoot = "C:\Users\ROCKENG\Desktop\코랩 머신러닝\data";
imgRoot  = fullfile(dataRoot, "images");
jsonDir  = fullfile(dataRoot, "json");

saveModel = fullfile(dataRoot, "svm_weathering_simple.mat");
predCSV   = fullfile(dataRoot, "predictions_all_simple.csv");

%% ====== Options ======
rng(0,'twister');
holdoutRate = 0.2;          % 80/20 split

% 데이터 증강 관련
params.targetSize   = [256 256];   % HOG/LPB 계산용 기본 크기
params.hogCell      = [16 16];
params.lbpCell      = [32 32];
params.maxRepPerCls = 5;           % 한 샘플당 최대 증강 횟수 제한

%% ====== Guards ======
assert(isfolder(imgRoot),  ...
    sprintf("Image folder not found: %s", imgRoot));
assert(isfolder(jsonDir),  ...
    sprintf("JSON folder not found: %s", jsonDir));

%% ====== 1. JSON → (imageRef, label) 읽기 ======
recs = collectJsonRecords(jsonDir);
assert(~isempty(recs), sprintf("No usable JSON records under: %s", jsonDir));

[yAll, classList, labelMap, labelsNumeric] = normalizeLabels({recs.labelRaw});
K = numel(classList);
fprintf("Detected %d classes. (labelsNumeric = %d)\n", K, labelsNumeric);
disp(labelMap);

%% ====== 2. 이미지 목록 인덱싱 ======
imgDB = listAllImages(imgRoot);
assert(~isempty(imgDB), sprintf("No images found under: %s", imgRoot));

%% ====== 3. JSON 레코드 → 실제 이미지 파일 매칭 ======
nRec     = numel(recs);
resolved = strings(nRec,1);
hitType  = strings(nRec,1);  % "json-path","exact","contains","fuzzy","missing"

for i = 1:nRec
    [pth, how] = resolveImageForRecord(recs(i), imgRoot, imgDB);
    resolved(i) = pth;
    hitType(i)  = how;
end

missing = (hitType=="missing") | (resolved=="");
if any(missing)
    fprintf(2, "[Warn] %d records could not be matched to an image. Examples:\n", sum(missing));
    ex = find(missing, min(5,sum(missing)), 'first');
    for k = 1:numel(ex)
        fprintf(2, "  rec %d: label=%s\n", ex(k), string(recs(ex(k)).labelRaw));
    end
end

imgFiles = resolved(~missing);
y        = yAll(~missing);

assert(~isempty(imgFiles), "No valid (image,label) pairs after resolution.");

fprintf("Match types: json-path=%d, exact=%d, contains=%d, fuzzy=%d, missing=%d\n", ...
    sum(hitType=="json-path"), sum(hitType=="exact"), sum(hitType=="contains"), ...
    sum(hitType=="fuzzy"), sum(hitType=="missing"));

%% ====== 4. Holdout 분할 (아직 특징 X, 라벨만) ======
cv = cvpartition(y, 'Holdout', holdoutRate);
tr = training(cv);
te = test(cv);

%% ====== 5. 특징 차원 확인 (훈련셋에서 feature 추출 성공할 때까지) ======
idxTrAll = find(tr);
assert(~isempty(idxTrAll), 'No training samples found.');

D = [];
for k = 1:numel(idxTrAll)
    i = idxTrAll(k);
    try
        f1 = extractFeatSimple(imgFiles(i), params, 1);
        D  = numel(f1);
        break;
    catch ME
        warning("Feature probe failed @%s: %s", imgFiles(i), ME.message);
    end
end
assert(~isempty(D), 'Failed to extract features from any training sample.');
fprintf("Feature dimension D = %d\n", D);

%% ====== 6. TRAIN: 클래스별 증강으로 균형 맞추기 ======
classes   = unique(y(tr));                      % 실제 훈련에 쓰이는 클래스
nPerClass = arrayfun(@(c) sum(y(tr)==c), classes);
nMax      = max(nPerClass);

repPerCls = ceil(nMax ./ nPerClass);           % 필요한 반복 수
repPerCls = min(repPerCls, params.maxRepPerCls);

fprintf("Class counts (train only):\n");
for i = 1:numel(classes)
    fprintf("  Class %d: n=%d, rep=%d\n", classes(i), nPerClass(i), repPerCls(i));
end

totalNtr = sum(nPerClass .* repPerCls);
Ftr      = zeros(totalNtr, D, 'single');
yTrAug   = zeros(totalNtr,1);

ptr = 1;
for ci = 1:numel(classes)
    c   = classes(ci);
    rep = repPerCls(ci);
    idxs = find((y==c) & tr);
    for ii = 1:numel(idxs)
        i = idxs(ii);
        for r = 1:rep
            try
                f = extractFeatSimple(imgFiles(i), params, r);
            catch ME
                warning("Train feature failed @%s: %s", imgFiles(i), ME.message);
                continue;
            end
            Ftr(ptr,:) = single(f);
            yTrAug(ptr) = c;
            ptr = ptr + 1;
        end
    end
end

Ftr    = Ftr(1:ptr-1,:);
yTrAug = yTrAug(1:ptr-1);
fprintf("Augmented train N (after skipping failures) = %d\n", size(Ftr,1));
assert(~isempty(yTrAug), 'No training samples after feature extraction.');

%% ====== 7. TEST: 증강 없이 원본만 (실패 샘플은 제거) ======
idxTe = find(te);
Nte   = numel(idxTe);

Fte   = zeros(Nte, D, 'single');
yTe   = zeros(Nte,1);
imgTe = strings(Nte,1);

ptr = 1;
for j = 1:Nte
    i = idxTe(j);
    try
        f = extractFeatSimple(imgFiles(i), params, 1);
    catch ME
        warning("Test feature failed @%s: %s", imgFiles(i), ME.message);
        continue;
    end
    Fte(ptr,:) = single(f);
    yTe(ptr)   = y(i);
    imgTe(ptr) = imgFiles(i);
    ptr = ptr + 1;
end

Fte   = Fte(1:ptr-1,:);
yTe   = yTe(1:ptr-1);
imgTe = imgTe(1:ptr-1);
fprintf("Test N (after skipping failures) = %d\n", size(Fte,1));
assert(~isempty(yTe), 'No test samples after feature extraction.');

%% ====== 8. ECOC SVM (RBF) 학습 & Holdout 평가 ======
t = templateSVM('KernelFunction','rbf', 'KernelScale','auto', ...
                'BoxConstraint',1, 'Standardize',true);

classesTrEff = unique(yTrAug);  % 실제 학습에 사용된 클래스만
Mdl_ho = fitcecoc(Ftr, yTrAug, 'Learners', t, ...
    'ClassNames', classesTrEff, 'Coding','onevsone');

yp     = predict(Mdl_ho, Fte);
acc_ho = mean(yp == yTe);
bal_ho = mean(perClassRecall(yTe, yp, 1:K));
fprintf("[Holdout] Acc=%.2f%% | BalAcc=%.2f%%\n", acc_ho*100, bal_ho*100);

try
    figure; confusionchart(yTe, yp);
    title(sprintf("Holdout – Acc %.2f%% / BalAcc %.2f%%", acc_ho*100, bal_ho*100));
catch
    % GUI 없는 환경에서도 깨지지 않도록
end

%% ====== 9. 전체 데이터 증강 + 최종 모델 학습 ======
classesAll   = unique(y);
nPerClassAll = arrayfun(@(c) sum(y==c), classesAll);
nMaxAll      = max(nPerClassAll);
repPerClsAll = ceil(nMaxAll ./ nPerClassAll);
repPerClsAll = min(repPerClsAll, params.maxRepPerCls);

fprintf("Class counts (ALL, before feature failures):\n");
for i = 1:numel(classesAll)
    fprintf("  Class %d: n=%d, rep=%d\n", classesAll(i), nPerClassAll(i), repPerClsAll(i));
end

totalNall = sum(nPerClassAll .* repPerClsAll);
Fall      = zeros(totalNall, D, 'single');
yAllAug   = zeros(totalNall,1);
ptr = 1;

for ci = 1:numel(classesAll)
    c   = classesAll(ci);
    rep = repPerClsAll(ci);
    idxs = find(y==c);
    for ii = 1:numel(idxs)
        i = idxs(ii);
        for r = 1:rep
            try
                f = extractFeatSimple(imgFiles(i), params, r);
            catch ME
                warning("ALL-train feature failed @%s: %s", imgFiles(i), ME.message);
                continue;
            end
            Fall(ptr,:) = single(f);
            yAllAug(ptr) = c;
            ptr = ptr + 1;
        end
    end
end

Fall    = Fall(1:ptr-1,:);
yAllAug = yAllAug(1:ptr-1);
fprintf("Augmented ALL N (after skipping failures) = %d\n", size(Fall,1));
assert(~isempty(yAllAug), 'No samples for final training after feature extraction.');

classesAllEff = unique(yAllAug);

Mdl_all = fitcecoc(Fall, yAllAug, 'Learners', t, ...
    'ClassNames', classesAllEff, 'Coding','onevsone');

meta = struct('datetime',string(datetime('now')), ...
              'matlab',string(version), 'params',params, ...
              'classList', {classList}, 'labelMap', labelMap, ...
              'labelsNumeric', labelsNumeric, ...
              'imgRoot',imgRoot,'jsonDir',jsonDir);
save(saveModel, 'Mdl_all', 'meta', '-v7.3');
fprintf("Saved model: %s\n", saveModel);

%% ====== 10. 전체 샘플 원본 기준 예측 CSV (실패 샘플 제외) ======
Nall = numel(y);
F_all   = zeros(Nall, D, 'single');
y_all   = zeros(Nall,1);
img_all = strings(Nall,1);
ptr = 1;

for i = 1:Nall
    try
        f = extractFeatSimple(imgFiles(i), params, 1);
    catch ME
        warning("Predict-all feature failed @%s: %s", imgFiles(i), ME.message);
        continue;
    end
    F_all(ptr,:) = single(f);
    y_all(ptr)   = y(i);
    img_all(ptr) = imgFiles(i);
    ptr = ptr + 1;
end

F_all   = F_all(1:ptr-1,:);
y_all   = y_all(1:ptr-1);
img_all = img_all(1:ptr-1);
fprintf("Predict-all N (after skipping failures) = %d\n", size(F_all,1));

yhat_all = predict(Mdl_all, F_all);
T = table(img_all, y_all, yhat_all, 'VariableNames', {'image','label','pred'});
writetable(T, predCSV);
fprintf("Saved CSV: %s\n", predCSV);

%% ====== 끝 ======



%% ================= Local functions (스크립트 아래) =================

function recs = collectJsonRecords(jsonDir)
    J = getJsonFiles(jsonDir);
    recs = struct('imageRef',"",'labelRaw',[]);
    recs(1) = []; % empty
    for i = 1:numel(J)
        S = readJsonSafely(J{i});
        if isempty(S), continue; end

        if isstruct(S) && (isfield(S,'data') || isfield(S,'images'))
            if isfield(S,'data'), A = S.data; else, A = S.images; end
        elseif isstruct(S)
            A = S;
        else
            A = S;
        end

        if isstruct(A) && numel(A) > 1
            fn = fieldnames(A);
            imgKey = firstField(fn, ["image","img","path","filepath","file","filename","name"]);
            labKey = firstField(fn, ["label","class","grade","y","target","weather","weathering","level","score"]);
            if ~isempty(imgKey) && ~isempty(labKey)
                for k = 1:numel(A)
                    r.imageRef = string(A(k).(imgKey));
                    r.labelRaw = A(k).(labKey);
                    recs(end+1) = r; %#ok<AGROW>
                end
                continue;
            end
        end

        if isstruct(A)
            fn     = fieldnames(A);
            labKey = firstField(fn, ["label","class","grade","y","target","weather","weathering","level","score"]);
            imgKey = firstField(fn, ["image","img","path","filepath","file","filename","name"]);
            r.labelRaw = [];
            r.imageRef = "";

            if ~isempty(labKey), r.labelRaw = A.(labKey); end
            if ~isempty(imgKey), r.imageRef = string(A.(imgKey)); end

            [~,base,~] = fileparts(J{i});
            if r.imageRef=="" || r.imageRef=="<none>"
                r.imageRef = string(base);
            end
            if ~isempty(r.labelRaw)
                recs(end+1) = r; %#ok<AGROW>
            end
        end
    end
end

function files = getJsonFiles(jsonDir)
    L = dir(fullfile(jsonDir,"**","*.json"));
    files = cell(numel(L),1);
    for i=1:numel(L)
        files{i} = fullfile(L(i).folder, L(i).name);
    end
end

function S = readJsonSafely(p)
    S = [];
    try
        raw = fileread(p);
        S = jsondecode(raw);
    catch
        warning("Invalid JSON skipped: %s", p);
    end
end

function [pth, how] = resolveImageForRecord(rec, imgRoot, imgDB)
    how = "missing";
    pth = "";

    if rec.imageRef~=""
        ref = rec.imageRef;
        if startsWith(ref, filesep) || ~isempty(regexp(ref,'^[A-Za-z]:[\\/]', 'once'))
            cand = ref;
        else
            cand = fullfile(imgRoot, ref);
        end
        if isfile(cand)
            pth = cand; how = "json-path"; return;
        end

        [~,base,ext] = fileparts(ref);
        if ext~="", key = lower(base); else, key = lower(string(ref)); end
    else
        key = "";
    end

    if key==""
        how = "missing"; pth = ""; return;
    end

    idx = find(strcmp({imgDB.baseLower}, key), 1, 'first');
    if ~isempty(idx)
        pth = imgDB(idx).path; how = "exact"; return;
    end

    idxs = find(contains({imgDB.baseLower}, key) | contains(key, {imgDB.baseLower}));
    if ~isempty(idxs)
        pth = imgDB(idxs(1)).path; how = "contains"; return;
    end

    bestIdx = 0; bestD = inf;
    for i = 1:numel(imgDB)
        d = levenshtein(char(key), char(imgDB(i).baseLower));
        if d < bestD
            bestD = d; bestIdx = i;
        end
    end
    thr = max(1, floor(strlength(key)*0.3));
    if bestD <= thr && bestIdx>0
        pth = imgDB(bestIdx).path; how = "fuzzy"; return;
    end
end

function db = listAllImages(imgRoot)
    exts = {".png",".jpg",".jpeg",".bmp",".tif",".tiff",".gif"};
    L = dir(fullfile(imgRoot,"**","*"));
    db = struct('path',"",'base',"",'baseLower',"");
    db(1) = [];
    for i = 1:numel(L)
        if ~L(i).isdir
            [~,b,ext] = fileparts(L(i).name);
            if any(strcmpi(ext, exts))
                rec.path = fullfile(L(i).folder, L(i).name);
                rec.base = b;
                rec.baseLower = lower(b);
                db(end+1) = rec; %#ok<AGROW>
            end
        end
    end
end

function [y, classList, labelMap, isNumeric] = normalizeLabels(lblRaw)
    % lblRaw: cell/array, 숫자+문자 혼재 가능
    S = string(lblRaw(:));
    num = str2double(S);

    isAllNumeric = all(~isnan(num));
    if isAllNumeric
        % 전부 숫자로 해석 가능 → 숫자 라벨
        [classListNum, ~, y] = unique(num, 'stable');
        classList = string(classListNum(:))';
        isNumeric = true;
    else
        % 하나라도 숫자 아님 → 전부 문자열로 보고 categorical 처리
        C = categorical(S);
        [cats, ~, y] = unique(C, 'stable');
        classList = string(cats(:))';
        isNumeric = false;
    end
    labelMap = table((1:numel(classList))', classList', ...
        'VariableNames', {'Mapped','Original'});
end

function r = perClassRecall(yTrue, yPred, classes)
    r = zeros(numel(classes),1);
    for i = 1:numel(classes)
        idx = (yTrue == classes(i));
        if any(idx)
            r(i) = sum(yPred(idx)==classes(i)) / sum(idx);
        else
            r(i) = NaN;
        end
    end
    r = fillmissing(r, 'constant', 0);
end

function d = levenshtein(a,b)
    a = char(a); b = char(b);
    m = length(a); n = length(b);
    D = zeros(m+1, n+1);
    D(:,1) = (0:m)'; 
    D(1,:) = 0:n;
    for i = 2:m+1
        ai = a(i-1);
        for j = 2:n+1
            cost = ~(ai == b(j-1));
            D(i,j) = min([ D(i-1,j) + 1, ...
                           D(i,j-1) + 1, ...
                           D(i-1,j-1) + cost ]);
        end
    end
    d = D(m+1,n+1);
end

function name = firstField(fieldNames, candidates)
    name = "";
    for c = candidates
        if any(strcmpi(fieldNames, c))
            name = string(c);
            return;
        end
    end
end

function f = extractFeatSimple(pth, params, augIdx)
    % pth: string/char 이미지 경로
    % augIdx: 1 → 원본, 2..n → 회전/반전 증강

    % imread 에러 대비
    try
        I = imread(pth);
    catch ME
        error("FeatureExtraction:ReadError", ...
            "Failed to read image: %s (%s)", pth, ME.message);
    end

    if size(I,3) > 1
        Igray = rgb2gray(I);
    else
        Igray = I;
    end
    Igray = imresize(Igray, params.targetSize);

    if nargin >= 3 && augIdx > 1
        Igray = applyAugment(Igray, augIdx);
    end

    try
        hog = extractHOGFeatures(Igray, 'CellSize', params.hogCell);
        lbp = extractLBPFeatures(Igray, 'Upright', true, 'CellSize', params.lbpCell);
    catch ME
        error("FeatureExtraction:ComputeError", ...
            "Failed to extract features from image: %s (%s)", pth, ME.message);
    end

    f = [hog(:)' lbp(:)'];
end

function Iaug = applyAugment(Igray, k)
    switch mod(k-1, 4)
        case 0
            Iaug = Igray;
        case 1
            Iaug = fliplr(Igray);
        case 2
            Iaug = flipud(Igray);
        case 3
            Iaug = imrotate(Igray, 10, 'bilinear', 'crop');
    end
end
