function rocktype_resnet18_train_19()
% rocktype_resnet18_train (ONE-FILE, R2025a)
% -----------------------------------------------------------------------------
% ResNet-18 기반 "세부분류(모드2)" 암종 분류 (JSON/CSV 입력 사용 안 함)
% 라벨은 오직 파일명에서 정규식으로 추출.
%
% 라벨 규칙(오탐 방지 정규식)
%   패턴:  ...-R0x-yy-...
%   정규식: (?i)-(R0[1-3])-(\d{2})-
%
% 예)
%   001-BH6(1)-R01-01-90(NA)-D-1-10  -> R01(화성암), 01(화강암) -> R01_01_granite
%
% 클래스(총 19개)
%   R01: 01 granite, 02 diorite, 03 gabbro, 04 rhyolite, 05 andesite, 06 basalt, 07 tuff
%   R02: 01 gneiss,  02 schist,  03 phyllite, 04 slate,  05 quartzite, 06 marble
%   R03: 01 conglomerate, 02 sandstone, 03 mudstone, 04 shale, 05 limestone, 06 dolomite
%
% 요구: Train 균형샘플링
%   - Train 테이블을 클래스별 동일 샘플 수(target)로 맞춤
%   - 최소 클래스당 300장 (Train 기준)
%   - 부족한 클래스는 "복원추출(중복 허용)" upsample -> (증강과 함께 사용)
%
% 견고성
%   (1) Precheck: 깨진 이미지 제거(가능 시 병렬 parfor)
%   (2) ReadFcn: 런타임 imread 실패 시 더미 이미지 반환 + bad_images_runtime.csv 기록
%
% 결과 저장(강제)
%   C:\Users\ROCKENG\Desktop\코랩 머신러닝\results\rocktype_resnet18_train\<RunTag>\
% -----------------------------------------------------------------------------

clc; close all;

%% =========================
% 0) 경로 설정 (고정)
%% =========================
imgTrainRoots = [
 "C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\images-igneous rock\TS_1, 기반암 암종 분류 데이터_1. 화성암_1"
 "C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\images-igneous rock\TS_1, 기반암 암종 분류 데이터_1. 화성암_2"
 "C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\images-metamorphic rock\TS_1, 기반암 암종 분류 데이터_2. 변성암_2"
 "C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\images-sedimentary\TS_1, 기반암 암종 분류 데이터_3. 퇴적암"
];

imgValRoots = [
 "C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\images-igneous rock\VS_1. 기반암 암종 분류 데이터_1. 화성암"
 "C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\images-metamorphic rock\VS_1. 기반암 암종 분류 데이터_2. 변성암"
 "C:\Users\ROCKENG\Desktop\코랩 머신러닝\data\images-sedimentary\VS_1. 기반암 암종 분류 데이터_3. 퇴적암"
];

%% =========================
% 1) 실행/학습 옵션
%% =========================
opts = struct();
opts.SEED               = 7;
opts.TestRatioFromTrain = 0.15;

opts.MiniBatchSize      = 64;   % OOM 나면 32/16으로 낮추세요.
opts.MaxEpochs          = 40;
opts.InitialLearnRate   = 1e-4;
opts.L2Regularization   = 1e-4;
opts.Momentum           = 0.9;

opts.UsePiecewiseLR     = true; % StepLR 대응
opts.LRDropFactor       = 0.1;
opts.LRDropPeriod       = 10;

opts.Augment            = true;
opts.Verbose            = true;
opts.ShowTrainingPlot   = true;

opts.VALIDATE_IMAGES_PRECHECK = true;

% ====== 균형 샘플링 ======
opts.BalanceTrain           = true;
opts.MinPerClassTrain       = 300;
opts.MaxPerClassTrain       = 5000;
opts.BalanceTargetMode      = "medianCap"; % median(countsNZ) 기반

% ====== 병렬 Precheck ======
opts.UseParallelPrecheck     = true;
opts.ParallelPoolType        = "Processes";  % "Processes" or "Threads"
opts.ParallelWorkers         = 0;            % 0 => 자동
opts.MaxAutoWorkers          = 6;            % i7-8700 물리 6코어 기준 상한 권장
opts.ClosePoolBeforeTraining = true;         % GPU 학습 안정성 위해 권장

% ====== 결과 저장(강제) ======
opts.OutRoot     = "C:\Users\ROCKENG\Desktop\코랩 머신러닝\results";
opts.ProjectName = "rocktype_resnet18_train";
opts.RunTag      = char(datetime("now","Format","yyyyMMdd_HHmmss"));

% outDir 생성 + 쓰기 테스트(강제 경로, 실패 시 즉시 에러)
outDir = wx_prepare_outdir_strict(opts);

% 런타임 badlog 경로 등록 및 헤더 생성
wx_set_runtime_badlog(fullfile(outDir, "bad_images_runtime.csv"));

%% =========================
% 2) GPU 강제
%% =========================
wx_require_gpu();

%% =========================
% 3) 데이터 수집
%% =========================
classesAll = wx_mode2_classes_all();
Ttr = wx_collect_images(imgTrainRoots, "trainCand");
Tva = wx_collect_images(imgValRoots,   "val");

if height(Ttr) == 0
    error("TrainCand 이미지 0장. TS_* 폴더 경로/확장자 확인 필요.");
end

%% =========================
% 4) 파일명 기반 라벨링(모드2)
%% =========================
[Ttr, unl1] = wx_label_by_filename_mode2(Ttr, classesAll);
[Tva, unl2] = wx_label_by_filename_mode2(Tva, classesAll);

unlabeledT = [unl1; unl2];
wx_writetable_strict(unlabeledT, fullfile(outDir,"unlabeled_files.csv"), 'Encoding','UTF-8');

if height(Ttr) == 0
    error("파일명 라벨링 후 TrainCand 0장. 파일명에 -R01-01- 같은 패턴 포함 여부 확인 필요.");
end

% Train에 존재하는 클래스만 사용
present = string(categories(removecats(Ttr.label)));
maskPresent = ismember(classesAll, present);
classes = classesAll(maskPresent);

if numel(classes) < 2
    error("TrainCand에서 유효 클래스가 너무 적습니다. (present=%d)", numel(classes));
end

% Val에서 train에 없는 클래스 제외
Tva = wx_filter_unknown_classes(Tva, classes, fullfile(outDir,"dropped_val_unknown.csv"));

%% =========================
% 5) Precheck(깨진 이미지 제거) + 병렬풀 자동할당
%% =========================
badPre = table('Size',[0 4], ...
    'VariableTypes', {'string','string','string','string'}, ...
    'VariableNames', {'split','label','image_path','message'});

if opts.VALIDATE_IMAGES_PRECHECK
    disp("== Precheck: validating images (corrupt JPEG will be excluded) ==");

    usedParallel = false;

    if opts.UseParallelPrecheck && wx_can_parallel()
        try
            wx_ensure_parpool(opts);
            [Ttr, bad1] = wx_filter_bad_images_parallel(Ttr);
            [Tva, bad2] = wx_filter_bad_images_parallel(Tva);
            usedParallel = true;
        catch ME
            warning("rocktype:PrecheckParallelFailed", "%s", ...
                char("병렬 Precheck 실패 → 직렬 fallback: " + string(ME.message)));
            [Ttr, bad1] = wx_filter_bad_images(Ttr);
            [Tva, bad2] = wx_filter_bad_images(Tva);
        end
    else
        [Ttr, bad1] = wx_filter_bad_images(Ttr);
        [Tva, bad2] = wx_filter_bad_images(Tva);
    end

    badPre = [badPre; bad1; bad2];
    wx_writetable_strict(badPre, fullfile(outDir, "bad_images_precheck.csv"), 'Encoding','UTF-8');

    fprintf("Precheck removed: %d files (TrainCand+Val)\n", height(badPre));
    fprintf("Precheck mode    : %s\n", ternary(usedParallel,"parallel","serial"));

    if height(Ttr) == 0
        error("Precheck 이후 TrainCand 0장. 데이터가 전부 손상 또는 경로 문제.");
    end
end

% GPU 학습 전 풀 닫기(권장)
if opts.ClosePoolBeforeTraining
    wx_close_parpool();
end

%% =========================
% 6) Train/Test 층화 분할
%% =========================
rng(opts.SEED);
[Ttrain, Ttest] = wx_stratified_split_safe(Ttr, opts.TestRatioFromTrain, categorical(classes));

Ttest = wx_filter_unknown_classes(Ttest, classes, fullfile(outDir,"dropped_test_unknown.csv"));

wx_write_counts(Ttrain, classes, fullfile(outDir,"counts_train_before_balance.csv"));
wx_write_counts(Tva,    classes, fullfile(outDir,"counts_val.csv"));
wx_write_counts(Ttest,  classes, fullfile(outDir,"counts_test.csv"));

%% =========================
% 7) Train 균형 샘플링
%% =========================
if opts.BalanceTrain
    [TtrainBal, balInfo] = wx_balance_train_table(Ttrain, classes, opts);
    Ttrain = TtrainBal;
    wx_writetable_strict(balInfo, fullfile(outDir,"balance_info.csv"), 'Encoding','UTF-8');
    wx_write_counts(Ttrain, classes, fullfile(outDir,"counts_train_after_balance.csv"));
end

%% =========================
% 8) 데이터셋 로그(출력용)
%% =========================
Tall = [wx_add_split(Ttrain,"train"); wx_add_split(Tva,"val"); wx_add_split(Ttest,"test")];
wx_writetable_strict(Tall, fullfile(outDir,"dataset_table.csv"), 'Encoding','UTF-8');

%% =========================
% 9) Datastore 구성
%% =========================
net = resnet18;
inputSize = net.Layers(1).InputSize;

imdsTrain = imageDatastore(Ttrain.image_path, 'ReadFcn', @wx_read_image_uint8_safe);
imdsTrain.Labels = categorical(string(Ttrain.label), classes);

imdsVal = imageDatastore(Tva.image_path, 'ReadFcn', @wx_read_image_uint8_safe);
imdsVal.Labels = categorical(string(Tva.label), classes);

imdsTest = imageDatastore(Ttest.image_path, 'ReadFcn', @wx_read_image_uint8_safe);
imdsTest.Labels = categorical(string(Ttest.label), classes);

if numel(imdsTest.Files) == 0
    error("Test set이 0장입니다. TestRatioFromTrain 또는 데이터 분포를 확인하세요.");
end

if opts.Augment
    augmenter = imageDataAugmenter( ...
        'RandXReflection', true, ...
        'RandRotation', [-6 6], ...
        'RandXTranslation', [-10 10], ...
        'RandYTranslation', [-10 10], ...
        'RandScale', [0.95 1.05]);
    augTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, 'DataAugmentation', augmenter);
else
    augTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain);
end

hasVal = numel(imdsVal.Files) > 0;
if hasVal
    augVal  = augmentedImageDatastore(inputSize(1:2), imdsVal);
end
augTest = augmentedImageDatastore(inputSize(1:2), imdsTest);

%% =========================
% 10) 전이학습 네트워크 구성
%% =========================
lgraph = layerGraph(net);
[learnableLayer, classLayer] = wx_find_layers_to_replace(lgraph);

numClasses = numel(classes);

newLearnable = fullyConnectedLayer(numClasses, ...
    'Name','fc_new', ...
    'WeightLearnRateFactor',10, ...
    'BiasLearnRateFactor',10);

counts = countcats(categorical(string(Ttrain.label), classes));
den    = max(counts, ones(size(counts)));
classW = median(counts) ./ den;

newClassLayer = classificationLayer( ...
    'Name','classoutput', ...
    'Classes', categorical(classes), ...
    'ClassWeights', classW);

lgraph = replaceLayer(lgraph, learnableLayer.Name, newLearnable);
lgraph = replaceLayer(lgraph, classLayer.Name,    newClassLayer);

%% =========================
% 11) 학습 옵션
%% =========================
valFreq = max(1, floor(numel(imdsTrain.Files) / opts.MiniBatchSize));

ckptDir = fullfile(outDir,"checkpoints");
wx_mkdir(ckptDir);

plt = "none";
if opts.ShowTrainingPlot
    plt = "training-progress";
end

commonArgs = { ...
    'MiniBatchSize', opts.MiniBatchSize, ...
    'MaxEpochs', opts.MaxEpochs, ...
    'InitialLearnRate', opts.InitialLearnRate, ...
    'L2Regularization', opts.L2Regularization, ...
    'Momentum', opts.Momentum, ...
    'Shuffle','every-epoch', ...
    'Verbose', opts.Verbose, ...
    'Plots', plt, ...
    'ExecutionEnvironment','gpu', ...
    'CheckpointPath', ckptDir};

if opts.UsePiecewiseLR
    commonArgs = [commonArgs, { ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropFactor', opts.LRDropFactor, ...
        'LearnRateDropPeriod', opts.LRDropPeriod ...
    }];
end

if hasVal
    options = trainingOptions('sgdm', commonArgs{:}, ...
        'ValidationData', augVal, ...
        'ValidationFrequency', valFreq);
else
    options = trainingOptions('sgdm', commonArgs{:});
end

%% =========================
% 12) 학습
%% =========================
disp("== Training starts ==");
[trainedNet, trainInfo] = trainNetwork(augTrain, lgraph, options);

save(fullfile(outDir,"trainedNet.mat"), "trainedNet", "-v7.3");
save(fullfile(outDir,"trainInfo.mat"),  "trainInfo",  "-v7.3");

%% =========================
% 13) Test 평가 + 리포트 저장
%% =========================
disp("== Test evaluation ==");
YPred = classify(trainedNet, augTest, 'ExecutionEnvironment','gpu');
YTrue = imdsTest.Labels;

acc = mean(YPred == YTrue);

C = confusionmat(YTrue, YPred, 'Order', categorical(classes));

fig = figure('Visible','off');
cc = confusionchart(YTrue, YPred, 'Order', categorical(classes));
cc.Title = sprintf("ResNet-18 Rock Subtype (Test) | Acc = %.2f%%", acc*100);
exportgraphics(fig, fullfile(outDir,"confusion_test.png"));
close(fig);

perClass = wx_per_class_metrics(C, classes);
wx_writetable_strict(perClass, fullfile(outDir,"per_class_metrics.csv"), 'Encoding','UTF-8');

fid = fopen(fullfile(outDir,"metrics.txt"), 'w');
fprintf(fid, "OutDir: %s\n", outDir);
fprintf(fid, "RunTag: %s\n\n", opts.RunTag);

fprintf(fid, "Label rule(regex): (?i)-(R0[1-3])-(\\d{2})-\n");
fprintf(fid, "Mode2 classes used: %d\n\n", numel(classes));

fprintf(fid, "Counts:\n");
fprintf(fid, "  Train: %d\n", height(Ttrain));
fprintf(fid, "  Val  : %d\n", height(Tva));
fprintf(fid, "  Test : %d\n\n", height(Ttest));

fprintf(fid, "Train class counts: %s\n", mat2str(counts));
fprintf(fid, "ClassWeights      : %s\n\n", mat2str(classW));

fprintf(fid, "Test Accuracy: %.6f (%.2f%%)\n\n", acc, acc*100);
fprintf(fid, "Confusion Matrix (rows=true, cols=pred), order=classes\n");
fprintf(fid, "%s\n", mat2str(C));
fclose(fid);

disp("== DONE ==");
fprintf("Saved to: %s\n", outDir);

end

%% =========================================================================
% Local Functions
%% =========================================================================
function out = ternary(cond, a, b)
if cond, out = a; else, out = b; end
end

function outDir = wx_prepare_outdir_strict(opts)
% 반드시 opts.OutRoot 아래에 생성한다. 실패 시 fallback 금지(요구사항).
outRoot = char(string(opts.OutRoot));
proj    = char(string(opts.ProjectName));
tag     = char(string(opts.RunTag));

% 1) 상위 폴더 생성
if ~isfolder(outRoot)
    [ok,msg] = mkdir(outRoot);
    if ~ok
        error("results 폴더 생성 실패: %s\n경로=%s\nWindows 보안(제어된 폴더 액세스) 또는 권한을 확인하세요.", string(msg), outRoot);
    end
end

projDir = fullfile(outRoot, proj);
if ~isfolder(projDir)
    [ok,msg] = mkdir(projDir);
    if ~ok
        error("프로젝트 폴더 생성 실패: %s\n경로=%s", string(msg), projDir);
    end
end

outDir = fullfile(projDir, tag);
if ~isfolder(outDir)
    [ok,msg] = mkdir(outDir);
    if ~ok
        error("RunTag 폴더 생성 실패: %s\n경로=%s", string(msg), outDir);
    end
end

% 2) 쓰기 테스트(여기서 막히면 Desktop 보호/권한 문제)
testFile = fullfile(outDir, "__write_test__.tmp");
fid = fopen(testFile, "w");
if fid < 0
    error( ...
        "결과 폴더에 파일 쓰기 실패(권한/보안 문제).\n" + ...
        "경로: " + string(outDir) + "\n\n" + ...
        "대응:\n" + ...
        " - Windows Defender > 랜섬웨어 방지 > 제어된 폴더 액세스가 켜져 있으면 MATLAB을 허용 앱에 추가\n" + ...
        " - 또는 results 폴더를 OneDrive Desktop이 아닌 로컬 폴더로 변경(하지만 본 코드는 요구상 Desktop 고정)\n" );
else
    fclose(fid);
    delete(testFile);
end
end

function wx_writetable_strict(T, filePath, varargin)
% outDir은 이미 쓰기 테스트 완료 상태여야 한다.
folder = fileparts(filePath);
if ~isfolder(folder)
    [ok,msg] = mkdir(folder);
    if ~ok
        error("writetable 대상 폴더 생성 실패: %s\n경로=%s", string(msg), string(folder));
    end
end
writetable(T, filePath, varargin{:});
end

function canPar = wx_can_parallel()
canPar = false;
if exist("parpool","file") ~= 2
    return;
end
try
    canPar = license("test","Distrib_Computing_Toolbox");
catch
    canPar = false;
end
end

function wx_ensure_parpool(opts)
p = gcp('nocreate');
if ~isempty(p), return; end

poolType = string(opts.ParallelPoolType);
nWorkers = wx_pick_workers(opts);

if poolType == "Threads"
    parpool("threads", nWorkers);
else
    parpool("local", nWorkers);
end

fprintf("Parallel pool started: %s | Workers=%d\n", poolType, nWorkers);
end

function n = wx_pick_workers(opts)
try
    c = parcluster("local");
    nMax = c.NumWorkers;
catch
    nMax = 12;
end

if opts.ParallelWorkers > 0
    n = opts.ParallelWorkers;
else
    try
        nPhys = feature("numcores");  % 물리 코어 수
    catch
        nPhys = 4;
    end
    n = max(1, nPhys - 1);
end

n = min([n, nMax, opts.MaxAutoWorkers]);
n = max(1, n);
end

function wx_close_parpool()
p = gcp('nocreate');
if ~isempty(p)
    delete(p);
end
end

function wx_require_gpu()
try
    g = gpuDevice;
    fprintf("GPU: %s | CC: %s | Mem(GB): %.2f\n", g.Name, g.ComputeCapability, g.TotalMemory/1024^3);
catch ME
    error("GPU가 필요합니다. gpuDevice 실패: %s", ME.message);
end
end

function wx_mkdir(p)
p = char(string(p));
if ~isfolder(p)
    mkdir(p);
end
end

function T = wx_collect_images(folders, splitStr)
folders = string(folders);
files = string.empty(0,1);

for i = 1:numel(folders)
    f = folders(i);
    if ~isfolder(f)
        warning("rocktype:MissingFolder", "%s", char("폴더 없음: " + f));
        continue;
    end
    files = [files; wx_list_images_in_folder(f)]; %#ok<AGROW>
end

files = unique(files, 'stable');
T = table(files, repmat(string(splitStr),numel(files),1), ...
    'VariableNames', {'image_path','split'});
end

function files = wx_list_images_in_folder(folder)
exts = [".jpg",".jpeg",".png",".bmp",".tif",".tiff"];
files = string.empty(0,1);

for e = exts
    d = dir(fullfile(char(folder), "**", "*" + e));
    if ~isempty(d)
        files = [files; string(fullfile({d.folder}', {d.name}'))]; %#ok<AGROW>
    end
end
files = unique(files, 'stable');
end

function classes = wx_mode2_classes_all()
classes = [ ...
 "R01_01_granite"
 "R01_02_diorite"
 "R01_03_gabbro"
 "R01_04_rhyolite"
 "R01_05_andesite"
 "R01_06_basalt"
 "R01_07_tuff"
 "R02_01_gneiss"
 "R02_02_schist"
 "R02_03_phyllite"
 "R02_04_slate"
 "R02_05_quartzite"
 "R02_06_marble"
 "R03_01_conglomerate"
 "R03_02_sandstone"
 "R03_03_mudstone"
 "R03_04_shale"
 "R03_05_limestone"
 "R03_06_dolomite" ];
end

function [T2, unlT] = wx_label_by_filename_mode2(T, classesAll)
n = height(T);
labelStr = strings(n,1);
majorStr = strings(n,1);
minorStr = strings(n,1);
ok = false(n,1);

for i = 1:n
    p = T.image_path(i);
    [~, name, ~] = fileparts(char(p));

    tok = regexp(name, '(?i)-(R0[1-3])-(\d{2})-', 'tokens', 'once');
    if isempty(tok)
        continue;
    end

    maj = upper(string(tok{1}));
    minc = string(tok{2});

    key = maj + "_" + minc + "_" + wx_minor_name(maj, minc);
    if ismember(key, classesAll)
        labelStr(i) = key;
        majorStr(i) = maj;
        minorStr(i) = minc;
        ok(i) = true;
    end
end

unlT = table(string(T.split(~ok)), string(T.image_path(~ok)), ...
    'VariableNames', {'split','image_path'});

T2 = T(ok,:);
T2.major_code = majorStr(ok);
T2.minor_code = minorStr(ok);
T2.label = categorical(labelStr(ok), classesAll);
end

function nm = wx_minor_name(major, minor)
major = string(major); minor = string(minor);
nm = "";

if major == "R01"
    switch minor
        case "01", nm = "granite";
        case "02", nm = "diorite";
        case "03", nm = "gabbro";
        case "04", nm = "rhyolite";
        case "05", nm = "andesite";
        case "06", nm = "basalt";
        case "07", nm = "tuff";
        otherwise, nm = "";
    end
elseif major == "R02"
    switch minor
        case "01", nm = "gneiss";
        case "02", nm = "schist";
        case "03", nm = "phyllite";
        case "04", nm = "slate";
        case "05", nm = "quartzite";
        case "06", nm = "marble";
        otherwise, nm = "";
    end
elseif major == "R03"
    switch minor
        case "01", nm = "conglomerate";
        case "02", nm = "sandstone";
        case "03", nm = "mudstone";
        case "04", nm = "shale";
        case "05", nm = "limestone";
        case "06", nm = "dolomite";
        otherwise, nm = "";
    end
end
end

function T2 = wx_filter_unknown_classes(T, classes, outCsv)
if height(T) == 0
    T2 = T; return;
end

keep = ismember(string(T.label), string(classes));
drop = T(~keep,:);
T2 = T(keep,:);

if height(drop) > 0
    colsWanted = ["split","image_path","label","major_code","minor_code"];
    colsHave = colsWanted(ismember(colsWanted, string(drop.Properties.VariableNames)));
    if isempty(colsHave)
        tmp = drop(:, "image_path");
    else
        tmp = drop(:, colsHave);
    end
    wx_writetable_strict(tmp, outCsv, 'Encoding','UTF-8');
end
end

function [Tgood, badT] = wx_filter_bad_images(T)
n = height(T);
good = true(n,1);

badT = table('Size',[0 4], ...
    'VariableTypes', {'string','string','string','string'}, ...
    'VariableNames', {'split','label','image_path','message'});

hasLabel = ismember("label", string(T.Properties.VariableNames));

for i = 1:n
    p = T.image_path(i);
    try
        wx_try_imread(p);
    catch ME
        good(i) = false;
        lbl = "";
        if hasLabel
            lbl = string(T.label(i));
        end
        newRow = table(string(T.split(i)), lbl, string(p), string(ME.message), ...
            'VariableNames', badT.Properties.VariableNames);
        badT = [badT; newRow]; %#ok<AGROW>
    end
end

Tgood = T(good,:);
end

function [Tgood, badT] = wx_filter_bad_images_parallel(T)
n = height(T);
good = true(n,1);
msg  = strings(n,1);

paths = string(T.image_path);
spl   = string(T.split);

hasLabel = ismember("label", string(T.Properties.VariableNames));
if hasLabel
    lbl = string(T.label);
else
    lbl = repmat("", n, 1);
end

parfor i = 1:n
    try
        wx_try_imread(paths(i));
    catch ME
        good(i) = false;
        msg(i)  = string(ME.message);
    end
end

Tgood = T(good,:);
badIdx = find(~good);
badT = table( ...
    spl(badIdx), lbl(badIdx), paths(badIdx), msg(badIdx), ...
    'VariableNames', {'split','label','image_path','message'} );
end

function wx_try_imread(filename)
ws = warning;
cleanupObj = onCleanup(@() warning(ws)); 
warning('off','MATLAB:imagesci:jpg:libraryMessage');
warning('off','MATLAB:imagesci:jpg:corruptData');
I = imread(char(filename)); %#ok<NASGU>
end

function [Ttrain, Ttest] = wx_stratified_split_safe(T, testRatio, classesCat)
Y = categorical(string(T.label), categories(classesCat));
idxTest = false(height(T),1);

cats = categories(Y);
for c = 1:numel(cats)
    ic = find(Y == cats{c});
    nC = numel(ic);
    if nC <= 1
        continue;
    end

    nT = round(nC * testRatio);
    nT = max(0, nT);
    nT = min(nC-1, nT); % train 최소 1장 보장

    if nT > 0
        rp = ic(randperm(nC));
        idxTest(rp(1:nT)) = true;
    end
end

Ttest  = T(idxTest,:);
Ttrain = T(~idxTest,:);
end

function [Tbal, infoT] = wx_balance_train_table(Ttrain, classes, opts)
rng(opts.SEED);

counts = wx_counts_vec(Ttrain, classes);
countsNZ = counts(counts>0);
if isempty(countsNZ)
    error("Balance 실패: Train에 유효 클래스가 없습니다.");
end

switch string(opts.BalanceTargetMode)
    case "medianCap"
        target = round(median(countsNZ));
    case "min"
        target = min(countsNZ);
    otherwise
        target = round(median(countsNZ));
end

target = max(target, opts.MinPerClassTrain);
target = min(target, opts.MaxPerClassTrain);

parts = repmat({Ttrain([],:)}, numel(classes), 1);
infoRows = cell(0,4);

for k = 1:numel(classes)
    cls = classes(k);
    idx = find(string(Ttrain.label) == string(cls));
    n = numel(idx);

    if n == 0
        infoRows(end+1,:) = {string(cls), n, 0, "absent"}; %#ok<AGROW>
        continue;
    end

    if n == target
        pick = idx;
        action = "keep";
    elseif n > target
        pick = idx(randperm(n, target));
        action = "downsample";
    else
        need = target - n;
        add = idx(randi(n, need, 1));   % 복원추출(중복 허용)
        pick = [idx; add];
        action = "upsample";
    end

    parts{k} = Ttrain(pick,:);
    infoRows(end+1,:) = {string(cls), n, target, string(action)}; %#ok<AGROW>
end

Tbal = vertcat(parts{:});
if height(Tbal) > 0
    Tbal = Tbal(randperm(height(Tbal)),:); % shuffle
end

infoT = cell2table(infoRows, 'VariableNames', {'class','n_before','n_after','action'});
end

function c = wx_counts_vec(T, classes)
c = zeros(numel(classes),1);
if height(T)==0, return; end
for k=1:numel(classes)
    c(k) = sum(string(T.label) == string(classes(k)));
end
end

function wx_write_counts(T, classes, outCsv)
counts = wx_counts_vec(T, classes);
tbl = table(string(classes), counts, 'VariableNames', {'class','count'});
wx_writetable_strict(tbl, outCsv, 'Encoding','UTF-8');
end

function T2 = wx_add_split(T, splitName)
T2 = T;
T2.split = repmat(string(splitName), height(T2), 1);
end

function wx_set_runtime_badlog(pathStr)
wx_runtime_badlog(pathStr);
p = wx_runtime_badlog();
if strlength(p) > 0 && ~isfile(p)
    fid = fopen(char(p), 'w');
    fprintf(fid, "timestamp,image_path,message\n");
    fclose(fid);
end
end

function p = wx_runtime_badlog(newPath)
persistent P
if nargin > 0
    P = string(newPath);
end
if isempty(P)
    P = "";
end
p = P;
end

function I = wx_read_image_uint8_safe(filename)
pLog = wx_runtime_badlog();

ws = warning;
cleanupObj = onCleanup(@() warning(ws)); 
warning('off','MATLAB:imagesci:jpg:libraryMessage');
warning('off','MATLAB:imagesci:jpg:corruptData');

try
    I = imread(char(filename));

    if ismatrix(I)
        I = repmat(I, [1 1 3]);
    elseif ndims(I) == 3 && size(I,3) > 3
        I = I(:,:,1:3);
    end

    if ~isa(I,'uint8')
        I = im2uint8(I);
    end

catch ME
    I = uint8(zeros(224,224,3)); % 더미(검정)

    if strlength(pLog) > 0
        try
            fid = fopen(char(pLog), 'a');
            ts = char(datetime("now","Format","yyyy-MM-dd HH:mm:ss"));
            msg = string(ME.message);
            msg = replace(msg, """", "'");
            fprintf(fid, """%s"",""%s"",""%s""\n", ts, char(filename), msg);
            fclose(fid);
        catch
        end
    end
end
end

function [learnableLayer, classLayer] = wx_find_layers_to_replace(lgraph)
layers = lgraph.Layers;

classLayer = [];
for i = numel(layers):-1:1
    if isa(layers(i),'nnet.cnn.layer.ClassificationOutputLayer') || ...
       isa(layers(i),'nnet.cnn.layer.ClassificationLayer')
        classLayer = layers(i);
        break;
    end
end
if isempty(classLayer)
    error("Classification layer를 찾지 못했습니다.");
end

learnableLayer = [];
for i = numel(layers):-1:1
    if isa(layers(i),'nnet.cnn.layer.FullyConnectedLayer') || ...
       isa(layers(i),'nnet.cnn.layer.Convolution2DLayer')
        learnableLayer = layers(i);
        break;
    end
end
if isempty(learnableLayer)
    error("Learnable layer(FC/Conv)를 찾지 못했습니다.");
end
end

function perClass = wx_per_class_metrics(C, classes)
K = size(C,1);
prec = zeros(K,1);
rec  = zeros(K,1);
f1   = zeros(K,1);
supp = sum(C,2);

for i=1:K
    tp = C(i,i);
    fp = sum(C(:,i)) - tp;
    fn = sum(C(i,:)) - tp;

    prec(i) = tp / max(tp+fp,1);
    rec(i)  = tp / max(tp+fn,1);
    f1(i)   = 2*prec(i)*rec(i) / max(prec(i)+rec(i), eps);
end

perClass = table(string(classes), supp, prec, rec, f1, ...
    'VariableNames', {'class','support','precision','recall','f1'});
end
