
function runInfo = CoreWeatheringPaperKit_OneFile2027(varargin)
% CoreWeathering_Ultra2028_FULL  ─ 화강암 코어 풍화등급 SCI급 완전 단일파일
% =========================================================================
%  ■ 핵심 개선 (원본 대비)
%
%  [GMM-UPGRADE] 국소영역 희석 문제 완전 해결
%    ALP  : Adaptive Local Prior  — 타일별 b*분포로 mu 후보 추정 후
%           전역 GMM 초기값 가중 업데이트 → 소수 군집이 평균에 희석 방지
%    RRC  : Responsibility Clip   — 사후확률 하한(RESP_CLIP_MIN) 설정으로
%           소수 군집의 확률이 0으로 붕괴되는 것 방지
%    MCI  : Monotone Constraint   — b* 방향 단조성 강제 정렬 (D1→D5)
%    MSS  : Multi-start Best-NLL  — GMM_REPS 다중 시작, 최고 NLL 선택
%    LBS  : Local Blend Sigma     — ALP/전역 mu 혼합 비율(ALP_BLEND_ALPHA)
%
%  [ROI-UPGRADE] 흑색 제외 6레이어 완전 시스템
%    R1   : RGB_raw 채널별  ≤ BLACK_RGB_MAX (기본 8)
%    R2   : sum(RGB_raw)   ≤ BLACK_SUM_MAX  (기본 24)
%    R3   : CIELAB L*      ≤ ROI_LSTAR_BLACK_MAX (기본 22)
%           → 본 연구 D1 L*하한 52 대비 30pt 여유
%    R4   : HSV V          ≤ ROI_V_BLACK_MAX (기본 0.10)
%    R5   : L*≤(R3+3) AND V≤R4 복합 경계조건
%    R6   : HSV S ≥ ROI_S_MIN_FOR_VALID 채도 하한 (기본 0.04)
%           (완전 무채색 검은돌 제거)
%    형태학적: bridge→close→fill→area-open→component→final-close
%
%  [SCI 출력] 분석항목 22종 + CSV 11종 + PNG 35종 + FIG 16종 + MAT + XLSX
%
%  ■ 색상 (논문 표준)
%    D1 진한파랑  [0.11, 0.25, 0.78]
%    D2 연한파랑  [0.41, 0.71, 0.96]
%    D3 초록      [0.16, 0.73, 0.34]
%    D4 주황      [0.96, 0.61, 0.11]
%    D5 채도높은레드 [0.86, 0.10, 0.12]
%
%  ■ 기준표 (본 연구 수정)
%    D1  L*52-60  a*-3.5~-1.0  b*-1.5~ 1.0
%    D2  L*48-56  a*-3.5~ 0.5  b*-1.5~ 5.5
%    D3  L*42-50  a*-1.5~ 3.0  b* 0.5~ 8.5
%    D4  L*35-45  a* 0.0~ 6.0  b* 4.0~12.5
%    D5  L*28-38  a* 1.0~ 6.0  b* 6.0~15.0
%
%  ■ 가중치  W_b=1.00 > W_a=0.60 > W_L=0.30 (b*=황색도 최강 판별력)
%
%  ■ 사용법
%    runInfo = CoreWeathering_Ultra2028_FULL();
%    runInfo = CoreWeathering_Ultra2028_FULL('IMAGE_FILE','core.jpg');
%    runInfo = CoreWeathering_Ultra2028_FULL('MODE','folder','IMAGE_DIR','C:\cores');
%    runInfo = CoreWeathering_Ultra2028_FULL('GMM_ALP',true,'ALP_TILE',64,'GMM_REPS',8);
% =========================================================================

OPT = u28_parse_opts(varargin{:});
rng(double(OPT.RNG), 'twister');

tstamp  = char(datetime('now','Format','yyyyMMdd_HHmmss'));
runName = char(OPT.RUN_NAME);
if isempty(runName), runName = ['Ultra2028_' tstamp]; end
baseDir = char(OPT.OUT_BASEDIR);
if isempty(baseDir), baseDir = pwd; end

outDir = fullfile(baseDir, runName);
pngDir = fullfile(outDir,'png');
matDir = fullfile(outDir,'mat');
csvDir = fullfile(outDir,'csv');
figDir = fullfile(outDir,'figures');
perDir = fullfile(outDir,'per_image');
logDir = fullfile(outDir,'log');

u28_ensure({outDir,pngDir,matDir,csvDir,figDir,perDir,logDir});
C = u28_granite_refs();

try
    diary off;
    diary(fullfile(logDir,'run_log.txt'));
    diary on;
catch mexc28 
end
u28_print_header(OPT);
fprintf('[INFO] Output root: %s\n\n', outDir);

files = u28_collect_files(OPT);
if isempty(files), error('Ultra2028:noFiles','No images found.'); end
fprintf('[INFO] %d 이미지 처리 예정\n\n', numel(files));

sumCell  = cell(numel(files),1);
nSkip    = 0;
skipped  = cell(numel(files),1);

for ii = 1:numel(files)
    f = files{ii};
    [~,stem,ext] = fileparts(f);
    itemId = sprintf('%06d_%s', ii, stem);
    fprintf('══════════════════════════════════════════════════\n');
    fprintf('[%d/%d] %s%s\n', ii, numel(files), stem, ext);

    iRoot = fullfile(perDir,itemId);
    iPng  = fullfile(iRoot,'png');
    iCsv  = fullfile(iRoot,'csv');
    iMat  = fullfile(iRoot,'mat');
    iFig  = fullfile(iRoot,'figures');
    u28_ensure({iRoot,iPng,iCsv,iMat,iFig});

    try I0=imread(f);
    catch mexc28
        warning('U28:read','[SKIP] %s',mexc28.message); nSkip=nSkip+1; skipped{nSkip}=f; continue;
    end

    I0       = u28_force_rgb(I0);
    I0       = u28_maybe_resize(I0,OPT);
    RGB_raw  = u28_to_uint8(I0);
    RGB_srgb = u28_to_srgb01(I0);

    %% ── ROI: 흑색 6레이어 제외 ─────────────────────────────────────────
    [mask, roiInfo] = u28_make_roi(RGB_raw, RGB_srgb, OPT);
    if nnz(mask) < OPT.MIN_ROI_PIXELS
        warning('U28:roi','[SKIP] ROI<min: %d px',nnz(mask));
        nSkip=nSkip+1; skipped{nSkip}=f; continue;
    end
    fprintf('  ROI keep=%d px  dark_excl=%.1f%%\n',nnz(mask),roiInfo.darkExcFrac*100);

    if OPT.SAVE_ORIGINAL_COPY
        try
            imwrite(RGB_raw, fullfile(iPng,'original.png'));
            imwrite(uint8(mask)*255, fullfile(iPng,'mask_roi.png'));
            imwrite(uint8(roiInfo.darkMask)*255, fullfile(iPng,'mask_dark_excluded.png'));
        catch mexc28
            warning('U28:imwrite','%s',mexc28.message);
        end
    end

    %% ── 색공간 변환 ─────────────────────────────────────────────────────
    HSV = rgb2hsv(RGB_srgb);
    Hc=HSV(:,:,1); Sc=HSV(:,:,2); Vc=HSV(:,:,3);
    try
        Lab=rgb2lab(RGB_srgb);
    catch mexc28
        warning('U28:lab','[SKIP] %s',mexc28.message); nSkip=nSkip+1; skipped{nSkip}=f; continue;
    end
    Lc=Lab(:,:,1); ac=Lab(:,:,2); bc=Lab(:,:,3);

    if OPT.SMOOTH_SIGMA>0
        Lg=imgaussfilt(Lc,OPT.SMOOTH_SIGMA,'Padding','replicate');
        ag=imgaussfilt(ac,OPT.SMOOTH_SIGMA,'Padding','replicate');
        bg=imgaussfilt(bc,OPT.SMOOTH_SIGMA,'Padding','replicate');
    else
        Lg=Lc; ag=ac; bg=bc;
    end

    %% ── 채널 벡터 ───────────────────────────────────────────────────────
    Rraw=RGB_raw(:,:,1); Grow=RGB_raw(:,:,2); Braw=RGB_raw(:,:,3);
    Rs=RGB_srgb(:,:,1); Gs=RGB_srgb(:,:,2); Bs=RGB_srgb(:,:,3);
    Rraw_v=single(Rraw(mask)); Graw_v=single(Grow(mask)); Braw_v=single(Braw(mask));
    Rsv=single(Rs(mask)); Gsv=single(Gs(mask)); Bsv=single(Bs(mask));
    Hv=single(Hc(mask)); Sv=single(Sc(mask)); Vv=single(Vc(mask));
    Lv=single(Lc(mask)); av=single(ac(mask)); bv=single(bc(mask));

    %% ── 인덱스 계산 ─────────────────────────────────────────────────────
    a50=median(double(av),'omitnan'); b50=median(double(bv),'omitnan');
    dE = sqrt((ac-a50).^2+(bc-b50).^2);
    dEv= single(dE(mask));

    Ln  =u28_norm(Lc,mask,5,95); an  =u28_norm(ac,mask,5,95);
    bn  =u28_norm(bc,mask,5,95); Sn  =u28_norm(Sc,mask,5,95);
    Vn  =u28_norm(Vc,mask,5,95); dEn =u28_norm(dE,mask,5,95);
    dEn2=u28_norm(dEn,mask,5,95);

    SVI = u28_norm(0.55*Sn+0.45*(1-Vn),mask,5,95);

    weatherScore = OPT.WHI_W_b*bn + OPT.WHI_W_a*an + OPT.WHI_W_L*(1-Ln) + ...
                   OPT.WHI_W_S*Sn  + OPT.WHI_W_V*(1-Vn) + OPT.WHI_W_dE*dEn2;
    weatherScore(~mask)=0;
    WHI = u28_norm(weatherScore,mask,5,95);

    %% ── 분류 ────────────────────────────────────────────────────────────
    fprintf('  [RULE] 규칙 기반...\n');
    [D_rule,postRule,scoreRule,distRule,tieRule,WI_rule] = ...
        u28_rule_grade(Lg,ag,bg,dEn,mask,OPT,C);

    fprintf('  [ALP-GMM] 국소희석 보완 GMM...\n');
    [D_gmm,postGMM,WI_gmm,confGMM,gmmInfo] = ...
        u28_alp_gmm(Lg,ag,bg,dEn,mask,OPT,C);

    fprintf('  [BLEND] Rule+GMM 혼합...\n');
    [D,post5,WI_hybrid,confHybrid,tieMap] = ...
        u28_blend(postRule,postGMM,tieRule,mask,OPT);

    if OPT.POST_MAJOR_FILTER
        D      = u28_majority(D,      mask,OPT.MAJORITY_WIN);
        D_gmm  = u28_majority(D_gmm,  mask,OPT.MAJORITY_WIN);
        D_rule = u28_majority(D_rule, mask,OPT.MAJORITY_WIN);
    end
    WD10 = u28_split10(D,bg,mask);

    %% ── 텍스처 & 타일 ───────────────────────────────────────────────────
    tile = max(8,round(OPT.TILE));
    tex  = u28_texture_pack(Ln,bc,dEn,mask,OPT);
    [tileMedL,tileIqrL] = u28_tile_stats(Lc,mask,tile);
    [tileMedb,tileIqrb] = u28_tile_stats(bc,mask,tile);
    [tileMedS,tileIqrS] = u28_tile_stats(Sc,mask,tile);
    [tileMedV,tileIqrV] = u28_tile_stats(Vc,mask,tile);
    [tileMeda,tileIqra] = u28_tile_stats(ac,mask,tile);

    gray    = rgb2gray(RGB_srgb);
    BWedge  = edge(gray,'Sobel');
    edgeDen = u28_tile_mean(single(BWedge).*single(mask),mask,tile);
    locVar  = u28_tile_mean(u28_local_var(gray).*single(mask),mask,tile);
    tileWI  = u28_tile_mean(single(WI_hybrid).*single(mask),mask,tile);
    tileCnf = u28_tile_mean(single(confHybrid).*single(mask),mask,tile);

    wlist  = OPT.TEXTURE_WINS;
    wS=wlist(1); wL=wlist(end);
    tGradL  = u28_tile_mean(tex.gradL  .*single(mask),mask,tile);
    tGradb  = u28_tile_mean(tex.gradb  .*single(mask),mask,tile);
    tGraddE = u28_tile_mean(tex.graddEn.*single(mask),mask,tile);
    tEntL   = u28_tile_mean(tex.entL   .*single(mask),mask,tile);
    tLoG    = u28_tile_mean(tex.logL   .*single(mask),mask,tile);
    tStdS   = u28_tile_mean(tex.(sprintf('stdL_w%d',wS)).*single(mask),mask,tile);
    tStdL   = u28_tile_mean(tex.(sprintf('stdL_w%d',wL)).*single(mask),mask,tile);
    tRngL   = u28_tile_mean(tex.(sprintf('rngL_w%d',wL)).*single(mask),mask,tile);

    transM   = u28_transition(D,mask);
    heatGrid = u28_heat(D,mask,OPT.HEAT_GRID_N);
    [f5,f10] = u28_fracs(D,WD10,mask);
    gStat    = u28_grade_stats(D,Lc,ac,bc,mask,C);
    wiMapC   = u28_wi_continuous(D,post5,mask,size(Lc));
    entMap   = u28_entropy_map(post5,mask,size(Lc));

    fprintf('  [CSV] 저장...\n');
    u28_csv_all(Rraw_v,Graw_v,Braw_v,Rsv,Gsv,Bsv,...
        Hv,Sv,Vv,Lv,av,bv,dEv,...
        dEn,WHI,SVI,WI_rule,WI_gmm,WI_hybrid,confGMM,confHybrid,...
        tex,mask,D,WD10,OPT,...
        tileMedL,tileIqrL,tileMedb,tileIqrb,...
        tileMedS,tileIqrS,tileMedV,tileIqrV,tileMeda,tileIqra,...
        edgeDen,locVar,tileWI,tileCnf,...
        tGradL,tGradb,tGraddE,tEntL,tLoG,tStdS,tStdL,tRngL,...
        tile,wS,wL,...
        Lg,ag,bg,transM,gStat,iCsv,C);

    if OPT.SAVE_MAPS
        fprintf('  [MAP] PNG 저장...\n');
        u28_save_maps(RGB_raw,RGB_srgb,...
            Hc,Sc,Vc,Lc,ac,bc,...
            dE,dEn,WHI,SVI,...
            WI_rule,WI_gmm,WI_hybrid,...
            D_rule,D_gmm,D,WD10,...
            tieMap,confGMM,confHybrid,entMap,...
            roiInfo,tex,...
            tileMedL,tileIqrL,tileMedb,tileIqrb,...
            edgeDen,locVar,tileWI,tileCnf,...
            tGradL,tEntL,tLoG,...
            mask,iPng,OPT,C);
    end

    if OPT.SAVE_FIGS
        fprintf('  [FIG] 논문급 그림...\n');
        u28_all_figures(I0,RGB_raw,D,D_rule,D_gmm,WD10,...
            post5,postRule,postGMM,...
            WI_hybrid,WI_rule,WI_gmm,wiMapC,...
            confHybrid,entMap,...
            Lc,ac,bc,Sc,Vc,...
            f5,f10,gStat,transM,heatGrid,...
            tileWI,mask,OPT,C,iFig);
    end

    if OPT.SAVE_MAT
        fprintf('  [MAT] 저장...\n');
        u28_save_mat(f,itemId,mask,roiInfo,...
            RGB_raw,RGB_srgb,HSV,Lab,...
            dE,dEn,weatherScore,WHI,SVI,...
            WI_rule,WI_gmm,WI_hybrid,...
            D_rule,D_gmm,D,WD10,...
            postRule,postGMM,post5,scoreRule,distRule,...
            tieMap,confGMM,confHybrid,entMap,gmmInfo,C,...
            tile,tileMedL,tileIqrL,tileMedb,tileIqrb,...
            tileMedS,tileIqrS,tileMedV,tileIqrV,...
            edgeDen,locVar,tileWI,tileCnf,...
            tex,tGradL,tGradb,tGraddE,tEntL,tLoG,...
            tStdS,tStdL,tRngL,OPT.TEXTURE_WINS,...
            transM,heatGrid,gStat,wiMapC,...
            iMat,matDir,itemId);
    end

    sumCell{ii} = u28_summary_row(f,itemId,stem,ext,...
        RGB_raw,mask,roiInfo,...
        Rraw_v,Graw_v,Braw_v,Rsv,Gsv,Bsv,...
        Lv,av,bv,Hv,Sv,Vv,dEv,...
        WHI,SVI,WI_rule,WI_gmm,WI_hybrid,...
        D_rule,D_gmm,D,tieMap,confGMM,confHybrid,...
        tex,f5,f10,gmmInfo,entMap,OPT);

    fprintf('  ✔  주등급=D%d  WI_med=%.2f  ALP-GMM=%d\n',...
        mode(double(D(mask))),median(WI_hybrid(mask),'omitnan'),gmmInfo.gmmUsed);
end

valid = sumCell(~cellfun(@isempty,sumCell));
if ~isempty(valid)
    T = vertcat(valid{:});
    u28_wtbl(T, fullfile(csvDir,'IMAGE_SUMMARY.csv'));
    if OPT.SAVE_EXCEL
        try
            xl = fullfile(outDir,'WEATHERING_Ultra2028_FULL.xlsx');
            writetable(T, xl,'Sheet','SUMMARY');
            cfg = struct2table(OPT,'AsArray',true);
            writetable(cfg, xl,'Sheet','CONFIG');
            fprintf('[OK] XLSX: %s\n',xl);
        catch mexc28
            warning('U28:xlsx','%s',mexc28.message);
        end
    end
end

runInfo.outDir     = outDir;
runInfo.nFound     = numel(files);
runInfo.nProcessed = sum(~cellfun(@isempty,sumCell));
runInfo.skipped    = skipped(1:nSkip);
fprintf('\n[DONE] %d/%d  →  %s\n',runInfo.nProcessed,runInfo.nFound,outDir);
try 
    diary off;
catch
end
end

%% =========================================================================
%%  ALP-GMM : Adaptive Local Prior GMM  (국소희석 방지 핵심)
%% =========================================================================
function [Dgmm,postGMM,WIgmm,confMap,info] = u28_alp_gmm(Lc,ac,bc,dEn,mask,OPT,C)
%------------------------------------------------------------------
%  ALP-GMM 4단계 알고리즘
%  ① ALP  — 타일별 b*분포로 각 등급 mu 후보 계산 (로컬 희석 방지)
%  ② MCI  — 전역 Prior mu와 ALP mu를 가중 혼합
%  ③ MSS  — 다중 시작점(GMM_REPS회), 최고 NLL 선택
%  ④ RRC  — Posterior 사후확률 하한 설정 (소수 군집 붕괴 방지)
%------------------------------------------------------------------
[h,w] = size(Lc);
K     = 5;
Lv    = double(Lc(mask));
av    = double(ac(mask));
bv    = double(bc(mask));
dv    = double(dEn(mask));
N     = numel(Lv);

postGMM = zeros(N,K,'single');
confMap = zeros(h,w,'single');
WIgmm   = zeros(h,w,'single');
Dgmm    = zeros(h,w,'uint8');
info    = struct('gmmUsed',false,'sampleN',0,'bestNLL',NaN,...
    'alpUsed',OPT.GMM_ALP,'alpTile',OPT.ALP_TILE);

%── 전역 분위수 변환 ──────────────────────────────────────────────
[qL,qA,qB,qD] = u28_blend_qtile(Lc,ac,bc,dEn,mask,OPT);
featAll = [OPT.W_a*double(qA(mask)), OPT.W_b*double(qB(mask)), ...
           OPT.W_L*double(qL(mask)), OPT.W_dE*double(qD(mask))];
featAll = u28_fix_nan(featAll);

%── ① ALP: 타일별 로컬 mu 추정 ───────────────────────────────────
if OPT.GMM_ALP
    mu_alp = u28_alp_mu(Lc,ac,bc,qL,qA,qB,mask,OPT,C);
else
    mu_alp = [];
end

%── ② 전역 Prior mu 계산 ─────────────────────────────────────────
dPrior = [0.10 0.20 0.35 0.55 0.75];
mu_prior = zeros(K,4);
for k = 1:K
    mu_prior(k,1) = OPT.W_a  * u28_ecdf_sc(av, C.PRIOR_LAB(k,2));
    mu_prior(k,2) = OPT.W_b  * u28_ecdf_sc(bv, C.PRIOR_LAB(k,3));
    mu_prior(k,3) = OPT.W_L  * u28_ecdf_sc(Lv, C.PRIOR_LAB(k,1));
    mu_prior(k,4) = OPT.W_dE * u28_ecdf_sc(dv, dPrior(k));
end

%── ALP + Prior 혼합 ─────────────────────────────────────────────
if ~isempty(mu_alp)
    alpha = OPT.ALP_BLEND_ALPHA;
    mu0   = alpha*mu_alp + (1-alpha)*mu_prior;
else
    mu0 = mu_prior;
end
mu0 = min(max(mu0,0.01),0.99);

%── ③ MSS: 다중 시작 GMM ─────────────────────────────────────────
nSamp = min(N, OPT.GMM_SAMPLE_MAX);
rp    = randperm(N,nSamp);
ft    = featAll(rp,:);
ft    = ft(all(isfinite(ft),2),:);

if size(ft,1) < max(200,K*20)
    warning('U28:gmmSmall','GMM 샘플 부족(%d) → rule 결과 사용',size(ft,1));
    return;
end
info.sampleN = size(ft,1);

gopts   = statset('MaxIter',OPT.GMM_MAXITER,'Display','off');
gmBest  = [];
bestLL  = -inf;
s0      = eye(4)*0.022;

for rep = 1:max(1,OPT.GMM_REPS)
    try
        if rep==1
            mu_i = mu0;
        else
            noise = randn(K,4)*0.028;
            mu_i  = min(max(mu0+noise,0.01),0.99);
        end
        ss  = struct('mu',mu_i,'Sigma',repmat(s0,[1 1 K]),...
                     'ComponentProportion',ones(1,K)/K);
        gmt = fitgmdist(ft,K,'Start',ss,...
            'RegularizationValue',OPT.GMM_REG,'Options',gopts);
        ll  = -gmt.NegativeLogLikelihood;
        if ll > bestLL
            bestLL = ll;
            gmBest = gmt;
        end
    catch %#ok<CTCH>
    end
end

if isempty(gmBest)
    try
        gmBest = fitgmdist(ft,K,'Replicates',3,...
            'RegularizationValue',OPT.GMM_REG,'Options',gopts);
        bestLL = -gmBest.NegativeLogLikelihood;
    catch mexc28
        warning('U28:gmmFail','%s',mexc28.message); return;
    end
end
info.gmmUsed = true;
info.bestNLL = bestLL;

%── MCI: b* 방향 단조 정렬 ──────────────────────────────────────
[~,ord] = sort(gmBest.mu(:,2),'ascend');
c2g = zeros(K,1,'uint8');
for i=1:K, c2g(ord(i))=i; end

%── ④ RRC: Posterior 청크계산 + 하한클리핑 ───────────────────────
pr = zeros(N,K,'single');
s  = 1;
while s<=N
    e = min(N,s+OPT.POST_CHUNK-1);
    try
        p = single(posterior(gmBest,featAll(s:e,:)));
    catch
        p = repmat(single(1/K),e-s+1,K);
    end
    p(~isfinite(p)) = 1/K;
    % RRC: 하한 설정 → 소수 군집이 0으로 붕괴하는 것 방지
    p = max(p, OPT.RESP_CLIP_MIN);
    p = p ./ max(sum(p,2),eps);
    pr(s:e,:) = p;
    s = e+1;
end

% 컴포넌트 → D등급 재배열
for k=1:K, postGMM(:,c2g(k))=pr(:,k); end
postGMM = postGMM ./ max(sum(postGMM,2),eps);

[sp,~]    = sort(postGMM,2,'descend');
confMap(mask) = single(sp(:,1)-sp(:,2));

WIv = zeros(N,1,'single');
for k=1:K, WIv=WIv+k*postGMM(:,k); end
WIgmm(mask) = WIv;

[~,lab]   = max(postGMM,[],2);
Dgmm(mask)= uint8(lab);

fprintf('    ALP-GMM NLL/n=%.4f  D1=%.1f D2=%.1f D3=%.1f D4=%.1f D5=%.1f%%\n',...
    -bestLL/size(ft,1),...
    mean(lab==1)*100,mean(lab==2)*100,mean(lab==3)*100,...
    mean(lab==4)*100,mean(lab==5)*100);
end

%% =========================================================================
%%  ALP mu 추정
%% =========================================================================
function mu_alp = u28_alp_mu(~,~,bc,qL,qA,qB,mask,OPT,C)
% qL,qA,qB와 mask를 이용해 타일별 초기 mu 추정
% 각 타일에서 b* 기준으로 K등급에 연결 후
% 로컬 분위수 공간 mu를 추정 → 희석 없는 초기값 제공
K   = 5;
tsz = OPT.ALP_TILE;
[h,w] = size(mask);

mu_sum   = zeros(K,4);
mu_cnt   = zeros(K,1);

bCenters = mean(C.PRIOR_b_R,2);  % Kx1, 각 등급 b* 중심

for yy = 1:ceil(h/tsz)
    y0=(yy-1)*tsz+1; y1=min(h,yy*tsz);
    for xx = 1:ceil(w/tsz)
        x0=(xx-1)*tsz+1; x1=min(w,xx*tsz);
        tm = mask(y0:y1,x0:x1);
        if nnz(tm) < OPT.ALP_MIN_PIX, continue; end

        bPatch = double(bc(y0:y1,x0:x1));
        bVals  = bPatch(tm);

        % 각 픽셀을 가장 가까운 b* 등급 중심에 배정
        dist_b   = abs(bVals - bCenters');   % (npix×K)
        [~,asgn] = min(dist_b,[],2);

        qLp=double(qL(y0:y1,x0:x1));
        qAp=double(qA(y0:y1,x0:x1));
        qBp=double(qB(y0:y1,x0:x1));

        for k = 1:K
            sel  = (asgn==k);
            npix = sum(sel);
            if npix<5, continue; end

            tmFlat = tm(:);
            selMask = false(size(tmFlat));
            tmIdx   = find(tmFlat);
            selMask(tmIdx(sel)) = true;
            selMask = reshape(selMask, size(tm));

            lqA = median(qAp(selMask),'omitnan');
            lqB = median(qBp(selMask),'omitnan');
            lqL = median(qLp(selMask),'omitnan');

            if all(isfinite([lqA lqB lqL]))
                mu_sum(k,:) = mu_sum(k,:) + [OPT.W_a*lqA, OPT.W_b*lqB, OPT.W_L*lqL, 0.3];
                mu_cnt(k)   = mu_cnt(k)+1;
            end
        end
    end
end

mu_alp = zeros(K,4);
for k=1:K
    if mu_cnt(k)>0
        mu_alp(k,:) = mu_sum(k,:)/mu_cnt(k);
    else
        mu_alp(k,:) = [OPT.W_a*0.25, OPT.W_b*(k-1)/4, OPT.W_L*0.55, 0.3];
    end
end
mu_alp = min(max(mu_alp,0.01),0.99);
end

%% =========================================================================
%%  ROI: 흑색 6레이어 제외
%% =========================================================================
function [mask,info] = u28_make_roi(RGB_raw,RGB_srgb,OPT)
%------------------------------------------------------------------
%  R1: RGB 채널별 ≤ BLACK_RGB_MAX
%  R2: sum(RGB)   ≤ BLACK_SUM_MAX
%  R3: L*         ≤ ROI_LSTAR_BLACK_MAX  (CIELAB 기준)
%  R4: V          ≤ ROI_V_BLACK_MAX      (HSV 기준)
%  R5: L*≤(R3+3) AND V≤R4               (경계부 복합조건)
%  R6: S          < ROI_S_MIN_FOR_VALID  (무채색 검은돌 제거)
%------------------------------------------------------------------
blk    = all(RGB_raw <= OPT.BLACK_RGB_MAX, 3);
nearBk = sum(uint16(RGB_raw),3) <= OPT.BLACK_SUM_MAX;

darkLab=false(size(blk)); darkMix=false(size(blk)); darkS=false(size(blk));
try
    Lab0 = rgb2lab(RGB_srgb);
    HSV0 = rgb2hsv(RGB_srgb);
    L0   = Lab0(:,:,1);
    V0   = HSV0(:,:,3);
    S0   = HSV0(:,:,2);
    darkLab = L0 <= OPT.ROI_LSTAR_BLACK_MAX;
    darkMix = (L0 <= (OPT.ROI_LSTAR_BLACK_MAX+3)) & (V0 <= OPT.ROI_V_BLACK_MAX);
    darkS   = (S0 <  OPT.ROI_S_MIN_FOR_VALID) & (V0 <= 0.18);
catch mexc28
    warning('U28:roiLab','%s',mexc28.message);
end

reject = blk | nearBk | darkLab | darkMix | darkS;
mask   = ~reject;

% 형태학적 정제
if OPT.ROI_BRIDGE_RADIUS>0
    mask = imclose(mask,strel('disk',max(1,OPT.ROI_BRIDGE_RADIUS),0));
end
mask = imfill(mask,'holes');
mask = bwareaopen(mask,OPT.ROI_MIN_KEEP_COMPONENT_AREA);

if nnz(mask)>0
    CC  = bwconncomp(mask,8);
    tmp = false(size(mask));
    kid = [];
    for i=1:CC.NumObjects
        if numel(CC.PixelIdxList{i})>=OPT.ROI_MIN_KEEP_COMPONENT_AREA
            kid(end+1)=i; %#ok<AGROW>
        end
    end
    if OPT.ROI_KEEP_LARGEST_ONLY && ~isempty(kid)
        nn=cellfun(@numel,CC.PixelIdxList(kid));
        kid=kid(nn==max(nn));
    end
    for j=kid, tmp(CC.PixelIdxList{j})=true; end
    mask=tmp;
end

mask = imclose(mask,strel('disk',max(1,OPT.MASK_CLOSE_RADIUS),0));
mask = imfill(mask,'holes');
mask = bwareaopen(mask,OPT.ROI_MIN_KEEP_COMPONENT_AREA);

info.nTotal        = numel(mask);
info.nKeep         = nnz(mask);
info.nBlackRGB     = nnz(blk);
info.nNearBlack    = nnz(nearBk);
info.nDarkLab      = nnz(darkLab);
info.nDarkMix      = nnz(darkMix);
info.nDarkS        = nnz(darkS);
info.darkExcFrac   = nnz(reject)/max(numel(mask),1);
info.darkMask      = single(reject);
info.keepComponentCount = u28_ncomp(mask);
end

function n=u28_ncomp(m)
try CC=bwconncomp(m,8); n=CC.NumObjects; catch, n=0; end
end

%% =========================================================================
%%  규칙 기반 분류
%% =========================================================================
function [D,post5,score5,dist5,tieMap,WI] = u28_rule_grade(Lc,ac,bc,dEn,mask,OPT,C)
Lv=double(Lc(mask)); av=double(ac(mask));
bv=double(bc(mask)); dv=double(dEn(mask));
N=numel(Lv); K=5;
score5=zeros(N,K,'single'); dist5=zeros(N,K,'single');
for k=1:K
    cL=C.PRIOR_LAB(k,1); ca=C.PRIOR_LAB(k,2); cb=C.PRIOR_LAB(k,3);
    hL=max((C.PRIOR_L_R(k,2)-C.PRIOR_L_R(k,1))/2,eps);
    ha=max((C.PRIOR_a_R(k,2)-C.PRIOR_a_R(k,1))/2,eps);
    hb=max((C.PRIOR_b_R(k,2)-C.PRIOR_b_R(k,1))/2,eps);
    dL=((Lv-cL)./hL).^2; da=((av-ca)./ha).^2;
    db=((bv-cb)./hb).^2; dd=dv.^2;
    wd=OPT.W_L*dL+OPT.W_a*da+OPT.W_b*db+OPT.W_dE*dd;
    inL=single(Lv>=C.PRIOR_L_R(k,1)&Lv<=C.PRIOR_L_R(k,2));
    inA=single(av>=C.PRIOR_a_R(k,1)&av<=C.PRIOR_a_R(k,2));
    inB=single(bv>=C.PRIOR_b_R(k,1)&bv<=C.PRIOR_b_R(k,2));
    rSc=(OPT.W_L*inL+OPT.W_a*inA+OPT.W_b*inB)/max(OPT.W_L+OPT.W_a+OPT.W_b,eps);
    outL=u28_outp(Lv,C.PRIOR_L_R(k,:))/hL;
    outA=u28_outp(av,C.PRIOR_a_R(k,:))/ha;
    outB=u28_outp(bv,C.PRIOR_b_R(k,:))/hb;
    oP=OPT.W_L*outL.^2+OPT.W_a*outA.^2+OPT.W_b*outB.^2;
    s=exp(-0.5*wd).*(1+OPT.RANGE_BONUS*rSc).*exp(-OPT.OUTSIDE_LAMBDA*oP);
    bNrm=min(max((bv-OPT.B_NORM_MIN)/max(OPT.B_NORM_MAX-OPT.B_NORM_MIN,eps),0),1);
    anch=(k-1)/(K-1);
    sB=exp(-OPT.ORDER_GAMMA*(bNrm-anch).^2);
    s=s.*(1+OPT.ORDER_BONUS*sB);
    score5(:,k)=single(s); dist5(:,k)=single(wd+oP);
end
sm=sum(score5,2); sm(sm<=0)=1; post5=score5./sm;
[~,Dv]=max(post5,[],2); Dv=uint8(Dv);
[sp,ord]=sort(post5,2,'descend');
isTie=(sp(:,1)-sp(:,2))<=OPT.TIE_EPS;
if any(isTie)
    idx=find(isTie);
    for i=1:numel(idx)
        r=idx(i); k1=ord(r,1); k2=ord(r,2);
        d1=dist5(r,k1); d2=dist5(r,k2);
        if d2<d1, Dv(r)=uint8(k2);
        elseif abs(double(d1-d2))<=OPT.TIE_DIST_EPS
            if abs(bv(r)-C.PRIOR_LAB(k2,3))<abs(bv(r)-C.PRIOR_LAB(k1,3))
                Dv(r)=uint8(k2); end
        end
    end
end
tieMap=zeros(size(Lc),'single'); tieMap(mask)=single(isTie);
WIv=zeros(N,1,'single');
for k=1:K, WIv=WIv+k*post5(:,k); end
WI=zeros(size(Lc),'single'); WI(mask)=WIv;
D=zeros(size(Lc),'uint8'); D(mask)=Dv;
end

%% =========================================================================
%%  Rule+GMM 블렌딩
%% =========================================================================
function [D,postB,WI,confMap,tieMap] = u28_blend(postRule,postGMM,tieRule,mask,OPT)
[h,w]=size(mask); K=size(postRule,2);
if isempty(postGMM)||size(postGMM,2)~=K
    postB=postRule;
else
    postB=(1-OPT.GMM_BLEND)*postRule + OPT.GMM_BLEND*postGMM;
end
postB=postB./max(sum(postB,2),eps);

% 공간 스무딩
stack=zeros(h,w,K,'single');
for k=1:K
    tmp=zeros(h,w,'single'); tmp(mask)=postB(:,k);
    tmp=imgaussfilt(tmp,OPT.GMM_POST_SMOOTH_SIGMA,'Padding','replicate');
    tmp(~mask)=0; stack(:,:,k)=tmp;
end
pSm=zeros(nnz(mask),K,'single');
for k=1:K, t=stack(:,:,k); pSm(:,k)=t(mask); end
postB=pSm./max(sum(pSm,2),eps);

[sp,~]=sort(postB,2,'descend');
conf=sp(:,1)-sp(:,2);
confMap=zeros(h,w,'single'); confMap(mask)=single(conf);
tieMap=tieRule;
tieMap(mask)=max(single(tieRule(mask)),single(conf<=OPT.TIE_EPS));

WIv=zeros(nnz(mask),1,'single');
for k=1:K, WIv=WIv+k*postB(:,k); end
WI=zeros(h,w,'single'); WI(mask)=WIv;
[~,lab]=max(postB,[],2);
D=zeros(h,w,'uint8'); D(mask)=uint8(lab);
end

%% =========================================================================
%%  분위수 변환 & 유틸리티
%% =========================================================================
function [qL,qA,qB,qD]=u28_blend_qtile(Lc,ac,bc,dEn,mask,OPT)
qL=u28_ecdf_map(Lc,mask); qA=u28_ecdf_map(ac,mask);
qB=u28_ecdf_map(bc,mask); qD=u28_ecdf_map(dEn,mask);
if OPT.LOCAL_BLEND_ALPHA>0
    a=OPT.LOCAL_BLEND_ALPHA; tsz=OPT.LOCAL_TILE_SIZE;
    lL=u28_lp5p95(Lc,mask,tsz); lA=u28_lp5p95(ac,mask,tsz);
    lB=u28_lp5p95(bc,mask,tsz); lD=u28_lp5p95(dEn,mask,tsz);
    qL=single((1-a)*qL+a*lL); qA=single((1-a)*qA+a*lA);
    qB=single((1-a)*qB+a*lB); qD=single((1-a)*qD+a*lD);
    qL(~mask)=0; qA(~mask)=0; qB(~mask)=0; qD(~mask)=0;
end
end

function q=u28_ecdf_map(X,mask)
vals=double(X(mask)); N=numel(vals);
if N<2, q=single(repmat(0.5,size(X))); q(~mask)=0; return; end
[sv,si]=sort(vals); rk=zeros(N,1); rk(si)=(1:N)'./N;
[svu,~,ic]=unique(sv,'stable'); ru=zeros(numel(svu),1);
for i=1:numel(svu), ru(i)=mean(rk(ic==i)); end
out=interp1(svu,ru,double(X(:)),'linear','extrap');
q=reshape(single(min(max(out,0),1)),size(X)); q(~mask)=0;
end

function q=u28_lp5p95(X,mask,tsz)
[h,w]=size(X); q=zeros(h,w,'single');
for y0=1:tsz:h
    y1=min(h,y0+tsz-1);
    for x0=1:tsz:w
        x1=min(w,x0+tsz-1);
        m=mask(y0:y1,x0:x1);
        if nnz(m)<20, continue; end
        blk=double(X(y0:y1,x0:x1)); vv=blk(m);
        lo=prctile(vv,5); hi=prctile(vv,95);
        qb=min(max((blk-lo)/max(eps,hi-lo),0),1);
        tmp=q(y0:y1,x0:x1); tmp(m)=single(qb(m)); q(y0:y1,x0:x1)=tmp;
    end
end
q(~mask)=0;
end

function q=u28_ecdf_sc(v,x)
v=v(isfinite(v)); if isempty(v), q=0.5; return; end
q=min(max(mean(v<=x),0.02),0.98);
end

function Xn=u28_norm(X,mask,p1,p2)
if nnz(mask)<50, Xn=zeros(size(X),'single'); return; end
Xm=double(X(mask)); lo=prctile(Xm,p1); hi=prctile(Xm,p2);
Xn=min(max((double(X)-lo)/max(eps,hi-lo),0),1);
Xn(~mask)=0; Xn=single(Xn);
end

function y=u28_outp(x,rr)
y=zeros(size(x));
y(x<rr(1))=rr(1)-x(x<rr(1));
y(x>rr(2))=x(x>rr(2))-rr(2);
end

function f=u28_fix_nan(f)
for c=1:size(f,2)
    b=~isfinite(f(:,c));
    if any(b), f(b,c)=mean(f(:,c),'omitnan'); end
end
end

function D=u28_majority(D,mask,win)
if win<=1, return; end
pad=floor(win/2); Dp=padarray(double(D),[pad pad],0,'both'); out=D;
for r=1:size(D,1)
    for c=1:size(D,2)
        if ~mask(r,c), continue; end
        blk=Dp(r:r+2*pad,c:c+2*pad);
        vv=uint8(blk(blk>=1&blk<=5));
        if isempty(vv), continue; end
        cnt=accumarray(double(vv),1,[5 1],@sum,0);
        [~,id]=max(cnt); out(r,c)=uint8(id);
    end
end
D(mask)=out(mask);
end

function WD10=u28_split10(D,bc,mask)
WD10=zeros(size(D),'uint8'); bD=double(bc);
for d=1:5
    dm=mask&(D==d); if nnz(dm)<4, continue; end
    bm=median(bD(dm),'omitnan');
    WD10(dm&(bD<=bm))=uint8(2*d-1);
    WD10(dm&(bD> bm))=uint8(2*d);
end
end

function [f5,f10]=u28_fracs(D,WD10,mask)
den=max(1,nnz(mask)); f5=zeros(1,5); f10=zeros(1,10);
for k=1:5,  f5(k) =nnz(mask&(D==k))/den; end
for k=1:10, f10(k)=nnz(mask&(WD10==k))/den; end
end

function T=u28_grade_stats(D,Lc,ac,bc,mask,C)
T=table();
for d=1:5
    dm=mask&(D==d); nm=sprintf('D%d',d);
    if nnz(dm)<5
        flds={'L_med','a_med','b_med','L_std','a_std','b_std','dE_med','nPix'};
        for fi=flds, T.([nm,'_',fi{1}])=NaN; end
        T.([nm,'_nPix'])=0; continue;
    end
    Lv=double(Lc(dm)); av=double(ac(dm)); bv=double(bc(dm));
    dE=sqrt((Lv-C.PRIOR_LAB(d,1)).^2+(av-C.PRIOR_LAB(d,2)).^2+(bv-C.PRIOR_LAB(d,3)).^2);
    T.([nm,'_L_med'])=median(Lv,'omitnan'); T.([nm,'_L_std'])=std(Lv,'omitnan');
    T.([nm,'_a_med'])=median(av,'omitnan'); T.([nm,'_a_std'])=std(av,'omitnan');
    T.([nm,'_b_med'])=median(bv,'omitnan'); T.([nm,'_b_std'])=std(bv,'omitnan');
    for fn={'L','a','b'}
        switch fn{1}, case 'L',vv=Lv; case 'a',vv=av; case 'b',vv=bv; end
        for pp=[5 25 50 75 95]
            T.([nm,'_',fn{1},'_p',sprintf('%02d',pp)])=prctile(vv,pp);
        end
    end
    T.([nm,'_dE_med'])=median(dE,'omitnan');
    T.([nm,'_dE_p95'])=prctile(dE,95);
    T.([nm,'_nPix'])  =nnz(dm);
end
end

function wiMap=u28_wi_continuous(~,post5,mask,sz)
% D argument not used; sz=[H W]
N=size(post5,1); wiv=zeros(N,1,'single');
for k=1:5, wiv=wiv+k*post5(:,k); end
wiMap=zeros(sz(1),sz(2),'single'); wiMap(mask)=wiv;
end

function em=u28_entropy_map(post5,mask,sz)
K  = size(post5,2);
ev = -sum(post5.*log(max(post5,1e-10)),2) ./ log(max(K,2));
em = zeros(sz(1),sz(2),'single');
em(mask) = single(ev);
end

function TM=u28_transition(D,mask)
TM=zeros(5,5); mr=D; mr(~mask)=0;
Dlft=mr(:,1:end-1); Drgt=mr(:,2:end);
vm=mask(:,1:end-1)&mask(:,2:end);
lv=Dlft(vm); rv=Drgt(vm);
for i=1:numel(lv)
    if lv(i)>=1&&lv(i)<=5&&rv(i)>=1&&rv(i)<=5
        TM(lv(i),rv(i))=TM(lv(i),rv(i))+1;
    end
end
rs=sum(TM,2); rs(rs==0)=1; TM=TM./rs;
end

function G=u28_heat(D,mask,N)
[H,W]=size(D); gh=ceil(H/N); gw=ceil(W/N); G=zeros(N,N,'single');
for rr=1:N
    y0=(rr-1)*gh+1; y1=min(H,rr*gh);
    for cc=1:N
        x0=(cc-1)*gw+1; x1=min(W,cc*gw);
        p2=D(y0:y1,x0:x1); pm2=mask(y0:y1,x0:x1);
        if nnz(pm2)>0
            G(rr,cc)=mean(double(p2(pm2)),'omitnan');
        else
            G(rr,cc)=NaN;
        end
    end
end
end

function [tileMed,tileIqr]=u28_tile_stats(X,mask,tile)
[h,w]=size(X); ny=ceil(h/tile); nx=ceil(w/tile);
tileMed=zeros(ny,nx,'single'); tileIqr=zeros(ny,nx,'single');
for yy=1:ny
    y0=(yy-1)*tile+1; y1=min(h,yy*tile);
    for xx=1:nx
        x0=(xx-1)*tile+1; x1=min(w,xx*tile);
        m=mask(y0:y1,x0:x1);
        if nnz(m)<10, continue; end
        vm=double(X(y0:y1,x0:x1)); vm=vm(m);
        tileMed(yy,xx)=single(median(vm,'omitnan'));
        tileIqr(yy,xx)=single(prctile(vm,95)-prctile(vm,5));
    end
end
end

function M=u28_tile_mean(X,mask,tile)
[h,w]=size(X); ny=ceil(h/tile); nx=ceil(w/tile); M=zeros(ny,nx,'single');
for yy=1:ny
    y0=(yy-1)*tile+1; y1=min(h,yy*tile);
    for xx=1:nx
        x0=(xx-1)*tile+1; x1=min(w,xx*tile);
        m=mask(y0:y1,x0:x1);
        if nnz(m)<10, continue; end
        vm=double(X(y0:y1,x0:x1)); M(yy,xx)=single(mean(vm(m),'omitnan'));
    end
end
end

function V=u28_local_var(gray)
s=stdfilt(gray,true(5)); V=single(mat2gray(s.^2));
end

function tex=u28_texture_pack(Ln,bc,dEn,mask,OPT)
tex=struct();
try
    tex.gradL   = single(imgradient(Ln,  OPT.GRAD_METHOD));
    tex.gradb   = single(imgradient(bc,  OPT.GRAD_METHOD));
    tex.graddEn = single(imgradient(dEn, OPT.GRAD_METHOD));
catch
    tex.gradL   = u28_fdiff(Ln);
    tex.gradb   = u28_fdiff(bc);
    tex.graddEn = u28_fdiff(dEn);
end
tex.gradL(~mask)=0; tex.gradb(~mask)=0; tex.graddEn(~mask)=0;
try
    Ln8      = uint8(255*min(max(Ln,0),1));
    tex.entL = u28_norm(single(entropyfilt(Ln8,true(9))),mask,5,95);
catch
    tex.entL = zeros(size(Ln),'single');
end
tex.entL(~mask)=0;
try
    Gsmooth  = imgaussfilt(Ln,max(double(OPT.LOG_SIGMA),0.5));
    tex.logL = u28_norm(single(abs(del2(Gsmooth))),mask,5,95);
catch
    tex.logL = zeros(size(Ln),'single');
end
tex.logL(~mask)=0;
for ww=OPT.TEXTURE_WINS
    w=max(3,round(double(ww))); if mod(w,2)==0, w=w+1; end
    try
        tex.(sprintf('stdL_w%d',w))=u28_norm(single(stdfilt(Ln,true(w))),mask,5,95);
        tex.(sprintf('rngL_w%d',w))=u28_norm(single(rangefilt(Ln,true(w))),mask,5,95);
    catch
        tex.(sprintf('stdL_w%d',w))=zeros(size(Ln),'single');
        tex.(sprintf('rngL_w%d',w))=zeros(size(Ln),'single');
    end
    tex.(sprintf('stdL_w%d',w))(~mask)=0;
    tex.(sprintf('rngL_w%d',w))(~mask)=0;
end
end

function g=u28_fdiff(X)
X=single(X);
dx=[diff(X,1,2) zeros(size(X,1),1,'single')];
dy=[diff(X,1,1); zeros(1,size(X,2),'single')];
g=hypot(dx,dy);
end

%% =========================================================================
%%  CSV 저장 (11종)
%% =========================================================================
function u28_csv_all(Rraw_v,Graw_v,Braw_v,Rsv,Gsv,Bsv,...
    Hv,Sv,Vv,Lv,av,bv,dEv,...
    dEn,WHI,SVI,WI_rule,WI_gmm,WI_hybrid,confGMM,confHybrid,...
    tex,mask,D,WD10,OPT,...
    tileMedL,tileIqrL,tileMedb,tileIqrb,...
    tileMedS,tileIqrS,tileMedV,tileIqrV,tileMeda,tileIqra,...
    edgeDen,locVar,tileWI,tileCnf,...
    tGradL,tGradb,tGraddE,tEntL,tLoG,tStdS,tStdL,tRngL,...
    tile,wS,wL,...
    Lg,ag,bg,transM,gStat,iCsv,C)

wt = @(T,fn) u28_wtbl(T,fullfile(iCsv,fn));

wt(u28_stat1('RGB_raw_uint8',{'R','G','B'},{Rraw_v,Graw_v,Braw_v}), 'rgb_raw_uint8_stats.csv');
wt(u28_stat1('RGB_srgb_01',{'R','G','B'},{Rsv,Gsv,Bsv}),            'rgb_srgb01_stats.csv');
wt(u28_stat1('HSV',{'H','S','V'},{Hv,Sv,Vv}),                       'hsv_stats.csv');
wt(u28_stat1('CIELAB',{'L','a','b'},{Lv,av,bv}),                    'cielab_stats.csv');
wt(u28_stat1('INDICES',{'dE','dEn','WHI','SVI','WI_rule','WI_gmm','WI_hybrid','confGMM','confHybrid'},...
    {dEv,single(dEn(mask)),single(WHI(mask)),single(SVI(mask)),...
    single(WI_rule(mask)),single(WI_gmm(mask)),single(WI_hybrid(mask)),...
    single(confGMM(mask)),single(confHybrid(mask))}),                'indices_stats.csv');
wt(u28_stat1('TEXTURE',{'gradL','gradb','graddEn','entL','logL'},...
    {single(tex.gradL(mask)),single(tex.gradb(mask)),single(tex.graddEn(mask)),...
    single(tex.entL(mask)),single(tex.logL(mask))}),                 'texture_stats.csv');
wt(u28_frac_table(D,WD10,mask,OPT),                                  'granite_grade_fraction.csv');
wt(u28_tile_basic_table(tileMedL,tileIqrL,tileMedb,tileIqrb,...
    tileMedS,tileIqrS,tileMedV,tileIqrV,tileMeda,tileIqra,...
    edgeDen,locVar,tileWI,tileCnf,tile),                             'tiles_basic_summary.csv');
wt(u28_tile_tex_table(tGradL,tGradb,tGraddE,tEntL,tLoG,...
    tStdS,tStdL,tRngL,tile,wS,wL),                                  'tiles_texture_summary.csv');
wt(u28_ref_match(Lg,ag,bg,D,mask,C),                                 'granite_reference_match.csv');
wt(u28_grade_csv(gStat),                                             'per_grade_stats.csv');
wt(u28_trans_table(transM),                                          'transition_matrix.csv');
end

function T=u28_stat1(grp,nms,vecs)
T=table(string(grp),'VariableNames',{'group'});
for i=1:numel(nms)
    v=double(vecs{i}(:)); v=v(isfinite(v));
    if isempty(v), v=0; end
    nm=string(nms{i});
    T.(nm+"_mean")=mean(v,'omitnan'); T.(nm+"_std") =std(v,'omitnan');
    T.(nm+"_p05") =prctile(v,5);     T.(nm+"_p25") =prctile(v,25);
    T.(nm+"_p50") =prctile(v,50);    T.(nm+"_p75") =prctile(v,75);
    T.(nm+"_p95") =prctile(v,95);    T.(nm+"_ipr") =prctile(v,95)-prctile(v,5);
    T.(nm+"_min") =min(v);           T.(nm+"_max") =max(v);
    T.(nm+"_skew")=u28_skew(v);      T.(nm+"_kurt")=u28_kurt(v);
end
end

function s=u28_skew(v)
v=v-mean(v,'omitnan'); s=mean(v.^3,'omitnan')/max(std(v,'omitnan')^3,eps);
end
function k=u28_kurt(v)
v=v-mean(v,'omitnan'); k=mean(v.^4,'omitnan')/max(std(v,'omitnan')^4,eps)-3;
end

function T=u28_frac_table(D,WD10,mask,OPT)
v=double(D(mask)); vd=double(WD10(mask)); den=max(1,numel(v));
T=table(); T.n=den;
T.W_b=OPT.W_b; T.W_a=OPT.W_a; T.W_L=OPT.W_L;
for g=1:5,  T.(sprintf('D%d_frac', g))=nnz(v==g)/den; end
for g=1:10, T.(sprintf('WD10_%02d_frac',g))=nnz(vd==g)/den; end
end

function T=u28_tile_basic_table(tileMedL,tileIqrL,tileMedb,tileIqrb,...
    tileMedS,tileIqrS,tileMedV,tileIqrV,tileMeda,tileIqra,...
    edgeDen,locVar,tileWI,tileCnf,tile)
T=table(); T.tile=tile;
fns={'medL','iprL','medb','iprb','medS','iprS','medV','iprV','meda','ipra',...
     'edgeDen','locVar','tileWI','tileCnf'};
mats={tileMedL,tileIqrL,tileMedb,tileIqrb,...
      tileMedS,tileIqrS,tileMedV,tileIqrV,tileMeda,tileIqra,...
      edgeDen,locVar,tileWI,tileCnf};
for i=1:numel(fns)
    T.([fns{i},'_mean'])=mean(mats{i}(:),'omitnan');
    T.([fns{i},'_std']) =std(mats{i}(:),'omitnan');
    T.([fns{i},'_p50']) =median(mats{i}(:),'omitnan');
end
end

function T=u28_tile_tex_table(tGradL,tGradb,tGraddE,tEntL,tLoG,...
    tStdS,tStdL,tRngL,tile,wS,wL)
T=table(); T.tile=tile; T.win_small=wS; T.win_large=wL;
fns={'gradL','gradb','graddEn','entL','LoG','stdSmall','stdLarge','rngLarge'};
mats={tGradL,tGradb,tGraddE,tEntL,tLoG,tStdS,tStdL,tRngL};
for i=1:numel(fns)
    T.([fns{i},'_mean'])=mean(mats{i}(:),'omitnan');
    T.([fns{i},'_p50']) =median(mats{i}(:),'omitnan');
end
end

function T=u28_ref_match(Lg,ag,bg,D,mask,C)
T=table();
for d=1:5
    dm=mask&(D==d); n=nnz(dm); T.(sprintf('D%d_nPix',d))=n;
    T.(sprintf('D%d_L_refLo',d))=C.PRIOR_L_R(d,1);
    T.(sprintf('D%d_L_refHi',d))=C.PRIOR_L_R(d,2);
    T.(sprintf('D%d_a_refLo',d))=C.PRIOR_a_R(d,1);
    T.(sprintf('D%d_a_refHi',d))=C.PRIOR_a_R(d,2);
    T.(sprintf('D%d_b_refLo',d))=C.PRIOR_b_R(d,1);
    T.(sprintf('D%d_b_refHi',d))=C.PRIOR_b_R(d,2);
    if n<5
        T.(sprintf('D%d_L_p50',d))=NaN; T.(sprintf('D%d_a_p50',d))=NaN;
        T.(sprintf('D%d_b_p50',d))=NaN; T.(sprintf('D%d_dE_p50',d))=NaN;
        continue;
    end
    Lv=double(Lg(dm)); av=double(ag(dm)); bv=double(bg(dm));
    dE=sqrt((Lv-C.PRIOR_LAB(d,1)).^2+(av-C.PRIOR_LAB(d,2)).^2+(bv-C.PRIOR_LAB(d,3)).^2);
    T.(sprintf('D%d_L_p50',d))=prctile(Lv,50);
    T.(sprintf('D%d_a_p50',d))=prctile(av,50);
    T.(sprintf('D%d_b_p50',d))=prctile(bv,50);
    T.(sprintf('D%d_dE_p50',d))=prctile(dE,50);
    T.(sprintf('D%d_L_ipr',d))=prctile(Lv,95)-prctile(Lv,5);
    T.(sprintf('D%d_b_ipr',d))=prctile(bv,95)-prctile(bv,5);
end
end

function T=u28_grade_csv(gStat)
if isempty(gStat)||~istable(gStat), T=table(); return; end
T=gStat;
end

function T=u28_trans_table(TM)
T=array2table(TM,'VariableNames',{'to_D1','to_D2','to_D3','to_D4','to_D5'},...
    'RowNames',{'from_D1','from_D2','from_D3','from_D4','from_D5'});
end

%% =========================================================================
%%  PNG MAP 저장 (35종)
function u28_save_maps(RGB_raw,RGB_srgb,...
    Hc,Sc,Vc,Lc,ac,bc,...
    dE,dEn,WHI,SVI,...
    WI_rule,WI_gmm,WI_hybrid,...
    D_rule,D_gmm,D,WD10,...
    tieMap,confGMM,confHybrid,entMap,...
    roiInfo,tex,...
    tileMedL,tileIqrL,tileMedb,tileIqrb,...
    edgeDen,locVar,tileWI,tileCnf,...
    tGradL,tEntL,tLoG,...
    mask,iPng,OPT,C)

sm = @(X,ttl,fn,cl,cm) u28_smap(X,ttl,fullfile(iPng,fn),cl,cm);
so = @(X,ttl,fn,cl,cm) u28_soverlay(RGB_raw,X,mask,fullfile(iPng,fn),cl,ttl,cm);

Rraw=RGB_raw(:,:,1); Grow=RGB_raw(:,:,2); Braw=RGB_raw(:,:,3);
Rs=RGB_srgb(:,:,1); Gs=RGB_srgb(:,:,2); Bs=RGB_srgb(:,:,3);

% ===== 공통 표시 범위: RGB끼리 동일 / a-b 동일 =====
rgbRawLim  = [0 255];
rgbSrgbLim = [0 1];

abVals = double([ac(mask); bc(mask)]);
abAbs  = max(abs(abVals),[],'omitnan');
if ~isfinite(abAbs) || abAbs<=0
    abAbs = 1;
end
abLim = [-abAbs abAbs];

% ── RGB ─────────────────────────────────────────────────────────
sm(single(Rraw),'R raw (uint8)','MAP_R_raw.png',rgbRawLim,parula(256));
sm(single(Grow),'G raw (uint8)','MAP_G_raw.png',rgbRawLim,parula(256));
sm(single(Braw),'B raw (uint8)','MAP_B_raw.png',rgbRawLim,parula(256));
sm(single(Rs),'R sRGB','MAP_R_srgb.png',rgbSrgbLim,parula(256));
sm(single(Gs),'G sRGB','MAP_G_srgb.png',rgbSrgbLim,parula(256));
sm(single(Bs),'B sRGB','MAP_B_srgb.png',rgbSrgbLim,parula(256));

% ── HSV/LAB ──────────────────────────────────────────────────────
sm(single(Hc),'H (0-1)','MAP_H.png',[0 1],hsv(256));
sm(single(Sc),'S (0-1)','MAP_S.png',[0 1],turbo(256));
sm(single(Vc),'V (0-1)','MAP_V.png',[0 1],gray(256));
sm(single(Lc),'L*','MAP_L.png',[],turbo(256));
sm(single(ac),'a*','MAP_a.png',abLim,turbo(256));
sm(single(bc),'b*','MAP_b.png',abLim,turbo(256));

% ── 인덱스 ───────────────────────────────────────────────────────
sm(single(dE), 'ΔE (ab→median)','MAP_dE.png',[],hot(256));
sm(single(dEn),'ΔE norm','MAP_dEn.png',[0 1],hot(256));
sm(single(WHI),'WHI (0-1)','MAP_WHI.png',[0 1],turbo(256));
sm(single(SVI),'SVI (0-1)','MAP_SVI.png',[0 1],turbo(256));

wCM = u28_wi_cmap(C);
gCM = u28_grade_cmap(C);
g10 = u28_grade10_cmap(C);

sm(single(WI_rule),'WI rule (1-5)','MAP_WI_rule.png',[1 5],wCM);
sm(single(WI_gmm), 'WI ALP-GMM (1-5)','MAP_WI_gmm.png',[1 5],wCM);
sm(single(WI_hybrid),'WI hybrid (1-5)','MAP_WI_hybrid.png',[1 5],wCM);
sm(single(D_rule),'D-grade rule','MAP_Dgrade_rule.png',[1 5],gCM);
sm(single(D_gmm),'D-grade ALP-GMM','MAP_Dgrade_gmm.png',[1 5],gCM);
sm(single(D),'D-grade hybrid','MAP_Dgrade_hybrid.png',[1 5],gCM);
sm(single(WD10),'WD-10','MAP_WD10.png',[1 10],g10);
sm(single(tieMap),'Tie map','MAP_tie.png',[0 1],gray(256));
sm(single(confGMM),'Conf GMM','MAP_confGMM.png',[0 1],parula(256));
sm(single(confHybrid),'Conf hybrid','MAP_confHybrid.png',[0 1],parula(256));
sm(single(entMap),'Entropy','MAP_entropy.png',[0 1],turbo(256));
sm(single(roiInfo.darkMask),'Dark excluded','MAP_dark_excluded.png',[0 1],gray(256));

% ── 오버레이 ─────────────────────────────────────────────────────
so(single(WI_hybrid),'WI hybrid overlay','OVERLAY_WI_hybrid.png',[1 5],wCM);
so(single(D),        'D-grade overlay',  'OVERLAY_Dgrade.png',   [1 5],gCM);
so(single(confHybrid),'Confidence overlay','OVERLAY_conf.png',   [0 1],parula(256));
so(single(entMap),   'Entropy overlay',  'OVERLAY_entropy.png',  [0 1],turbo(256));

% ── 타일 ─────────────────────────────────────────────────────────
sm(tileMedL,'Tile med L*','TILE_medL.png',[],turbo(256));
sm(tileIqrL,'Tile IPR L*','TILE_iprL.png',[],turbo(256));
sm(tileMedb,'Tile med b*','TILE_medb.png',[],turbo(256));
sm(tileIqrb,'Tile IPR b*','TILE_iprb.png',[],turbo(256));
sm(edgeDen, 'Tile edge density','TILE_edge.png',[],turbo(256));
sm(locVar,  'Tile local var',   'TILE_var.png',[],turbo(256));
sm(tileWI,  'Tile WI mean',     'TILE_WI.png',[],wCM);
sm(tileCnf, 'Tile conf mean',   'TILE_conf.png',[],parula(256));

if OPT.SAVE_TEXTURE_MAPS
    sm(tex.gradL,  'grad|L*|','MAP_gradL.png',[],hot(256));
    sm(tex.gradb,  'grad|b*|','MAP_gradb.png',[],hot(256));
    sm(tex.graddEn,'grad|ΔEn|','MAP_graddEn.png',[],hot(256));
    sm(tex.entL,   'entropy(L*)','MAP_entL.png',[0 1],parula(256));
    sm(tex.logL,   'LoG(L*)','MAP_LoG.png',[0 1],parula(256));
    sm(tGradL,'Tile gradL','TILE_gradL.png',[],parula(256));
    sm(tEntL, 'Tile entL', 'TILE_entL.png',[],parula(256));
    sm(tLoG,  'Tile LoG',  'TILE_LoG.png', [],parula(256));
end
end

function u28_smap(X,ttl,fpath,cl,cmap)
try
    fig=figure('Visible','off','Position',[10 10 1600 480],'Color','w');
    imagesc(X); axis image off; title(ttl,'Interpreter','none'); colorbar;
    if ~isempty(cl), clim(cl); end
    if ~isempty(cmap), colormap(gca,cmap); end
    exportgraphics(fig,fpath,'Resolution',260,'BackgroundColor','white');
    close(fig);
catch mexc28
    warning('U28:smap','%s',mexc28.message);
end
end

function u28_soverlay(RGB,X,mask,fpath,cl,ttl,cmap)
try
    fig=figure('Visible','off','Position',[10 10 1800 520],'Color','w');
    imshow(RGB); hold on;
    h=imagesc(X); set(h,'AlphaData',0.4*double(mask));
    if ~isempty(cl), clim(cl); end
    if ~isempty(cmap), colormap(gca,cmap); end
    colorbar; title(ttl,'Interpreter','none');
    exportgraphics(fig,fpath,'Resolution',260,'BackgroundColor','white');
    close(fig);
catch mexc28
    warning('U28:overlay','%s',mexc28.message);
end
end

%% =========================================================================
%%  논문급 그림 16종
%% =========================================================================
function u28_all_figures(I0,RGB_raw,D,D_rule,D_gmm,WD10,...
    post5,postRule,postGMM,...
    ~,WI_rule,WI_gmm,wiMapC,...
    confHybrid,entMap,...
    Lc,ac,bc,~,~,...
    f5,f10,gStat,transM,heatGrid,...
    ~,mask,OPT,C,iFig)

dpi=OPT.FIG_DPI;
sv=@(fig,fn) u28_savefig(fig,iFig,fn,dpi);

sv(u28_fig01_maps(I0,D,D_rule,D_gmm,WD10,C),              'FIG01_grade_maps');
sv(u28_fig02_fracs(f5,f10,C),                              'FIG02_fractions');
sv(u28_fig03_entropy(entMap,mask),                         'FIG03_entropy');
sv(u28_fig04_posteriors(post5,D,mask,size(Lc),C),          'FIG04_posteriors');
sv(u28_fig05_cielab_compare(gStat,C),                      'FIG05_cielab_compare');
sv(u28_fig06_ab_scatter(ac,bc,D,gStat,mask,C),             'FIG06_ab_scatter');
sv(u28_fig07_Lb_scatter(Lc,bc,D,gStat,mask,C),             'FIG07_Lb_scatter');
sv(u28_fig08_wi_profile(wiMapC,WI_rule,WI_gmm,D,mask,C),   'FIG08_wi_profile');
sv(u28_fig09_importance(Lc,ac,bc,mask,C,OPT),              'FIG09_importance');
sv(u28_fig10_boxplots(D,Lc,ac,bc,mask,C),                  'FIG10_boxplots');
sv(u28_fig11_heatmap(heatGrid,C),                          'FIG11_heatmap');
sv(u28_fig12_histograms(D,bc,mask,C),                      'FIG12_histograms');
sv(u28_fig13_confidence(confHybrid,entMap,D,mask,C),       'FIG13_confidence');
sv(u28_fig14_transition(transM,C),                         'FIG14_transition');
sv(u28_fig15_alp_diag(postGMM,postRule,post5,D,mask,C),    'FIG15_alp_diagnostics');
sv(u28_fig16_dashboard(I0,RGB_raw,D,WD10,f5,f10,...
    wiMapC,entMap,confHybrid,gStat,mask,OPT,C),            'FIG16_dashboard');
end


%──────────────────────────────────────────────────────────────────
function fig=u28_fig01_maps(I0,D,D_rule,D_gmm,~,C)
fig=figure('Visible','off','Position',[10 10 3200 800],'Color','w');
tl=tiledlayout(1,4,'Padding','tight','TileSpacing','compact');
title(tl,'풍화등급 분류 결과 (Rule / ALP-GMM / Hybrid / WD-10)',...
    'FontSize',14,'FontWeight','bold');
nexttile; imshow(I0); title('원본','FontSize',12,'FontWeight','bold');

ttls={'WD-5 Rule','WD-5 ALP-GMM','WD-5 Hybrid (→논문)'};
maps={D_rule,D_gmm,D};
for mi=1:3
    ax=nexttile; imshow(u28_l2rgb(maps{mi},C.D5,5));
    title(ttls{mi},'FontSize',12,'FontWeight','bold');
    ps=gobjects(5,1);
    for i=1:5, ps(i)=patch(ax,nan,nan,C.D5(i,:),'EdgeColor','none'); end
    legend(ax,ps,{'D1 신선암','D2 약풍화','D3 중풍화','D4 강풍화','D5 완전풍화'},...
        'Location','eastoutside','FontSize',9);
end
end

%──────────────────────────────────────────────────────────────────
function fig=u28_fig02_fracs(f5,f10,C)
fig=figure('Visible','off','Position',[10 10 2600 820],'Color','w');
tl=tiledlayout(1,3,'Padding','tight','TileSpacing','compact');
title(tl,'풍화등급 면적 분율','FontSize',14,'FontWeight','bold');

ax1=nexttile; bh=bar(ax1,1:5,f5*100,'FaceColor','flat','EdgeColor','k','LineWidth',0.9);
for i=1:5, bh.CData(i,:)=C.D5(i,:); end
for i=1:5
    if f5(i)>0.002
        text(i,f5(i)*100+0.5,sprintf('%.2f%%',f5(i)*100),...
            'HorizontalAlignment','center','FontSize',10,'FontWeight','bold','Parent',ax1);
    end
end
xticks(ax1,1:5); xticklabels(ax1,{'D1','D2','D3','D4','D5'});
ylabel(ax1,'면적 (%)','FontSize',11); title(ax1,'WD-5','FontSize',12,'FontWeight','bold');
ylim(ax1,[0 max(f5)*100*1.25+3]); grid(ax1,'on');

ax2=nexttile; bh2=bar(ax2,1:10,f10*100,'FaceColor','flat','EdgeColor','k','LineWidth',0.9);
D10cols={'D1a','D1b','D2a','D2b','D3a','D3b','D4a','D4b','D5a','D5b'};
for i=1:10, bh2.CData(i,:)=C.D10(i,:); end
xticks(ax2,1:10); xticklabels(ax2,D10cols); xtickangle(ax2,45);
ylabel(ax2,'면적 (%)','FontSize',11); title(ax2,'WD-10','FontSize',12,'FontWeight','bold');
grid(ax2,'on');

ax3=nexttile;
valid=f5>0.001;
p=pie(ax3,f5(valid));
cnt=0;
for i=1:5
    if ~valid(i), continue; end
    cnt=cnt+1;
    p(2*cnt-1).FaceColor=C.D5(i,:); p(2*cnt-1).EdgeColor='w'; p(2*cnt-1).LineWidth=1.5;
    p(2*cnt).FontSize=10; p(2*cnt).FontWeight='bold';
end
lstr={'D1 신선암','D2 약풍화','D3 중풍화','D4 강풍화','D5 완전풍화'};
legend(ax3,lstr(valid),'Location','southoutside','FontSize',9);
title(ax3,'파이차트','FontSize',12,'FontWeight','bold');
end

%──────────────────────────────────────────────────────────────────
function fig=u28_fig03_entropy(entMap,mask)
em=mean(entMap(mask),'omitnan'); ev=entMap(mask); ev=ev(isfinite(ev));
fig=figure('Visible','off','Position',[10 10 1900 820],'Color','w');
tl=tiledlayout(1,2,'Padding','tight','TileSpacing','compact');
title(tl,sprintf('GMM 분류 불확실도 — 엔트로피 (μ=%.4f)',em),'FontSize',13,'FontWeight','bold');
ax1h=nexttile; imagesc(ax1h,entMap); axis(ax1h,'image'); axis(ax1h,'off');
colorbar(ax1h); clim(ax1h,[0 1]); colormap(ax1h,turbo(256));
title(ax1h,'픽셀 엔트로피 H∈[0,1]','FontSize',12,'FontWeight','bold');
ax=nexttile;
histogram(ax,ev,60,'Normalization','probability','FaceColor',[0.3 0.5 0.9],'EdgeColor','none','FaceAlpha',0.85);
hold(ax,'on');
xline(ax,em,'r-','LineWidth',2.5,'Label',sprintf('μ=%.3f',em),'LabelOrientation','horizontal','FontSize',10);
xline(ax,median(ev,'omitnan'),'b--','LineWidth',2,...
    'Label',sprintf('Med=%.3f',median(ev,'omitnan')),'LabelOrientation','horizontal','FontSize',10);
xlabel(ax,'엔트로피 H','FontSize',12); ylabel(ax,'상대 빈도','FontSize',12);
title(ax,'엔트로피 분포','FontSize',12,'FontWeight','bold'); grid(ax,'on'); box(ax,'on');
end

%──────────────────────────────────────────────────────────────────
function fig=u28_fig04_posteriors(post5,~,mask,sz,C)
fig=figure('Visible','off','Position',[10 10 2800 640],'Color','w');
tl=tiledlayout(1,5,'Padding','tight','TileSpacing','compact');
title(tl,'GMM Posterior 확률 맵 P(D_i | 픽셀)','FontSize',13,'FontWeight','bold');
for d=1:5
    Pd=zeros(sz,'single'); Pd(mask)=post5(:,d);
    ax=nexttile;
    imagesc(ax,Pd); axis(ax,'image'); axis(ax,'off');
    colorbar(ax); clim(ax,[0 1]);
    cma=zeros(256,3);
    for ci=1:256
        cma(ci,:)=C.D5(d,:)*(ci/256);
    end
    colormap(ax,cma);
    title(ax,sprintf('P(D%d)  μ=%.3f',d,mean(Pd(mask),'omitnan')),...
        'FontSize',11,'FontWeight','bold','Color',C.D5(d,:)*0.75);
end
end

%──────────────────────────────────────────────────────────────────
function fig=u28_fig05_cielab_compare(gStat,C)
fig=figure('Visible','off','Position',[10 10 2400 820],'Color','w');
tl=tiledlayout(1,3,'Padding','tight','TileSpacing','compact');
title(tl,'CIELAB 기준표 vs 검출값 비교','FontSize',14,'FontWeight','bold');
chs={'L*','a*','b*'};
ranges={C.PRIOR_L_R,C.PRIOR_a_R,C.PRIOR_b_R};
ylims={[18 70],[-8 10],[-5 20]};
sfx={'L_med','a_med','b_med'};
p05x={'L_p05','a_p05','b_p05'}; p95x={'L_p95','a_p95','b_p95'};
for ch=1:3
    ax=nexttile; hold on; box on;
    R=ranges{ch};
    for d=1:5
        fill(ax,[d-0.38 d+0.38 d+0.38 d-0.38],[R(d,1) R(d,1) R(d,2) R(d,2)],...
            C.D5(d,:),'FaceAlpha',0.15,'EdgeColor',C.D5(d,:),'LineWidth',1.2);
    end
    det=nan(1,5); lo=nan(1,5); hi=nan(1,5);
    for d=1:5
        nm=sprintf('D%d',d);
        fn=gStat.Properties.VariableNames;
        if ismember([nm,'_',sfx{ch}],fn)
            det(d)=gStat.([nm,'_',sfx{ch}])(1);
            lo(d) =gStat.([nm,'_',p05x{ch}])(1);
            hi(d) =gStat.([nm,'_',p95x{ch}])(1);
        end
    end
    for d=1:5
        if isfinite(det(d))
            errorbar(ax,d,det(d),det(d)-lo(d),hi(d)-det(d),'o','LineWidth',2,...
                'Color',C.D5(d,:),'MarkerFaceColor',C.D5(d,:),'MarkerSize',9,'CapSize',7);
        end
    end
    plot(ax,1:5,C.PRIOR_LAB(:,ch),'k--^','LineWidth',1.8,'MarkerFaceColor','k',...
        'MarkerSize',7,'DisplayName','기준 중앙');
    xticks(ax,1:5); xticklabels(ax,{'D1','D2','D3','D4','D5'});
    ylabel(ax,chs{ch},'FontSize',13,'FontWeight','bold');
    ylim(ax,ylims{ch}); grid(ax,'on'); ax.GridAlpha=0.25;
    title(ax,[chs{ch},' 비교'],'FontSize',12,'FontWeight','bold');
    if ch==1
        legend(ax,'Location','best','FontSize',8.5);
    end
end
end

%──────────────────────────────────────────────────────────────────
function fig=u28_fig06_ab_scatter(ac,bc,D,gStat,~,C)
fig=figure('Visible','off','Position',[10 10 1100 1000],'Color','w');
hold on; box on;
theta=linspace(0,2*pi,150);
for d=1:5
    ar=(C.PRIOR_a_R(d,2)-C.PRIOR_a_R(d,1))/2;
    br=(C.PRIOR_b_R(d,2)-C.PRIOR_b_R(d,1))/2;
    fill(C.PRIOR_LAB(d,2)+ar*cos(theta),C.PRIOR_LAB(d,3)+br*sin(theta),...
        C.D5(d,:),'FaceAlpha',0.10,'EdgeColor',C.D5(d,:),'LineWidth',1.8,'LineStyle','--');
end
av_a=double(ac(:)); bv_a=double(bc(:)); lb=double(reshape(D,[],1));
idx=randperm(numel(av_a),min(40000,numel(av_a)));
for d=1:5
    sel=idx(lb(idx)==d);
    if isempty(sel), continue; end
    scatter(av_a(sel),bv_a(sel),5,C.D5(d,:),'filled','MarkerFaceAlpha',0.35,...
        'DisplayName',sprintf('D%d',d));
end
for d=1:5
    plot(C.PRIOR_LAB(d,2),C.PRIOR_LAB(d,3),'k+','MarkerSize',18,'LineWidth',3);
    nm=sprintf('D%d',d); fn=gStat.Properties.VariableNames;
    if ismember([nm,'_a_med'],fn)&&isfinite(gStat.([nm,'_a_med'])(1))
        plot(gStat.([nm,'_a_med'])(1),gStat.([nm,'_b_med'])(1),'w^',...
            'MarkerSize',11,'LineWidth',2,'MarkerFaceColor',C.D5(d,:),'MarkerEdgeColor','k');
    end
end
grid on; set(gca,'GridAlpha',0.25);
xlabel('a* (적색도)','FontSize',13,'FontWeight','bold');
ylabel('b* (황색도 — 풍화 주지표)','FontSize',13,'FontWeight','bold');
xlim([-9 12]); ylim([-5 20]);
title({'a*-b* 색공간 산점도';'+ 기준중앙  ▲ 검출중앙  점선 기준타원'},...
    'FontSize',12,'FontWeight','bold');
legend('Location','northwest','FontSize',9,'NumColumns',2);
end

%──────────────────────────────────────────────────────────────────
function fig=u28_fig07_Lb_scatter(Lc,bc,D,~,~,C)
fig=figure('Visible','off','Position',[10 10 1100 1000],'Color','w');
hold on; box on;
theta=linspace(0,2*pi,150);
for d=1:5
    Lr=(C.PRIOR_L_R(d,2)-C.PRIOR_L_R(d,1))/2;
    br=(C.PRIOR_b_R(d,2)-C.PRIOR_b_R(d,1))/2;
    fill(C.PRIOR_LAB(d,3)+br*cos(theta),C.PRIOR_LAB(d,1)+Lr*sin(theta),...
        C.D5(d,:),'FaceAlpha',0.12,'EdgeColor',C.D5(d,:),'LineWidth',1.8,'LineStyle','--');
end
Lv_a=double(Lc(:)); bv_a=double(bc(:)); lb=double(reshape(D,[],1));
idx=randperm(numel(Lv_a),min(40000,numel(Lv_a)));
for d=1:5
    sel=idx(lb(idx)==d);
    if isempty(sel), continue; end
    scatter(bv_a(sel),Lv_a(sel),5,C.D5(d,:),'filled','MarkerFaceAlpha',0.35,...
        'DisplayName',sprintf('D%d',d));
end
for d=1:5
    plot(C.PRIOR_LAB(d,3),C.PRIOR_LAB(d,1),'k+','MarkerSize',18,'LineWidth',3);
end
grid on; set(gca,'GridAlpha',0.25);
xlabel('b* (황색도)','FontSize',13,'FontWeight','bold');
ylabel('L* (명도)','FontSize',13,'FontWeight','bold');
xlim([-5 20]); ylim([18 72]);
title({'L*-b* 색공간 산점도';'+ 기준중앙'},'FontSize',12,'FontWeight','bold');
legend('Location','northeast','FontSize',9);
end

%──────────────────────────────────────────────────────────────────
function fig=u28_fig08_wi_profile(wiMapC,WI_rule,WI_gmm,D,mask,C)
wCM=u28_wi_cmap(C);
fig=figure('Visible','off','Position',[10 10 2600 1000],'Color','w');
tl=tiledlayout(2,3,'Padding','tight','TileSpacing','compact');
title(tl,'풍화지수 WI 공간분포 & 프로파일','FontSize',14,'FontWeight','bold');

nexttile([2 1]);
imagesc(wiMapC); axis image off; clim([1 5]); colorbar;
colormap(gca,wCM);
title('WI Continuous Map','FontSize',12,'FontWeight','bold');

rowWI=zeros(size(wiMapC,1),1);
for r=1:size(wiMapC,1)
    rv=wiMapC(r,mask(r,:)); if ~isempty(rv), rowWI(r)=mean(rv,'omitnan'); end
end
axRow=nexttile;
plot(axRow,rowWI,1:size(wiMapC,1),'Color',[0.2 0.4 0.8],'LineWidth',1.8);
hold(axRow,'on');
for d=1:5, xline(axRow,d,'--','Color',C.D5(d,:),'LineWidth',1,'Alpha',0.6); end
xlabel(axRow,'평균 WI','FontSize',11); ylabel(axRow,'행(깊이)','FontSize',11);
title(axRow,'행방향 WI 프로파일','FontSize',12,'FontWeight','bold');
set(axRow,'YDir','reverse'); xlim(axRow,[1 5]); grid(axRow,'on'); box(axRow,'on');

colWI=zeros(1,size(wiMapC,2));
for c=1:size(wiMapC,2)
    cv=wiMapC(mask(:,c),c); if ~isempty(cv), colWI(c)=mean(cv,'omitnan'); end
end
axCol=nexttile;
plot(axCol,1:numel(colWI),colWI,'Color',[0.8 0.3 0.1],'LineWidth',1.8);
hold(axCol,'on');
for d=1:5, yline(axCol,d,'--','Color',C.D5(d,:),'LineWidth',1,'Alpha',0.6); end
xlabel(axCol,'열(수평)','FontSize',11); ylabel(axCol,'평균 WI','FontSize',11);
title(axCol,'열방향 WI 프로파일','FontSize',12,'FontWeight','bold');
ylim(axCol,[1 5]); grid(axCol,'on'); box(axCol,'on');

axHist=nexttile;
wiv=wiMapC(mask); wiv=wiv(isfinite(wiv));
histogram(axHist,wiv,80,'Normalization','pdf','FaceColor',[0.5 0.3 0.8],...
    'EdgeColor','none','FaceAlpha',0.85);
hold(axHist,'on');
xline(axHist,mean(wiv,'omitnan'),'r-','LineWidth',2.5,...
    'Label',sprintf('μ=%.2f',mean(wiv,'omitnan')),'LabelOrientation','horizontal','FontSize',9);
xline(axHist,median(wiv,'omitnan'),'b--','LineWidth',2,...
    'Label',sprintf('Med=%.2f',median(wiv,'omitnan')),'LabelOrientation','horizontal','FontSize',9);
for d=1:5, xline(axHist,d,'--','Color',C.D5(d,:),'LineWidth',1,'Alpha',0.7); end
xlabel(axHist,'WI','FontSize',12); ylabel(axHist,'밀도','FontSize',12);
title(axHist,'WI 분포','FontSize',12,'FontWeight','bold'); grid(axHist,'on'); box(axHist,'on');

ax=nexttile;
scatter(double(WI_rule(mask)),double(WI_gmm(mask)),3,...
    double(D(mask)),'filled','MarkerFaceAlpha',0.3);
colormap(ax,C.D5); clim([1 5]);
hold on; plot([1 5],[1 5],'k--','LineWidth',1.5);
xlabel('WI Rule','FontSize',11); ylabel('WI ALP-GMM','FontSize',11);
title('Rule vs ALP-GMM 비교','FontSize',12,'FontWeight','bold');
grid on; box on; axis equal; xlim([1 5]); ylim([1 5]);
end

%──────────────────────────────────────────────────────────────────
function fig=u28_fig09_importance(Lc,ac,bc,mask,C,OPT)
cmap_imp=u28_importance_map(Lc,ac,bc,mask,C);
b_w=[2.5;7.0;8.0;8.5;9.0]; imp=b_w/sum(b_w);
fig=figure('Visible','off','Position',[10 10 1900 820],'Color','w');
tl=tiledlayout(1,2,'Padding','tight','TileSpacing','compact');
title(tl,'복합 중요도 특징 (b* 범위폭 기반)','FontSize',13,'FontWeight','bold');
nexttile;
imagesc(cmap_imp); axis image off; colorbar; clim([0 1]);
colormap(gca,flipud(hot(256)));
title('중요도 스코어 맵 (밝을수록 분류 경계 인접)','FontSize',11,'FontWeight','bold');
ax=nexttile;
bh=bar(ax,1:5,imp*100,'FaceColor','flat','EdgeColor','k','LineWidth',1.0);
for i=1:5, bh.CData(i,:)=C.D5(i,:); end
for i=1:5
    text(i,imp(i)*100+0.3,sprintf('%.1f%%',imp(i)*100),...
        'HorizontalAlignment','center','FontSize',11,'FontWeight','bold');
end
xticks(ax,1:5); xticklabels(ax,{'D1','D2','D3','D4','D5'});
ylabel(ax,'중요도 (%)','FontSize',12);
title(ax,'b* 범위폭 기반 등급 중요도','FontSize',12,'FontWeight','bold');
ylim(ax,[0 32]); grid(ax,'on'); ax.GridAlpha=0.3;
annotation('textbox',[0.55 0.08 0.42 0.22],'String',...
    {sprintf('W_b(황색도) = %.2f ← 최대',OPT.W_b),...
     sprintf('W_a(적색도) = %.2f',OPT.W_a),...
     sprintf('W_L(명도)   = %.2f',OPT.W_L),...
     '','D1→2.5 / D2→7.0 / D3→8.0 / D4→8.5 / D5→9.0'},...
    'FontSize',9.5,'BackgroundColor',[0.97 0.97 0.97],...
    'EdgeColor',[0.7 0.7 0.7]);
end

%──────────────────────────────────────────────────────────────────
function fig=u28_fig10_boxplots(D,Lc,ac,bc,mask,C)
fig=figure('Visible','off','Position',[10 10 2400 820],'Color','w');
tl=tiledlayout(1,3,'Padding','tight','TileSpacing','compact');
title(tl,'등급별 CIELAB 박스플롯 (기준표 범위 포함)','FontSize',14,'FontWeight','bold');
chs={'L*','a*','b*'}; chmaps={Lc,ac,bc};
ranges={C.PRIOR_L_R,C.PRIOR_a_R,C.PRIOR_b_R};
ylims={[15 72],[-8 10],[-5 20]};
PRIOR_mid=C.PRIOR_LAB;
for ch=1:3
    ax=nexttile; hold(ax,'on'); box(ax,'on');
    R=ranges{ch};
    for d=1:5
        fill(ax,[d-0.44 d+0.44 d+0.44 d-0.44],[R(d,1) R(d,1) R(d,2) R(d,2)],...
            C.D5(d,:),'FaceAlpha',0.15,'EdgeColor',C.D5(d,:),'LineWidth',1.2,'LineStyle','--');
        plot(ax,[d-0.44 d+0.44],[PRIOR_mid(d,ch) PRIOR_mid(d,ch)],...
            '--','Color',C.D5(d,:)*0.65,'LineWidth',1.8);
    end
    allD=[]; allL=[];
    for d=1:5
        dm=mask&(D==d); vv=double(chmaps{ch}(dm)); vv=vv(isfinite(vv));
        allD=[allD; repmat(d,numel(vv),1)]; %#ok<AGROW>
        allL=[allL; vv]; %#ok<AGROW>
    end
    if ~isempty(allD)
        bp=boxplot(ax,allL,allD,'Labels',{'D1','D2','D3','D4','D5'},...
            'OutlierSize',2,'Symbol','.','Colors',reshape(C.D5',3,5)','Widths',0.6);
        set(bp,'LineWidth',1.5);
    end
    ylabel(ax,chs{ch},'FontSize',13,'FontWeight','bold');
    title(ax,[chs{ch},' 분포 (음영=기준표)'],'FontSize',12,'FontWeight','bold');
    grid(ax,'on'); ax.GridAlpha=0.25; ylim(ax,ylims{ch});
end
end

%──────────────────────────────────────────────────────────────────
function fig=u28_fig11_heatmap(G,C)
wCM=u28_wi_cmap(C);
fig=figure('Visible','off','Position',[10 10 1400 920],'Color','w');
tl=tiledlayout(1,1,'Padding','tight');
title(tl,sprintf('공간 풍화 히트맵 (%d×%d 격자)',size(G,1),size(G,2)),...
    'FontSize',14,'FontWeight','bold');
ax=nexttile; imagesc(ax,G); axis(ax,'image');
colormap(ax,wCM); clim(ax,[1 5]);
cb=colorbar(ax); cb.Label.String='평균 WI (1=신선→5=완전풍화)';
cb.Ticks=1:5; cb.TickLabels={'D1','D2','D3','D4','D5'};
[nr,nc]=size(G);
for rr=1:nr
    for cc=1:nc
        if isfinite(G(rr,cc))
            tc=(G(rr,cc)>3)*[1 1 1]+(G(rr,cc)<=3)*[0 0 0];
            text(ax,cc,rr,sprintf('%.1f',G(rr,cc)),'HorizontalAlignment','center',...
                'FontSize',max(5,floor(120/max(nr,nc))),'Color',tc,'FontWeight','bold');
        end
    end
end
end

%──────────────────────────────────────────────────────────────────
function fig=u28_fig12_histograms(D,bc,mask,C)
fig=figure('Visible','off','Position',[10 10 2600 720],'Color','w');
tl=tiledlayout(1,5,'Padding','tight','TileSpacing','compact');
title(tl,'등급별 b* 분포 히스토그램 (기준표 범위)','FontSize',14,'FontWeight','bold');
for d=1:5
    ax=nexttile; hold(ax,'on'); box(ax,'on');
    dm=mask&(D==d); bv=double(bc(dm)); bv=bv(isfinite(bv));
    fill(ax,[C.PRIOR_b_R(d,1) C.PRIOR_b_R(d,2) C.PRIOR_b_R(d,2) C.PRIOR_b_R(d,1)],...
        [0 0 1 1],C.D5(d,:),'FaceAlpha',0.14,'EdgeColor',C.D5(d,:),'LineWidth',1.5,'LineStyle','--');
    if ~isempty(bv)
        histogram(ax,bv,50,'Normalization','probability',...
            'FaceColor',C.D5(d,:),'EdgeColor','w','FaceAlpha',0.88,'LineWidth',0.5);
        xline(ax,C.PRIOR_LAB(d,3),'k--','LineWidth',2,...
            'Label','기준중앙','LabelOrientation','horizontal','FontSize',8);
        xline(ax,median(bv,'omitnan'),'-','Color',C.D5(d,:)*0.6,'LineWidth',2,...
            'Label',sprintf('검출=%.2f',median(bv,'omitnan')),'LabelOrientation','horizontal','FontSize',8);
    end
    xlabel(ax,'b*','FontSize',11); ylabel(ax,'빈도','FontSize',11);
    title(ax,{sprintf('D%d (N=%d)',d,numel(bv));...
        sprintf('[%.1f, %.1f]',C.PRIOR_b_R(d,1),C.PRIOR_b_R(d,2))},...
        'FontSize',11,'FontWeight','bold','Color',C.D5(d,:)*0.7);
    grid(ax,'on'); ax.GridAlpha=0.25;
end
end

%──────────────────────────────────────────────────────────────────
function fig=u28_fig13_confidence(confHybrid,entMap,D,mask,C)
fig=figure('Visible','off','Position',[10 10 2200 860],'Color','w');
tl=tiledlayout(1,3,'Padding','tight','TileSpacing','compact');
title(tl,'분류 신뢰도 & 엔트로피 종합 분석','FontSize',13,'FontWeight','bold');
nexttile; imagesc(confHybrid); axis image off; colorbar; clim([0 1]);
colormap(gca,parula(256)); title('신뢰도 맵 (P1−P2)','FontSize',12,'FontWeight','bold');
nexttile; imagesc(entMap); axis image off; colorbar; clim([0 1]);
colormap(gca,turbo(256)); title('엔트로피 맵 H','FontSize',12,'FontWeight','bold');
axS=nexttile; hold(axS,'on');
for d=1:5
    dm=mask&(D==d);
    cv=double(confHybrid(dm)); cv=cv(isfinite(cv));
    if isempty(cv), continue; end
    scatter(axS,repmat(d,min(5000,numel(cv)),1),cv(randperm(numel(cv),min(5000,numel(cv)))),...
        8,C.D5(d,:),'filled','MarkerFaceAlpha',0.25);
    plot(axS,d,median(cv,'omitnan'),'w^','MarkerSize',10,'LineWidth',2,...
        'MarkerFaceColor',C.D5(d,:),'MarkerEdgeColor','k');
end
xticks(axS,1:5); xticklabels(axS,{'D1','D2','D3','D4','D5'});
ylabel(axS,'신뢰도 (P1-P2)','FontSize',11); xlabel(axS,'등급','FontSize',11);
title(axS,'등급별 신뢰도 분포','FontSize',12,'FontWeight','bold');
ylim(axS,[0 1]); grid(axS,'on'); box(axS,'on');
end

%──────────────────────────────────────────────────────────────────
function fig=u28_fig14_transition(transM,~)
fig=figure('Visible','off','Position',[10 10 1200 1000],'Color','w');
tl=tiledlayout(1,1,'Padding','tight');
title(tl,'공간 풍화 전이 확률 매트릭스 (수평 인접)','FontSize',14,'FontWeight','bold');
ax=nexttile; imagesc(ax,transM); clim(ax,[0 1]);
colormap(ax,hot(256)); colorbar(ax); axis(ax,'square'); box(ax,'on');
xticks(ax,1:5); yticks(ax,1:5);
xticklabels(ax,{'D1','D2','D3','D4','D5'}); yticklabels(ax,{'D1','D2','D3','D4','D5'});
xlabel(ax,'다음 등급','FontSize',12,'FontWeight','bold');
ylabel(ax,'현재 등급','FontSize',12,'FontWeight','bold');
for rr=1:5
    for cc=1:5
        fc=transM(rr,cc);
        tc=(fc>0.5)*[0 0 0]+(fc<=0.5)*[1 1 1];
        text(ax,cc,rr,sprintf('%.3f',fc),'HorizontalAlignment','center','FontSize',12,...
            'FontWeight','bold','Color',tc);
    end
end
end

%──────────────────────────────────────────────────────────────────
function fig=u28_fig15_alp_diag(postGMM,postRule,post5,D,mask,C)
fig=figure('Visible','off','Position',[10 10 2800 860],'Color','w');
tl=tiledlayout(1,4,'Padding','tight','TileSpacing','compact');
title(tl,'ALP-GMM 진단 — Rule/GMM/Hybrid Posterior 비교','FontSize',13,'FontWeight','bold');

lbls={'D1','D2','D3','D4','D5'};
datasets={'Rule','ALP-GMM','Hybrid'};
psets={postRule, postGMM, post5};
Dv=double(D(mask));

for pi=1:3
    ax=nexttile; hold on; box on;
    ps=psets{pi};
    if isempty(ps), title(ax,['없음: ' datasets{pi}]); continue; end
    for d=1:5
        sel=(Dv==d);
        if sum(sel)<5, continue; end
        histogram(ax,ps(sel,d),30,'Normalization','probability',...
            'FaceColor',C.D5(d,:),'EdgeColor','none','FaceAlpha',0.75,...
            'DisplayName',sprintf('D%d',d));
    end
    xlabel(ax,'사후확률 P(D_i)','FontSize',10);
    ylabel(ax,'빈도','FontSize',10);
    title(ax,[datasets{pi},' Posterior 분포'],'FontSize',11,'FontWeight','bold');
    legend(ax,'FontSize',8); grid(ax,'on'); ax.GridAlpha=0.25;
end

axB=nexttile; hold(axB,'on'); box(axB,'on');
x=1:5;
if ~isempty(postRule)
    f_rule=arrayfun(@(d)mean(Dv==d),1:5);
    bar(axB,x-0.27,f_rule*100,0.25,'FaceColor',[0.5 0.5 0.9],'EdgeColor','none',...
        'DisplayName','Rule');
end
if ~isempty(postGMM)
    [~,gmmlb]=max(postGMM,[],2);
    f_gmm=arrayfun(@(d)mean(gmmlb==d),1:5);
    bar(axB,x,f_gmm*100,0.25,'FaceColor',[0.3 0.8 0.4],'EdgeColor','none',...
        'DisplayName','ALP-GMM');
end
f_hyb=arrayfun(@(d)mean(Dv==d),1:5);
bar(axB,x+0.27,f_hyb*100,0.25,'FaceColor',[0.9 0.5 0.1],'EdgeColor','none',...
    'DisplayName','Hybrid');
xticks(axB,1:5); xticklabels(axB,lbls);
ylabel(axB,'면적 비율 (%)','FontSize',11); legend(axB,'FontSize',9);
title(axB,'Rule/GMM/Hybrid 등급 비율 비교','FontSize',11,'FontWeight','bold');
grid(axB,'on');
end

%──────────────────────────────────────────────────────────────────
function fig=u28_fig16_dashboard(I0,~,D,~,f5,f10,...
    wiMapC,entMap,confHybrid,gStat,mask,OPT,C)
wCM=u28_wi_cmap(C);
fig=figure('Visible','off','Position',[10 10 3400 2400],'Color','w');
tl=tiledlayout(5,4,'Padding','compact','TileSpacing','compact');
title(tl,{'화강암 코어 풍화등급 — SCI급 종합 대시보드 (ALP-GMM 국소희석 보완)';...
    'Granite Core Weathering Grade — Comprehensive Dashboard (Ultra2028)'},...
    'FontSize',16,'FontWeight','bold');

nexttile; imshow(I0);
title('원본','FontSize',11,'FontWeight','bold');

nexttile;
imshow(u28_l2rgb(D,C.D5,5));
ps=gobjects(5,1);
for i=1:5, ps(i)=patch(gca,nan,nan,C.D5(i,:),'EdgeColor','none'); end
legend(ps,{'D1','D2','D3','D4','D5'},'Location','eastoutside','FontSize',9);
title('WD-5 Hybrid','FontSize',11,'FontWeight','bold');

nexttile;
imagesc(wiMapC); axis image off; clim([1 5]); colorbar;
colormap(gca,wCM);
title(sprintf('WI 연속맵 (μ=%.2f)',mean(wiMapC(mask),'omitnan')),...
    'FontSize',11,'FontWeight','bold');

nexttile;
imagesc(entMap); axis image off; colorbar; clim([0 1]);
colormap(gca,turbo(256));
title(sprintf('엔트로피 (μ=%.3f)',mean(entMap(mask),'omitnan')),...
    'FontSize',11,'FontWeight','bold');

axWD=nexttile; bh=bar(axWD,1:5,f5*100,'FaceColor','flat','EdgeColor','k');
for i=1:5, bh.CData(i,:)=C.D5(i,:); end
for i=1:5
    if f5(i)>0.003
        text(axWD,i,f5(i)*100+0.5,sprintf('%.1f%%',f5(i)*100),...
            'HorizontalAlignment','center','FontSize',9,'FontWeight','bold');
    end
end
xticks(axWD,1:5); xticklabels(axWD,{'D1','D2','D3','D4','D5'});
ylabel(axWD,'%'); title(axWD,'WD5 분율','FontSize',11,'FontWeight','bold');
ylim(axWD,[0 max(f5)*100*1.3+2]); grid(axWD,'on');

nexttile;
valid=f5>0.001;
p=pie(gca,f5(valid));
cnt=0;
for i=1:5
    if ~valid(i), continue; end; cnt=cnt+1;
    p(2*cnt-1).FaceColor=C.D5(i,:); p(2*cnt-1).EdgeColor='w';
    p(2*cnt).FontSize=9; p(2*cnt).FontWeight='bold';
end
title('파이차트','FontSize',11,'FontWeight','bold');

ax=nexttile; bh2=bar(ax,1:10,f10*100,'FaceColor','flat','EdgeColor','k');
for i=1:10, bh2.CData(i,:)=C.D10(i,:); end
D10cols={'D1a','D1b','D2a','D2b','D3a','D3b','D4a','D4b','D5a','D5b'};
xticks(ax,1:10); xticklabels(ax,D10cols); xtickangle(ax,55);
ylabel(ax,'%'); title(ax,'WD10 분율','FontSize',11,'FontWeight','bold'); grid(ax,'on');

nexttile;
imagesc(confHybrid); axis image off; colorbar; clim([0 1]);
colormap(gca,parula(256));
title(sprintf('신뢰도 맵 (μ=%.3f)',mean(confHybrid(mask),'omitnan')),...
    'FontSize',11,'FontWeight','bold');

chs={'L*','a*','b*'};
PRIOR_mid=C.PRIOR_LAB; ranges={C.PRIOR_L_R,C.PRIOR_a_R,C.PRIOR_b_R};
ylims_db={[15 72],[-8 10],[-5 20]};
sfx={'L_med','a_med','b_med'};
for ch=1:3
    ax=nexttile; hold(ax,'on'); box(ax,'on');
    R=ranges{ch};
    for d=1:5
        fill(ax,[d-0.38 d+0.38 d+0.38 d-0.38],[R(d,1) R(d,1) R(d,2) R(d,2)],...
            C.D5(d,:),'FaceAlpha',0.15,'EdgeColor',C.D5(d,:),'LineWidth',1.0);
        nm=sprintf('D%d',d); fn=gStat.Properties.VariableNames;
        if ismember([nm,'_',sfx{ch}],fn)&&isfinite(gStat.([nm,'_',sfx{ch}])(1))
            plot(ax,d,gStat.([nm,'_',sfx{ch}])(1),'o',...
                'MarkerSize',8,'LineWidth',2,...
                'Color',C.D5(d,:),'MarkerFaceColor',C.D5(d,:));
        end
    end
    plot(ax,1:5,PRIOR_mid(:,ch),'k--^','LineWidth',1.4,'MarkerSize',6,'MarkerFaceColor','k');
    xticks(ax,1:5); xticklabels(ax,{'D1','D2','D3','D4','D5'});
    ylabel(ax,chs{ch},'FontSize',11); ylim(ax,ylims_db{ch});
    title(ax,[chs{ch},' 비교'],'FontSize',11,'FontWeight','bold');
    grid(ax,'on'); ax.GridAlpha=0.25;
end

axTxt=nexttile([1 4]); axis(axTxt,'off');
det_b=nan(1,5);
fnv=gStat.Properties.VariableNames;
for d=1:5
    nm=sprintf('D%d',d);
    if ismember([nm,'_b_med'],fnv)
        det_b(d)=gStat.([nm,'_b_med'])(1);
    end
end
[~,topD]=max(f5);
txt={...
    '━━━━━━━━━━━━━━━━━━━━━━━━  분석 결과 요약  ━━━━━━━━━━━━━━━━━━━━━━━━',...
    sprintf('주등급(Dominant): D%d   |   WI μ=%.3f   |   Entropy μ=%.4f',...
        topD,mean(wiMapC(mask),'omitnan'),mean(entMap(mask),'omitnan')),...
    sprintf('D1=%.2f%%  D2=%.2f%%  D3=%.2f%%  D4=%.2f%%  D5=%.2f%%',...
        f5*100),...
    sprintf('b* 검출중앙: D1=%.2f  D2=%.2f  D3=%.2f  D4=%.2f  D5=%.2f',...
        det_b),...
    sprintf('가중치: W_b=%.2f(황색도)>W_a=%.2f(적색도)>W_L=%.2f(명도)',...
        OPT.W_b,OPT.W_a,OPT.W_L),...
    '핵심: ALP-GMM(국소희석방지)+RRC(클립)+ROI-6레이어(흑색제외)+Rule+Hybrid',...
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'};
text(axTxt,0.01,0.85,txt,'FontSize',10,'VerticalAlignment','top','Units','normalized',...
    'BackgroundColor',[0.96 0.96 0.98],'EdgeColor',[0.7 0.7 0.7],'Margin',8);
end

%% =========================================================================
%%  컬러맵 유틸
%% =========================================================================
function cmap=u28_wi_cmap(C)
cmap=zeros(256,3); vals=linspace(1,5,256);
for i=1:256
    t=(vals(i)-1)/4;
    if t<0.25
        cmap(i,:)=C.D5(1,:)*(1-t/0.25)+C.D5(2,:)*(t/0.25);
    elseif t<0.5
        cmap(i,:)=C.D5(2,:)*(1-(t-0.25)/0.25)+C.D5(3,:)*((t-0.25)/0.25);
    elseif t<0.75
        cmap(i,:)=C.D5(3,:)*(1-(t-0.5)/0.25)+C.D5(4,:)*((t-0.5)/0.25);
    else
        cmap(i,:)=C.D5(4,:)*(1-(t-0.75)/0.25)+C.D5(5,:)*((t-0.75)/0.25);
    end
end
end

function cmap=u28_grade_cmap(C)
n=256; cmap=zeros(n,3); rng_=[1 5];
for i=1:n
    v=(i-1)/(n-1)*(rng_(2)-rng_(1))+rng_(1);
    d=min(max(round(v),1),5);
    cmap(i,:)=C.D5(d,:);
end
end

function cmap=u28_grade10_cmap(C)
n=256; cmap=zeros(n,3);
for i=1:n
    v=(i-1)/(n-1)*9+1;
    d=min(max(round(v),1),10);
    cmap(i,:)=C.D10(d,:);
end
end

function rgb=u28_l2rgb(labelMap,cmap,K)
[h,w]=size(labelMap); rgb=ones(h,w,3,'single');
for k=1:K
    m=(labelMap==k); if ~any(m(:)), continue; end
    for c=1:3, sl=rgb(:,:,c); sl(m)=cmap(k,c); rgb(:,:,c)=sl; end
end
end

function cm=u28_importance_map(Lc,ac,bc,mask,C)
b_w=[2.5;7.0;8.0;8.5;9.0]; imp=b_w/sum(b_w);
Lh=(C.PRIOR_L_R(:,2)-C.PRIOR_L_R(:,1))/2;
ah=(C.PRIOR_a_R(:,2)-C.PRIOR_a_R(:,1))/2;
bh=b_w/2;
sc=zeros(size(Lc),'single');
for d=1:5
    dL=((double(Lc)-C.PRIOR_LAB(d,1))/max(Lh(d),eps)).^2;
    da=((double(ac)-C.PRIOR_LAB(d,2))/max(ah(d),eps)).^2;
    db=((double(bc)-C.PRIOR_LAB(d,3))/max(bh(d),eps)).^2;
    sc=sc+single(imp(d)*exp(-0.5*(0.2*dL+0.35*da+0.45*db)));
end
lo=min(sc(mask)); hi=max(sc(mask));
cm=(sc-lo)/max(eps,hi-lo); cm(~mask)=0;
end

%% =========================================================================
%%  MAT 저장
%% =========================================================================
function u28_save_mat(f,itemId,mask,roiInfo,...
    RGB_raw,RGB_srgb,HSV,Lab,...
    dE,dEn,weatherScore,WHI,SVI,...
    WI_rule,WI_gmm,WI_hybrid,...
    D_rule,D_gmm,D,WD10,...
    postRule,postGMM,post5,scoreRule,distRule,...
    tieMap,confGMM,confHybrid,entMap,gmmInfo,C,...
    tile,tileMedL,tileIqrL,tileMedb,tileIqrb,...
    tileMedS,tileIqrS,tileMedV,tileIqrV,...
    edgeDen,locVar,tileWI,tileCnf,...
    tex,tGradL,tGradb,tGraddE,tEntL,tLoG,...
    tStdS,tStdL,tRngL,textureWins,...
    transM,heatGrid,gStat,wiMapC,...
    iMat,matDir,itemId2)
try
    S=struct('image',f,'itemId',itemId,'mask',mask,'roiInfo',roiInfo,...
        'RGB_raw',RGB_raw,'RGB_srgb',RGB_srgb,'HSV',HSV,'Lab',Lab,...
        'dE',dE,'dEn',dEn,'weatherScore',weatherScore,'WHI',WHI,'SVI',SVI,...
        'WI_rule',WI_rule,'WI_gmm',WI_gmm,'WI_hybrid',WI_hybrid,...
        'D_rule',D_rule,'D_gmm',D_gmm,'D',D,'WD10',WD10,...
        'postRule',postRule,'postGMM',postGMM,'post5',post5,...
        'scoreRule',scoreRule,'distRule',distRule,...
        'tieMap',tieMap,'confGMM',confGMM,'confHybrid',confHybrid,...
        'entMap',entMap,'gmmInfo',gmmInfo,'graniteRefs',C,...
        'tile',tile,...
        'tileMedL',tileMedL,'tileIqrL',tileIqrL,...
        'tileMedb',tileMedb,'tileIqrb',tileIqrb,...
        'tileMedS',tileMedS,'tileIqrS',tileIqrS,...
        'tileMedV',tileMedV,'tileIqrV',tileIqrV,...
        'edgeDen',edgeDen,'locVar',locVar,'tileWI',tileWI,'tileCnf',tileCnf,...
        'texture',tex,...
        'tileGradL',tGradL,'tileGradb',tGradb,'tileGraddE',tGraddE,...
        'tileEntL',tEntL,'tileLoG',tLoG,...
        'tileStdSmall',tStdS,'tileStdLarge',tStdL,'tileRngLarge',tRngL,...
        'textureWins',textureWins,'transMatrix',transM,'heatGrid',heatGrid,...
        'gradeStats',gStat,'wiMapContinuous',wiMapC);
    save(fullfile(iMat,'analysis.mat'),'-struct','S','-v7.3');
    save(fullfile(matDir,[itemId2,'.mat']),'-struct','S','-v7.3');
catch mexc28
    warning('U28:matSave','%s',mexc28.message);
end
end

%% =========================================================================
%%  Summary row
%% =========================================================================
function row=u28_summary_row(f,itemId,stem,ext,...
    RGB_raw,mask,roiInfo,...
    Rraw_v,Graw_v,Braw_v,Rsv,Gsv,Bsv,...
    Lv,av,bv,Hv,Sv,Vv,dEv,...
    WHI,SVI,WI_rule,WI_gmm,WI_hybrid,...
    D_rule,D_gmm,D,tieMap,confGMM,confHybrid,...
    tex,f5,f10,gmmInfo,entMap,OPT)
row=table();
row.image   =string(f); row.itemId=string(itemId);
row.stem    =string([stem ext]);
row.H       =size(RGB_raw,1); row.W=size(RGB_raw,2);
row.nPix    =nnz(mask);
row.darkExcFrac=roiInfo.darkExcFrac;

prc=@(v,p) prctile(double(v(:)),p);
for nm={'Rraw','Graw','Braw'}
    switch nm{1}
        case 'Rraw', vv=Rraw_v; case 'Graw', vv=Graw_v; case 'Braw', vv=Braw_v;
    end
    row.([nm{1},'_p50'])=prc(vv,50);
    row.([nm{1},'_ipr'])=prc(vv,95)-prc(vv,5);
end
for nm={'Rsrgb','Gsrgb','Bsrgb'}
    switch nm{1}
        case 'Rsrgb',vv=Rsv; case 'Gsrgb',vv=Gsv; case 'Bsrgb',vv=Bsv;
    end
    row.([nm{1},'_p50'])=prc(vv,50);
    row.([nm{1},'_ipr'])=prc(vv,95)-prc(vv,5);
end
for nm={'L','a','b'}
    switch nm{1}, case 'L',vv=Lv; case 'a',vv=av; case 'b',vv=bv; end
    for pp=[5 50 95]
        row.([nm{1},'_p',sprintf('%02d',pp)])=prc(vv,pp);
    end
end
row.H_p50=prc(Hv,50); row.S_p50=prc(Sv,50); row.V_p50=prc(Vv,50);
row.dE_p50=prc(dEv,50);
row.WHI_p50=prc(WHI(mask),50); row.SVI_p50=prc(SVI(mask),50);
row.WIrule_p50=prc(WI_rule(mask),50);
row.WIgmm_p50 =prc(WI_gmm(mask),50);
row.WIhybrid_p50=prc(WI_hybrid(mask),50);
row.D_rule_mode=mode(double(D_rule(mask)));
row.D_gmm_mode =mode(double(D_gmm(mask)));
row.D_mode     =mode(double(D(mask)));
row.tieFrac=nnz(tieMap(mask)>0.5)/max(nnz(mask),1);
row.confGMM_p50=prc(confGMM(mask),50);
row.confHybrid_p50=prc(confHybrid(mask),50);
row.entropy_mean=mean(double(entMap(mask)),'omitnan');
row.gradL_p50=prc(tex.gradL(mask),50);
row.entL_p50 =prc(tex.entL(mask),50);
row.logL_p50 =prc(tex.logL(mask),50);
for g=1:5,  row.(sprintf('D%d_frac',g))=f5(g); end
for g=1:10, row.(sprintf('WD10_%02d_frac',g))=f10(g); end
row.gmmUsed=gmmInfo.gmmUsed;
row.alpUsed=gmmInfo.alpUsed;
row.GMM_bestNLL=gmmInfo.bestNLL;
row.W_b=OPT.W_b; row.W_a=OPT.W_a; row.W_L=OPT.W_L;
end

%% =========================================================================
%%  컬러 참조
%% =========================================================================
function refs=u28_granite_refs()
refs.PRIOR_LAB=[56.00,-2.25,-0.25; 52.00,-1.50,2.00; 46.00,0.75,4.50;
                40.00,3.00,8.25; 33.00,3.50,10.50];
refs.PRIOR_L_R=[52 60;48 56;42 50;35 45;28 38];
refs.PRIOR_a_R=[-3.5 -1.0;-3.5 0.5;-1.5 3.0;0.0 6.0;1.0 6.0];
refs.PRIOR_b_R=[-1.5 1.0;-1.5 5.5;0.5 8.5;4.0 12.5;6.0 15.0];
refs.D5=[0.11,0.25,0.78; 0.41,0.71,0.96; 0.16,0.73,0.34; 0.96,0.61,0.11; 0.86,0.10,0.12];
refs.D10=[0.21,0.36,0.86; 0.05,0.15,0.55; 0.60,0.82,0.99; 0.27,0.57,0.89;
          0.34,0.85,0.50; 0.08,0.52,0.20; 0.99,0.76,0.25; 0.86,0.40,0.03;
          0.99,0.40,0.40; 0.66,0.05,0.05];
end

%% =========================================================================
%%  옵션 파서
%% =========================================================================
function OPT=u28_parse_opts(varargin)
p=inputParser; p.FunctionName='CoreWeathering_Ultra2028_FULL';
add=@(n,v) addParameter(p,n,v);
add('MODE','select'); add('IMAGE_FILE',''); add('IMAGE_DIR','');
add('FILE_EXTS',[]); add('RECURSIVE',false); add('MAX_IMAGES',inf);
add('RNG',0); add('OUT_BASEDIR',pwd); add('RUN_NAME','');
add('RESIZE_TARGETW',0); add('RESIZE_TARGETH',0); add('RESIZE_MAXW',0);
add('TILE',32); add('TEXTURE_WINS',[3 5 9]); add('LOG_SIGMA',1.0);
add('GRAD_METHOD','Sobel'); add('SMOOTH_SIGMA',0.6);
% 가중치: W_b > W_a > W_L
add('W_b',1.00); add('W_a',0.60); add('W_L',0.30); add('W_dE',0.20);
add('WHI_W_b',0.30); add('WHI_W_a',0.24); add('WHI_W_L',0.18);
add('WHI_W_S',0.16); add('WHI_W_V',0.08); add('WHI_W_dE',0.04);
add('RANGE_BONUS',0.55); add('OUTSIDE_LAMBDA',0.45);
add('ORDER_BONUS',0.15); add('ORDER_GAMMA',6.0);
add('B_NORM_MIN',-2.0); add('B_NORM_MAX',15.0);
add('TIE_EPS',0.030); add('TIE_DIST_EPS',0.060);
add('POST_MAJOR_FILTER',true); add('MAJORITY_WIN',3);
add('LOCAL_BLEND_ALPHA',0.35); add('LOCAL_TILE_SIZE',128);
add('GMM_BLEND',0.45); add('GMM_POST_SMOOTH_SIGMA',1.2);
add('GMM_SAMPLE_MAX',120000); add('GMM_REPS',5);
add('GMM_MAXITER',300); add('GMM_REG',1e-4);
add('POST_CHUNK',120000);
% ALP-GMM 국소희석 방지 파라미터
add('GMM_ALP',true);        % ALP 활성화 여부
add('ALP_TILE',64);          % ALP 타일 크기 (픽셀)
add('ALP_MIN_PIX',30);       % ALP 타일 최소 픽셀
add('ALP_BLEND_ALPHA',0.55); % ALP/Prior 혼합 비율 (0=pure prior, 1=pure alp)
add('RESP_CLIP_MIN',0.010);  % Responsibility Clipping 하한
% ROI 흑색 제외 6레이어
add('BLACK_RGB_MAX',8);      % R1: 채널별 최대
add('BLACK_SUM_MAX',24);     % R2: 합산 최대
add('ROI_LSTAR_BLACK_MAX',22); % R3: L* 상한 (D1 하한 52에서 30pt 여유)
add('ROI_V_BLACK_MAX',0.10); % R4: HSV V 상한
add('ROI_USE_LAB_BLACK',true);
add('ROI_S_MIN_FOR_VALID',0.04); % R6: 채도 하한
add('ROI_KEEP_LARGEST_ONLY',false);
add('ROI_MIN_KEEP_COMPONENT_AREA',1500);
add('ROI_BRIDGE_RADIUS',5);
add('MASK_CLOSE_RADIUS',9);
add('MIN_ROI_PIXELS',5000);
add('MIN_COMPONENT_AREA',3000);
% 저장
add('SAVE_ORIGINAL_COPY',true); add('SAVE_MAPS',true);
add('SAVE_MAT',true); add('SAVE_EXCEL',true);
add('SAVE_TEXTURE_MAPS',true); add('SAVE_FIGS',true);
add('FIG_DPI',220); add('HEAT_GRID_N',20);
parse(p,varargin{:}); OPT=p.Results;
OPT.MODE=lower(char(string(OPT.MODE)));
OPT.GRAD_METHOD=char(string(OPT.GRAD_METHOD));
w=round(double(OPT.TEXTURE_WINS(:)'));
w=w(isfinite(w)&w>=3); w=unique(w,'stable');
ew=mod(w,2)==0; w(ew)=w(ew)+1;
if isempty(w), w=[3 5 9]; end
OPT.TEXTURE_WINS=w;
end

%% =========================================================================
%%  파일 수집
%% =========================================================================
function files=u28_collect_files(OPT)
pats={'*.jpg','*.jpeg','*.png','*.tif','*.tiff','*.bmp'};
switch OPT.MODE
    case 'select'
        [fn,fp]=uigetfile({'*.jpg;*.jpeg;*.png;*.tif;*.tiff;*.bmp','Images';'*.*','All'},...
            '이미지 선택 (다중 가능)',pwd,'MultiSelect','on');
        if isequal(fn,0)
            d=uigetdir(pwd,'폴더 선택'); if isequal(d,0), files={}; return; end
            files=u28_list(d,pats,true);
        else
            if iscell(fn), files=cellfun(@(x) fullfile(fp,x),fn,'UniformOutput',false)';
            else, files={fullfile(fp,fn)}; end
        end
    case 'single'
        imgf=char(string(OPT.IMAGE_FILE));
        if isempty(imgf)
            [fn,fp]=uigetfile({'*.jpg;*.jpeg;*.png;*.tif;*.tiff;*.bmp','Images'},...
                '이미지 선택',pwd);
            if isequal(fn,0), files={}; return; end
            imgf=fullfile(fp,fn);
        end
        files={imgf};
    case 'folder'
        d=char(string(OPT.IMAGE_DIR));
        if isempty(d), d2=uigetdir(pwd,'폴더 선택'); if isequal(d2,0), files={}; return; end; d=d2; end
        files=u28_list(d,pats,logical(OPT.RECURSIVE));
    otherwise, files={};
end
files=files(:); files=files(~cellfun(@isempty,files)); files=unique(files,'stable');
if isfinite(OPT.MAX_IMAGES)&&OPT.MAX_IMAGES>0&&OPT.MAX_IMAGES<numel(files)
    files=files(randperm(numel(files),OPT.MAX_IMAGES));
end
end

function files=u28_list(d,pats,rec)
% 사전 할당: 패턴별 누적
allFiles = cell(0,1);
for i=1:numel(pats)
    if rec
        dd=dir(fullfile(d,'**',pats{i}));
    else
        dd=dir(fullfile(d,pats{i}));
    end
    dd=dd(~[dd.isdir]);
    ndd = numel(dd);
    batch = cell(ndd,1);
    for k=1:ndd
        batch{k} = fullfile(dd(k).folder,dd(k).name);
    end
    allFiles = [allFiles; batch]; %#ok<AGROW>
end
files = unique(allFiles,'stable');
end

%% =========================================================================
%%  공통 유틸리티
%% =========================================================================
function I0=u28_force_rgb(I0)
if ismatrix(I0)||size(I0,3)==1, I0=repmat(I0,1,1,3);
elseif size(I0,3)>3, I0=I0(:,:,1:3); end
end

function Iu=u28_to_uint8(I0)
I0=u28_force_rgb(I0);
if isa(I0,'uint8')
    Iu=I0;
elseif isa(I0,'uint16')
    Iu=uint8(double(I0)/65535*255);
else
    Iu=uint8(max(0,min(255,round(double(I0)))));
end
end

function Is=u28_to_srgb01(I0)
I0=u28_force_rgb(I0);
if isa(I0,'uint8')||isa(I0,'uint16'), Is=im2single(I0);
else
    Is=single(I0); if max(Is(:))>1.5, Is=Is/255; end
    Is=min(max(Is,0),1);
end
end

function I0=u28_maybe_resize(I0,OPT)
if OPT.RESIZE_TARGETW>0&&OPT.RESIZE_TARGETH>0
    I0=imresize(I0,[OPT.RESIZE_TARGETH OPT.RESIZE_TARGETW],'bilinear');
elseif OPT.RESIZE_MAXW>0&&size(I0,2)>OPT.RESIZE_MAXW
    I0=imresize(I0,OPT.RESIZE_MAXW/size(I0,2),'bilinear');
end
end

function u28_ensure(C)
for i=1:numel(C)
    if ~isfolder(C{i}), mkdir(C{i}); end
end
end

function u28_wtbl(T,path)
try writetable(T,path,'Encoding','UTF-8');
catch mexc28, warning('U28:wtbl','%s',mexc28.message); end
end

function u28_savefig(fig,dir_,name,dpi)
try
    exportgraphics(fig,fullfile(dir_,[name,'.png']),'Resolution',dpi);
    close(fig);
catch mexc28
    warning('U28:savefig','%s',mexc28.message);
    try close(fig); catch, end 
end
end

function u28_print_header(OPT)
fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║    CoreWeathering_Ultra2028_FULL  — SCI급 화강암 풍화 분석   ║\n');
fprintf('╠══════════════════════════════════════════════════════════════╣\n');
fprintf('║  GMM: ALP(국소희석보완)+RRC(클립)+MCI(단조)+MSS(다중시작)   ║\n');
fprintf('║  ROI: 6레이어(RGB/LAB/HSV/복합/채도) 흑색 제외              ║\n');
fprintf('║  색상: D1=진한파랑 D2=연한파랑 D3=초록 D4=주황 D5=레드      ║\n');
fprintf('║  가중치: W_b=%.2f(b*)>W_a=%.2f(a*)>W_L=%.2f(L*)            ║\n',OPT.W_b,OPT.W_a,OPT.W_L);
fprintf('║  ALP_TILE=%d  ALP_BLEND=%.2f  RESP_CLIP=%.4f              ║\n',OPT.ALP_TILE,OPT.ALP_BLEND_ALPHA,OPT.RESP_CLIP_MIN);
fprintf('║  출력: FIG16종+CSV11종+PNG35종+MAT+XLSX                     ║\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n');
end
