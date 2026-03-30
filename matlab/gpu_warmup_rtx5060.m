function gpu_warmup_rtx5060()
% GPU_WARMUP_RTX5060
% -------------------------------------------------------------------------
% One-file GPU warm-up for RTX 50xx (RTX 5060 8 GB, CC 12.0) on MATLAB R2025b
% - Enable forward compatibility (env + session)
% - Run validateGPU if available
% - Check device availability and give clear next steps in Korean
% - Run small GPU kernels (BLAS, elementwise, dlconv) and reset device safely
%
% 한글 설명:
% - RTX 50xx(CC 12.0) 계열에서 MATLAB이 GPU를 'available'로 열지 못할 때
%   취해야 할 단계(환경변수, 세션 설정, 재시작 권장)를 출력합니다.
% - GPU가 사용 가능하면 작은 연산들을 실행하여 "warm-up" 합니다.
% - 대상 GPU: RTX 5060 VRAM 8 GB → 메모리 보수적으로 운용
% -------------------------------------------------------------------------

clc;
fprintf("=== GPU Warm-up  [MATLAB R2025b / RTX 5060 8 GB] ===\n");
fprintf("[Info] MATLAB version: %s\n", version);

%% ── 환경 변수 설정 (startup.m에 추가하면 재시작 이후에도 유지됨) ──────────
try
    setenv("MW_CUDA_FORWARD_COMPATIBILITY", "1");
    setenv("CUDA_CACHE_MAXSIZE", "536870912");   % JIT 캐시 512 MB
    fprintf("[Info] Environment variables set (current session).\n");
catch
    fprintf("[Warn] Could not set environment variables.\n");
end

%% ── Forward compatibility 활성화 (R2025b에서도 CC 12.0은 필요할 수 있음) ──
if exist("parallel.gpu.enableCUDAForwardCompatibility", "file") == 2
    try
        parallel.gpu.enableCUDAForwardCompatibility(true);
        fprintf("[Info] CUDA forward compatibility enabled.\n");
    catch ME
        fprintf("[Warn] enableCUDAForwardCompatibility failed: %s\n", ME.message);
    end
else
    fprintf("[Warn] parallel.gpu.enableCUDAForwardCompatibility not found in this build.\n");
end

%% ── 디바이스 수 확인 ──────────────────────────────────────────────────────
nAll   = NaN;
nAvail = NaN;
try nAll   = gpuDeviceCount("all");       catch, end
try nAvail = gpuDeviceCount("available"); catch, end
fprintf("[Info] gpuDeviceCount  all=%g  available=%g\n", nAll, nAvail);

%% ── validateGPU (R2025b에 있으면 실행) ─────────────────────────────────────
if exist("validateGPU", "file") == 2
    try
        fprintf("[Info] Running validateGPU...\n");
        try
            validateGPU("all");
        catch
            validateGPU;
        end
    catch ME
        fprintf("[Warn] validateGPU: %s\n", ME.message);
    end
end

% validateGPU 이후 재확인
try nAvail = gpuDeviceCount("available"); catch, end
fprintf("[Info] After checks: available=%g\n", nAvail);

%% ── 드라이버 레벨에서 GPU 미감지 → 중단 ─────────────────────────────────
if isnan(nAll) || nAll < 1
    error("[STOP] GPU가 OS 레벨에서 감지되지 않습니다.\n" + ...
          "       nvidia-smi 명령으로 드라이버 상태를 먼저 확인하세요.");
end

%% ── MATLAB에서 available 아님 → 한글 안내 출력 후 종료 ──────────────────
if isnan(nAvail) || nAvail < 1
    fprintf("\n[STOP] MATLAB이 RTX 5060(CC 12.0)을 'available'로 열지 못했습니다.\n\n");
    fprintf("  ▶ 권장 조치 (순서대로 시도)\n");
    fprintf("  1) startup.m 에 아래 2줄 추가 후 MATLAB 완전 재시작:\n");
    fprintf("       setenv('MW_CUDA_FORWARD_COMPATIBILITY','1');\n");
    fprintf("       parallel.gpu.enableCUDAForwardCompatibility(true);\n\n");
    fprintf("  2) NVIDIA 드라이버를 최신 버전(560+)으로 업데이트\n");
    fprintf("       nvidia-smi 로 드라이버 버전 및 CUDA 버전 확인\n\n");
    fprintf("  3) MATLAB R2025b Update 패치가 있으면 적용 후 재시도\n\n");
    fprintf("  4) Parallel Computing Toolbox가 설치되어 있는지 확인:\n");
    fprintf("       ver  →  'Parallel Computing Toolbox' 항목 존재 여부\n\n");
    return;
end

%% ── Warm-up 커널 실행 (보수적 메모리: 8 GB VRAM 기준) ────────────────────
g = [];
try
    g = gpuDevice(1);
    fprintf("\n[OK] GPU: %s  CC=%s  VRAM=%.2f GB\n", ...
        g.Name, string(g.ComputeCapability), g.TotalMemory / 1024^3);
    fprintf("     Available mem at start: %.2f GB\n\n", g.AvailableMemory / 1024^3);

    % ── (1) BLAS: 행렬 곱 ───────────────────────────────────────────────
    fprintf("[1/3] BLAS warm-up (matrix multiply 2048×2048 single)...\n");
    A = gpuArray.rand(2048, 2048, "single");
    B = gpuArray.rand(2048, 2048, "single");
    C = A * B; %#ok<NASGU>
    wait(g);
    clear A B C;
    fprintf("      → done  (free mem: %.2f GB)\n", g.AvailableMemory / 1024^3);

    % ── (2) Element-wise ops ─────────────────────────────────────────────
    fprintf("[2/3] Elementwise warm-up (4096×1024 single)...\n");
    X = gpuArray.rand(4096, 1024, "single");
    Y = log1p(X) + sqrt(X); %#ok<NASGU>
    wait(g);
    clear X Y;
    fprintf("      → done  (free mem: %.2f GB)\n", g.AvailableMemory / 1024^3);

    % ── (3) dlconv (Deep Learning Toolbox 필요) ──────────────────────────
    fprintf("[3/3] dlconv warm-up (if Deep Learning Toolbox available)...\n");
    if exist("dlarray", "file") == 2 && exist("dlconv", "file") == 2
        % 8 GB 기준 안전값: 입력 224×224×3, 배치 4, 필터 32개
        Xdl = dlarray(gpuArray.rand(224, 224, 3, 4, "single"), "SSCB");
        W   = gpuArray.rand(7, 7, 3, 32, "single");
        Ydl = dlconv(Xdl, W, [], "Stride", 2, "Padding", "same"); %#ok<NASGU>
        wait(g);
        clear Xdl W Ydl;
        fprintf("      → done  (free mem: %.2f GB)\n", g.AvailableMemory / 1024^3);
    else
        fprintf("      → [Skip] dlarray / dlconv 함수를 찾을 수 없습니다.\n");
        fprintf("               Deep Learning Toolbox가 설치되어 있는지 확인하세요.\n");
    end

    fprintf("\n=== GPU Warm-up 완료 ===\n");

catch ME
    fprintf("\n[Error] GPU warm-up 실패: %s\n", ME.message);
    fprintf("[Hint]  드라이버·CUDA 런타임·MATLAB 버전·startup.m 설정 확인 후 재시작 권장.\n");
end

%% ── 디바이스 해제 (항상 실행) ────────────────────────────────────────────
try
    if ~isempty(g)
        reset(g);
        fprintf("[Info] GPU device reset.\n");
    end
catch
    try
        gpuDevice([]);   % 선택 해제만이라도 시도
    catch
    end
end

end