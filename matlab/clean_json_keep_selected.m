function clean_json_keep_selected()
% clean_json_keep_selected
% ---------------------------------------------------------
% 폴더 선택 → 하위 폴더 포함 모든 JSON 탐색 →
% 필요한 필드만 남긴 JSON을 새 폴더에 생성
%
% 출력 폴더: <선택폴더>\_json_clean_minimal
%
% 남기는 필드:
%   rw, ISO, lux, shutter_speed, f-stop, humidity,
%   rock_type1, rock_type2
%
% 인수 없이 실행 가능

% ---------- 폴더 선택 ----------
jsonInRoot = uigetdir(pwd, "원본 JSON 폴더 선택");
if jsonInRoot == 0
    error("JSON 폴더가 선택되지 않았습니다.");
end

jsonOutRoot = fullfile(jsonInRoot, "_json_clean_minimal");
if ~isfolder(jsonOutRoot)
    mkdir(jsonOutRoot);
end

% ---------- JSON 파일 재귀 수집 ----------
files = listAllJsonFiles(jsonInRoot);
fprintf("[Info] JSON files found: %d\n", numel(files));

if isempty(files)
    warning("JSON:EMPTY", "선택한 폴더에서 JSON 파일을 찾지 못했습니다.");
    return;
end

% ---------- 처리 ----------
for i = 1:numel(files)
    inFile  = fullfile(files(i).folder, files(i).name);
    outFile = fullfile(jsonOutRoot, files(i).name);

    try
        s = jsondecode(fileread(inFile));
        out = struct();

        % --- 기본 필드 ---
        out.rw            = getFieldSafe(s, "rw");
        out.ISO           = getFieldSafe(s, "ISO");
        out.lux           = getFieldSafe(s, "lux");
        out.shutter_speed = getFieldSafe(s, "shutter_speed");
        out.fstop         = getFieldSafe(s, "f-stop");
        out.humidity      = getFieldSafe(s, "humidity");

        % --- rock_type 구조 평탄화 ---
        if isfield(s,"rock_type") && ~isempty(s.rock_type)
            rt = s.rock_type(1);
            out.rock_type1 = getFieldSafe(rt, "rock_type1");
            out.rock_type2 = getFieldSafe(rt, "rock_type2");
        else
            out.rock_type1 = missing;
            out.rock_type2 = missing;
        end

        % --- 저장 ---
        fid = fopen(outFile, 'w');
        fwrite(fid, jsonencode(out, "PrettyPrint", true), 'char');
        fclose(fid);

    catch ME
        warning("JSON:CLEANFAIL", "%s", ME.message);
    end
end

fprintf("[Done] Clean JSON saved to:\n%s\n", jsonOutRoot);

end

% =========================================================
% 하위 함수들
% =========================================================

function files = listAllJsonFiles(root)
% 재귀적으로 모든 *.json 수집 (한글/공백/버전 무관)
    files = dir(fullfile(root, "*.json"));
    subs  = dir(root);
    subs  = subs([subs.isdir] & ~startsWith({subs.name}, "."));

    for k = 1:numel(subs)
        subpath = fullfile(root, subs(k).name);
        files = [files; listAllJsonFiles(subpath)]; %#ok<AGROW>
    end
end

function v = getFieldSafe(s, fname)
% 필드가 없으면 missing 반환
    if isfield(s, fname)
        v = s.(fname);
    else
        v = missing;
    end
end
