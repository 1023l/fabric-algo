# -*- coding: utf-8 -*-
"""模型管理：上传训练平台导出的模型包 -> 自动转换 -> 热替换，同一服务内提供接口。

接口（与 demo UI / 第三方 API 同进程，prefix=/api/models）：
  GET  /api/models/current        当前生效模型版本 + 后台任务状态
  POST /api/models/upload         上传 model_bundle_*.zip（训练平台"一键导出"产物，3 个模型）
  POST /api/models/upload_single  上传单个模型（fabric/text 的 .pt；rec 的 rec*.zip，内含 ONNX）
  GET  /api/models/task           后台转换任务进度（step + 日志尾部）

上传后的自动流水线（后台线程）：
  1. 部署源文件（.pt / rec ONNX）到 models/
  2. 更新 tools/config.yaml 的 versions
  3. export_onnx.py det（fabric / text 的 .pt -> ONNX）
  4. build_engine.py --force（ONNX -> TRT engine，带计时缓存）
  5. 先试加载新 engine（失败则保留旧模型并报 failed），成功后热替换
     web_server / algo_routes 的模型单例，并写 models/current.json

rec 说明：rec 的 pdparams -> ONNX 导出已收敛到训练平台侧（paddle 环境），
模型包与单个下载直接携带 ONNX，本模块（及 Docker 容器）无需 paddle 环境。

约定：当前生效版本记录在 models/current.json（{"fabric": "fabric20260828V2", ...}），
     不存在时回退到下方 _FALLBACK（与历史硬编码一致）。
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from utils import ROOT

TOOLS_DIR = ROOT / "tools"
MODEL_DIR = ROOT / "models"
CURRENT_JSON = MODEL_DIR / "current.json"
INCOMING_DIR = MODEL_DIR / "incoming"

# 与旧版 web_server.py / algo_routes.py 硬编码一致（current.json 缺失时兜底）
_FALLBACK = {"fabric": "fabric20260828V2", "text": "text20260831V1", "rec": "rec20260828V3"}
_KEYS = ("fabric", "text", "rec")

router = APIRouter(prefix="/api/models", tags=["models-admin"])


# ============================================================
# 当前版本（models/current.json）
# ============================================================
def current_versions() -> dict:
    """当前生效模型 stem（如 fabric20260828V2）。"""
    try:
        d = json.loads(CURRENT_JSON.read_text(encoding="utf-8"))
        return {k: str(d[k]) for k in _KEYS}
    except (OSError, ValueError, KeyError, TypeError):
        return dict(_FALLBACK)


def engine_paths(versions: dict | None = None) -> dict:
    """由版本 stem 得到 3 个 engine 路径。"""
    v = versions or current_versions()
    return {k: MODEL_DIR / k / f"{v[k]}.engine" for k in _KEYS}


# ============================================================
# 后台转换任务状态
# ============================================================
_task: dict = {
    "status": "idle",       # idle / running / ok / failed
    "step": "",
    "versions": None,       # 本次上传的目标版本 {"fabric": "20260905V1", ...}
    "log": [],              # 日志尾部（最多 _LOG_MAX 行）
    "started": None,
    "finished": None,
}
_task_lock = threading.Lock()
_LOG_MAX = 300


def _log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with _task_lock:
        _task["log"].append(line)
        if len(_task["log"]) > _LOG_MAX:
            _task["log"] = _task["log"][-_LOG_MAX:]


def _step(step: str) -> None:
    with _task_lock:
        _task["step"] = step
    _log(f"== {step}")


# ============================================================
# 上传接口
# ============================================================
@router.get("/current")
def get_current():
    return JSONResponse({"current": current_versions(), "task": _task_snapshot()})


@router.get("/task")
def get_task():
    return JSONResponse(_task_snapshot())


def _task_snapshot() -> dict:
    with _task_lock:
        return {**_task, "log": list(_task["log"][-80:])}


def _safe_extract(zf: zipfile.ZipFile, dst: Path) -> None:
    """解压（防 zip slip：拒绝绝对路径 / ..）。"""
    for name in zf.namelist():
        p = Path(name)
        if p.is_absolute() or ".." in p.parts:
            raise HTTPException(400, f"非法压缩包条目: {name}")
    zf.extractall(dst)


@router.post("/upload")
async def upload_model(file: UploadFile = File(...)):
    """上传训练平台导出的模型包（manifest.json + det/*.pt + ocr/rec*/）。

    同步完成：保存 zip -> 解压 -> 解析 manifest -> 部署源文件清单校验，
    然后后台线程跑 转换流水线，立即返回目标版本供前端轮询 /api/models/task。
    """
    with _task_lock:
        if _task["status"] == "running":
            raise HTTPException(409, "已有转换任务在执行，请等待完成后再上传")
    if not (file.filename or "").lower().endswith(".zip"):
        raise HTTPException(400, "请上传 .zip 模型包（训练平台-模型管理-一键导出）")

    ts = time.strftime("%Y%m%d_%H%M%S")
    INCOMING_DIR.mkdir(parents=True, exist_ok=True)
    deploy = INCOMING_DIR / f"bundle_{ts}"
    deploy.mkdir(parents=True, exist_ok=True)
    zip_path = deploy / "upload.zip"   # zip 放进 staging，成功/失败统一清目录即可
    size = 0
    try:
        with zip_path.open("wb") as w:
            while True:
                chunk = await file.read(1 << 20)
                if not chunk:
                    break
                size += len(chunk)
                w.write(chunk)
        if size < 1024:
            raise HTTPException(400, "上传内容为空或过小")

        with zipfile.ZipFile(zip_path) as zf:
            names = set(zf.namelist())
            if "manifest.json" not in names:
                raise HTTPException(400, "不是模型包（缺 manifest.json），请用训练平台『一键导出3个模型』生成")
            _safe_extract(zf, deploy)

        manifest = json.loads((deploy / "manifest.json").read_text(encoding="utf-8"))
        versions = {k: str(manifest["models"][k]["version"]) for k in _KEYS}

        # 校验包内源文件齐全（此处仅检查，真正部署在后台线程做）
        for k, sub in (("fabric", f"det/fabric{versions['fabric']}.pt"),
                       ("text", f"det/text{versions['text']}.pt"),
                       ("rec", f"ocr/rec{versions['rec']}_onnx.onnx")):
            if not (deploy / sub).is_file():
                raise HTTPException(400, f"包内缺少 {sub}")

        # 防重复上传同一版本组合
        cur = current_versions()
        same = all(cur[k] == f"{k}{versions[k]}" for k in _KEYS)
        if same:
            raise HTTPException(400, f"该版本组合已是当前生效版本"
                                     f"（fabric{versions['fabric']} / text{versions['text']} / rec{versions['rec']}）"
                                     f"；如需重建请先在包内改版本号")

        with _task_lock:
            _task.update({"status": "running", "step": "等待后台流水线启动",
                          "versions": versions, "log": [],
                          "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                          "finished": None})
        _log(f"收到模型包 {file.filename}（{size / 1e6:.1f} MB），"
             f"目标版本 fabric{versions['fabric']} / text{versions['text']} / rec{versions['rec']}")
        threading.Thread(target=_pipeline, args=(deploy, versions, _KEYS),
                         name="model-pipeline", daemon=True).start()
        return JSONResponse({"ok": True, "versions": versions,
                             "msg": "已开始转换（导出ONNX -> 构建engine -> 热替换），"
                                    "请通过 /api/models/task 轮询进度"})
    except HTTPException:
        _cleanup(deploy)
        raise
    except (OSError, ValueError, KeyError, TypeError) as e:
        _cleanup(deploy)
        raise HTTPException(400, f"模型包解析失败: {e}")


def _cleanup(staging: Path) -> None:
    """删除 staging 目录（zip 与解压产物都在其中，统一清理）。"""
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)


async def _save_upload(f: UploadFile, dst: Path, min_size: int = 1024) -> int:
    """把 UploadFile 流式落盘到 dst，返回字节数。"""
    size = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as w:
        while True:
            chunk = await f.read(1 << 20)
            if not chunk:
                break
            size += len(chunk)
            w.write(chunk)
    if size < min_size:
        raise HTTPException(400, f"上传内容为空或过小: {dst.name}")
    return size


# ============================================================
# 单模型上传（fabric/text 传 .pt；rec 传训练平台下载的 rec*.zip，内含 ONNX）
# ============================================================
@router.post("/upload_single")
async def upload_single(files: list[UploadFile] = File(...)):
    """上传单个模型并只转换该模型（其余两个保持当前版本不动）。

    支持两种形式：
      a) 单个 .pt：文件名须为 fabric{YYYYMMDD}V{n}.pt / text{YYYYMMDD}V{n}.pt
      b) 单个 .zip：训练平台 rec 模型下载产物（文件名 rec{ver}.zip，内含 rec{ver}_onnx.onnx）

    rec 的 pdparams -> ONNX 导出在训练平台侧完成，本接口不再接收 pdparams。
    """
    with _task_lock:
        if _task["status"] == "running":
            raise HTTPException(409, "已有转换任务在执行，请等待完成后再上传")
    if not files:
        raise HTTPException(400, "未收到文件")
    if len(files) != 1:
        raise HTTPException(400, "一次只上传一个文件（fabric*.pt / text*.pt / rec*.zip）")

    ts = time.strftime("%Y%m%d_%H%M%S")
    staging = INCOMING_DIR / f"single_{ts}"
    size = 0
    try:
        kind: str | None = None
        ver: str | None = None

        f = files[0]
        name = (f.filename or "").replace("\\", "/").strip("/")
        low = name.lower()
        # 版本号约定 V 大写，须用原始文件名匹配（不能先 lowercase）
        m = re.match(r"^(fabric|text)(\d{8}V\d+)\.pt$", name)
        if m:
            # ---- a) fabric / text 的 .pt ----
            kind, ver = m.group(1), m.group(2)
            size = await _save_upload(f, staging / "det" / f"{kind}{ver}.pt")
        elif low.endswith(".zip"):
            # ---- b) rec 的 zip（内含 rec{ver}_onnx.onnx）----
            zip_path = staging / "upload.zip"
            size = await _save_upload(f, zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                names = [n.replace("\\", "/").strip("/") for n in zf.namelist()]
                ver, sub = None, ""
                zm = re.match(r"^rec(\d{8}V\d+)\.zip$", name)
                for n in names:
                    om = re.match(r"^(?:rec\d{8}V\d+/)?rec(\d{8}V\d+)_onnx\.onnx$", n)
                    if om:
                        ver, sub = om.group(1), n
                        break
                if not ver:
                    raise HTTPException(400, "zip 内未找到 rec*_onnx.onnx；"
                                             "rec 的 ONNX 由训练平台导出，请从新版训练平台"
                                             "「模型管理」重新下载 rec*.zip")
                kind = "rec"
                tmp = staging / "_z"
                _safe_extract(zf, tmp)
                onnx_src = tmp / sub
                dst = staging / "ocr" / f"rec{ver}_onnx.onnx"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(onnx_src), str(dst))
                shutil.rmtree(tmp, ignore_errors=True)
            zip_path.unlink()
        else:
            raise HTTPException(400, "请上传 fabric*.pt / text*.pt，"
                                     "或训练平台下载的 rec*.zip（内含 ONNX）")

        # 防重复：该模型版本 == 当前生效版本
        cur = current_versions()
        if cur[kind] == f"{kind}{ver}":
            raise HTTPException(400, f"{kind}{ver} 已是当前生效版本；如需重建请改版本号后重新上传")

        # 完整目标版本 = 当前版本 + 本次替换的模型
        cur_vers = {k: v[len(k):] for k, v in cur.items()}     # stem 去前缀 -> 短版本
        versions = {**cur_vers, kind: ver}

        with _task_lock:
            _task.update({"status": "running", "step": "等待后台流水线启动",
                          "versions": versions, "log": [],
                          "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                          "finished": None})
        _log(f"收到单模型上传 {kind}{ver}（{size / 1e6:.1f} MB），其余模型保持当前版本")
        threading.Thread(target=_pipeline, args=(staging, versions, (kind,)),
                         name="model-pipeline", daemon=True).start()
        return JSONResponse({"ok": True, "kind": kind, "version": ver, "versions": versions,
                             "msg": f"已开始转换 {kind}{ver}（构建engine -> 热替换），"
                                    "请通过 /api/models/task 轮询进度"})
    except HTTPException:
        _cleanup(staging)
        raise
    except (OSError, ValueError, KeyError, TypeError) as e:
        _cleanup(staging)
        raise HTTPException(400, f"单模型上传解析失败: {e}")


# ============================================================
# 后台流水线
# ============================================================
def _pipeline(staging: Path, versions: dict, kinds: tuple) -> None:
    """后台转换流水线。

    staging:  bundle 布局的源文件目录（det/*.pt、ocr/rec{ver}_onnx.onnx）
    versions: 3 个模型的完整目标版本（短版本号；未变化的保持当前值）
    kinds:    本次实际要转换的模型子集（fabric / text / rec 的任意组合）
    """
    try:
        _step("部署模型源文件到 models/")
        _deploy_sources(staging, versions, kinds)

        _step("更新 tools/config.yaml versions")
        _update_cfg_versions({k: versions[k] for k in kinds})

        if "fabric" in kinds:
            _step(f"导出 fabric ONNX（{versions['fabric']}）")
            _run_tool("export_onnx.py", ["det", "--model", "fabric",
                                         "--version", versions["fabric"],
                                         "--pt", str(staging / "det" / f"fabric{versions['fabric']}.pt")])

        if "text" in kinds:
            _step(f"导出 text ONNX（{versions['text']}）")
            _run_tool("export_onnx.py", ["det", "--model", "text",
                                         "--version", versions["text"],
                                         "--pt", str(staging / "det" / f"text{versions['text']}.pt")])

        # rec 的 ONNX 由训练平台侧导出并随包携带，此处直接进 engine 构建
        for k in kinds:
            _step(f"构建 TRT engine（{k}{versions[k]}，--force 重建）")
            _run_tool("build_engine.py", ["--model", k, "--force"])

        stems = {k: f"{k}{versions[k]}" for k in _KEYS}
        _step("试加载新 engine 并热替换")
        _hot_swap(engine_paths(stems))

        CURRENT_JSON.write_text(json.dumps(stems, ensure_ascii=False, indent=2),
                                encoding="utf-8")
        _cleanup(staging)
        _log(f"current.json 已更新: {stems}")
        _step("完成，新模型已生效，可直接开始检测")
        with _task_lock:
            _task["status"] = "ok"
            _task["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:  # noqa: BLE001
        _log(f"[failed] {type(e).__name__}: {e}")
        _log("旧模型仍在线继续服务；请排查后重新上传")
        with _task_lock:
            _task["status"] = "failed"
            _task["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    finally:
        with _task_lock:
            _task["step"] = "" if _task["status"] == "ok" else _task["step"]


def _deploy_sources(staging: Path, versions: dict, kinds: tuple = _KEYS) -> None:
    """pt -> models/{fabric,text}/，rec ONNX -> models/rec/（只部署 kinds 内的）。"""
    for kind in ("fabric", "text"):
        if kind not in kinds:
            continue
        src = staging / "det" / f"{kind}{versions[kind]}.pt"
        dst = MODEL_DIR / kind / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        _log(f"{src.name} -> {dst.relative_to(ROOT)}")
    if "rec" in kinds:
        src = staging / "ocr" / f"rec{versions['rec']}_onnx.onnx"
        dst = MODEL_DIR / "rec" / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        _log(f"{src.name} -> {dst.relative_to(ROOT)}")


def _update_cfg_versions(versions: dict) -> None:
    """把新版本号写进 tools/config.yaml 的 versions 段（保留注释，仅替换值）。"""
    p = TOOLS_DIR / "config.yaml"
    txt = p.read_text(encoding="utf-8")
    for k, v in versions.items():
        txt, n = re.subn(rf"(?m)^(\s+{k}:\s+)\S+.*$", rf"\g<1>{v}", txt, count=1)
        if n != 1:
            raise RuntimeError(f"config.yaml 未找到 versions.{k} 条目")
    p.write_text(txt, encoding="utf-8")
    _log("tools/config.yaml versions 已更新")


def _run_tool(script: str, args: list[str]) -> None:
    cmd = [sys.executable, str(TOOLS_DIR / script), *args]
    _log("$ " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=str(TOOLS_DIR))
    for line in (r.stdout or "").splitlines():
        _log(line)
    for line in (r.stderr or "").splitlines():
        _log(line)
    if r.returncode != 0:
        raise RuntimeError(f"{script} 退出码 {r.returncode}（详见上方日志）")


def _hot_swap(engines: dict) -> None:
    """先在旁路试加载新 engine（任一失败即抛异常，旧模型不受影响），
    成功后替换 web_server / algo_routes 的模型单例。"""
    import web_server
    import algo_routes
    from onnx_engine import RecTrtEngine

    if web_server.state["running"]:
        raise RuntimeError("检测运行中，不能替换模型；请先点『停止』再重试热替换")

    new_set = web_server.build_model_set(engines)   # 失败抛异常 -> 保留旧模型
    with web_server.state["lock"]:
        web_server._models.clear()
        web_server._models.update(new_set)
    _log("web_server 模型已热替换（fabric/text/rec×%d）" % len(new_set["rec"]))

    # algo_routes 用独立的 rec 实例（TRT context 不能跨线程共享）
    algo_rec = RecTrtEngine.load(engines["rec"],
                                 score_thresh=algo_routes.DEFAULT_REC_THRESH,
                                 max_width=algo_routes.DEFAULT_REC_MAX_W)
    with algo_routes._lock:
        algo_routes._models.update({"fabric": new_set["fabric"],
                                    "text": new_set["text"],
                                    "rec": [algo_rec], "rec_n": 1})
    _log("algo_routes 模型已热替换")
