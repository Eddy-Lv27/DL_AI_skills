"""模块四：模型拆解→可视化→再生成→两步验证→标准化入库（模块详细设计六章 6.1~6.6）。

四个后台任务 + 两个同步入口：
- decompose        6.1 agent 解析模型 → IR（validate_ir 硬校验后落盘 reports/ir.json）
- decompose_trace  6.2 项目环境跑模型 hook 回填缺失形状（scripts/trace_shapes.py）
- decompose_verify 6.4 结构+数值两步验证（scripts/verify_decompose.py，项目环境执行）
- module_ingest    6.5 验证通过 → 模块包入库 + 结构化项目（画布可编辑，graph.json）
- regenerate（同步端点）6.3 IR → 自包含 PyTorch 代码（ir_codegen.py，不产生任务）
- 查询/调参（同步）: read_ir / get_verification / update_node_params

验证不过（overall=failed）时任务本身仍 success，结果经 run_record(status=failed,
error=failure_reason) 可检索供 agent/用户改进；入库前置强校验 ir_hash 防 stale。
"""
import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.config import MODULES_DIR, PROJECT_ROOT
from app.services import (
    agent_service, analysis_service, ir_codegen, ir_graphir, ir_schema,
    knowledge_service, project_manager, task_manager,
)
from app.services.ir_codegen import IrIncompleteError
from app.services.ir_schema import SCHEMA_VERSION, TASK_TYPES, _as_ir, ir_hash, nodes_by_id, validate_ir

TASK_DECOMPOSE = "decompose"
TASK_TRACE = "decompose_trace"
TASK_VERIFY = "decompose_verify"
TASK_INGEST = "module_ingest"

TRACE_SCRIPT = PROJECT_ROOT / "scripts" / "trace_shapes.py"
VERIFY_SCRIPT = PROJECT_ROOT / "scripts" / "verify_decompose.py"

# 数值比对阈值（实施约定；仿 REPRO_DEVIATION_* 环境变量覆盖先例）
DECOMPOSE_NUM_RTOL = float(os.getenv("DECOMPOSE_NUM_RTOL", "1e-5"))
DECOMPOSE_NUM_ATOL = float(os.getenv("DECOMPOSE_NUM_ATOL", "1e-6"))
DECOMPOSE_NUM_SEEDS = os.getenv("DECOMPOSE_NUM_SEEDS", "42,1337,2024")
DECOMPOSE_AGENT_TIMEOUT_S = 1800
DECOMPOSE_TRACE_TIMEOUT_S = 600
DECOMPOSE_VERIFY_TIMEOUT_S = 1800


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ws(project: dict) -> Path:
    return Path(project["workspace_path"])


def _require_original(project_id: str) -> dict:
    return project_manager.require_type(project_id, {"original"})


def _read_ir(project: dict) -> Optional[dict]:
    p = _ws(project) / "reports" / "ir.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def read_ir(project_id: str) -> Optional[dict]:
    return _read_ir(_require_original(project_id))


def _write_ir(project: dict, ir: dict) -> None:
    p = _ws(project) / "reports" / "ir.json"
    p.write_text(json.dumps(ir, ensure_ascii=False, indent=2), encoding="utf-8")


def get_verification(project_id: str) -> Optional[dict]:
    p = _ws(_require_original(project_id)) / "reports" / "verification.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def verification_status(project_id: str) -> tuple[str, Optional[dict]]:
    """验证新鲜度：none=未验证 / valid=ir_hash 一致 / stale=IR 已修改未重验。"""
    ir = read_ir(project_id)
    verification = get_verification(project_id)
    if verification is None:
        return "none", None
    if ir is not None and verification.get("ir_hash") == ir_hash(ir):
        return "valid", verification
    return "stale", verification


# --------------------------- 同步入口 ---------------------------

def decompose(project_id: str, entry_class: str | None = None) -> str:
    _require_original(project_id)
    params: dict = {"project_id": project_id}
    if entry_class:
        params["entry_class"] = entry_class
    return task_manager.create_task(TASK_DECOMPOSE, project_id=project_id, params=params)


def trace(project_id: str) -> str:
    _require_original(project_id)
    return task_manager.create_task(TASK_TRACE, project_id=project_id, params={"project_id": project_id})


def verify(project_id: str) -> str:
    _require_original(project_id)
    return task_manager.create_task(TASK_VERIFY, project_id=project_id, params={"project_id": project_id})


def ingest(project_id: str) -> str:
    _require_original(project_id)
    return task_manager.create_task(TASK_INGEST, project_id=project_id, params={"project_id": project_id})


def regenerate(project_id: str) -> str:
    """6.3 同步再生成：IR 结构错误/不完整 → IrIncompleteError（message 为缺失项清单）。"""
    ir = read_ir(project_id)
    if ir is None:
        raise LookupError("ir not found：请先 POST /api/projects/{id}/decompose")
    return ir_codegen.generate(ir)


def update_node_params(project_id: str, node_id: str, params: dict) -> dict:
    """PUT 调参回写（6.2「用户能手动调节层级的参数」）：写回后旧验证经 ir_hash 变 stale。"""
    project = _require_original(project_id)
    ir = _read_ir(project)
    if ir is None:
        raise LookupError("ir not found：请先 POST /api/projects/{id}/decompose")
    node = nodes_by_id(ir).get(node_id)
    if node is None:
        raise LookupError(f"node not found: {node_id}")
    node["params"] = params
    _write_ir(project, ir)
    return node


# --------------------------- 6.1 拆解 ---------------------------

def _decompose_prompt(hierarchy: list[dict], entry_class: str | None = None) -> str:
    roots = [h for h in hierarchy if h.get("parent") == "Module"]
    parts = [
        "任务：阅读深度学习项目代码，把入口 PyTorch 模型精确拆解为结构化 IR（中间表示），"
        "用于后续确定性代码再生成与两步验证。\n\n"
        "【红线】只允许读取文件（Read/Glob/Grep），禁止修改、创建、删除项目内任何文件。\n\n"
        "【静态结构报告】识别到的 nn.Module 子类：\n"
        f"{json.dumps(hierarchy, ensure_ascii=False)}\n",
    ]
    if entry_class:
        parts.append(
            f"【指定入口类】用户已指定 entry_class={entry_class}，必须以此类为入口模型根节点"
            "（其他类按第 3 条作为子模块展开）。\n"
        )
    elif roots:
        parts.append(
            f"其中直接继承 nn.Module 的类：{json.dumps([h['class'] for h in roots], ensure_ascii=False)}；"
            "请从这些类中确定模型的实际入口类作为 entry_class（通常就是它）。\n"
        )
    parts.append(
        "【IR 规范】\n"
        "1. source_file=入口类所在文件（相对路径）；entry_class=入口类名；task_type 从 "
        f"{json.dumps(list(TASK_TYPES), ensure_ascii=False)} 中选；"
        'input_spec 填入口模型输入（shape 数组 + dtype），如 {"shape":[1,3,32,32],"dtype":"float32"}；'
        "root_id 指向入口类对应的节点。\n"
        "2. 每个 nn.Module 子类实例一个节点。入口模型实例为根节点（kind=module，parent_id=null，module_path=\"\"）。\n"
        "3. 自定义 nn.Module 子类（如 BasicBlock）必须作为 kind=module 节点，其内部子模块作为子节点"
        "（parent_id 指向它），逐层展开到叶子；class_name 用源码类名。\n"
        "4. 叶子层为 torch.nn 内置层：kind=leaf，class_name 用 nn.X 形式（nn.Conv2d、nn.BatchNorm2d、"
        "nn.ReLU、nn.Dropout 等）。params 为该层构造参数（PyTorch 构造器关键字，如 "
        '{"in_channels":64,"out_channels":128,"kernel_size":3}），必须从源码读出真实值，不要臆造；'
        "读不出就留空并置 uncertain=true（后续可人工补参数）。\n"
        "5. nn.Sequential 为 kind=container、class_name=\"nn.Sequential\"，其成员按顺序作为子节点。"
        "nn.ModuleList 展开为父模块的直接子节点（不建 container）。\n"
        "6. forward 中的函数式操作建 kind=op 节点：class_name 用小写名（白名单: add/sub/mul/div/"
        "matmul/bmm/cat/relu/sigmoid/tanh/softmax/flatten/mean/max/min/sum/view/reshape/permute）；"
        "白名单外的操作填 code_hint 内联表达式模板，输入变量用 {inputs} 占位。op 是叶子（无子节点）；"
        "op 可有多条入边（如残差相加），非 op 节点最多 1 条入边。\n"
        "7. 数据流用 edges 表达（from/to 为节点 id）。同层节点按 forward 执行顺序连边；残差/跳跃连接"
        "直接连到汇合 op（如 add）。Sequential 内部成员之间不连边（顺序由声明序决定），Sequential 整体"
        "与外部节点的边连在 container 节点上。\n"
        "8. 节点 id 全局唯一，须为合法 Python 标识符（[A-Za-z_][A-Za-z0-9_]*，不能含点/连字符）；"
        "建议直接用源码中的属性名（如 conv1、bn1、layer1），重名时加前缀区分。\n"
        "9. module_path 填该实例在 named_modules() 中的完整路径（如 layer1.0.conv1、downsample.0），"
        "用于形状追踪与验证对齐；根节点 module_path=\"\"。\n"
        "10. input_shape/output_shape 能静态推出就填数组（元素为整数），推不出填 null。拿不准的节点置 uncertain=true。\n"
        "11. 输出必须是单层 JSON 对象（不要用 {\"ir\":...} 包裹），顶层键：source_file、entry_class、"
        "task_type、input_spec、root_id、nodes、edges。\n"
        '示例（片段）：{"root_id":"net","nodes":[{"id":"net","kind":"module","class_name":"Net",'
        '"parent_id":null,"module_path":""},{"id":"conv1","kind":"leaf","class_name":"nn.Conv2d",'
        '"parent_id":"net","module_path":"conv1","params":{"in_channels":3,"out_channels":16,"kernel_size":3}},'
        '{"id":"skip_add","kind":"op","class_name":"add","parent_id":"net"}],'
        '"edges":[{"from":"conv1","to":"skip_add"}]}'
    )
    return "".join(parts)


async def _run_decompose(params: dict, task_id: str) -> None:
    project_id = params["project_id"]
    project = _require_original(project_id)
    ws = _ws(project)
    source = ws / "source"
    report_path = ws / "reports" / "structure_report.json"
    if not report_path.exists():
        raise RuntimeError("尚未完成代码结构分析：请先 POST /api/projects/{id}/analyze")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    hierarchy = report.get("module_hierarchy") or []
    if not hierarchy:
        raise RuntimeError("结构分析报告未识别到 nn.Module 子类，无法拆解")

    started = _now()
    entry_class = params.get("entry_class")
    try:
        result = await agent_service.run_sync(
            _decompose_prompt(hierarchy, entry_class),
            cwd=str(source),
            output_schema=ir_schema.IR_SCHEMA,
            max_turns=60,
            timeout_s=DECOMPOSE_AGENT_TIMEOUT_S,
        )
    except Exception as e:  # noqa: BLE001 —— agent 失败记 run_record 供检索（先例 4.3）
        knowledge_service.record_run({
            "project_id": project_id, "task_id": task_id, "run_type": "decompose",
            "command": "agent parse (structure → IR)",
            "status": "failed", "error": str(e),
            "started_at": started, "finished_at": _now(),
        })
        raise
    ir = _as_ir(result.get("structured_output"))
    if ir is None:
        raise RuntimeError("agent 未产出有效 IR 结构（structured_output 缺失）")
    errors = validate_ir(ir)
    if errors:
        raise RuntimeError("IR 结构校验失败: " + "；".join(errors))
    ir["schema_version"] = SCHEMA_VERSION
    ir["project_id"] = project_id
    ir_path = ws / "reports" / "ir.json"
    ir_path.write_text(json.dumps(ir, ensure_ascii=False, indent=2), encoding="utf-8")

    knowledge_service.record_run({
        "project_id": project_id, "task_id": task_id, "run_type": "decompose",
        "command": "agent parse (structure → IR)",
        "params": {"entry_class": ir.get("entry_class"), "source_file": ir.get("source_file")},
        "status": "success",
        "metrics": {
            "nodes": len(ir["nodes"]),
            "edges": len(ir["edges"]),
            "uncertain": sum(1 for n in ir["nodes"] if n.get("uncertain")),
        },
        "artifact_path": str(ir_path),
        "started_at": started, "finished_at": _now(),
    })
    task_manager.update_progress(task_id, {
        "entry_class": ir.get("entry_class"),
        "nodes": len(ir["nodes"]),
        "edges": len(ir["edges"]),
        "ir_path": str(ir_path),
    })


# --------------------------- 6.2 形状追踪 ---------------------------

def _merge_shapes(ir: dict, shapes: dict) -> int:
    """把 hook 捕获的形状回填 IR 的缺失项（ir_hash 不含形状，回填不产生 stale）。"""
    filled = 0
    for n in ir["nodes"]:
        mp = n.get("module_path")
        if mp is None:
            if n["id"] == ir["root_id"]:
                mp = ""
            else:
                continue
        rec = shapes.get(mp)
        if not rec:
            continue
        if n.get("input_shape") is None and rec.get("input_shape"):
            n["input_shape"] = rec["input_shape"]
            filled += 1
        if n.get("output_shape") is None and rec.get("output_shape"):
            n["output_shape"] = rec["output_shape"]
            filled += 1
    return filled


async def _run_trace(params: dict, task_id: str) -> None:
    project_id = params["project_id"]
    project = _require_original(project_id)
    ws = _ws(project)
    source = ws / "source"
    ir = _read_ir(project)
    if ir is None:
        raise RuntimeError("尚未拆解：请先 POST /api/projects/{id}/decompose")
    python = analysis_service._project_python(ws)
    if python is None:
        raise RuntimeError("项目环境未就绪：未找到独立环境解释器，请先完成环境创建（模块一 env）")

    run_dir = ws / "runs" / "decompose" / task_id
    run_dir.mkdir(parents=True, exist_ok=True)
    out_json = run_dir / "shapes.json"
    ir_path = ws / "reports" / "ir.json"
    started = _now()
    command = f"{python} {TRACE_SCRIPT} <source> {ir_path.name} {out_json.name}"
    try:
        proc = await asyncio.to_thread(
            subprocess.run,
            [python, str(TRACE_SCRIPT), str(source), str(ir_path), str(out_json)],
            cwd=str(run_dir), capture_output=True, text=True,
            timeout=DECOMPOSE_TRACE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        knowledge_service.record_run({
            "project_id": project_id, "task_id": task_id, "run_type": "decompose_trace",
            "command": command, "status": "failed",
            "error": f"形状追踪超时（>{DECOMPOSE_TRACE_TIMEOUT_S}s）",
            "started_at": started, "finished_at": _now(),
        })
        raise RuntimeError("形状追踪超时")
    log = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        knowledge_service.record_run({
            "project_id": project_id, "task_id": task_id, "run_type": "decompose_trace",
            "command": command, "status": "failed", "error": log[-2000:],
            "log_path": None, "started_at": started, "finished_at": _now(),
        })
        raise RuntimeError("形状追踪失败: " + log[-1500:])

    shapes = json.loads(out_json.read_text(encoding="utf-8"))
    filled = _merge_shapes(ir, shapes)
    _write_ir(project, ir)
    knowledge_service.record_run({
        "project_id": project_id, "task_id": task_id, "run_type": "decompose_trace",
        "command": command, "status": "success",
        "metrics": {"captured_paths": len(shapes), "filled": filled},
        "artifact_path": str(out_json),
        "started_at": started, "finished_at": _now(),
    })
    task_manager.update_progress(task_id, {"captured_paths": len(shapes), "filled": filled})


# --------------------------- 6.4 两步验证 ---------------------------

def _seeds() -> list[int]:
    return [int(s) for s in DECOMPOSE_NUM_SEEDS.split(",") if s.strip()]


async def _run_verify(params: dict, task_id: str) -> None:
    project_id = params["project_id"]
    project = _require_original(project_id)
    ws = _ws(project)
    source = ws / "source"
    ir = _read_ir(project)
    if ir is None:
        raise RuntimeError("尚未拆解：请先 POST /api/projects/{id}/decompose")
    python = analysis_service._project_python(ws)
    if python is None:
        raise RuntimeError("项目环境未就绪：未找到独立环境解释器，请先完成环境创建（模块一 env）")

    # 再生成失败（IR 不完整）→ 任务失败，报缺失项
    try:
        code = ir_codegen.generate(ir)
    except IrIncompleteError as e:
        raise RuntimeError(f"IR 不完整，无法再生成代码: {e}")

    run_dir = ws / "runs" / "decompose" / task_id
    run_dir.mkdir(parents=True, exist_ok=True)
    regen_path = run_dir / "regenerated.py"
    regen_path.write_text(code, encoding="utf-8")
    out_json = run_dir / "verification_result.json"
    ir_path = ws / "reports" / "ir.json"
    started = _now()
    command = (
        f"{python} {VERIFY_SCRIPT} <source> {ir_path.name} regenerated.py {out_json.name} "
        f"<seeds> <rtol> <atol>"
    )
    seeds = _seeds()
    try:
        proc = await asyncio.to_thread(
            subprocess.run,
            [python, str(VERIFY_SCRIPT), str(source), str(ir_path), str(regen_path),
             str(out_json), DECOMPOSE_NUM_SEEDS, str(DECOMPOSE_NUM_RTOL), str(DECOMPOSE_NUM_ATOL)],
            cwd=str(run_dir), capture_output=True, text=True,
            timeout=DECOMPOSE_VERIFY_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        knowledge_service.record_run({
            "project_id": project_id, "task_id": task_id, "run_type": "decompose_verify",
            "command": command, "status": "failed",
            "error": f"验证运行超时（>{DECOMPOSE_VERIFY_TIMEOUT_S}s）",
            "started_at": started, "finished_at": _now(),
        })
        raise RuntimeError("验证运行超时")
    if proc.returncode != 0:
        log = (proc.stdout or "") + (proc.stderr or "")
        knowledge_service.record_run({
            "project_id": project_id, "task_id": task_id, "run_type": "decompose_verify",
            "command": command, "status": "failed", "error": log[-2000:],
            "started_at": started, "finished_at": _now(),
        })
        raise RuntimeError("验证脚本异常: " + log[-1500:])

    result = json.loads(out_json.read_text(encoding="utf-8"))
    verification = {
        "verified_at": _now(),
        "ir_hash": ir_hash(ir),
        "seeds": seeds,
        "tolerance": {"rtol": DECOMPOSE_NUM_RTOL, "atol": DECOMPOSE_NUM_ATOL},
        **result,
    }
    verification_path = ws / "reports" / "verification.json"
    verification_path.write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")

    overall = verification["overall"]
    # 比对不过：任务 success、run_record failed（可检索供 agent 改进）
    knowledge_service.record_run({
        "project_id": project_id, "task_id": task_id, "run_type": "decompose_verify",
        "command": command,
        "params": {"seeds": seeds, "rtol": DECOMPOSE_NUM_RTOL, "atol": DECOMPOSE_NUM_ATOL},
        "status": "success" if overall == "passed" else "failed",
        "error": verification.get("failure_reason"),
        "metrics": {
            "overall": overall,
            "max_rel_err": max((r["max_rel_err"] for r in verification["numeric"]["per_seed"]), default=None),
            "max_abs_err": max((r["max_abs_err"] for r in verification["numeric"]["per_seed"]), default=None),
        },
        "artifact_path": str(verification_path),
        "started_at": started, "finished_at": _now(),
    })
    task_manager.update_progress(task_id, {
        "overall": overall,
        "structure_passed": verification["structure"]["passed"],
        "numeric_passed": verification["numeric"]["passed"],
        "failure_reason": verification.get("failure_reason"),
    })


# --------------------------- 6.5 模块入库 ---------------------------

def _module_signature(ir: dict) -> str:
    """结构签名（实施约定）：入口类+input_spec+节点(类名,排序参数)+边(类对)，不含 id/位置/形状。"""
    node_map = nodes_by_id(ir)

    def _node_sig(n):
        params = dict(sorted((n.get("params") or {}).items(), key=lambda kv: str(kv[0])))
        return [n.get("kind"), ir_schema.normalize_class_name(n.get("class_name") or ""), params]

    def _edge_sig(e):
        return [
            ir_schema.normalize_class_name(node_map[e["from"]].get("class_name") or ""),
            ir_schema.normalize_class_name(node_map[e["to"]].get("class_name") or ""),
        ]

    parts = [
        ir["entry_class"],
        ir.get("input_spec") or {},
        [_node_sig(n) for n in ir["nodes"]],
        sorted(_edge_sig(e) for e in ir["edges"]),
    ]
    blob = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


async def _run_ingest(params: dict, task_id: str) -> None:
    project_id = params["project_id"]
    project = _require_original(project_id)
    ws = _ws(project)
    ir = _read_ir(project)
    if ir is None:
        raise RuntimeError("尚未拆解：请先 POST /api/projects/{id}/decompose")
    verification = get_verification(project_id)
    if verification is None:
        raise RuntimeError("尚未验证：请先 POST /api/projects/{id}/decompose/verify")
    if verification.get("overall") != "passed":
        raise RuntimeError("验证未通过，不能入库；请调整 IR 后重新验证")
    if verification.get("ir_hash") != ir_hash(ir):
        raise RuntimeError("IR 已修改（调参）但未重新验证，验证结果已过期；请重新执行 verify")

    started = _now()
    code = ir_codegen.generate(ir)  # 验证已通过，此处不应再缺项；缺则任务失败
    module_id = "mod_" + _module_signature(ir)
    module_version = knowledge_service.next_module_version(module_id)

    # 结构化项目（画布可编辑，graph.json 落工作区根）
    structured_id = project_manager.create_project(
        "structured", source="decompose", name=ir["entry_class"], parent_project_id=project_id
    )
    graph = ir_graphir.ir_to_graphir(ir)
    graph_path = _ws(project_manager.get_project(structured_id)) / "graph.json"
    graph_path.write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")
    project_manager.update_status(structured_id, "ready")

    # 模块包：临时目录 → DB 事务 → rename 正式目录（实施约定）
    root = nodes_by_id(ir)[ir["root_id"]]
    package_dir = MODULES_DIR / module_id / module_version
    tmp_dir = MODULES_DIR / f".tmp_{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True)
    description = (
        f"拆解自项目 {project_id} 的标准化模块（入口类 {ir['entry_class']}，"
        f"任务类型 {ir.get('task_type')}），经两步验证通过"
    )
    module_json = {
        "module_id": module_id,
        "module_version": module_version,
        "name": ir["entry_class"],
        "description": description,
        "source_project_id": project_id,
        "source_paper_id": None,  # 实施约定：本阶段 project 表无论文关联列，置 null
        "task_type": ir.get("task_type"),
        "input_spec": ir.get("input_spec") or {},
        "output_spec": {"shape": root.get("output_shape")} if root.get("output_shape") else None,
        "params_schema": root.get("params") or {},
        "tags": [t for t in (ir.get("task_type"), "decompose") if t],
        "verification": verification,
        "saved_module_compat": {
            "id": f"{module_id}:{module_version}",
            "name": ir["entry_class"],
            "version": module_version,
            "description": description,
            "graph": graph,
            "handles": {"inputs": ["in"], "outputs": ["out"]},
            "graph_ref": structured_id,
            "createdAt": _now(),
            "updatedAt": _now(),
        },
        "path": str(package_dir / "module.json"),
    }
    (tmp_dir / "module.json").write_text(
        json.dumps(module_json, ensure_ascii=False, indent=2), encoding="utf-8")
    (tmp_dir / "model.py").write_text(code, encoding="utf-8")
    (tmp_dir / "ir.json").write_text(json.dumps(ir, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        knowledge_service.record_module(module_json)
    except Exception:  # noqa: BLE001
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    try:
        tmp_dir.rename(package_dir)
    except OSError:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    knowledge_service.record_run({
        "project_id": project_id, "task_id": task_id, "run_type": "module_ingest",
        "command": "module package → MODULES_DIR",
        "status": "success",
        "metrics": {
            "module_id": module_id,
            "module_version": module_version,
            "structured_project_id": structured_id,
        },
        "artifact_path": str(package_dir / "module.json"),
        "started_at": started, "finished_at": _now(),
    })
    task_manager.update_progress(task_id, {
        "module_id": module_id,
        "module_version": module_version,
        "structured_project_id": structured_id,
        "path": str(package_dir),
    })


def register() -> None:
    task_manager.register_handler(TASK_DECOMPOSE, _run_decompose)
    task_manager.register_handler(TASK_TRACE, _run_trace)
    task_manager.register_handler(TASK_VERIFY, _run_verify)
    task_manager.register_handler(TASK_INGEST, _run_ingest)
