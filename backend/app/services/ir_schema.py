"""模块四 IR：schema、校验与规范化（模块详细设计 6.1）。

IR 是拆解模块的核心数据结构（思想与基底 graphIR 一致，7.1）。
层级树由 parent_id 隐式表达 + root_id 标入口，不双写显式树（实施约定）。

校验分两层：validate_ir 为结构硬错误（decompose 闸门）；incomplete_ir 为
再生成阻断项（regenerate/verify/ingest 闸门，6.3「IR 不完整 → 拒绝生成」）。
"""
import hashlib
import json
import re
from typing import Optional

SCHEMA_VERSION = "1.0"

# 节点 id 将用作再生成代码的类名/属性名（Decomp_{id}、self.{id}），必须为合法标识符
_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

KINDS = ("module", "leaf", "container", "op")

TASK_TYPES = (
    "classification", "regression", "segmentation", "detection",
    "generation", "embedding", "other",
)

# op 节点无需 code_hint 即可再生成的已知类名（与 ir_codegen._OP_TEMPLATES 保持一致）
OP_WHITELIST = frozenset({
    "add", "sub", "mul", "div", "matmul", "bmm", "cat",
    "relu", "sigmoid", "tanh", "softmax",
    "flatten", "mean", "max", "min", "sum",
    "view", "reshape", "permute",
})

IR_SCHEMA = {
    "type": "object",
    "properties": {
        "source_file": {"type": "string", "description": "模型定义文件（相对 source 根的路径）"},
        "entry_class": {"type": "string", "description": "入口 nn.Module 子类名"},
        "task_type": {"type": "string", "enum": list(TASK_TYPES)},
        "input_spec": {
            "type": "object",
            "properties": {
                "shape": {"type": "array", "items": {"type": "integer"}},
                "dtype": {"type": "string"},
            },
        },
        "root_id": {"type": "string"},
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "kind": {"type": "string", "enum": list(KINDS)},
                    "class_name": {"type": "string"},
                    "module_file": {"type": "string"},
                    "module_path": {"type": "string", "description": "named_modules 路径，如 layer1.0.conv1"},
                    "params": {"type": "object"},
                    "parent_id": {"type": ["string", "null"]},
                    "input_shape": {"type": ["array", "null"], "items": {"type": "integer"}},
                    "output_shape": {"type": ["array", "null"], "items": {"type": "integer"}},
                    "code_hint": {"type": ["string", "null"], "description": "op 节点内联表达式模板，输入用 {inputs} 占位"},
                    "uncertain": {"type": "boolean"},
                },
                "required": ["id", "kind", "class_name"],
            },
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                    "tensor_shape": {"type": ["array", "null"], "items": {"type": "integer"}},
                },
                "required": ["from", "to"],
            },
        },
    },
    "required": ["source_file", "entry_class", "task_type", "root_id", "nodes", "edges"],
}

_IR_WRAPPER_KEYS = ("ir", "result", "model_ir")


def _as_ir(structured) -> Optional[dict]:
    """规整 agent 解析输出（形状漂移兼容，先例 paper_service._as_items）。

    兼容：裸 IR 对象、{"ir": {...}}/{"result": {...}} 包装、以及含 nodes 键的列表元素。
    """
    if isinstance(structured, dict):
        if "nodes" in structured and "edges" in structured:
            return structured
        for key in _IR_WRAPPER_KEYS:
            val = structured.get(key)
            if isinstance(val, dict):
                return _as_ir(val)
    if isinstance(structured, list):
        for item in structured:
            if isinstance(item, dict):
                found = _as_ir(item)
                if found:
                    return found
    return None


def nodes_by_id(ir: dict) -> dict[str, dict]:
    return {n["id"]: n for n in ir.get("nodes", [])}


def children_of(ir: dict, node_id: str) -> list[dict]:
    """某节点的子节点（parent_id 归属，保持 IR 原始顺序）。"""
    return [n for n in ir.get("nodes", []) if n.get("parent_id") == node_id]


def in_edges(ir: dict, node_id: str) -> list[dict]:
    """指向某节点的边（保持 edges 声明顺序，多输入的顺序语义由声明序决定）。"""
    return [e for e in ir.get("edges", []) if e["to"] == node_id]


def out_edges(ir: dict, node_id: str) -> list[dict]:
    return [e for e in ir.get("edges", []) if e["from"] == node_id]


def normalize_class_name(name: str) -> str:
    """归一化类名：去 torch. 前缀（leaf/container 另去 nn. 前缀）。"""
    return name[len("torch."):] if name.startswith("torch.") else name


def validate_ir(ir: dict) -> list[str]:
    """结构硬错误校验（decompose 闸门），返回错误清单，空 = 通过。"""
    errors: list[str] = []
    nodes = ir.get("nodes") or []
    edges = ir.get("edges") or []

    if not nodes:
        return ["nodes 为空"]
    ids = [n.get("id") for n in nodes]
    if len(set(ids)) != len(ids):
        errors.append("节点 id 不唯一")
    id_set = set(ids)

    root_id = ir.get("root_id")
    if root_id not in id_set:
        errors.append(f"root_id 不存在: {root_id}")
    if not ir.get("entry_class"):
        errors.append("entry_class 缺失")

    for n in nodes:
        nid = n.get("id", "?")
        if not _ID_RE.match(str(nid)):
            errors.append(f"节点 id 非法（须为 Python 标识符，再生成代码将用作类名/变量名）: {nid}")
        if n.get("kind") not in KINDS:
            errors.append(f"节点 {nid} kind 非法: {n.get('kind')}")
            continue
        pid = n.get("parent_id")
        if pid is not None and pid not in id_set:
            errors.append(f"节点 {nid} 的 parent_id 不存在: {pid}")

        if n["kind"] == "leaf":
            cls = normalize_class_name(n.get("class_name") or "")
            if not cls.startswith("nn."):
                errors.append(f"leaf 节点 {nid} class_name 应为 nn.* 形式: {n.get('class_name')}")
        elif n["kind"] == "container":
            if normalize_class_name(n.get("class_name") or "") != "nn.Sequential":
                errors.append(
                    f"container 节点 {nid} 仅支持 nn.Sequential（nn.ModuleList 请展开为父模块子节点）: {n.get('class_name')}"
                )
            for c in nodes:
                if c.get("parent_id") == nid and c.get("kind") == "op":
                    errors.append(f"container 节点 {nid} 的子节点不能是 op: {c.get('id')}")
        elif n["kind"] == "module":
            if not any(c.get("parent_id") == nid for c in nodes):
                errors.append(f"module 节点 {nid} 无子节点，无法重构其内部结构")
        elif n["kind"] == "op":
            if not in_edges(ir, nid):
                errors.append(f"op 节点 {nid} 至少需要一条入边")

        if len(in_edges(ir, nid)) > 1 and n["kind"] != "op":
            errors.append(f"非 op 节点 {nid} 有多条入边（PyTorch 模块单张量输入，多输入汇合请用 op 节点）")

    # container 子节点的边只能在同 Sequential 子节点之间：进出 Sequential 的
    # 数据流由 container 节点自身的边表达（codegen 将子节点内联进 Sequential，
    # 子节点不可被外部引用）
    for n in nodes:
        if n.get("kind") != "container":
            continue
        cids = {c.get("id") for c in nodes if c.get("parent_id") == n["id"]}
        for cid in cids:
            for e in edges:
                if e["to"] == cid and e["from"] not in cids:
                    errors.append(
                        f"container 子节点 {cid} 的入边必须来自同 Sequential 内"
                        f"（整体输入由 container 节点 {n['id']} 的边表达）: {e}"
                    )
                if e["from"] == cid and e["to"] not in cids:
                    errors.append(
                        f"container 子节点 {cid} 的出边必须指向同 Sequential 内"
                        f"（整体输出由 container 节点 {n['id']} 的边表达）: {e}"
                    )

    for e in edges:
        if e["from"] not in id_set or e["to"] not in id_set:
            errors.append(f"边引用不存在的节点: {e}")
        elif e["from"] == e["to"]:
            errors.append(f"边自环: {e}")

    return errors


def incomplete_ir(ir: dict) -> list[str]:
    """再生成阻断项（6.3「IR 不完整 → 拒绝生成」），返回缺失项清单，空 = 可生成。"""
    items: list[str] = []
    for n in ir.get("nodes") or []:
        nid = n.get("id", "?")
        kind = n.get("kind")
        if kind == "leaf":
            # 参数缺失且 agent 自己标注不确定 → 无法构造层，阻断再生成
            if not (n.get("params") or n.get("code_hint")) and n.get("uncertain"):
                items.append(f"leaf 节点 {nid}（{n.get('class_name')}）参数缺失且标注 uncertain")
        elif kind == "op":
            cls = normalize_class_name(n.get("class_name") or "")
            if cls not in OP_WHITELIST and not n.get("code_hint"):
                items.append(f"op 节点 {nid}（{n.get('class_name')}）既不在白名单也无 code_hint")
    return items


def canonical_ir(ir: dict) -> dict:
    """规范化副本（ir_hash 计算用）：去掉易漂移字段，键排序保证稳定。"""
    out = {k: ir.get(k) for k in
           ("schema_version", "project_id", "source_file", "entry_class", "task_type", "input_spec", "root_id")}
    out["nodes"] = [
        {k: n.get(k) for k in
         ("id", "kind", "class_name", "module_file", "module_path", "params", "parent_id", "code_hint")}
        for n in ir.get("nodes") or []
    ]
    out["edges"] = [
        {k: e.get(k) for k in ("from", "to")} for e in ir.get("edges") or []
    ]
    return out


def ir_hash(ir: dict) -> str:
    """IR 规范化哈希（PUT 调参后变化 → 旧验证变 stale 的判据）。"""
    return hashlib.sha256(
        json.dumps(canonical_ir(ir), ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
