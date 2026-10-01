"""模块四代码再生成引擎（模块详细设计 6.3；实施约定：后端 Python 单一确定性实现）。

按 IR 连接关系生成**自包含** PyTorch 代码（继承 nn.Module 的类），不 import 源项目——
入库后脱离原项目必须能独立运行；数值比对同时证明模块包的自包含性。
纯函数、无 IO/无 subprocess，宿主解释器内执行；同 IR 恒产出同代码。

与前端 codeCompile.ts 的关系（实施约定）：IR schema 与画布 GraphIR 不同、移植无收益，
本引擎为唯一实现，前端仅用 CodeViewer 展示返回代码。
"""
import json

from app.services.ir_schema import (
    OP_WHITELIST, children_of, in_edges, incomplete_ir, nodes_by_id, normalize_class_name,
    out_edges, validate_ir,
)

HEADER = (
    "import torch\n"
    "import torch.nn as nn\n"
    "import torch.nn.functional as F\n\n\n"
)


class IrIncompleteError(ValueError):
    """IR 不完整：拒绝生成（6.3 异常与边界），message 携带缺失项清单。"""

    def __init__(self, message):
        if isinstance(message, (list, tuple)):
            message = "；".join(message)
        super().__init__(message)


# 已知 op 类名 → 表达式模板；{inputs} 代输入变量，{dim}/{start_dim}/{end_dim}/{shape}/{dims} 代 params
_OP_TEMPLATES = {
    "add": "torch.add({inputs})",
    "sub": "torch.sub({inputs})",
    "mul": "torch.mul({inputs})",
    "div": "torch.div({inputs})",
    "matmul": "torch.matmul({inputs})",
    "bmm": "torch.bmm({inputs})",
    "cat": "torch.cat([{inputs}], dim={dim})",
    "relu": "torch.relu({inputs})",
    "sigmoid": "torch.sigmoid({inputs})",
    "tanh": "torch.tanh({inputs})",
    "softmax": "torch.softmax({inputs}, dim={dim})",
    "flatten": "torch.flatten({inputs}, start_dim={start_dim}, end_dim={end_dim})",
    "mean": "torch.mean({inputs}, dim={dim})",
    "max": "torch.max({inputs})",
    "min": "torch.min({inputs})",
    "sum": "torch.sum({inputs}, dim={dim})",
    "view": "{inputs}.view({shape})",
    "reshape": "{inputs}.reshape({shape})",
    "permute": "{inputs}.permute({dims})",
}
assert set(_OP_TEMPLATES) == set(OP_WHITELIST), "op 模板与 ir_schema.OP_WHITELIST 不同步"


def _py_value(v) -> str:
    """JSON 值 → Python 字面量（数组转元组，PyTorch 构造参数大多要求 tuple）。"""
    if isinstance(v, list):
        return "(" + ", ".join(_py_value(x) for x in v) + ")" if v else "()"
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k!r}: {_py_value(x)}" for k, x in v.items()) + "}"
    return repr(v)


def _render_kwargs(params: dict) -> str:
    return ", ".join(f"{k}={_py_value(v)}" for k, v in (params or {}).items())


def _leaf_name(class_name: str) -> str:
    """leaf 类名去 nn./torch.nn. 前缀（validate 已保证前缀合法）。"""
    return normalize_class_name(class_name)[len("nn."):]


def _render_op(node: dict, in_vars: list[str]) -> str:
    cls = normalize_class_name(node.get("class_name") or "")
    tpl = _OP_TEMPLATES.get(cls) or node.get("code_hint")
    if not tpl:
        raise IrIncompleteError([f"op 节点 {node['id']}（{cls}）既不在白名单也无 code_hint"])
    if "{inputs}" not in tpl:
        raise IrIncompleteError([f"op 节点 {node['id']} 的 code_hint 必须含 {{inputs}} 占位符: {tpl}"])
    params = node.get("params") or {}

    def _p(key, default):
        v = params.get(key)
        return default if v is None else v

    return (
        tpl.replace("{inputs}", in_vars[0] if len(in_vars) == 1 else ", ".join(in_vars))
        .replace("{dim}", _py_value(_p("dim", 1)))
        .replace("{start_dim}", _py_value(_p("start_dim", 1)))
        .replace("{end_dim}", _py_value(_p("end_dim", -1)))
        .replace("{shape}", _py_value(_p("shape", [])))
        .replace("{dims}", _py_value(_p("dims", [])))
    )


def _depth(ir: dict, node_id: str) -> int:
    d, cur = 0, nodes_by_id(ir).get(node_id)
    while cur and cur.get("parent_id"):
        d += 1
        cur = nodes_by_id(ir).get(cur["parent_id"])
    return d


def _subtree_ids(ir: dict, root_id: str) -> set[str]:
    """某节点子树内全部节点 id（不含根自身）。"""
    out = set()
    for c in children_of(ir, root_id):
        out.add(c["id"])
        out |= _subtree_ids(ir, c["id"])
    return out


def _subtree_order(ir: dict, root_id: str) -> list[dict]:
    """子树顺序：按边拓扑序（隔离节点按声明序），同深度深者在前（子先于父）。"""
    node_map = nodes_by_id(ir)
    ids = _subtree_ids(ir, root_id)
    edge_set = {(e["from"], e["to"]) for e in ir["edges"] if e["from"] in ids and e["to"] in ids}
    indeg = {nid: 0 for nid in ids}
    for _f, t in edge_set:
        indeg[t] += 1
    ready = sorted((nid for nid in ids if indeg[nid] == 0),
                   key=lambda nid: ir["nodes"].index(node_map[nid]))
    order: list[str] = []
    while ready:
        cur = ready.pop(0)
        order.append(cur)
        for nid in ids:
            if (cur, nid) in edge_set:
                indeg[nid] -= 1
                if indeg[nid] == 0:
                    ready.append(nid)
                    ready.sort(key=lambda nid: ir["nodes"].index(node_map[nid]))
    order += [nid for nid in ids if nid not in order]
    return sorted([node_map[nid] for nid in order], key=lambda n: -_depth(ir, n["id"]))


def _inline_init(ir: dict, node: dict) -> str:
    """container 子节点内联实例化（Sequential 参数位置，不单独注册属性——
    同一 module 实例注册两次会导致 named_modules/state_dict 重复，破坏结构比对）。"""
    nid = node["id"]
    if node["kind"] == "leaf":
        return f"nn.{_leaf_name(node['class_name'])}({_render_kwargs(node.get('params') or {})})"
    if node["kind"] == "module":
        return f"Decomp_{nid}()"
    if node["kind"] == "container":
        gc = [c["id"] for c in children_of(ir, nid)]
        return f"nn.Sequential({', '.join(_inline_init(ir, nodes_by_id(ir)[c]) for c in gc)})"
    raise IrIncompleteError([f"container 子节点 {nid} kind 非法: {node['kind']}"])


def _module_class(ir: dict, node: dict) -> str:
    node_map = nodes_by_id(ir)
    subtree = _subtree_ids(ir, node["id"])
    ordered = _subtree_order(ir, node["id"])
    order_index = {n["id"]: i for i, n in enumerate(ordered)}

    # container 子节点内联进 Sequential，不单独实例化/不单独出 forward 行
    container_children = set()
    for n in ordered:
        if n["kind"] == "container":
            container_children |= {c["id"] for c in children_of(ir, n["id"])}

    init_lines, fwd_lines = [], []
    for n in ordered:
        nid = n["id"]
        if nid in container_children:
            continue
        if n["kind"] == "leaf":
            init_lines.append(
                f"        self.{nid} = nn.{_leaf_name(n['class_name'])}({_render_kwargs(n.get('params') or {})})"
            )
        elif n["kind"] == "container":
            gc = [c["id"] for c in children_of(ir, nid)]
            gc.sort(key=lambda cid: order_index.get(cid, 0))
            init_lines.append(
                "        self.{0} = nn.Sequential({1})".format(
                    nid, ", ".join(_inline_init(ir, node_map[c]) for c in gc))
            )
        elif n["kind"] == "module":
            init_lines.append(f"        self.{nid} = Decomp_{nid}()")

    for n in ordered:
        nid = n["id"]
        if nid in container_children:
            continue
        ins = in_edges(ir, nid)
        in_vars = [f"var_{e['from']}" if e["from"] in subtree else "x" for e in ins]
        if n["kind"] == "op":
            fwd_lines.append(f"        var_{nid} = {_render_op(n, in_vars)}")
        else:
            in_var = in_vars[0] if in_vars else "x"
            fwd_lines.append(f"        var_{nid} = self.{nid}({in_var})")

    sinks = [n["id"] for n in ordered
             if n["id"] not in container_children
             and not any(e["to"] in subtree for e in out_edges(ir, n["id"]))]
    if len(sinks) != 1:
        raise IrIncompleteError(
            [f"module 节点 {node['id']} 子树应有唯一输出，实际汇点 {len(sinks)} 个: {sinks}"
             "（多分支输出请用 op 节点汇合）"]
        )

    return (
        f"class Decomp_{node['id']}(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        + "\n".join(init_lines) + "\n\n"
        "    def forward(self, x):\n"
        + "\n".join(fwd_lines) + "\n"
        f"        return var_{sinks[0]}\n"
    )


def generate(ir: dict) -> str:
    """IR → 自包含 PyTorch 代码。IR 结构错误或缺失再生成所需项 → IrIncompleteError。"""
    errors = validate_ir(ir) + incomplete_ir(ir)
    if errors:
        raise IrIncompleteError("；".join(errors))
    node_map = nodes_by_id(ir)
    if node_map[ir["root_id"]]["kind"] != "module":
        raise IrIncompleteError(
            [f"root 节点 {ir['root_id']} 应为 kind=module（当前 {node_map[ir['root_id']]['kind']}）"]
        )
    module_nodes = [n for n in ir["nodes"] if n["kind"] == "module"]
    module_nodes.sort(key=lambda n: -_depth(ir, n["id"]))
    return HEADER + "\n\n".join(_module_class(ir, n) for n in module_nodes) + "\n"
