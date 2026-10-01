"""IR → 画布 GraphIR v2（结构化项目 graph.json；模块详细设计 6.2/6.4 与 7.1）。

与前端 utils/graphIR.ts::buildGraphIR 同构（version/句柄/display/data 字段对齐），
前端 applyGraphIR 可直接还原为 React Flow nodes/edges。所有节点 type="ir"（画布
IrNode 通用渲染，前端注册），层级经 parentId 表达（applyGraphIR 对 parentId 节点
强制 extent="parent"）。

布局为确定性分层算法：顶层节点按最长路径分层排 x、层内声明序排 y；子节点相对父
节点两列网格排布，父节点 data.layout_hint 给出建议尺寸（画布渲染器据此定框）。
"""
import json
from datetime import datetime, timezone

from app.services.ir_schema import children_of, in_edges, nodes_by_id, out_edges

GRAPH_VERSION = 2

# 顶层布局参数
_LAYER_DX = 320.0
_LAYER_DY = 150.0
# 父节点内子节点网格参数
_GRID_COLS = 2
_GRID_DX = 240.0
_GRID_DY = 100.0
_GRID_ORIGIN_X = 10.0
_GRID_ORIGIN_Y = 30.0
_PARENT_WIDTH = 500.0
_PARENT_PAD_BOTTOM = 20.0


def _shape_str(shape) -> str:
    return f"shape: [{','.join(str(v) for v in shape)}]"


def _display(node: dict) -> dict:
    """display 与前端 formatDisplay 对齐：title=类名，params=参数 JSON 文本，shape=形状文本。"""
    params = node.get("params")
    shape = node.get("output_shape") or node.get("input_shape")
    return {
        "title": node.get("class_name") or node["id"],
        "params": json.dumps(params, ensure_ascii=False, separators=(",", ":")) if params else None,
        "shape": _shape_str(shape) if shape else None,
    }


def _handle_ids(n_ins: int, n_outs: int) -> tuple[list[str], list[str]]:
    """单入/单出用裸 "in"/"out"，多入/多出用 "in0".."inN"（buildGraphIR 约定）。"""
    ins = ["in"] if n_ins <= 1 else [f"in{i}" for i in range(n_ins)]
    outs = ["out"] if n_outs <= 1 else [f"out{i}" for i in range(n_outs)]
    return ins, outs


def _handles(ins: list[str], outs: list[str]) -> list[dict]:
    hs = [{"id": i, "kind": "input", "order": k} for k, i in enumerate(ins)]
    hs += [{"id": o, "kind": "output", "order": k} for k, o in enumerate(outs)]
    return hs


def _layers(ir: dict) -> dict[str, int]:
    """最长路径分层（边方向），无边节点（根、Sequential 内成员）为 0 层。"""
    layers: dict[str, int] = {}
    for n in ir["nodes"]:
        ins = in_edges(ir, n["id"])
        layers[n["id"]] = 1 + max((layers.get(e["from"], 0) for e in ins), default=0)
    return layers


def _layout(ir: dict) -> dict[str, dict]:
    """确定性布局：返回 {node_id: {x, y}}；子节点坐标相对父节点原点。"""
    layers = _layers(ir)
    positions: dict[str, dict] = {}

    # 顶层节点：x 按层，y 按层内声明序
    tops = [n for n in ir["nodes"] if not n.get("parent_id")]
    per_layer: dict[int, list[str]] = {}
    for n in tops:
        per_layer.setdefault(layers[n["id"]], []).append(n["id"])
    for layer, ids in sorted(per_layer.items()):
        for i, nid in enumerate(ids):
            positions[nid] = {"x": layer * _LAYER_DX, "y": i * _LAYER_DY}

    # 子节点：相对父节点两列网格
    for n in ir["nodes"]:
        children = children_of(ir, n["id"])
        for i, c in enumerate(children):
            positions[c["id"]] = {
                "x": _GRID_ORIGIN_X + (i % _GRID_COLS) * _GRID_DX,
                "y": _GRID_ORIGIN_Y + (i // _GRID_COLS) * _GRID_DY,
            }
    return positions


def _layout_hint(n_children: int) -> dict | None:
    if not n_children:
        return None
    rows = (n_children + _GRID_COLS - 1) // _GRID_COLS
    return {
        "width": _PARENT_WIDTH,
        "height": _GRID_ORIGIN_Y + rows * _GRID_DY + _PARENT_PAD_BOTTOM,
    }


def _edge_kind(ir: dict, e: dict) -> str:
    """边视觉类别：多输入目标的非首条入边、或同父跳过相邻兄弟的边 → skip（残差类）。"""
    node_map = nodes_by_id(ir)
    src, tgt = e["from"], e["to"]
    ins = in_edges(ir, tgt)
    if len(ins) > 1 and ins[0]["from"] != src:
        return "skip"
    parent = node_map[src].get("parent_id")
    if parent and parent == node_map[tgt].get("parent_id"):
        sibs = [c["id"] for c in children_of(ir, parent)]
        if src in sibs and tgt in sibs and sibs.index(tgt) != sibs.index(src) + 1:
            return "skip"
    return "data"


def _depth_of(ir: dict, node_id: str) -> int:
    d, cur = 0, nodes_by_id(ir).get(node_id)
    while cur and cur.get("parent_id"):
        d += 1
        cur = nodes_by_id(ir).get(cur["parent_id"])
    return d


def ir_to_graphir(ir: dict) -> dict:
    """IR → GraphIR v2 快照（父节点排在子节点前，避免画布加载时子节点脱离）。"""
    node_map = nodes_by_id(ir)
    positions = _layout(ir)
    n_ins = {nid: len(in_edges(ir, nid)) for nid in node_map}
    n_outs = {nid: len(out_edges(ir, nid)) for nid in node_map}

    graph_nodes = []
    for n in sorted(ir["nodes"], key=lambda n: _depth_of(ir, n["id"])):
        nid = n["id"]
        ins, outs = _handle_ids(n_ins[nid], n_outs[nid])
        data = {
            "kind": n.get("kind"),
            "class_name": n.get("class_name"),
            "module_path": n.get("module_path"),
            "params": n.get("params") or {},
            "ir_id": nid,
            "input_shape": n.get("input_shape"),
            "output_shape": n.get("output_shape"),
            "__shape": n.get("output_shape") or n.get("input_shape"),
        }
        hint = _layout_hint(len(children_of(ir, nid)))
        if hint:
            data["layout_hint"] = hint
        graph_nodes.append({
            "id": nid,
            "type": "ir",
            "label": n.get("class_name") or nid,
            "display": _display(n),
            "handles": _handles(ins, outs),
            "position": positions[nid],
            "parentId": n.get("parent_id"),
            "extent": "parent" if n.get("parent_id") else None,
            "data": data,
        })

    src_count: dict[str, int] = {}
    tgt_count: dict[str, int] = {}
    graph_edges = []
    for i, e in enumerate(ir["edges"]):
        src, tgt = e["from"], e["to"]
        _, src_outs = _handle_ids(n_ins[src], n_outs[src])
        tgt_ins, _ = _handle_ids(n_ins[tgt], n_outs[tgt])
        si = src_count.get(src, 0)
        src_count[src] = si + 1
        ti = tgt_count.get(tgt, 0)
        tgt_count[tgt] = ti + 1
        data = {"tensor_shape": e["tensor_shape"]} if e.get("tensor_shape") else {}
        graph_edges.append({
            "id": f"e{i}",
            "source": src,
            "target": tgt,
            "sourceHandle": src_outs[min(si, len(src_outs) - 1)],
            "targetHandle": tgt_ins[min(ti, len(tgt_ins) - 1)],
            "kind": _edge_kind(ir, e),
            "data": data,
        })

    return {
        "version": GRAPH_VERSION,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "nodes": graph_nodes,
        "edges": graph_edges,
    }
