// utils/irAdapter.ts — IR ↔ 画布 GraphIR 转换（模块四 B2/B3）。
// irToGraphIR 与后端 ir_graphir.py 同构（布局/句柄/skip 判定/display 字段对齐），
// 后端入库时同样产出一份 graph.json，前端只需对打开的画布做 graphIRToFlow 还原。

import type { Edge, Node } from "@xyflow/react";
import type { IrEdge, IrGraph, IrNode } from "../api/client";
import type { GraphEdge, GraphHandle, GraphIR, GraphNode } from "../types/graph";
import { applyGraphIR } from "./graphIR";

const GRAPH_VERSION = 2;

// 顶层布局参数（与后端一致）
const LAYER_DX = 320;
const LAYER_DY = 150;
// 父节点内子节点网格参数
const GRID_COLS = 2;
const GRID_DX = 240;
const GRID_DY = 100;
const GRID_ORIGIN_X = 10;
const GRID_ORIGIN_Y = 30;
const PARENT_WIDTH = 500;
const PARENT_PAD_BOTTOM = 20;

const inEdgesOf = (ir: IrGraph, id: string): IrEdge[] => ir.edges.filter(e => e.to === id);
const outEdgesOf = (ir: IrGraph, id: string): IrEdge[] => ir.edges.filter(e => e.from === id);
const childrenOf = (ir: IrGraph, id: string): IrNode[] => ir.nodes.filter(n => n.parent_id === id);

function shapeStr(shape?: number[] | null): string | undefined {
    return shape ? `shape: [${shape.join(",")}]` : undefined;
}

/** display 与前端 formatDisplay/后端 _display 对齐：title=类名，params=参数 JSON，shape=形状文本。 */
function displayOf(n: IrNode) {
    const params = n.params && Object.keys(n.params).length ? JSON.stringify(n.params) : undefined;
    const shape = n.output_shape ?? n.input_shape;
    return { title: n.class_name || n.id, params, shape: shapeStr(shape) };
}

/** 单入/单出用裸 "in"/"out"，多入/多出用 "in0".."inN"（buildGraphIR 约定）。 */
function handleIds(nIns: number, nOuts: number): [string[], string[]] {
    const ins = nIns <= 1 ? ["in"] : Array.from({ length: nIns }, (_, i) => `in${i}`);
    const outs = nOuts <= 1 ? ["out"] : Array.from({ length: nOuts }, (_, i) => `out${i}`);
    return [ins, outs];
}

function handlesOf(ins: string[], outs: string[]): GraphHandle[] {
    return [
        ...ins.map((id, k) => ({ id, kind: "input" as const, order: k })),
        ...outs.map((id, k) => ({ id, kind: "output" as const, order: k })),
    ];
}

/** 最长路径分层（按节点声明序单趟计算，与后端 _layers 语义一致）。 */
function layersOf(ir: IrGraph): Map<string, number> {
    const layers = new Map<string, number>();
    for (const n of ir.nodes) {
        let m = 0;
        for (const e of inEdgesOf(ir, n.id)) m = Math.max(m, layers.get(e.from) ?? 0);
        layers.set(n.id, 1 + m);
    }
    return layers;
}

/** 确定性布局：顶层按层 x / 层内声明序 y；子节点相对父节点两列网格（与后端 _layout 一致）。 */
function layoutOf(ir: IrGraph, layers: Map<string, number>): Map<string, { x: number; y: number }> {
    const positions = new Map<string, { x: number; y: number }>();
    const perLayer = new Map<number, string[]>();
    for (const n of ir.nodes) {
        if (n.parent_id) continue;
        const layer = layers.get(n.id) ?? 0;
        const arr = perLayer.get(layer);
        if (arr) arr.push(n.id);
        else perLayer.set(layer, [n.id]);
    }
    for (const [layer, ids] of [...perLayer.entries()].sort((a, b) => a[0] - b[0])) {
        ids.forEach((nid, i) => positions.set(nid, { x: layer * LAYER_DX, y: i * LAYER_DY }));
    }
    for (const n of ir.nodes) {
        childrenOf(ir, n.id).forEach((c, i) =>
            positions.set(c.id, {
                x: GRID_ORIGIN_X + (i % GRID_COLS) * GRID_DX,
                y: GRID_ORIGIN_Y + Math.floor(i / GRID_COLS) * GRID_DY,
            })
        );
    }
    return positions;
}

/** 边视觉类别（与后端 _edge_kind 一致）：多输入目标的非首条入边、或同父跳过相邻兄弟 → skip。 */
function edgeKindOf(ir: IrGraph, e: IrEdge): GraphEdge["kind"] {
    const byId = new Map(ir.nodes.map(n => [n.id, n]));
    const ins = inEdgesOf(ir, e.to);
    if (ins.length > 1 && ins[0].from !== e.from) return "skip";
    const src = byId.get(e.from);
    const tgt = byId.get(e.to);
    const parent = src?.parent_id;
    if (parent && parent === tgt?.parent_id) {
        const sibs = childrenOf(ir, parent).map(c => c.id);
        const si = sibs.indexOf(e.from);
        const ti = sibs.indexOf(e.to);
        if (si >= 0 && ti >= 0 && ti !== si + 1) return "skip";
    }
    return "data";
}

function depthOf(ir: IrGraph, id: string): number {
    const byId = new Map(ir.nodes.map(n => [n.id, n]));
    let d = 0;
    let cur = byId.get(id);
    while (cur?.parent_id) {
        d++;
        cur = byId.get(cur.parent_id);
    }
    return d;
}

/** IR → GraphIR v2 快照（父节点排在子节点前，避免画布加载时子节点脱离）。 */
export function irToGraphIR(ir: IrGraph): GraphIR {
    const layers = layersOf(ir);
    const positions = layoutOf(ir, layers);
    const nIns = new Map<string, number>();
    const nOuts = new Map<string, number>();
    for (const n of ir.nodes) {
        nIns.set(n.id, inEdgesOf(ir, n.id).length);
        nOuts.set(n.id, outEdgesOf(ir, n.id).length);
    }

    const nodes: GraphNode[] = [...ir.nodes]
        .sort((a, b) => depthOf(ir, a.id) - depthOf(ir, b.id))
        .map(n => {
            const [ins, outs] = handleIds(nIns.get(n.id) ?? 0, nOuts.get(n.id) ?? 0);
            const data: Record<string, unknown> = {
                kind: n.kind,
                class_name: n.class_name,
                module_path: n.module_path,
                params: n.params || {},
                ir_id: n.id,
                input_shape: n.input_shape,
                output_shape: n.output_shape,
                __shape: n.output_shape ?? n.input_shape,
            };
            const kids = childrenOf(ir, n.id);
            if (kids.length) {
                const rows = Math.ceil(kids.length / GRID_COLS);
                data.layout_hint = {
                    width: PARENT_WIDTH,
                    height: GRID_ORIGIN_Y + rows * GRID_DY + PARENT_PAD_BOTTOM,
                };
            }
            const parentId = n.parent_id || undefined;
            return {
                id: n.id,
                type: "ir",
                label: n.class_name || n.id,
                display: displayOf(n),
                handles: handlesOf(ins, outs),
                position: positions.get(n.id),
                parentId,
                extent: parentId ? "parent" : undefined,
                data,
            };
        });

    const srcCount = new Map<string, number>();
    const tgtCount = new Map<string, number>();
    const edges: GraphEdge[] = ir.edges.map((e, i) => {
        const [, srcOuts] = handleIds(nIns.get(e.from) ?? 0, nOuts.get(e.from) ?? 0);
        const [tgtIns] = handleIds(nIns.get(e.to) ?? 0, nOuts.get(e.to) ?? 0);
        const si = srcCount.get(e.from) ?? 0;
        srcCount.set(e.from, si + 1);
        const ti = tgtCount.get(e.to) ?? 0;
        tgtCount.set(e.to, ti + 1);
        return {
            id: `e${i}`,
            source: e.from,
            target: e.to,
            sourceHandle: srcOuts[Math.min(si, srcOuts.length - 1)],
            targetHandle: tgtIns[Math.min(ti, tgtIns.length - 1)],
            kind: edgeKindOf(ir, e),
            data: e.tensor_shape ? { tensor_shape: e.tensor_shape } : {},
        };
    });

    return { version: GRAPH_VERSION, createdAt: new Date().toISOString(), nodes, edges };
}

/** GraphIR → React Flow：applyGraphIR 还原 + 把 handles 注入 data（IrNode 据此动态渲染句柄）。 */
export function graphIRToFlow(graph: GraphIR): { nodes: Node[]; edges: Edge[] } {
    const flow = applyGraphIR(graph);
    const byId = new Map(graph.nodes.map(n => [n.id, n]));
    return {
        nodes: flow.nodes.map(n => ({
            ...n,
            data: { ...(n.data || {}), handles: byId.get(n.id)?.handles ?? [] },
        })),
        edges: flow.edges,
    };
}
