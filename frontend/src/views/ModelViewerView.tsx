// views/ModelViewerView.tsx — 模型查看器（模块四 B2，模块详细设计 6.1/6.2/6.3/6.4）。
// 操作链：① 结构分析 → ② 拆解 IR → ③ 补形状 → ④ 再生成代码 → ⑤ 两步验证 → ⑥ 入库。
// 左：IR 层级结构图（只读 ReactFlow，parentId 嵌套）；右：参数面板（PUT 调参）/
// 代码（CodeViewer）/ 验证结果；另有数据流图（DiagramView + ELK，GraphIR 投影）。

import { ReactFlow, Background, type Node, type NodeProps } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useCallback, useEffect, useMemo, useState, type CSSProperties, type ReactNode } from "react";
import {
    ApiError,
    getIr,
    getProject,
    getReport,
    getTask,
    listModules,
    listProjects,
    postAnalyze,
    postDecompose,
    postIngestModule,
    postRegenerate,
    postTrace,
    postVerifyDecompose,
    putNodeParams,
    type IrNode,
    type IrResponse,
    type ModuleItem,
    type Project,
    type Task,
    type Verification,
} from "../api/client";
import CodeViewer from "../components/CodeViewer";
import DiagramView from "../components/DiagramView";
import { graphIRToFlow, irToGraphIR } from "../utils/irAdapter";

const KIND_COLORS: Record<string, string> = {
    module: "#3b82f6",
    container: "#8b5cf6",
    leaf: "#10b981",
    op: "#f59e0b",
};
const KIND_LABELS: Record<string, string> = {
    module: "模块",
    container: "容器",
    leaf: "叶子层",
    op: "算子",
};

type TaskKind = "analyze" | "decompose" | "trace" | "verify" | "ingest";

export type ModelViewerViewProps = {
    projectId: string;
    onBack: () => void;
    onOpenCanvas: (structuredProjectId: string) => void;
};

// ---------------------------------------------------------------------------
// 结构图节点（只读）
// ---------------------------------------------------------------------------
function ViewerNode({ data, selected }: NodeProps<Node<Record<string, unknown>>>) {
    const d = (data || {}) as {
        kind?: string;
        class_name?: string;
        label?: string;
        __shape?: number[];
        uncertain?: boolean;
    };
    const color = KIND_COLORS[d.kind ?? "module"] ?? "#64748b";
    const shape = d.__shape && d.__shape.length ? `[${d.__shape.join(", ")}]` : null;
    return (
        <div
            style={{
                background: selected ? "#1e293b" : "#0f172a",
                border: `1.5px solid ${selected ? color : "#334155"}`,
                borderRadius: 6,
                padding: "6px 10px",
                fontSize: 12,
                color: "#e2e8f0",
                cursor: "pointer",
                maxWidth: 220,
            }}
        >
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <span style={{ background: color, color: "#0b1220", fontWeight: 700, fontSize: 10, borderRadius: 999, padding: "1px 6px", whiteSpace: "nowrap" }}>
                    {KIND_LABELS[d.kind ?? ""] ?? d.kind}
                </span>
                <span style={{ fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {d.class_name ?? d.label}
                </span>
                {d.uncertain ? <span title="agent 标注不确定">⚠</span> : null}
            </div>
            {shape && <div style={{ color: "#64748b", fontSize: 10, fontFamily: "monospace" }}>{shape}</div>}
        </div>
    );
}

// ---------------------------------------------------------------------------
// 参数编辑（PUT /ir/nodes/{id}）
// ---------------------------------------------------------------------------
function ParamsEditor({
    node,
    saving,
    onSave,
}: {
    node: IrNode;
    saving: boolean;
    onSave: (params: Record<string, unknown>) => Promise<void>;
}) {
    const original = node.params ?? {};
    const [drafts, setDrafts] = useState<Record<string, string>>(() =>
        Object.fromEntries(
            Object.entries(original).map(([k, v]) => [
                k,
                v == null ? "" : typeof v === "object" ? JSON.stringify(v) : String(v),
            ])
        )
    );
    const [newKey, setNewKey] = useState("");

    const parseValue = (k: string, raw: string): unknown => {
        const orig = original[k];
        if (typeof orig === "number") {
            const num = Number(raw);
            return Number.isNaN(num) ? raw : num;
        }
        if (typeof orig === "boolean") return raw === "true";
        if (orig !== null && typeof orig === "object") {
            try {
                return JSON.parse(raw);
            } catch {
                return raw;
            }
        }
        return raw;
    };

    const handleSave = async () => {
        const params: Record<string, unknown> = {};
        for (const [k, raw] of Object.entries(drafts)) params[k] = parseValue(k, raw);
        await onSave(params);
    };

    const addKey = () => {
        const k = newKey.trim();
        if (k && !(k in drafts)) setDrafts(d => ({ ...d, [k]: "" }));
        setNewKey("");
    };

    const entries = Object.entries(drafts);

    return (
        <div>
            {entries.length === 0 && (
                <div style={{ color: "#64748b", fontSize: 12, marginBottom: 8 }}>无参数</div>
            )}
            {entries.map(([k, raw]) => (
                <div key={k} style={{ display: "flex", gap: 6, alignItems: "center", marginBottom: 6 }}>
                    <span style={{ flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", fontSize: 12, color: "#94a3b8" }} title={k}>
                        {k}
                    </span>
                    <input
                        value={typeof original[k] === "boolean" ? undefined : raw}
                        type={typeof original[k] === "boolean" ? "checkbox" : "text"}
                        checked={typeof original[k] === "boolean" ? raw === "true" : undefined}
                        onChange={e => {
                            const next = typeof original[k] === "boolean" ? String(e.target.checked) : e.target.value;
                            setDrafts(d => ({ ...d, [k]: next }));
                        }}
                        style={{
                            width: 140,
                            border: "1px solid #334155",
                            background: "#0f172a",
                            color: "#e2e8f0",
                            borderRadius: 4,
                            padding: "3px 6px",
                            fontSize: 12,
                            fontFamily: "monospace",
                        }}
                    />
                    <button
                        title="删除参数"
                        onClick={() => setDrafts(d => Object.fromEntries(Object.entries(d).filter(([x]) => x !== k)))}
                        style={{ border: "none", background: "transparent", color: "#64748b", cursor: "pointer", fontSize: 12 }}
                    >
                        ×
                    </button>
                </div>
            ))}
            <div style={{ display: "flex", gap: 6, marginTop: 10 }}>
                <input
                    placeholder="新增参数名"
                    value={newKey}
                    onChange={e => setNewKey(e.target.value)}
                    onKeyDown={e => e.key === "Enter" && addKey()}
                    style={{ ...inputStyle, flex: 1 }}
                />
                <button style={btnStyle} onClick={addKey}>＋</button>
            </div>
            <button
                style={{ ...btnStyle, width: "100%", marginTop: 12 }}
                disabled={saving}
                onClick={() => void handleSave()}
            >
                {saving ? "保存中…" : "保存参数（调参后需重新验证）"}
            </button>
        </div>
    );
}

// ---------------------------------------------------------------------------
// IR 层级树（<details> 折叠）
// ---------------------------------------------------------------------------
function IrTree({
    nodes,
    selectedNodeId,
    onSelect,
}: {
    nodes: IrNode[];
    selectedNodeId: string | null;
    onSelect: (id: string) => void;
}) {
    const childrenOf = (pid: string | null) => nodes.filter(n => (n.parent_id ?? null) === pid);
    const render = (pid: string | null, depth: number): ReactNode =>
        childrenOf(pid).map(n => (
            <div key={n.id}>
                <div
                    onClick={() => onSelect(n.id)}
                    style={{
                        padding: "3px 8px",
                        marginLeft: depth * 14,
                        borderRadius: 4,
                        cursor: "pointer",
                        fontSize: 12,
                        background: selectedNodeId === n.id ? "#1e293b" : "transparent",
                        color: n.uncertain ? "#f59e0b" : "#cbd5e1",
                    }}
                >
                    <span style={{ color: KIND_COLORS[n.kind] ?? "#94a3b8", marginRight: 6 }}>●</span>
                    {n.class_name || n.id}
                </div>
                {childrenOf(n.id).length > 0 && render(n.id, depth + 1)}
            </div>
        ));
    return <div>{render(null, 0)}</div>;
}

// ---------------------------------------------------------------------------
// 主视图
// ---------------------------------------------------------------------------
export default function ModelViewerView({ projectId, onBack, onOpenCanvas }: ModelViewerViewProps) {
    const [project, setProject] = useState<Project | null>(null);
    const [reportOk, setReportOk] = useState<boolean | null>(null);
    const [irResp, setIrResp] = useState<IrResponse | null>(null);
    const [code, setCode] = useState<string | null>(null);
    const [regenError, setRegenError] = useState<{ error: string; missing: string[] } | null>(null);
    const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
    const [tab, setTab] = useState<"params" | "code" | "verify" | "tree">("params");
    const [showDiagram, setShowDiagram] = useState(false);
    const [savingParams, setSavingParams] = useState(false);
    const [task, setTask] = useState<{ id: string; kind: TaskKind; after: () => Promise<void> } | null>(null);
    const [taskInfo, setTaskInfo] = useState<Task | null>(null);
    const [banner, setBanner] = useState<string | null>(null);
    const [flash, setFlash] = useState<string | null>(null);
    const [ingestedModules, setIngestedModules] = useState<ModuleItem[]>([]);
    const [structuredChild, setStructuredChild] = useState<Project | null>(null);
    const [entryClass, setEntryClass] = useState("");

    const ir = irResp?.ir ?? null;
    const verification: Verification | null = irResp?.verification ?? null;
    const vStatus = irResp?.verification_status ?? "none";

    // ---------------- 加载 ----------------
    useEffect(() => {
        void (async () => {
            try {
                setProject(await getProject(projectId));
            } catch (e) {
                setBanner(e instanceof Error ? e.message : String(e));
            }
            try {
                await getReport(projectId);
                setReportOk(true);
            } catch (e) {
                if (e instanceof ApiError && e.status === 404) setReportOk(false);
                // 其他错误不阻塞：拆解任务本身会校验结构报告
            }
            try {
                setIrResp(await getIr(projectId));
            } catch (e) {
                if (e instanceof ApiError && e.status === 404) setIrResp(null);
                else setBanner(`IR 读取失败：${e instanceof Error ? e.message : String(e)}`);
            }
        })();
    }, [projectId]);

    const refreshIr = useCallback(async () => {
        try {
            setIrResp(await getIr(projectId));
        } catch (e) {
            if (e instanceof ApiError && e.status === 404) setIrResp(null);
            else setBanner(`IR 读取失败：${e instanceof Error ? e.message : String(e)}`);
        }
    }, [projectId]);

    // ---------------- 任务轮询 ----------------
    useEffect(() => {
        if (!task) return;
        let cancelled = false;
        const timer = setInterval(async () => {
            let t: Task;
            try {
                t = await getTask(task.id);
            } catch (e) {
                if (!cancelled) {
                    setBanner(`任务查询失败：${e instanceof Error ? e.message : String(e)}`);
                    setTask(null);
                }
                return;
            }
            if (cancelled) return;
            setTaskInfo(t);
            if (["success", "failed", "cancelled"].includes(t.status)) {
                clearInterval(timer);
                const cur = task;
                setTask(null);
                if (t.status === "success") {
                    void cur.after();
                } else {
                    setBanner(t.error || `任务失败（${t.status}）`);
                }
            }
        }, 2000);
        return () => {
            cancelled = true;
            clearInterval(timer);
        };
    }, [task]);

    const runTask = useCallback(
        (kind: TaskKind, starter: () => Promise<{ task_id: string }>, after: () => Promise<void>) => {
            void (async () => {
                setBanner(null);
                try {
                    const { task_id } = await starter();
                    setTaskInfo(null);
                    setTask({ id: task_id, kind, after });
                } catch (e) {
                    setBanner(e instanceof Error ? e.message : String(e));
                }
            })();
        },
        []
    );

    // ---------------- 操作链 ----------------
    const afterAnalyze = async () => {
        setReportOk(true);
    };
    const afterDecompose = async () => {
        await refreshIr();
        setSelectedNodeId(null);
    };
    const afterTrace = async () => {
        await refreshIr();
        setFlash("形状回填完成");
    };
    const afterVerify = async () => {
        await refreshIr();
        setTab("verify");
    };
    const afterIngest = async () => {
        try {
            const mods = await listModules();
            setIngestedModules(mods.filter(m => m.source_project_id === projectId));
        } catch (e) {
            console.warn("模块列表读取失败", e);
        }
        try {
            const structured = await listProjects("structured");
            const child = structured
                .filter(p => p.parent_project_id === projectId)
                .sort((a, b) => b.created_at.localeCompare(a.created_at))[0];
            setStructuredChild(child ?? null);
        } catch (e) {
            console.warn("结构化项目列表读取失败", e);
        }
        setFlash("模块已入库");
    };

    const handleRegenerate = async () => {
        setBanner(null);
        setRegenError(null);
        try {
            const { code: c } = await postRegenerate(projectId);
            setCode(c);
            setTab("code");
        } catch (e) {
            if (e instanceof ApiError && e.status === 400) {
                try {
                    const detail = JSON.parse(e.message) as { error?: string; missing?: string[] };
                    setRegenError({ error: detail.error ?? e.message, missing: detail.missing ?? [] });
                } catch {
                    setRegenError({ error: e.message, missing: [] });
                }
            } else {
                setBanner(e instanceof Error ? e.message : String(e));
            }
        }
    };

    const handleSaveParams = async (params: Record<string, unknown>) => {
        if (!selectedNodeId) return;
        setSavingParams(true);
        try {
            await putNodeParams(projectId, selectedNodeId, params);
            await refreshIr();
            setFlash("参数已保存，验证已失效（stale），入库前需重新验证");
        } catch (e) {
            setBanner(e instanceof Error ? e.message : String(e));
        } finally {
            setSavingParams(false);
        }
    };

    // ---------------- 派生 ----------------
    const missingShapes = useMemo(
        () => (ir ? ir.nodes.filter(n => n.kind !== "op" && (!n.input_shape || !n.output_shape)).length : 0),
        [ir]
    );
    const uncertainCount = useMemo(() => (ir ? ir.nodes.filter(n => n.uncertain).length : 0), [ir]);

    const selectedNode = useMemo(
        () => (ir && selectedNodeId ? ir.nodes.find(n => n.id === selectedNodeId) ?? null : null),
        [ir, selectedNodeId]
    );

    const graph = useMemo(() => (ir ? irToGraphIR(ir) : null), [ir]);
    const flow = useMemo(() => (graph ? graphIRToFlow(graph) : null), [graph]);
    const viewerEdges = useMemo(() => {
        if (!flow || !graph) return [];
        const kindById = new Map(graph.edges.map(e => [e.id, e.kind]));
        // 剥离 "custom" edge 类型（查看器未注册编辑器边组件），skip 边画虚线
        return flow.edges.map(e => ({
            ...e,
            type: undefined as unknown as string,
            style: kindById.get(e.id) === "skip" ? { strokeDasharray: "6 4" } : undefined,
        }));
    }, [flow, graph]);

    const step = (
        key: TaskKind | "regenerate",
        title: string,
        desc: string,
        ready: boolean,
        done: boolean,
        action: () => void,
        hint?: string
    ) => {
        const running = task?.kind === key;
        const disabled = !ready || running || !!task;
        const style: CSSProperties = {
            flex: 1,
            minWidth: 130,
            border: done ? "1px solid #16a34a" : running ? "1px solid #d97706" : "1px solid #334155",
            background: done ? "#0f2d1f" : running ? "#2d1f0f" : disabled ? "#0f172a" : "#0f172a",
            color: disabled ? "#475569" : "#e2e8f0",
            borderRadius: 8,
            padding: "8px 12px",
            cursor: disabled ? "not-allowed" : "pointer",
            fontSize: 12,
            textAlign: "left" as const,
            opacity: disabled ? 0.65 : 1,
        };
        return (
            <button key={key} style={style} onClick={action} disabled={disabled} title={hint ?? desc}>
                <div style={{ fontWeight: 700, marginBottom: 2 }}>
                    {done ? "✓ " : running ? "… " : ""}
                    {title}
                </div>
                <div style={{ fontSize: 11, color: "#94a3b8" }}>{running ? "运行中" : hint ?? desc}</div>
            </button>
        );
    };

    const steps = (
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            {step("analyze", "① 结构分析", "生成 structure_report.json（拆解前置）", true, reportOk === true, () =>
                runTask("analyze", () => postAnalyze(projectId), afterAnalyze)
            )}
            {step("decompose", "② 拆解 IR", "agent 解析模型为 IR", reportOk === true, !!ir, () =>
                runTask("decompose", () => postDecompose(projectId, entryClass.trim() || undefined), afterDecompose),
                reportOk !== true ? "需先完成结构分析" : undefined
            )}
            {step("trace", "③ 补形状", "项目环境跑 hook 回填形状", !!ir, !!ir && missingShapes === 0, () =>
                runTask("trace", () => postTrace(projectId), afterTrace),
                !ir ? "需先拆解" : missingShapes === 0 ? "所有形状已知（可跳过）" : `${missingShapes} 个节点缺形状`
            )}
            {step("regenerate", "④ 再生成", "后端确定性引擎生成代码", !!ir, !!code, () => void handleRegenerate(), !ir ? "需先拆解" : undefined)}
            {step("verify", "⑤ 两步验证", "结构 + 数值比对", !!code, !!verification, () =>
                runTask("verify", () => postVerifyDecompose(projectId), afterVerify),
                !code ? "需先再生成代码" : undefined
            )}
            {step(
                "ingest",
                "⑥ 入库",
                "生成模块包 + 结构化项目",
                !!verification && verification.overall === "passed" && vStatus === "valid",
                !!structuredChild,
                () => runTask("ingest", () => postIngestModule(projectId), afterIngest),
                !verification
                    ? "需先通过两步验证"
                    : verification.overall !== "passed"
                      ? "验证未通过，不可入库"
                      : vStatus === "stale"
                        ? "调参后验证已失效，需重新验证"
                        : undefined
            )}
        </div>
    );

    // ---------------- 渲染 ----------------
    return (
        <div style={{ height: "100vh", display: "flex", flexDirection: "column", background: "#0b1220", color: "#e2e8f0", overflow: "hidden" }}>
            {/* 头部 */}
            <div style={{ display: "flex", alignItems: "center", gap: 14, padding: "10px 18px", borderBottom: "1px solid #1f2937", flexWrap: "wrap" }}>
                <button style={btnStyle} onClick={onBack}>← 返回</button>
                <div>
                    <div style={{ fontWeight: 700, fontSize: 14 }}>
                        {project?.name || "(未命名)"}
                        <span style={{ color: "#64748b", fontFamily: "monospace", fontSize: 11, marginLeft: 8 }}>{projectId}</span>
                    </div>
                    <div style={{ fontSize: 11, color: "#94a3b8" }}>
                        {project?.source ?? ""}
                        {ir ? ` · 入口 ${ir.entry_class ?? "?"} · ${ir.nodes.length} 节点 / ${ir.edges.length} 边 · ${ir.task_type ?? ""}` : ""}
                        {ir && missingShapes > 0 ? ` · ${missingShapes} 缺形状` : ""}
                        {uncertainCount > 0 ? ` · ${uncertainCount} 处不确定` : ""}
                    </div>
                </div>
                <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
                    {ir && (
                        <span style={badge(vStatus === "valid" ? "#16a34a" : vStatus === "stale" ? "#d97706" : "#64748b")}>
                            验证：{vStatus === "none" ? "未验证" : vStatus === "stale" ? "已失效" : "有效"}
                        </span>
                    )}
                    {verification && (
                        <span style={badge(verification.overall === "passed" ? "#16a34a" : "#dc2626")}>
                            比对：{verification.overall === "passed" ? "通过 ✓" : "未通过 ✗"}
                        </span>
                    )}
                </div>
            </div>

            {/* 操作链 */}
            <div style={{ padding: "12px 18px 6px", borderBottom: "1px solid #1f2937" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                    <label style={{ fontSize: 11, color: "#94a3b8" }}>
                        入口类（可选，多个根类歧义时指定）：
                        <input
                            style={{ ...inputStyle, width: 180, marginLeft: 6 }}
                            placeholder="如 ResNet / MLP"
                            value={entryClass}
                            onChange={e => setEntryClass(e.target.value)}
                            disabled={!!ir}
                        />
                    </label>
                    {ir && <span style={{ fontSize: 11, color: "#64748b" }}>IR 已生成，入口 {ir.entry_class}</span>}
                </div>
                {steps}
            </div>

            {/* 任务/提示横幅 */}
            {(task || banner || flash) && (
                <div style={{ padding: "6px 18px 0" }}>
                    {task && taskInfo && (
                        <div style={{ ...infoBanner, background: "#1e293b", borderColor: "#334155" }}>
                            任务 {taskInfo.task_type}：{taskInfo.status}
                            {taskInfo.progress ? ` · ${taskInfo.progress}` : ""}
                        </div>
                    )}
                    {task && !taskInfo && <div style={infoBanner}>任务排队中…</div>}
                    {banner && (
                        <div style={{ ...infoBanner, background: "#3f1d1d", borderColor: "#b91c1c", color: "#fecaca" }}>
                            {banner}
                            <button style={{ ...btnStyle, marginLeft: 10 }} onClick={() => setBanner(null)}>关闭</button>
                        </div>
                    )}
                    {flash && (
                        <div style={infoBanner}>
                            {flash}
                            <button style={{ ...btnStyle, marginLeft: 10 }} onClick={() => setFlash(null)}>关闭</button>
                        </div>
                    )}
                </div>
            )}

            {/* 入库结果 */}
            {structuredChild && (
                <div style={{ padding: "8px 18px 0" }}>
                    <div style={{ ...infoBanner, background: "#0f2d1f", borderColor: "#16a34a" }}>
                        已生成结构化项目 <span style={{ fontFamily: "monospace" }}>{structuredChild.project_id}</span>
                        {ingestedModules.length > 0 && (
                            <span style={{ marginLeft: 8, color: "#94a3b8" }}>
                                模块 {ingestedModules[0].module_id}（{ingestedModules[0].module_version}）
                            </span>
                        )}
                        <button style={{ ...btnStyle, marginLeft: 12 }} onClick={() => onOpenCanvas(structuredChild.project_id)}>
                            打开画布
                        </button>
                    </div>
                </div>
            )}

            {/* 主体 */}
            <div style={{ flex: 1, display: "flex", minHeight: 0 }}>
                {/* 左：结构图 */}
                <div style={{ flex: 1, minWidth: 0, position: "relative" }}>
                    {ir && flow ? (
                        <ReactFlow
                            key={`${ir.nodes.length}-${ir.edges.length}`}
                            nodes={flow.nodes}
                            edges={viewerEdges}
                            nodeTypes={{ ir: ViewerNode }}
                            onNodeClick={(_, n) => setSelectedNodeId(n.id)}
                            onPaneClick={() => setSelectedNodeId(null)}
                            nodesDraggable={false}
                            nodesConnectable={false}
                            fitView
                            proOptions={{ hideAttribution: true }}
                        >
                            <Background color="#1e293b" gap={24} />
                        </ReactFlow>
                    ) : (
                        <div style={{ display: "flex", height: "100%", alignItems: "center", justifyContent: "center", color: "#64748b", fontSize: 13 }}>
                            尚未拆解 —— 请先完成 ① 结构分析与 ② 拆解 IR
                        </div>
                    )}
                    {/* 数据流图入口 */}
                    {ir && (
                        <button
                            style={{ ...btnStyle, position: "absolute", top: 12, right: 12, zIndex: 5 }}
                            onClick={() => setShowDiagram(true)}
                        >
                            数据流图（ELK 布局）
                        </button>
                    )}
                </div>

                {/* 右：面板 */}
                <div style={{ width: 380, borderLeft: "1px solid #1f2937", display: "flex", flexDirection: "column", minHeight: 0, background: "#0f172a" }}>
                    <div style={{ display: "flex", borderBottom: "1px solid #1f2937" }}>
                        {(["params", "code", "verify", "tree"] as const).map(t => (
                            <button
                                key={t}
                                onClick={() => setTab(t)}
                                style={{
                                    flex: 1,
                                    padding: "8px 4px",
                                    fontSize: 12,
                                    fontWeight: 600,
                                    border: "none",
                                    borderBottom: tab === t ? "2px solid #0f766e" : "2px solid transparent",
                                    background: "transparent",
                                    color: tab === t ? "#e2e8f0" : "#64748b",
                                    cursor: "pointer",
                                }}
                            >
                                {t === "params" ? "参数" : t === "code" ? "代码" : t === "verify" ? "验证" : "层级树"}
                            </button>
                        ))}
                    </div>
                    <div style={{ flex: 1, overflowY: "auto", padding: 14 }}>
                        {tab === "params" && (
                            selectedNode ? (
                                <div>
                                    <div style={{ fontWeight: 700, marginBottom: 4 }}>
                                        {selectedNode.class_name || selectedNode.id}
                                        {selectedNode.uncertain && <span style={{ color: "#f59e0b", marginLeft: 6 }}>⚠ agent 标注不确定</span>}
                                    </div>
                                    <div style={{ fontSize: 11, color: "#94a3b8", marginBottom: 10, fontFamily: "monospace" }}>
                                        kind={selectedNode.kind}
                                        {selectedNode.module_path ? ` · path="${selectedNode.module_path}"` : ""}
                                        <br />
                                        in {JSON.stringify(selectedNode.input_shape ?? null)} → out{" "}
                                        {JSON.stringify(selectedNode.output_shape ?? null)}
                                        {selectedNode.code_hint ? <><br />code_hint: {selectedNode.code_hint}</> : null}
                                    </div>
                                    <ParamsEditor key={selectedNode.id} node={selectedNode} saving={savingParams} onSave={handleSaveParams} />
                                </div>
                            ) : (
                                <div style={{ color: "#64748b", fontSize: 12 }}>在结构图中点击节点查看/编辑参数</div>
                            )
                        )}

                        {tab === "code" && (
                            code ? (
                                <div>
                                    <div style={{ fontSize: 11, color: "#94a3b8", marginBottom: 8 }}>
                                        后端确定性引擎输出（按当前 IR 生成）
                                        <button style={{ ...btnStyle, marginLeft: 8 }} onClick={() => void handleRegenerate()}>重新生成</button>
                                    </div>
                                    <CodeViewer
                                        code={code}
                                        spans={[]}
                                        onSelectionChange={() => {}}
                                        language="python"
                                        style={{ height: "calc(100vh - 320px)", minHeight: 260 }}
                                    />
                                </div>
                            ) : (
                                <div>
                                    <div style={{ color: "#64748b", fontSize: 12, marginBottom: 10 }}>
                                        尚未生成代码。在操作链点「④ 再生成」。
                                    </div>
                                    {regenError && (
                                        <div style={{ ...infoBanner, background: "#3f1d1d", borderColor: "#b91c1c", color: "#fecaca" }}>
                                            <div style={{ fontWeight: 700, marginBottom: 4 }}>{regenError.error}</div>
                                            {regenError.missing.length > 0 && (
                                                <div style={{ fontSize: 11, maxHeight: 160, overflowY: "auto" }}>
                                                    缺失项：{regenError.missing.join("；")}
                                                </div>
                                            )}
                                        </div>
                                    )}
                                </div>
                            )
                        )}

                        {tab === "verify" && (
                            verification ? (
                                <div style={{ fontSize: 12 }}>
                                    <div style={{ marginBottom: 10 }}>
                                        <span style={badge(verification.overall === "passed" ? "#16a34a" : "#dc2626")}>
                                            {verification.overall === "passed" ? "整体通过 ✓" : "整体未通过 ✗"}
                                        </span>
                                        <span style={{ marginLeft: 8, color: "#64748b", fontSize: 11 }}>
                                            {vStatus === "valid" ? "IR 一致（valid）" : vStatus === "stale" ? "IR 已变化（stale）" : ""}
                                        </span>
                                    </div>
                                    <div style={{ color: "#64748b", fontSize: 11, marginBottom: 10 }}>
                                        {new Date(verification.verified_at).toLocaleString()} · seeds=[{verification.seeds.join(", ")}] ·
                                        rtol={verification.tolerance.rtol} atol={verification.tolerance.atol}
                                    </div>

                                    <div style={{ fontWeight: 700, marginBottom: 4 }}>结构比对</div>
                                    <table style={{ width: "100%", fontSize: 11, marginBottom: 12 }}>
                                        <tbody>
                                            {[
                                                ["层序列匹配", verification.structure.layer_sequence_match],
                                                ["模块计数匹配", verification.structure.module_count_match],
                                                ["参数形状匹配", verification.structure.param_shapes_match],
                                                ["输出形状匹配", verification.structure.output_shape_match],
                                            ].map(([label, ok]) => (
                                                <tr key={label as string}>
                                                    <td style={tdStyle}>{label}</td>
                                                    <td style={{ ...tdStyle, textAlign: "right" }}>{ok ? "✓" : "✗"}</td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                    <div style={{ color: "#94a3b8", fontSize: 11, marginBottom: 12 }}>
                                        参数量：{typeof verification.structure.param_count === "object"
                                            ? JSON.stringify(verification.structure.param_count)
                                            : verification.structure.param_count}
                                        {" · "}层数：{verification.structure.layer_count}
                                    </div>

                                    <div style={{ fontWeight: 700, marginBottom: 4 }}>数值比对（逐种子）</div>
                                    <table style={{ width: "100%", fontSize: 11, marginBottom: 12 }}>
                                        <thead>
                                            <tr>
                                                <th style={tdStyle}>seed</th>
                                                <th style={tdStyle}>max_rel</th>
                                                <th style={tdStyle}>max_abs</th>
                                                <th style={tdStyle}>结果</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {verification.numeric.per_seed.map(s => (
                                                <tr key={s.seed}>
                                                    <td style={tdStyle}>{s.seed}</td>
                                                    <td style={tdStyle}>{s.max_rel_err?.toExponential(2)}</td>
                                                    <td style={tdStyle}>{s.max_abs_err?.toExponential(2)}</td>
                                                    <td style={{ ...tdStyle, textAlign: "right" }}>{s.passed ? "✓" : "✗"}</td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>

                                    {verification.failure_reason && (
                                        <div style={{ ...infoBanner, background: "#3f1d1d", borderColor: "#b91c1c", color: "#fecaca" }}>
                                            <div style={{ fontWeight: 700, marginBottom: 4 }}>失败原因</div>
                                            {verification.failure_reason}
                                        </div>
                                    )}
                                    {verification.structure.diff_layers.length > 0 && (
                                        <div style={{ fontSize: 11, color: "#94a3b8", marginBottom: 8 }}>
                                            <div style={{ fontWeight: 700, color: "#e2e8f0", marginBottom: 4 }}>差异层</div>
                                            {verification.structure.diff_layers.slice(0, 10).map((l, i) => (
                                                <div key={i} style={{ fontFamily: "monospace" }}>{l}</div>
                                            ))}
                                        </div>
                                    )}
                                    <div style={{ color: "#64748b", fontSize: 11, marginTop: 8 }}>
                                        入库条件：overall=passed 且验证未 stale（调参后需重新验证）
                                    </div>
                                </div>
                            ) : (
                                <div style={{ color: "#64748b", fontSize: 12 }}>尚未验证。在操作链点「⑤ 两步验证」。</div>
                            )
                        )}

                        {tab === "tree" && ir && (
                            <IrTree nodes={ir.nodes} selectedNodeId={selectedNodeId} onSelect={setSelectedNodeId} />
                        )}
                    </div>
                </div>
            </div>

            {showDiagram && ir && graph && flow && (
                <DiagramView
                    nodes={flow.nodes}
                    edges={viewerEdges}
                    direction="LR"
                    onClose={() => setShowDiagram(false)}
                    graph={graph}
                />
            )}
        </div>
    );
}

const btnStyle: CSSProperties = {
    border: "1px solid #334155",
    background: "#0f766e",
    color: "#e2e8f0",
    borderRadius: 6,
    padding: "4px 10px",
    fontSize: 12,
    fontWeight: 600,
    cursor: "pointer",
};
const inputStyle: CSSProperties = {
    border: "1px solid #334155",
    background: "#0f172a",
    color: "#e2e8f0",
    borderRadius: 4,
    padding: "3px 6px",
    fontSize: 12,
};
const tdStyle: CSSProperties = { padding: "3px 6px", borderBottom: "1px solid #1f2937" };
const infoBanner: CSSProperties = {
    background: "#0f2d1f",
    border: "1px solid #16a34a",
    color: "#bbf7d0",
    borderRadius: 8,
    padding: "8px 12px",
    fontSize: 12,
    marginBottom: 8,
};
const badge = (color: string): CSSProperties => ({
    border: `1px solid ${color}`,
    color,
    borderRadius: 999,
    padding: "2px 10px",
    fontSize: 11,
    fontWeight: 600,
    whiteSpace: "nowrap",
});
