"""模块四两步验证（模块详细设计 6.4，T7 固定脚本）：结构比对 + 数值比对。

以项目独立环境 python 运行（原模型依赖项目环境；再生成代码自包含，仅依赖 torch）。
用法: python verify_decompose.py <source_dir> <ir.json> <regenerated.py> <out.json> <seeds> <rtol> <atol>
退出码约定（实施约定）：比对不过 = 0 退出（业务结果在 JSON）；脚本异常 = 非 0。
"""
import importlib.util
import json
import sys
import traceback
from pathlib import Path

import torch

from _model_loader import instantiate, load_entry_class

_NUM_DIFF_LIMIT = 10  # 数值失败定位的差异层上报上限


def _load_generated(path: Path):
    """按文件路径加载再生成代码（类名 Decomp_{root_id}，由 ir_codegen 保证）。"""
    spec = importlib.util.spec_from_file_location("regenerated", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法解析再生成代码: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _param_modules(model):
    """带参数的子模块序列（named_modules 顺序，不含根），用于逐层类型比对。"""
    return [
        (path, sub) for path, sub in model.named_modules()
        if path and any(sub.parameters(recurse=False))
    ]


def _compare_structure(orig, regen, output_shape_match: bool) -> dict:
    o = _param_modules(orig)
    g = _param_modules(regen)
    layer_sequence_match = (
        len(o) == len(g)
        and all(oc.__class__.__name__ == gc.__class__.__name__ for (_, oc), (_, gc) in zip(o, g))
    )
    module_count_match = (
        len(list(orig.named_modules())) == len(list(regen.named_modules()))
    )
    param_count = sum(p.numel() for p in orig.parameters())
    regen_param_count = sum(p.numel() for p in regen.parameters())

    o_items = list(orig.state_dict().items())
    g_items = list(regen.state_dict().items())
    param_shapes_match = (
        len(o_items) == len(g_items)
        and all(v1.shape == v2.shape for (_, v1), (_, v2) in zip(o_items, g_items))
    )

    diff: list[dict] = []
    for i, ((op, oc), (gp, gc)) in enumerate(zip(o, g)):
        if oc.__class__.__name__ != gc.__class__.__name__:
            diff.append({
                "path": op,
                "original_class": oc.__class__.__name__,
                "regenerated_class": gc.__class__.__name__,
            })
    if len(o) != len(g):
        diff.append({"path": None, "original_layers": len(o), "regenerated_layers": len(g)})
    for i, ((k1, v1), (k2, v2)) in enumerate(zip(o_items, g_items)):
        if v1.shape != v2.shape:
            diff.append({
                "path": k1,
                "original_shape": list(v1.shape),
                "regenerated_shape": list(v2.shape),
            })

    return {
        "passed": layer_sequence_match and module_count_match and param_shapes_match
        and param_count == regen_param_count and output_shape_match,
        "layer_count": len(o),
        "module_count_match": module_count_match,
        "param_count": param_count,
        "regenerated_param_count": regen_param_count,
        "layer_sequence_match": layer_sequence_match,
        "param_shapes_match": param_shapes_match,
        "output_shape_match": output_shape_match,
        "diff_layers": diff[:_NUM_DIFF_LIMIT],
    }


def _run_capture(model, x):
    """前向并捕获各子模块输出（路径 → 张量），供数值失败时定位差异层。"""
    captured: dict[str, torch.Tensor] = {}

    def _hook(path):
        def _fn(_module, _args, output):
            if isinstance(output, torch.Tensor):
                captured[path] = output.detach()

        return _fn

    handles = [sub.register_forward_hook(_hook(path)) for path, sub in model.named_modules()]
    try:
        with torch.no_grad():
            y = model(x)
    finally:
        for h in handles:
            h.remove()
    return y, captured


def _max_errors(y1: torch.Tensor, y2: torch.Tensor):
    diff = (y1 - y2).abs()
    rel = diff / (y1.abs() + 1e-8)
    return float(rel.max()), float(diff.max())


def _numeric_pass(rel_max: float, abs_max: float, rtol: float, atol: float) -> bool:
    """逐元素相对+绝对误差双判据（满足其一即通过）。"""
    return rel_max <= rtol or abs_max <= atol


def _diff_paths(cap1: dict, cap2: dict, atol: float) -> list[dict]:
    """两模型共同路径上输出不一致的层，按层级深度降序（最深差异层在前）。"""
    out = []
    for path in sorted(set(cap1) & set(cap2)):
        y1, y2 = cap1[path], cap2[path]
        if y1.shape != y2.shape:
            out.append({"path": path, "max_abs_err": None, "note": "shape mismatch"})
            continue
        max_abs = float((y1 - y2).abs().max())
        if max_abs > atol:
            out.append({"path": path, "max_abs_err": max_abs})
    out.sort(key=lambda d: d["path"].count("."), reverse=True)
    return out[:_NUM_DIFF_LIMIT]


def main() -> None:
    if len(sys.argv) != 8:
        print(
            "usage: python verify_decompose.py <source_dir> <ir.json> <regenerated.py> "
            "<out.json> <seeds> <rtol> <atol>",
            file=sys.stderr,
        )
        sys.exit(2)
    source_dir, ir_path, regen_path, out_path = sys.argv[1:5]
    seeds = [int(s) for s in sys.argv[5].split(",") if s.strip()]
    rtol, atol = float(sys.argv[6]), float(sys.argv[7])

    ir = json.loads(Path(ir_path).read_text(encoding="utf-8"))
    cls = load_entry_class(source_dir, ir["source_file"], ir["entry_class"])
    spec = ir.get("input_spec") or {}
    shape = list(spec.get("shape") or [1, 3, 32, 32])
    dtype = getattr(torch, str(spec.get("dtype") or "float32"), torch.float32)

    orig = instantiate(cls).eval()
    regen = getattr(_load_generated(Path(regen_path)), f"Decomp_{ir['root_id']}")().eval()

    # 输出形状（种子 0 单次前向）+ 逐种子数值比对
    torch.manual_seed(seeds[0])
    x0 = torch.randn(*shape, dtype=dtype)
    with torch.no_grad():
        yo = orig(x0)
        yr = regen(x0)
    output_shape_match = list(yo.shape) == list(yr.shape)

    structure = _compare_structure(orig, regen, output_shape_match)

    per_seed: list[dict] = []
    first_fail: dict | None = None
    first_fail_diffs: list[dict] = []
    for seed in seeds:
        torch.manual_seed(seed)
        x = torch.randn(*shape, dtype=dtype)
        if seed == seeds[0]:
            y1, y2 = yo, yr
        else:
            y1, y2 = _run_capture(orig, x)[0], _run_capture(regen, x)[0]
        rel_max, abs_max = _max_errors(y1, y2)
        passed = _numeric_pass(rel_max, abs_max, rtol, atol)
        record = {"seed": seed, "max_rel_err": rel_max, "max_abs_err": abs_max, "passed": passed}
        per_seed.append(record)
        if not passed and first_fail is None:
            first_fail = record
            _, cap_o = _run_capture(orig, x)
            _, cap_r = _run_capture(regen, x)
            first_fail_diffs = _diff_paths(cap_o, cap_r, atol)

    numeric = {"passed": all(r["passed"] for r in per_seed), "per_seed": per_seed}

    failure_reason = None
    if not structure["passed"]:
        failure_reason = (
            f"结构比对失败: layer_sequence_match={structure['layer_sequence_match']}, "
            f"module_count_match={structure['module_count_match']}, "
            f"param_shapes_match={structure['param_shapes_match']}, "
            f"param_count={structure['param_count']}/{structure['regenerated_param_count']}, "
            f"output_shape_match={structure['output_shape_match']}; "
            f"差异: {json.dumps(structure['diff_layers'][:3], ensure_ascii=False)}"
        )
    elif not numeric["passed"] and first_fail is not None:
        failure_reason = (
            f"数值比对失败: seed={first_fail['seed']} "
            f"max_rel_err={first_fail['max_rel_err']:.3e} max_abs_err={first_fail['max_abs_err']:.3e}; "
            f"差异层: {json.dumps(first_fail_diffs, ensure_ascii=False)}"
            if first_fail_diffs else
            f"数值比对失败: seed={first_fail['seed']} "
            f"max_rel_err={first_fail['max_rel_err']:.3e} max_abs_err={first_fail['max_abs_err']:.3e}"
            "（无共同内部路径，无法定位差异层）"
        )

    result = {
        "structure": structure,
        "numeric": numeric,
        "overall": "passed" if structure["passed"] and numeric["passed"] else "failed",
        "failure_reason": failure_reason,
    }
    Path(out_path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"verify done: overall={result['overall']}")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 —— 脚本异常非 0 退出，宿主记 run_record
        traceback.print_exc()
        sys.exit(1)
