"""模块四形状追踪（模块详细设计 6.1）：项目环境中实例化入口模型，注册 forward hook
捕获各子模块的输入/输出形状，按 named_modules 路径回填 IR 的缺失形状。

用法: python trace_shapes.py <source_dir> <ir.json> <out.json>
宿主侧编排：以项目独立环境 python 运行本脚本（decompose_service._run_trace）。
退出码: 0=成功; 1=模型加载/前向失败（错误信息入 stderr，宿主记 run_record 供检索）。
"""
import json
import sys
from pathlib import Path

import torch

from _model_loader import instantiate, load_entry_class


def _capture(model, x) -> dict:
    """注册 hook 捕获各子模块（含根，路径 ""）的输入/输出形状。"""
    captured: dict[str, dict] = {}

    def _hook(path):
        def _fn(_module, args, output):
            rec = captured.setdefault(path, {})
            if args and isinstance(args[0], torch.Tensor):
                rec["input_shape"] = list(args[0].shape)
            if isinstance(output, torch.Tensor):
                rec["output_shape"] = list(output.shape)

        return _fn

    handles = [sub.register_forward_hook(_hook(path)) for path, sub in model.named_modules()]
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in handles:
            h.remove()
    return captured


def main() -> None:
    if len(sys.argv) < 4:
        print("usage: python trace_shapes.py <source_dir> <ir.json> <out.json>", file=sys.stderr)
        sys.exit(2)
    source_dir, ir_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

    ir = json.loads(Path(ir_path).read_text(encoding="utf-8"))
    cls = load_entry_class(source_dir, ir["source_file"], ir["entry_class"])
    model = instantiate(cls).eval()

    spec = ir.get("input_spec") or {}
    shape = list(spec.get("shape") or [1, 3, 32, 32])
    dtype = getattr(torch, str(spec.get("dtype") or "float32"), torch.float32)
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=dtype)

    captured = _capture(model, x)
    Path(out_path).write_text(json.dumps(captured, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"trace done: {len(captured)} module paths captured")


if __name__ == "__main__":
    main()
