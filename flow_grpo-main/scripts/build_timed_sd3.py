"""Generate an SD3 timing entry point without changing train_sd3.py."""
import ast
from pathlib import Path

from build_timed_flux import Instrument, wrap


class SD3Instrument(Instrument):
    def visit_With(self, node):
        self.generic_visit(node)
        if (ast.unparse(node.items[0].context_expr) == "autocast()"
                and "images, latents, log_probs = pipeline_with_logprob(" in ast.unparse(node)):
            return wrap([node], "sampling_including_vae")
        return node


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    tree = SD3Instrument().visit(ast.parse((root / "train_sd3.py").read_text(encoding="utf-8")))
    # The original script defines this setting but never enables it.
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    for index, node in enumerate(main.body):
        if isinstance(node, ast.Assign) and ast.unparse(node) == "transformer = pipeline.transformer":
            main.body[index + 1:index + 1] = ast.parse(
                "if config.activation_checkpointing:\n"
                "    transformer.enable_gradient_checkpointing()\n"
            ).body
            break
    else:
        raise RuntimeError("Could not locate transformer initialization")
    tree.body = ast.parse(
        'from phase_timing import PhaseTimer\n'
        'os_timing = __import__("os")\n'
        'os_timing.environ.setdefault("FLOW_TIMING_DIR", "logs/sd3_timing")\n'
        'timer = PhaseTimer()'
    ).body + tree.body
    ast.fix_missing_locations(tree)
    output = "# Generated from train_sd3.py by build_timed_sd3.py. Original is unchanged.\n" + ast.unparse(tree) + "\n"
    compile(output, str(root / "train_sd3_timed.py"), "exec")
    (root / "train_sd3_timed.py").write_text(output, encoding="utf-8")
