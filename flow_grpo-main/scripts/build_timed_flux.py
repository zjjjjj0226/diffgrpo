"""Generate a separate instrumented copy; never edits train_flux.py."""
import ast
from pathlib import Path


def wrap(nodes, phase, timestep=False):
    expression = f'timer.phase({phase!r}' + (', timestep=j)' if timestep else ')')
    return ast.fix_missing_locations(ast.With(items=[ast.withitem(context_expr=ast.parse(expression, mode='eval').body)], body=nodes))


class Instrument(ast.NodeTransformer):
    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id == 'wandb' and node.func.attr == 'log':
                node.func.value.id = 'timer'
        return node

    def visit_FunctionDef(self, node):
        if node.name != 'main':
            return node
        self.generic_visit(node)
        return node

    def visit_While(self, node):
        self.generic_visit(node)
        if isinstance(node.test, ast.Constant) and node.test.value is True:
            body = node.body
            increment = body.pop()
            assert isinstance(increment, ast.AugAssign) and increment.target.id == 'epoch'
            context = ast.parse('timer.iteration(epoch)', mode='eval').body
            node.body = [ast.fix_missing_locations(ast.With(items=[ast.withitem(context_expr=context)], body=body)), increment]
        return node

    def generic_visit(self, node):
        super().generic_visit(node)
        for field in ('body', 'orelse'):
            body = getattr(node, field, None)
            if not isinstance(body, list):
                continue
            result = []
            index = 0
            while index < len(body):
                item = body[index]
                source = ast.unparse(item)
                phase = None
                count = 1
                timestep = False
                if isinstance(item, ast.Assign) and source.startswith('prompt_embeds, pooled_prompt_embeds = compute_text_embeddings('):
                    phase = 'text_encoding'
                elif isinstance(item, ast.Assign) and source.startswith('prompts, prompt_metadata = next('):
                    phase = 'data_loading'
                elif isinstance(item, ast.With) and source.startswith('with autocast():') and 'images, latents, image_ids, text_ids, log_probs = pipeline_with_logprob(' in source:
                    phase = 'sampling_including_vae'
                elif isinstance(item, ast.For) and 'sample[\'rewards\'].result()' in source:
                    phase = 'reward_wait_and_transfer'
                elif isinstance(item, ast.With) and source.startswith('with autocast():') and 'prev_sample, log_prob, prev_sample_mean, std_dev_t = compute_log_prob(' in source:
                    phase = 'train_forward'
                    timestep = True
                elif isinstance(item, ast.Expr) and source == 'accelerator.backward(loss)':
                    phase = 'backward_call'
                    timestep = True
                elif isinstance(item, ast.If) and source.startswith('if accelerator.sync_gradients:') and 'accelerator.clip_grad_norm_' in source:
                    phase = 'optimizer_calls'
                    timestep = True
                    count = 3
                    assert ast.unparse(body[index + 1]) == 'optimizer.step()'
                    assert ast.unparse(body[index + 2]) == 'optimizer.zero_grad()'
                elif isinstance(item, ast.If) and source.startswith('if epoch % config.eval_freq == 0:'):
                    phase = 'evaluation'
                elif isinstance(item, ast.If) and source.startswith('if epoch % config.save_freq == 0'):
                    phase = 'checkpoint'
                elif isinstance(item, ast.Expr) and source.startswith('ema.step('):
                    phase = 'ema'
                result.append(wrap(body[index:index + count], phase, timestep) if phase else item)
                index += count
            setattr(node, field, result)
        return node


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    tree = Instrument().visit(ast.parse((root / 'train_flux.py').read_text(encoding='utf-8')))
    imports = ast.parse('from phase_timing import PhaseTimer\ntimer = PhaseTimer()').body
    tree.body = imports + tree.body
    ast.fix_missing_locations(tree)
    output = '# Generated from train_flux.py by build_timed_flux.py. Original is unchanged.\n' + ast.unparse(tree) + '\n'
    compile(output, str(root / 'train_flux_timed.py'), 'exec')
    (root / 'train_flux_timed.py').write_text(output, encoding='utf-8')
