import ast
import linecache
import random
import textwrap

import pytest

from liveavatar_assets import SOURCE
from liveavatar_grouped_benchmark import (
    StageRandomStreams,
    install_grouped_generate,
    stage_groups,
)


def test_stage_rng_matches_separate_native_processes():
    random.seed(420)
    outside = random.getstate()
    streams = StageRandomStreams(range(4))
    reference = random.Random(420)
    expected = [reference.randint(4, 30) for _ in range(100)]
    observed = [[] for _ in range(4)]
    for _ in range(100):
        for stage in range(4):
            observed[stage].append(streams.call(stage, random.randint, 4, 30))
    assert observed == [expected] * 4
    assert random.getstate() == outside


@pytest.mark.parametrize("gpus", range(1, 6))
def test_all_four_stages_are_preserved(gpus):
    groups = stage_groups(gpus)
    assert [step for group in groups for step in group] == [0, 1, 2, 3]
    assert len(groups) == max(1, gpus - 1)


@pytest.mark.parametrize("gpus", range(1, 6))
@pytest.mark.parametrize("shared_vae", [False, True])
def test_pinned_upstream_transform_compiles_without_loading_weights(gpus, shared_vae):
    if shared_vae and gpus == 5:
        return
    path = SOURCE / "liveavatar/models/wan/causal_s2v_pipeline_tpp.py"
    if not path.exists():
        pytest.skip("Pinned upstream checkout is not available")
    source = path.read_text()
    tree = ast.parse(source)
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WanS2V"
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "generate"
    )
    code = textwrap.dedent(
        "\n".join(source.splitlines()[method.lineno - 1 : method.end_lineno])
    )
    filename = "<test-pinned-generate>"
    linecache.cache[filename] = (len(code), None, code.splitlines(True), filename)
    namespace = {}
    exec(compile(code, filename, "exec"), namespace)  # noqa: S102
    model_class = type("PinnedWanS2V", (), {"generate": namespace["generate"]})
    expected = stage_groups(gpus + 1 if shared_vae else gpus)
    assert install_grouped_generate(model_class, gpus, shared_vae) == expected
