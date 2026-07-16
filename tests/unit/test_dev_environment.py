"""Guards against running the test suite against a stale package install.

A non-editable ``pip install .`` copies the package into site-packages at
install time; local edits to ``src/`` then silently stop taking effect for
anything that imports the package (including pytest), and the test suite can
pass green against dead code with zero indication anything is wrong. This
bit us once already this session. ``pip install -e .`` (or ``-e ".[dev]"``)
avoids it — see Readme.md's "Local dev setup" section.

Note: ``event_driven_rag_service`` has no ``__init__.py`` (implicit namespace
package), so its own ``__file__`` is always ``None`` — even under a correct
editable install — and can't be used for this check. A concrete submodule's
``__file__`` can.
"""
from pathlib import Path

import event_driven_rag_service.config.embedding_config as _canary_module


def test_package_resolves_to_src_not_a_stale_install():
    resolved = Path(_canary_module.__file__).resolve()
    repo_src = (
        Path(__file__).resolve().parents[2]
        / "src" / "event_driven_rag_service" / "config" / "embedding_config.py"
    )

    assert resolved == repo_src, (
        f"event_driven_rag_service.config.embedding_config resolved to {resolved}, "
        f"not {repo_src}. This usually means the package was installed non-editable "
        "(`pip install .` instead of `pip install -e .`), so local source "
        "changes aren't taking effect. Re-run `pip install -e \".[dev]\"`."
    )
