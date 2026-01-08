from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SRC_PATH = Path(__file__).parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

spec = importlib.util.spec_from_file_location("src_app", SRC_PATH / "app.py")
module = importlib.util.module_from_spec(spec)
sys.modules["src_app"] = module
assert spec.loader is not None
spec.loader.exec_module(module)

demo = module.demo

if __name__ == "__main__":
    demo.launch()
