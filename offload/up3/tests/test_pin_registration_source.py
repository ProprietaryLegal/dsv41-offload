"""Regression for S73 using the pinned image source, without model imports."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from offload.up3 import host_gates as h


def test_actual_first_registration_pins_but_can_emit_no_pin_log(tmp_path: Path) -> None:
    script = tmp_path/'first_register.py'
    script.write_text('''import hashlib, importlib.util, io, json, logging, os, sys, types
from pathlib import Path
source=Path('/opt/dsv41/data/src-image/vllm_exl3_moe/__init__.py')
expected=json.loads(Path(sys.argv[1]).read_text())['pin']['sha256']
assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
buffer=io.StringIO()
vllm=logging.getLogger('vllm')
vllm.handlers=[logging.StreamHandler(buffer)]
vllm.setLevel(logging.INFO)
vllm.propagate=False
logging.getLogger().setLevel(logging.WARNING)
spec=importlib.util.spec_from_file_location('vllm_exl3_moe',source,submodule_search_locations=[str(source.parent)])
module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=module
spec.loader.exec_module(module)
for name,function in [('config','register_config'),('rdma','install'),('router','install')]:
 stub=types.ModuleType('vllm_exl3_moe.'+name)
 setattr(stub,function,lambda:None)
 sys.modules[stub.__name__]=stub
os.environ['EXL3_CPU_PIN']='5-9,15-19'
os.sched_getaffinity=lambda tid:set(range(20))
original_listdir=os.listdir
os.listdir=lambda path:['100','101'] if path=='/proc/self/task' else original_listdir(path)
pins=[]
os.sched_setaffinity=lambda tid,cpus:pins.append([tid,sorted(cpus)])
module.register()
first=buffer.getvalue()
assert len(pins)==2 and module._registered
# Diagnostic only: real registration does not perform this second pass.
module._pin_cpus()
print(json.dumps({'first_log':first,'pins':pins[:2],'after_handlers':buffer.getvalue()}))
''')
    templates = Path(__file__).with_name('fixtures')/'logger-templates.json'
    result = json.loads(subprocess.check_output([sys.executable, str(script), str(templates)], text=True, timeout=10))
    assert result['first_log'] == ''
    assert result['pins'] == [[100, h.CPUS], [101, h.CPUS]]
    assert h.log_evidence(result['after_handlers'].splitlines())['pinned_pids']
