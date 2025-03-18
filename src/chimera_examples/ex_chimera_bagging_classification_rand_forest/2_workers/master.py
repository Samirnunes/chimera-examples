import os

import chimera
from chimera.nodes import AggregationMaster

os.environ["CHIMERA_WORKERS_NODES_NAMES"] = '["tree1", "tree2"]'
os.environ["CHIMERA_WORKERS_CPU_SHARES"] = "[2, 2]"
os.environ["CHIMERA_WORKERS_MAPPED_PORTS"] = "[81, 82]"

chimera.run(AggregationMaster(), 8082)
