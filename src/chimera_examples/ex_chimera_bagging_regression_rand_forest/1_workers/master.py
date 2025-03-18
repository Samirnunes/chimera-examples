import os

import chimera
from chimera.nodes import AggregationMaster

os.environ["CHIMERA_WORKERS_NODES_NAMES"] = '["tree1"]'
os.environ["CHIMERA_WORKERS_CPU_SHARES"] = "[2]"
os.environ["CHIMERA_WORKERS_MAPPED_PORTS"] = "[81]"

chimera.run(AggregationMaster(), 8082)
