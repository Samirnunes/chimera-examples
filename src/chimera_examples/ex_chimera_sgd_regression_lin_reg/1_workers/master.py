import os

import chimera
from chimera.nodes import ParameterServerMaster

os.environ["CHIMERA_WORKERS_NODES_NAMES"] = '["sgd1"]'
os.environ["CHIMERA_WORKERS_CPU_SHARES"] = "[2]"
os.environ["CHIMERA_WORKERS_MAPPED_PORTS"] = "[81]"

chimera.run(ParameterServerMaster("classifier", max_iter=200), 8080)
