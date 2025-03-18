from chimera.nodes.workers import RegressionWorker
from sklearn.tree import DecisionTreeRegressor

worker = RegressionWorker(
    DecisionTreeRegressor(max_depth=5, max_leaf_nodes=15), bootstrap=True
)

if __name__ == "__main__":
    worker.serve()
