import numpy as np
from scipy.sparse import coo_matrix
from scipy.io import mmwrite


# ====== 输入输出配置 ======
input_file = "cora/cora.cites"    # 输入边文件
output_file = "cora.mtx"

print(f"Reading edges from {input_file} ...")
edges = []
nodes = set()

# 读取边列表
with open(input_file, "r") as f:
    for line in f:
        if not line.strip():
            continue
        src, dst = line.strip().split()
        edges.append((src, dst))
        nodes.add(src)
        nodes.add(dst)


node_list = sorted(list(nodes))
id_map = {node: i for i, node in enumerate(node_list)}
n = len(node_list)

print(n)

# 转换为索引形式
rows = [id_map[s] for s, d in edges if s in id_map and d in id_map]
cols = [id_map[d] for s, d in edges if s in id_map and d in id_map]

# 可选：若为无向图，加反向边
rows += cols
cols += rows[:len(cols)]

# 构造稀疏矩阵
data = np.ones(len(rows))
A = coo_matrix((data, (rows, cols)), shape=(n, n))

print(f"Nodes: {n}, Edges: {A.nnz}")
print(f"Writing to {output_file} ...")
mmwrite(output_file, A)
print("✅ Done! Saved sparse adjacency matrix.")