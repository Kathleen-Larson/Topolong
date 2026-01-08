import os, time
import numpy as np
import surfa as sf

fbase = 'lh.sphere.ico2.reg'
mesh = sf.load_mesh(os.path.join('data', '.'.join([fbase, 'clean'])))

edges = np.diff(mesh.vertices[mesh.edges].transpose(0, 2, 1), axis=-1).squeeze()
min_edge_len = np.linalg.norm(edges, axis=1).min()

norms = mesh.vertex_normals
noise = np.random.uniform(low=-min_edge_len/4, high=min_edge_len/4, size=mesh.vertices.shape)

mesh.vertices += noise

mesh.save(os.path.join('data', '.'.join([fbase, 'noisy'])))
