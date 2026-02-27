"""
@file nng.py

@description NNG (Neural Network on Graphs) model with pure PyTorch graph operations (no DGL dependency).
@author Leizheng
@date 2026-02-27
@version 1.1.0

@changelog
  - 2026-02-27 Leizheng: v1.1.0 removed DGL dependency, replaced with pure PyTorch/numpy
  - 2026-02-27 Leizheng: v1.0.0 initial creation - adapted from NeuralFramework
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, List, Optional
from einops import rearrange, repeat, reduce

from sklearn.neighbors import NearestNeighbors


# ============================================================================
# Graph Operation Functions (DGL-free)
# ============================================================================

def concat_efeat(efeat, nfeat, src_idx, dst_idx):
    """Concatenate edge features with source and destination node features."""
    if isinstance(nfeat, Tensor):
        src_feat = dst_feat = nfeat
    else:
        src_feat, dst_feat = nfeat

    return torch.cat([efeat, src_feat[src_idx], dst_feat[dst_idx]], dim=-1)


def aggregate_and_concat(efeat, nfeat, dst_idx, num_dst_nodes, aggregation="sum"):
    """Aggregate edge features to destination nodes and concatenate with node features."""
    aggregated = torch.zeros(num_dst_nodes, efeat.shape[1],
                            device=efeat.device, dtype=efeat.dtype)

    if aggregation == "sum":
        aggregated.index_add_(0, dst_idx, efeat)
    elif aggregation == "mean":
        aggregated.index_add_(0, dst_idx, efeat)
        counts = torch.zeros(num_dst_nodes, device=efeat.device, dtype=torch.float32)
        counts.index_add_(0, dst_idx,
                         torch.ones(dst_idx.shape[0], device=efeat.device, dtype=torch.float32))
        aggregated = aggregated / counts.clamp(min=1).unsqueeze(-1)

    return torch.cat([aggregated, nfeat], dim=-1)


# ============================================================================
# Basic MLP Modules
# ============================================================================

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        norm_type: Optional[str] = "LayerNorm",
    ):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, output_dim))

        if norm_type == "LayerNorm":
            layers.append(nn.LayerNorm(output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class EdgeMLP(nn.Module):
    """Edge update MLP using edge indices instead of graph."""

    def __init__(
        self,
        efeat_dim: int,
        src_dim: int,
        dst_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        hidden_layers: int = 2,
    ):
        super().__init__()
        self.mlp = MLP(
            efeat_dim + src_dim + dst_dim, output_dim, hidden_dim, hidden_layers
        )

    def forward(self, efeat: Tensor, nfeat, src_idx: Tensor, dst_idx: Tensor) -> Tensor:
        if isinstance(nfeat, Tensor):
            src_feat = dst_feat = nfeat
        else:
            src_feat, dst_feat = nfeat

        cat_feat = concat_efeat(efeat, (src_feat, dst_feat), src_idx, dst_idx)
        return self.mlp(cat_feat)


# ============================================================================
# Temporal Encoding Module
# ============================================================================

class TemporalEncoder(nn.Module):
    """Temporal encoding for multi-frame processing."""

    def __init__(self, hidden_dim, input_len):
        super().__init__()
        self.input_len = input_len
        self.temporal_embedding = nn.Parameter(torch.randn(input_len, hidden_dim))
        self.temporal_mlp = MLP(hidden_dim * 2, hidden_dim, hidden_dim)

    def forward(self, x, time_step):
        temporal_emb = self.temporal_embedding[time_step].unsqueeze(0)
        temporal_emb = temporal_emb.expand(x.shape[0], -1)
        x_concat = torch.cat([x, temporal_emb], dim=-1)
        return self.temporal_mlp(x_concat)


# ============================================================================
# Encoder/Decoder/Processor Modules
# ============================================================================

class Embedder(nn.Module):
    def __init__(
        self, input_dim_grid=2, input_dim_mesh=3, input_dim_edges=4, hidden_dim=512,
        input_len=7, add_3d_dim=True
    ):
        super().__init__()
        grid_input_dim = input_dim_grid * 2 if add_3d_dim else input_dim_grid

        self.grid_mlp = MLP(grid_input_dim, hidden_dim, hidden_dim)
        self.mesh_mlp = MLP(input_dim_mesh, hidden_dim, hidden_dim)
        self.g2m_edge_mlp = MLP(input_dim_edges, hidden_dim, hidden_dim)
        self.mesh_edge_mlp = MLP(input_dim_edges, hidden_dim, hidden_dim)

        self.temporal_encoder = TemporalEncoder(hidden_dim, input_len)
        self.add_3d_dim = add_3d_dim

    def forward(self, grid_feat, mesh_feat, g2m_efeat, mesh_efeat, time_step=None):
        if self.add_3d_dim:
            pseudo_3d_feat = grid_feat.clone()
            pseudo_3d_feat = pseudo_3d_feat + 0.01
            grid_feat = torch.cat([grid_feat, pseudo_3d_feat], dim=-1)

        grid_emb = self.grid_mlp(grid_feat)

        if time_step is not None:
            grid_emb = self.temporal_encoder(grid_emb, time_step)

        return (
            grid_emb,
            self.mesh_mlp(mesh_feat),
            self.g2m_edge_mlp(g2m_efeat),
            self.mesh_edge_mlp(mesh_efeat),
        )


class Encoder(nn.Module):
    def __init__(self, hidden_dim=512, aggregation="sum"):
        super().__init__()
        self.aggregation = aggregation
        self.edge_mlp = EdgeMLP(
            hidden_dim, hidden_dim, hidden_dim, hidden_dim, hidden_dim
        )
        self.src_mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
        self.dst_mlp = MLP(hidden_dim * 2, hidden_dim, hidden_dim)

    def forward(self, efeat, grid_feat, mesh_feat, src_idx, dst_idx, num_dst_nodes):
        efeat_new = self.edge_mlp(efeat, (grid_feat, mesh_feat), src_idx, dst_idx)
        cat_feat = aggregate_and_concat(efeat_new, mesh_feat, dst_idx, num_dst_nodes, self.aggregation)
        mesh_feat_new = self.dst_mlp(cat_feat) + mesh_feat
        grid_feat_new = self.src_mlp(grid_feat) + grid_feat
        return grid_feat_new, mesh_feat_new


class Decoder(nn.Module):
    def __init__(self, hidden_dim=512, aggregation="sum", output_len=1):
        super().__init__()
        self.aggregation = aggregation
        self.edge_mlp = EdgeMLP(
            hidden_dim, hidden_dim, hidden_dim, hidden_dim, hidden_dim
        )
        self.node_mlp = MLP(hidden_dim * 2, hidden_dim, hidden_dim)
        self.m2g_edge_mlp = MLP(4, hidden_dim, hidden_dim)

        self.output_len = output_len
        self.frame_projections = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, hidden_dim) for _ in range(output_len)
        ])

    def forward(self, m2g_efeat, grid_feat, mesh_feat, src_idx, dst_idx, num_dst_nodes, frame_idx=0):
        m2g_efeat_emb = self.m2g_edge_mlp(m2g_efeat)
        efeat_new = self.edge_mlp(m2g_efeat_emb, (mesh_feat, grid_feat), src_idx, dst_idx)
        cat_feat = aggregate_and_concat(efeat_new, grid_feat, dst_idx, num_dst_nodes, self.aggregation)
        decoded = self.node_mlp(cat_feat) + grid_feat

        if frame_idx < len(self.frame_projections):
            decoded = self.frame_projections[frame_idx](decoded)

        return decoded


class ProcessorLayer(nn.Module):
    def __init__(self, hidden_dim=512, aggregation="sum"):
        super().__init__()
        self.aggregation = aggregation
        self.edge_mlp = EdgeMLP(
            hidden_dim, hidden_dim, hidden_dim, hidden_dim, hidden_dim
        )
        self.node_mlp = MLP(hidden_dim * 2, hidden_dim, hidden_dim)

    def forward(self, efeat, nfeat, src_idx, dst_idx, num_nodes):
        efeat_new = self.edge_mlp(efeat, nfeat, src_idx, dst_idx) + efeat
        cat_feat = aggregate_and_concat(efeat_new, nfeat, dst_idx, num_nodes, self.aggregation)
        nfeat_new = self.node_mlp(cat_feat) + nfeat
        return efeat_new, nfeat_new


class Processor(nn.Module):
    def __init__(self, num_layers=16, hidden_dim=512, aggregation="sum"):
        super().__init__()
        self.layers = nn.ModuleList(
            [ProcessorLayer(hidden_dim, aggregation) for _ in range(num_layers)]
        )

    def forward(self, efeat, nfeat, src_idx, dst_idx, num_nodes):
        for layer in self.layers:
            efeat, nfeat = layer(efeat, nfeat, src_idx, dst_idx, num_nodes)
        return efeat, nfeat


# ============================================================================
# Geometry Utility Functions
# ============================================================================

def deg2rad(deg):
    return deg * np.pi / 180


def rad2deg(rad):
    return rad * 180 / np.pi


def latlon2xyz(latlon: Tensor, radius: float = 1) -> Tensor:
    latlon_rad = deg2rad(latlon)
    lat, lon = latlon_rad[:, 0], latlon_rad[:, 1]
    x = radius * torch.cos(lat) * torch.cos(lon)
    y = radius * torch.cos(lat) * torch.sin(lon)
    z = radius * torch.sin(lat)
    return torch.stack([x, y, z], dim=1)


def xyz2latlon(xyz: Tensor, radius: float = 1) -> Tensor:
    lat = torch.arcsin(xyz[:, 2] / radius)
    lon = torch.arctan2(xyz[:, 1], xyz[:, 0])
    return torch.stack([rad2deg(lat), rad2deg(lon)], dim=1)


# ============================================================================
# Graph Utilities (DGL-free)
# ============================================================================

def make_bidirected_edges(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Make edges bidirectional by adding reverse edges and removing duplicates."""
    all_src = np.concatenate([src, dst])
    all_dst = np.concatenate([dst, src])
    edges = np.stack([all_src, all_dst], axis=1)
    unique_edges = np.unique(edges, axis=0)
    return unique_edges[:, 0], unique_edges[:, 1]


def compute_edge_features(src_idx, dst_idx, src_pos, dst_pos=None, normalize=True):
    """Compute edge features from positions without DGL.

    Uses simple displacement-based features (NNG style).

    Args:
        src_idx: Source node indices [num_edges]
        dst_idx: Destination node indices [num_edges]
        src_pos: Source positions [num_src, 3]
        dst_pos: Destination positions [num_dst, 3], defaults to src_pos
        normalize: Whether to normalize by max displacement

    Returns:
        Edge features [num_edges, 4] (3 displacement + 1 distance)
    """
    if dst_pos is None:
        dst_pos = src_pos

    s = src_pos[src_idx.long()]
    d = dst_pos[dst_idx.long()]

    disp = s - d
    disp_norm = torch.linalg.norm(disp, dim=-1, keepdim=True)

    if normalize:
        max_norm = torch.max(disp_norm)
        return torch.cat([disp / max_norm, disp_norm / max_norm], dim=-1)
    return torch.cat([disp, disp_norm], dim=-1)


def compute_node_features(pos):
    """Compute node features from 3D positions.

    Converts to lat/lon in radians, then computes [cos(lat), sin(lon), cos(lon)].
    """
    latlon = xyz2latlon(pos)
    lat, lon = deg2rad(latlon[:, 0]), deg2rad(latlon[:, 1])
    return torch.stack([torch.cos(lat), torch.sin(lon), torch.cos(lon)], dim=-1)


# ============================================================================
# Mesh Generation
# ============================================================================

class TriangularMesh:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        self.vertices = vertices
        self.faces = faces


def get_icosahedron() -> TriangularMesh:
    phi = (1 + np.sqrt(5)) / 2
    vertices = []
    for c1 in [1.0, -1.0]:
        for c2 in [phi, -phi]:
            vertices.extend([(c1, c2, 0.0), (0.0, c1, c2), (c2, 0.0, c1)])
    vertices = np.array(vertices, dtype=np.float32)
    vertices /= np.linalg.norm([1.0, phi])
    faces = [
        (0, 1, 2), (0, 6, 1), (8, 0, 2), (8, 4, 0), (3, 8, 2),
        (3, 2, 7), (7, 2, 1), (0, 4, 6), (4, 11, 6), (6, 11, 5),
        (1, 5, 7), (4, 10, 11), (4, 8, 10), (10, 8, 3), (10, 3, 9),
        (11, 10, 9), (11, 9, 5), (5, 9, 7), (9, 3, 7), (1, 6, 5),
    ]
    return TriangularMesh(vertices.astype(np.float32), np.array(faces, dtype=np.int32))


def split_triangles(mesh: TriangularMesh) -> TriangularMesh:
    vertices_list = list(mesh.vertices)
    child_vertex_map = {}

    def get_midpoint(idx1, idx2):
        key = tuple(sorted([idx1, idx2]))
        if key not in child_vertex_map:
            midpoint = (mesh.vertices[idx1] + mesh.vertices[idx2]) / 2
            midpoint /= np.linalg.norm(midpoint)
            child_vertex_map[key] = len(vertices_list)
            vertices_list.append(midpoint)
        return child_vertex_map[key]

    new_faces = []
    for i1, i2, i3 in mesh.faces:
        i12, i23, i31 = get_midpoint(i1, i2), get_midpoint(i2, i3), get_midpoint(i3, i1)
        new_faces.extend(
            [[i1, i12, i31], [i12, i2, i23], [i31, i23, i3], [i12, i23, i31]]
        )

    return TriangularMesh(
        np.array(vertices_list, dtype=np.float32), np.array(new_faces, dtype=np.int32)
    )


def get_mesh_hierarchy(splits: int) -> List[TriangularMesh]:
    meshes = [get_icosahedron()]
    for _ in range(splits):
        meshes.append(split_triangles(meshes[-1]))
    return meshes


def faces_to_edges(faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    src = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    dst = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    return src, dst


# ============================================================================
# Graph Construction (DGL-free)
# ============================================================================

class GraphStructure:
    """Build and store graph edge indices and features using pure numpy/PyTorch."""

    def __init__(
        self, lat_lon_grid, mesh_level=6, multimesh=True
    ):
        self.lat_lon_grid_flat = rearrange(lat_lon_grid, "h w c -> (h w) c")

        meshes = get_mesh_hierarchy(mesh_level)
        if multimesh:
            all_vertices = meshes[-1].vertices
            all_faces = np.concatenate([m.faces for m in meshes], axis=0)
        else:
            all_vertices = meshes[-1].vertices
            all_faces = meshes[-1].faces

        self.mesh_vertices = all_vertices
        self.mesh_faces = all_faces

        # Compute max edge length from finest mesh
        finest_src, finest_dst = faces_to_edges(meshes[-1].faces)
        src_coords = meshes[-1].vertices[finest_src]
        dst_coords = meshes[-1].vertices[finest_dst]
        self.max_edge_len = np.sqrt(
            np.max(np.sum((src_coords - dst_coords) ** 2, axis=1))
        )

    def build_mesh_graph(self):
        """Build mesh graph: returns (src_idx, dst_idx, nfeat, efeat)."""
        mesh_src, mesh_dst = faces_to_edges(self.mesh_faces)
        mesh_src_bi, mesh_dst_bi = make_bidirected_edges(mesh_src, mesh_dst)

        src_idx = torch.from_numpy(mesh_src_bi).long()
        dst_idx = torch.from_numpy(mesh_dst_bi).long()
        mesh_pos = torch.tensor(self.mesh_vertices, dtype=torch.float32)
        num_nodes = len(self.mesh_vertices)

        efeat = compute_edge_features(src_idx, dst_idx, mesh_pos)
        nfeat = compute_node_features(mesh_pos)

        return src_idx, dst_idx, num_nodes, nfeat, efeat

    def build_g2m_graph(self):
        """Build grid-to-mesh graph: returns (src_idx, dst_idx, efeat)."""
        cartesian_grid = latlon2xyz(self.lat_lon_grid_flat)
        mesh_pos = torch.tensor(self.mesh_vertices, dtype=torch.float32)

        nbrs = NearestNeighbors(n_neighbors=4).fit(self.mesh_vertices)
        distances, indices = nbrs.kneighbors(cartesian_grid.numpy())

        src, dst = [], []
        for i in range(len(cartesian_grid)):
            for j in range(4):
                if distances[i][j] <= 0.6 * self.max_edge_len:
                    src.append(i)
                    dst.append(indices[i][j])

        src_idx = torch.tensor(src, dtype=torch.long)
        dst_idx = torch.tensor(dst, dtype=torch.long)

        efeat = compute_edge_features(src_idx, dst_idx, cartesian_grid.float(), mesh_pos)
        return src_idx, dst_idx, efeat

    def build_m2g_graph(self):
        """Build mesh-to-grid graph: returns (src_idx, dst_idx, num_grid_nodes, efeat)."""
        cartesian_grid = latlon2xyz(self.lat_lon_grid_flat)
        mesh_pos = torch.tensor(self.mesh_vertices, dtype=torch.float32)

        centroids = np.array(
            [self.mesh_vertices[face].mean(axis=0) for face in self.mesh_faces],
            dtype=np.float32,
        )

        nbrs = NearestNeighbors(n_neighbors=1).fit(centroids)
        _, indices = nbrs.kneighbors(cartesian_grid.numpy())
        indices = indices.flatten()

        src = [int(p) for i in indices for p in self.mesh_faces[i]]
        dst = [i for i in range(len(cartesian_grid)) for _ in range(3)]

        src_idx = torch.tensor(src, dtype=torch.long)
        dst_idx = torch.tensor(dst, dtype=torch.long)
        num_grid_nodes = len(cartesian_grid)

        efeat = compute_edge_features(src_idx, dst_idx, mesh_pos, cartesian_grid.float())
        return src_idx, dst_idx, num_grid_nodes, efeat


# ============================================================================
# Main Model
# ============================================================================

class NNG(nn.Module):
    """Neural Network on Graphs for Ocean Velocity Prediction.
    Uses pure PyTorch operations - no DGL dependency.
    """

    def __init__(self, args):
        super().__init__()

        # Extract parameters from args
        self.input_len = args.get('input_len', 7)
        self.output_len = args.get('output_len', 1)
        self.in_channels = args.get('in_channels', 2)
        self.input_res = args.get('input_res', (240, 240))

        # Model architecture parameters
        self.mesh_level = args.get('mesh_level', 5)
        self.multimesh = args.get('multimesh', True)
        self.hidden_dim = args.get('hidden_dim', 128)
        self.processor_layers = args.get('processor_layers', 16)
        self.aggregation = args.get('aggregation', 'sum')
        self.add_3d_dim = args.get('add_3d_dim', True)

        # Create lat-lon grid
        lats = torch.linspace(-90, 90, steps=self.input_res[0] + 1)[:-1]
        lons = torch.linspace(-180, 180, steps=self.input_res[1] + 1)[1:]
        lat_grid, lon_grid = torch.meshgrid(lats, lons, indexing="ij")
        lat_lon_grid = torch.stack([lat_grid, lon_grid], dim=-1)

        # Build graph structure (pure PyTorch)
        graph_struct = GraphStructure(lat_lon_grid, self.mesh_level, self.multimesh)

        # Mesh graph
        mesh_src, mesh_dst, num_mesh, mesh_nfeat, mesh_efeat = graph_struct.build_mesh_graph()
        self.register_buffer("mesh_src_idx", mesh_src)
        self.register_buffer("mesh_dst_idx", mesh_dst)
        self.num_mesh_nodes = num_mesh
        self.register_buffer("mesh_ndata", mesh_nfeat)
        self.register_buffer("mesh_edata", mesh_efeat)

        # G2M graph
        g2m_src, g2m_dst, g2m_efeat = graph_struct.build_g2m_graph()
        self.register_buffer("g2m_src_idx", g2m_src)
        self.register_buffer("g2m_dst_idx", g2m_dst)
        self.register_buffer("g2m_edata", g2m_efeat)

        # M2G graph
        m2g_src, m2g_dst, num_grid, m2g_efeat = graph_struct.build_m2g_graph()
        self.register_buffer("m2g_src_idx", m2g_src)
        self.register_buffer("m2g_dst_idx", m2g_dst)
        self.num_grid_nodes = num_grid
        self.register_buffer("m2g_edata", m2g_efeat)

        # Model components
        self.embedder = Embedder(
            self.in_channels, 3, 4, self.hidden_dim,
            self.input_len, self.add_3d_dim
        )
        self.encoder = Encoder(self.hidden_dim, self.aggregation)
        self.processor = Processor(self.processor_layers, self.hidden_dim, self.aggregation)
        self.decoder = Decoder(self.hidden_dim, self.aggregation, self.output_len)

        # Output projection for each output frame
        self.output_mlps = nn.ModuleList([
            MLP(self.hidden_dim, self.in_channels, self.hidden_dim, norm_type=None)
            for _ in range(self.output_len)
        ])

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, T_in, C, H, W) input tensor
        Returns:
            (B, T_out, C, H, W) output tensor
        """
        B, T_in, C, H, W = x.shape

        outputs = []
        for b in range(B):
            batch_outputs = []
            accumulated_grid_feat = None

            for t in range(T_in):
                frame = x[b, t]  # (C, H, W)
                frame = rearrange(frame, "c h w -> (h w) c")

                # Embedding with temporal encoding
                grid_emb, mesh_emb, g2m_emb, mesh_edge_emb = self.embedder(
                    frame, self.mesh_ndata, self.g2m_edata, self.mesh_edata, time_step=t
                )

                # Encode: grid -> mesh
                grid_feat, mesh_feat = self.encoder(
                    g2m_emb, grid_emb, mesh_emb,
                    self.g2m_src_idx, self.g2m_dst_idx, self.num_mesh_nodes
                )

                # Accumulate features across time
                if accumulated_grid_feat is None:
                    accumulated_grid_feat = grid_feat
                else:
                    accumulated_grid_feat = accumulated_grid_feat + grid_feat * 0.5

                # Process mesh
                mesh_edge_feat, mesh_feat = self.processor(
                    mesh_edge_emb, mesh_feat,
                    self.mesh_src_idx, self.mesh_dst_idx, self.num_mesh_nodes
                )

            # Decode for each output frame
            for out_t in range(self.output_len):
                grid_out = self.decoder(
                    self.m2g_edata, accumulated_grid_feat, mesh_feat,
                    self.m2g_src_idx, self.m2g_dst_idx, self.num_grid_nodes,
                    frame_idx=out_t
                )

                output = self.output_mlps[out_t](grid_out)
                output = rearrange(
                    output, "(h w) c -> c h w", h=self.input_res[0], w=self.input_res[1]
                )
                batch_outputs.append(output)

            batch_output = torch.stack(batch_outputs, dim=0)
            outputs.append(batch_output)

        return torch.stack(outputs, dim=0)
