"""
@file graphcast.py

@description GraphCast model with pure PyTorch graph operations (no DGL dependency).
@author Leizheng
@date 2026-02-27
@version 1.1.0

@changelog
  - 2026-02-27 Leizheng: v1.1.0 removed DGL dependency, replaced with pure PyTorch/numpy
  - 2026-02-27 Leizheng: v1.0.0 initial creation - adapted from NeuralFramework
"""

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from typing import Tuple, List, Union, Optional
from scipy.spatial import transform
from sklearn.neighbors import NearestNeighbors


# =============================================================================
# Geometry Utilities
# =============================================================================

def deg2rad(deg: torch.Tensor) -> torch.Tensor:
    return deg * np.pi / 180


def rad2deg(rad: torch.Tensor) -> torch.Tensor:
    return rad * 180 / np.pi


def latlon2xyz(
    latlon: torch.Tensor, radius: float = 1, unit: str = "deg"
) -> torch.Tensor:
    """Convert lat/lon to 3D cartesian coordinates."""
    if unit == "deg":
        latlon = deg2rad(latlon)
    lat, lon = latlon[..., 0], latlon[..., 1]
    x = radius * torch.cos(lat) * torch.cos(lon)
    y = radius * torch.cos(lat) * torch.sin(lon)
    z = radius * torch.sin(lat)
    return torch.stack([x, y, z], dim=-1)


def xyz2latlon(xyz: torch.Tensor, radius: float = 1, unit: str = "deg") -> torch.Tensor:
    """Convert 3D cartesian to lat/lon coordinates."""
    lat = torch.arcsin(xyz[..., 2] / radius)
    lon = torch.arctan2(xyz[..., 1], xyz[..., 0])
    result = torch.stack([lat, lon], dim=-1)
    return rad2deg(result) if unit == "deg" else result


def geospatial_rotation(
    invar: torch.Tensor, theta: torch.Tensor, axis: str, unit: str = "rad"
) -> torch.Tensor:
    """Rotate coordinates around specified axis."""
    if unit == "deg":
        theta = deg2rad(theta)

    invar = rearrange(invar, "... d -> ... d 1")
    rotation = torch.zeros((*theta.shape, 3, 3), device=theta.device)
    cos, sin = torch.cos(theta), torch.sin(theta)

    if axis == "x":
        rotation[..., 0, 0] = 1.0
        rotation[..., 1, 1] = cos
        rotation[..., 1, 2] = -sin
        rotation[..., 2, 1] = sin
        rotation[..., 2, 2] = cos
    elif axis == "y":
        rotation[..., 0, 0] = cos
        rotation[..., 0, 2] = sin
        rotation[..., 1, 1] = 1.0
        rotation[..., 2, 0] = -sin
        rotation[..., 2, 2] = cos
    elif axis == "z":
        rotation[..., 0, 0] = cos
        rotation[..., 0, 1] = -sin
        rotation[..., 1, 0] = sin
        rotation[..., 1, 1] = cos
        rotation[..., 2, 2] = 1.0

    outvar = torch.matmul(rotation, invar)
    return rearrange(outvar, "... d 1 -> ... d")


# =============================================================================
# Mesh Generation
# =============================================================================

class TriangularMesh:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        self.vertices = vertices
        self.faces = faces


def get_icosahedron() -> TriangularMesh:
    """Generate icosahedron mesh."""
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

    # Rotation for proper alignment
    angle = 2 * np.arcsin(phi / np.sqrt(3))
    rotation_angle = (np.pi - angle) / 2
    rotation = transform.Rotation.from_euler(seq="y", angles=rotation_angle)
    vertices = vertices @ rotation.as_matrix().T

    return TriangularMesh(vertices.astype(np.float32), np.array(faces, dtype=np.int32))


def split_mesh(mesh: TriangularMesh) -> TriangularMesh:
    """Split each triangle into 4 smaller triangles."""
    vertex_cache = {}
    all_vertices = list(mesh.vertices)

    def get_midpoint(i1: int, i2: int) -> int:
        key = tuple(sorted([i1, i2]))
        if key not in vertex_cache:
            midpoint = (mesh.vertices[i1] + mesh.vertices[i2]) / 2
            midpoint /= np.linalg.norm(midpoint)
            vertex_cache[key] = len(all_vertices)
            all_vertices.append(midpoint)
        return vertex_cache[key]

    new_faces = []
    for i1, i2, i3 in mesh.faces:
        i12 = get_midpoint(i1, i2)
        i23 = get_midpoint(i2, i3)
        i31 = get_midpoint(i3, i1)
        new_faces.extend(
            [[i1, i12, i31], [i12, i2, i23], [i31, i23, i3], [i12, i23, i31]]
        )

    return TriangularMesh(
        np.array(all_vertices, dtype=np.float32), np.array(new_faces, dtype=np.int32)
    )


def get_mesh_hierarchy(splits: int) -> List[TriangularMesh]:
    """Generate hierarchy of meshes."""
    meshes = [get_icosahedron()]
    for _ in range(splits):
        meshes.append(split_mesh(meshes[-1]))
    return meshes


def merge_meshes(meshes: List[TriangularMesh]) -> TriangularMesh:
    """Merge multiple meshes into one."""
    return TriangularMesh(
        vertices=meshes[-1].vertices,
        faces=np.concatenate([m.faces for m in meshes], axis=0),
    )


def faces_to_edges(faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert face indices to edge indices."""
    src = np.concatenate([faces[:, i] for i in range(3)])
    dst = np.concatenate([faces[:, (i + 1) % 3] for i in range(3)])
    return src, dst


# =============================================================================
# Graph Utilities (DGL-free)
# =============================================================================

def make_bidirected_edges(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Make edges bidirectional by adding reverse edges and removing duplicates."""
    all_src = np.concatenate([src, dst])
    all_dst = np.concatenate([dst, src])
    edges = np.stack([all_src, all_dst], axis=1)
    unique_edges = np.unique(edges, axis=0)
    return unique_edges[:, 0], unique_edges[:, 1]


def compute_edge_features(
    src_idx: torch.Tensor, dst_idx: torch.Tensor,
    src_pos: torch.Tensor, dst_pos: torch.Tensor
) -> torch.Tensor:
    """Compute edge features based on node positions (GraphCast-style).

    Uses local coordinate transformation: rotates src to align dst with [1,0,0],
    then computes displacement from [1,0,0].

    Returns:
        Edge features [num_edges, 4] (3 displacement + 1 distance)
    """
    src_coords = src_pos[src_idx.long()]
    dst_coords = dst_pos[dst_idx.long()]

    # Convert to local coordinate system
    dst_latlon = xyz2latlon(dst_coords, unit="rad")
    theta_az = torch.where(
        dst_latlon[:, 1] >= 0, 2 * np.pi - dst_latlon[:, 1], -dst_latlon[:, 1]
    )
    theta_polar = torch.where(
        dst_latlon[:, 0] >= 0, dst_latlon[:, 0], 2 * np.pi + dst_latlon[:, 0]
    )

    src_rot = geospatial_rotation(src_coords, theta_az, "z", "rad")
    src_rot = geospatial_rotation(src_rot, theta_polar, "y", "rad")

    # Edge features: displacement from [1,0,0] + distance
    disp = src_rot - torch.tensor([1, 0, 0], device=src_rot.device, dtype=src_rot.dtype)
    dist = torch.norm(disp, dim=-1, keepdim=True)
    max_dist = dist.max()

    return torch.cat([disp / max_dist, dist / max_dist], dim=-1)


# =============================================================================
# Graph Operations (DGL-free)
# =============================================================================

def concat_efeat(
    efeat: torch.Tensor, src_feat: torch.Tensor, dst_feat: torch.Tensor,
    src_idx: torch.Tensor, dst_idx: torch.Tensor
) -> torch.Tensor:
    """Concatenate edge and node features using edge indices."""
    return torch.cat([efeat, src_feat[src_idx], dst_feat[dst_idx]], dim=-1)


def aggregate_and_concat(
    efeat: torch.Tensor, dst_feat: torch.Tensor,
    dst_idx: torch.Tensor, num_dst_nodes: int, agg: str = "sum"
) -> torch.Tensor:
    """Aggregate edge features to nodes and concatenate."""
    aggregated = torch.zeros(num_dst_nodes, efeat.shape[1],
                            device=efeat.device, dtype=efeat.dtype)

    if agg == "sum":
        aggregated.index_add_(0, dst_idx, efeat)
    else:  # mean
        aggregated.index_add_(0, dst_idx, efeat)
        counts = torch.zeros(num_dst_nodes, device=efeat.device, dtype=torch.float32)
        counts.index_add_(0, dst_idx, torch.ones(dst_idx.shape[0], device=efeat.device, dtype=torch.float32))
        aggregated = aggregated / counts.clamp(min=1).unsqueeze(-1)

    return torch.cat([aggregated, dst_feat], dim=-1)


# =============================================================================
# Neural Network Modules
# =============================================================================

class MLP(nn.Module):
    """Multi-layer perceptron with LayerNorm."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, n_layers: int = 1):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.extend([nn.Linear(hidden_dim, out_dim), nn.LayerNorm(out_dim)])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EdgeMLP(nn.Module):
    """MLP for edge updates with node feature concatenation."""

    def __init__(
        self,
        edge_dim: int,
        node_dim: int,
        out_dim: int,
        hidden_dim: int,
        n_layers: int = 1,
    ):
        super().__init__()
        self.mlp = MLP(edge_dim + 2 * node_dim, out_dim, hidden_dim, n_layers)

    def forward(
        self,
        efeat: torch.Tensor,
        src_feat: torch.Tensor,
        dst_feat: torch.Tensor,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
    ) -> torch.Tensor:
        cat_feat = concat_efeat(efeat, src_feat, dst_feat, src_idx, dst_idx)
        return self.mlp(cat_feat)


class Encoder(nn.Module):
    """Grid to mesh encoder."""

    def __init__(self, hidden_dim: int, n_layers: int, agg: str = "sum"):
        super().__init__()
        self.edge_mlp = EdgeMLP(
            hidden_dim, hidden_dim, hidden_dim, hidden_dim, n_layers
        )
        self.src_mlp = MLP(hidden_dim, hidden_dim, hidden_dim, n_layers)
        self.dst_mlp = MLP(hidden_dim + hidden_dim, hidden_dim, hidden_dim, n_layers)
        self.agg = agg

    def forward(
        self,
        g2m_efeat: torch.Tensor,
        grid_feat: torch.Tensor,
        mesh_feat: torch.Tensor,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        num_dst_nodes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        efeat = self.edge_mlp(g2m_efeat, grid_feat, mesh_feat, src_idx, dst_idx)
        agg_feat = aggregate_and_concat(efeat, mesh_feat, dst_idx, num_dst_nodes, self.agg)
        mesh_feat_new = mesh_feat + self.dst_mlp(agg_feat)
        grid_feat_new = grid_feat + self.src_mlp(grid_feat)
        return grid_feat_new, mesh_feat_new


class Decoder(nn.Module):
    """Mesh to grid decoder."""

    def __init__(self, hidden_dim: int, n_layers: int, agg: str = "sum"):
        super().__init__()
        self.edge_mlp = EdgeMLP(
            hidden_dim, hidden_dim, hidden_dim, hidden_dim, n_layers
        )
        self.node_mlp = MLP(hidden_dim + hidden_dim, hidden_dim, hidden_dim, n_layers)
        self.agg = agg

    def forward(
        self,
        m2g_efeat: torch.Tensor,
        grid_feat: torch.Tensor,
        mesh_feat: torch.Tensor,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        num_dst_nodes: int,
    ) -> torch.Tensor:
        efeat = self.edge_mlp(m2g_efeat, mesh_feat, grid_feat, src_idx, dst_idx)
        agg_feat = aggregate_and_concat(efeat, grid_feat, dst_idx, num_dst_nodes, self.agg)
        return grid_feat + self.node_mlp(agg_feat)


class ProcessorLayer(nn.Module):
    """Single processor layer for mesh."""

    def __init__(self, hidden_dim: int, n_layers: int, agg: str = "sum"):
        super().__init__()
        self.edge_mlp = EdgeMLP(
            hidden_dim, hidden_dim, hidden_dim, hidden_dim, n_layers
        )
        self.node_mlp = MLP(hidden_dim + hidden_dim, hidden_dim, hidden_dim, n_layers)
        self.agg = agg

    def forward(
        self, efeat: torch.Tensor, nfeat: torch.Tensor,
        src_idx: torch.Tensor, dst_idx: torch.Tensor, num_nodes: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        efeat_new = efeat + self.edge_mlp(efeat, nfeat, nfeat, src_idx, dst_idx)
        agg_feat = aggregate_and_concat(efeat_new, nfeat, dst_idx, num_nodes, self.agg)
        nfeat_new = nfeat + self.node_mlp(agg_feat)
        return efeat_new, nfeat_new


class Processor(nn.Module):
    """Mesh graph processor with multiple layers."""

    def __init__(
        self, hidden_dim: int, n_layers: int, mlp_layers: int, agg: str = "sum"
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [ProcessorLayer(hidden_dim, mlp_layers, agg) for _ in range(n_layers)]
        )

    def forward(
        self, efeat: torch.Tensor, nfeat: torch.Tensor,
        src_idx: torch.Tensor, dst_idx: torch.Tensor, num_nodes: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            efeat, nfeat = layer(efeat, nfeat, src_idx, dst_idx, num_nodes)
        return efeat, nfeat


# =============================================================================
# Graph Builder (DGL-free)
# =============================================================================

class GraphBuilder:
    """Build graphs for GraphCast using pure numpy/PyTorch."""

    def __init__(
        self, lat_lon_grid: torch.Tensor, mesh_level: int = 5, multimesh: bool = True
    ):
        self.lat_lon_flat = rearrange(lat_lon_grid, "h w c -> (h w) c")

        # Build mesh hierarchy
        meshes = get_mesh_hierarchy(mesh_level)
        finest = meshes[-1]
        mesh = merge_meshes(meshes) if multimesh else finest

        self.mesh_vertices = torch.from_numpy(mesh.vertices).float()
        mesh_src, mesh_dst = faces_to_edges(mesh.faces)

        # Mesh graph (bidirected edges)
        mesh_src_bi, mesh_dst_bi = make_bidirected_edges(mesh_src, mesh_dst)
        self.mesh_src_idx = torch.from_numpy(mesh_src_bi).long()
        self.mesh_dst_idx = torch.from_numpy(mesh_dst_bi).long()
        self.num_mesh_nodes = len(self.mesh_vertices)

        # Mesh edge features
        self.mesh_efeat = compute_edge_features(
            self.mesh_src_idx, self.mesh_dst_idx,
            self.mesh_vertices, self.mesh_vertices
        )

        # G2M graph (grid to mesh)
        grid_xyz = latlon2xyz(self.lat_lon_flat)
        max_edge_len = self._get_max_edge_len(finest)

        g2m_src, g2m_dst = self._find_bipartite_edges(
            grid_xyz, self.mesh_vertices, max_edge_len * 0.6
        )
        self.g2m_src_idx = torch.tensor(g2m_src, dtype=torch.long)
        self.g2m_dst_idx = torch.tensor(g2m_dst, dtype=torch.long)

        # G2M edge features
        self.g2m_efeat = compute_edge_features(
            self.g2m_src_idx, self.g2m_dst_idx,
            grid_xyz.float(), self.mesh_vertices
        )

        # M2G graph (mesh to grid)
        centroids = self._get_centroids(mesh)
        m2g_src, m2g_dst = self._find_m2g_edges(grid_xyz, centroids, mesh.faces)
        self.m2g_src_idx = torch.tensor(m2g_src, dtype=torch.long)
        self.m2g_dst_idx = torch.tensor(m2g_dst, dtype=torch.long)
        self.num_grid_nodes = len(grid_xyz)

        # M2G edge features
        self.m2g_efeat = compute_edge_features(
            self.m2g_src_idx, self.m2g_dst_idx,
            self.mesh_vertices, grid_xyz.float()
        )

    def _get_max_edge_len(self, mesh: TriangularMesh) -> float:
        src, dst = faces_to_edges(mesh.faces)
        diffs = mesh.vertices[src] - mesh.vertices[dst]
        return np.sqrt(np.max(np.sum(diffs**2, axis=1)))

    def _get_centroids(self, mesh: TriangularMesh) -> np.ndarray:
        return np.mean(mesh.vertices[mesh.faces], axis=1)

    def _find_bipartite_edges(
        self,
        src_pos: torch.Tensor,
        dst_pos: Union[np.ndarray, torch.Tensor],
        max_dist: float,
    ) -> Tuple[List[int], List[int]]:
        """Find bipartite edges using KNN within max_dist."""
        dst_pos_np = (
            dst_pos.cpu().numpy() if isinstance(dst_pos, torch.Tensor) else dst_pos
        )

        nbrs = NearestNeighbors(n_neighbors=4).fit(dst_pos_np)
        distances, indices = nbrs.kneighbors(src_pos.cpu().numpy())

        src_idx, dst_idx = [], []
        for i in range(len(src_pos)):
            for j in range(4):
                if distances[i][j] <= max_dist:
                    src_idx.append(i)
                    dst_idx.append(indices[i][j])

        return src_idx, dst_idx

    def _find_m2g_edges(
        self, grid_xyz: torch.Tensor, centroids: np.ndarray, faces: np.ndarray
    ) -> Tuple[List[int], List[int]]:
        """Find mesh-to-grid edges by nearest face centroid."""
        nbrs = NearestNeighbors(n_neighbors=1).fit(centroids)
        _, indices = nbrs.kneighbors(grid_xyz.cpu().numpy())
        indices = indices.flatten()

        src_idx = [int(p) for i in indices for p in faces[i]]
        dst_idx = [i for i in range(len(grid_xyz)) for _ in range(3)]

        return src_idx, dst_idx


# =============================================================================
# Main GraphCast Model for Ocean Velocity Prediction
# =============================================================================

class GraphCast(nn.Module):
    """
    GraphCast model adapted for ocean velocity prediction.
    Uses pure PyTorch operations - no DGL dependency.
    """

    def __init__(self, args):
        super().__init__()

        # Temporal configuration
        self.input_len = args.get('input_len', 7)
        self.output_len = args.get('output_len', 1)
        self.in_channels = args.get('in_channels', 2)  # u, v velocity components

        # Spatial configuration
        input_res = args.get('input_res', [240, 240])
        self.input_res = tuple(input_res) if isinstance(input_res, list) else input_res

        # Model architecture parameters
        self.hidden_dim = args.get('hidden_dim', 128)
        self.mesh_level = args.get('mesh_level', 4)
        self.processor_layers = args.get('processor_layers', 8)
        self.mlp_layers = args.get('mlp_layers', 1)
        self.multimesh = args.get('multimesh', True)
        self.aggregation = args.get('aggregation', 'sum')

        # Processing options
        self.add_3d_dim = args.get('add_3d_dim', True)
        self.temporal_encoding = args.get('temporal_encoding', 'concat')

        # Build graphs (pure PyTorch)
        lat = torch.linspace(-90, 90, self.input_res[0])
        lon = torch.linspace(-180, 180, self.input_res[1] + 1)[1:]
        lat_lon_grid = torch.stack(torch.meshgrid(lat, lon, indexing="ij"), dim=-1)

        graph_builder = GraphBuilder(lat_lon_grid, self.mesh_level, self.multimesh)

        # Register graph structure as buffers (moves with .to(device))
        self.register_buffer("mesh_pos", graph_builder.mesh_vertices)
        self.register_buffer("g2m_efeat", graph_builder.g2m_efeat)
        self.register_buffer("mesh_efeat", graph_builder.mesh_efeat)
        self.register_buffer("m2g_efeat", graph_builder.m2g_efeat)
        self.register_buffer("g2m_src_idx", graph_builder.g2m_src_idx)
        self.register_buffer("g2m_dst_idx", graph_builder.g2m_dst_idx)
        self.register_buffer("mesh_src_idx", graph_builder.mesh_src_idx)
        self.register_buffer("mesh_dst_idx", graph_builder.mesh_dst_idx)
        self.register_buffer("m2g_src_idx", graph_builder.m2g_src_idx)
        self.register_buffer("m2g_dst_idx", graph_builder.m2g_dst_idx)
        self.num_mesh_nodes = graph_builder.num_mesh_nodes
        self.num_grid_nodes = graph_builder.num_grid_nodes

        # Calculate input dimensions based on temporal encoding
        if self.temporal_encoding == 'concat':
            grid_input_dim = self.input_len * self.in_channels
            if self.add_3d_dim:
                grid_input_dim *= 2
        else:
            grid_input_dim = self.hidden_dim
            self.temporal_encoder = nn.GRU(
                input_size=self.in_channels,
                hidden_size=self.hidden_dim,
                num_layers=2,
                batch_first=True
            )

        # Feature embedders
        self.grid_embed = MLP(grid_input_dim, self.hidden_dim, self.hidden_dim, self.mlp_layers)
        self.mesh_node_embed = MLP(3, self.hidden_dim, self.hidden_dim, self.mlp_layers)
        self.g2m_edge_embed = MLP(4, self.hidden_dim, self.hidden_dim, self.mlp_layers)
        self.mesh_edge_embed = MLP(4, self.hidden_dim, self.hidden_dim, self.mlp_layers)
        self.m2g_edge_embed = MLP(4, self.hidden_dim, self.hidden_dim, self.mlp_layers)

        # Encoder, processor, decoder
        self.encoder = Encoder(self.hidden_dim, self.mlp_layers, self.aggregation)
        self.processor = Processor(
            self.hidden_dim, self.processor_layers, self.mlp_layers, self.aggregation
        )
        self.decoder = Decoder(self.hidden_dim, self.mlp_layers, self.aggregation)

        # Output head for multi-frame prediction
        output_channels = self.output_len * self.in_channels
        self.output_head = nn.Sequential(
            MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, self.mlp_layers),
            nn.Linear(self.hidden_dim, output_channels),
        )

    def prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prepare input for processing.

        Args:
            x: [B, T, C, H, W] input tensor
        Returns:
            [B, C_concat, H, W] or [B, H*W, C_encoded] depending on temporal encoding
        """
        B, T, C, H, W = x.shape

        if self.temporal_encoding == 'concat':
            x = rearrange(x, 'b t c h w -> b (t c) h w')
            if self.add_3d_dim:
                x_surface = x
                x_depth = x * 0.9
                x = torch.cat([x_surface, x_depth], dim=1)
            return x
        else:
            x = rearrange(x, 'b t c h w -> (b h w) t c')
            _, hidden = self.temporal_encoder(x)
            x_encoded = hidden[-1]
            return rearrange(x_encoded, '(b h w) d -> b (h w) d', b=B, h=H, w=W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GraphCast.

        Args:
            x: [B, T_in, C, H, W] input features
        Returns:
            [B, T_out, C, H, W] output predictions
        """
        B, T_in, C, H, W = x.shape

        # Handle batch processing
        outputs = []
        for b in range(B):
            x_single = x[b:b+1]

            # Prepare input
            x_processed = self.prepare_input(x_single)

            if self.temporal_encoding == 'concat':
                grid_feat = rearrange(x_processed, '1 c h w -> (h w) c')
            else:
                grid_feat = x_processed[0]

            # Embed features
            grid_feat = self.grid_embed(grid_feat)
            mesh_node_feat = self.mesh_node_embed(self.mesh_pos)
            g2m_efeat_embed = self.g2m_edge_embed(self.g2m_efeat)
            mesh_efeat_embed = self.mesh_edge_embed(self.mesh_efeat)
            m2g_efeat_embed = self.m2g_edge_embed(self.m2g_efeat)

            # Encode: grid -> mesh
            grid_feat, mesh_node_feat = self.encoder(
                g2m_efeat_embed, grid_feat, mesh_node_feat,
                self.g2m_src_idx, self.g2m_dst_idx, self.num_mesh_nodes
            )

            # Process on mesh
            mesh_efeat_embed, mesh_node_feat = self.processor(
                mesh_efeat_embed, mesh_node_feat,
                self.mesh_src_idx, self.mesh_dst_idx, self.num_mesh_nodes
            )

            # Decode: mesh -> grid
            grid_feat = self.decoder(
                m2g_efeat_embed, grid_feat, mesh_node_feat,
                self.m2g_src_idx, self.m2g_dst_idx, self.num_grid_nodes
            )

            # Output
            output = self.output_head(grid_feat)

            # Reshape: [N, C_out] -> [1, T_out, C, H, W]
            output = rearrange(
                output,
                '(h w) (t c) -> 1 t c h w',
                h=H, w=W, t=self.output_len, c=self.in_channels
            )

            outputs.append(output)

        return torch.cat(outputs, dim=0)
