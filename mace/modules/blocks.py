###########################################################################################
# Elementary Block for Building O(3) Equivariant Higher Order Message Passing Neural Network
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from abc import abstractmethod
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import math
import torch.nn.functional
from e3nn import nn, o3
from e3nn.util.jit import compile_mode
from torch_geometric.utils import softmax as scatter_softmax

from mace.modules.wrapper_ops import (
    CuEquivarianceConfig,
    FullyConnectedTensorProduct,
    Linear,
    SymmetricContractionWrapper,
    TensorProduct,
)
from mace.tools.compile import simplify_if_compile
from mace.tools.scatter import scatter_sum
from mace.tools.utils import LAMMPS_MP

from .irreps_tools import mask_head, reshape_irreps, tp_out_irreps_with_instructions
from .radial import (
    AgnesiTransform,
    BesselBasis,
    ChebychevBasis,
    GaussianBasis,
    PolynomialCutoff,
    SoftTransform,
)


@compile_mode("script")
class LinearNodeEmbeddingBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        super().__init__()
        self.linear = Linear(
            irreps_in=irreps_in, irreps_out=irreps_out, cueq_config=cueq_config
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
    ) -> torch.Tensor:  # [n_nodes, irreps]
        return self.linear(node_attrs)


@compile_mode("script")
class LinearReadoutBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irrep_out: o3.Irreps = o3.Irreps("0e"),
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        super().__init__()
        self.linear = Linear(
            irreps_in=irreps_in, irreps_out=irrep_out, cueq_config=cueq_config
        )

    def forward(
        self,
        x: torch.Tensor,
        heads: Optional[torch.Tensor] = None,  # pylint: disable=unused-argument
    ) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        return self.linear(x)  # [n_nodes, 1]


@simplify_if_compile
@compile_mode("script")
class NonLinearReadoutBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        MLP_irreps: o3.Irreps,
        gate: Optional[Callable],
        irrep_out: o3.Irreps = o3.Irreps("0e"),
        num_heads: int = 1,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        super().__init__()
        self.hidden_irreps = MLP_irreps
        self.num_heads = num_heads
        self.linear_1 = Linear(
            irreps_in=irreps_in, irreps_out=self.hidden_irreps, cueq_config=cueq_config
        )
        self.non_linearity = nn.Activation(irreps_in=self.hidden_irreps, acts=[gate])
        self.linear_2 = Linear(
            irreps_in=self.hidden_irreps, irreps_out=irrep_out, cueq_config=cueq_config
        )

    def forward(
        self, x: torch.Tensor, heads: Optional[torch.Tensor] = None
    ) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        x = self.non_linearity(self.linear_1(x))
        if hasattr(self, "num_heads"):
            if self.num_heads > 1 and heads is not None:
                x = mask_head(x, heads, self.num_heads)
        return self.linear_2(x)  # [n_nodes, len(heads)]


@compile_mode("script")
class LinearDipoleReadoutBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        dipole_only: bool = False,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        super().__init__()
        if dipole_only:
            self.irreps_out = o3.Irreps("1x1o")
        else:
            self.irreps_out = o3.Irreps("1x0e + 1x1o")
        self.linear = Linear(
            irreps_in=irreps_in, irreps_out=self.irreps_out, cueq_config=cueq_config
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        return self.linear(x)  # [n_nodes, 1]


@compile_mode("script")
class NonLinearDipoleReadoutBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        MLP_irreps: o3.Irreps,
        gate: Callable,
        dipole_only: bool = False,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        super().__init__()
        self.hidden_irreps = MLP_irreps
        if dipole_only:
            self.irreps_out = o3.Irreps("1x1o")
        else:
            self.irreps_out = o3.Irreps("1x0e + 1x1o")
        irreps_scalars = o3.Irreps(
            [(mul, ir) for mul, ir in MLP_irreps if ir.l == 0 and ir in self.irreps_out]
        )
        irreps_gated = o3.Irreps(
            [(mul, ir) for mul, ir in MLP_irreps if ir.l > 0 and ir in self.irreps_out]
        )
        irreps_gates = o3.Irreps([mul, "0e"] for mul, _ in irreps_gated)
        self.equivariant_nonlin = nn.Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[gate for _, ir in irreps_scalars],
            irreps_gates=irreps_gates,
            act_gates=[gate] * len(irreps_gates),
            irreps_gated=irreps_gated,
        )
        self.irreps_nonlin = self.equivariant_nonlin.irreps_in.simplify()
        self.linear_1 = Linear(
            irreps_in=irreps_in, irreps_out=self.irreps_nonlin, cueq_config=cueq_config
        )
        self.linear_2 = Linear(
            irreps_in=self.hidden_irreps,
            irreps_out=self.irreps_out,
            cueq_config=cueq_config,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        x = self.equivariant_nonlin(self.linear_1(x))
        return self.linear_2(x)  # [n_nodes, 1]


@compile_mode("script")
class AtomicEnergiesBlock(torch.nn.Module):
    atomic_energies: torch.Tensor

    def __init__(self, atomic_energies: Union[np.ndarray, torch.Tensor]):
        super().__init__()
        # assert len(atomic_energies.shape) == 1

        self.register_buffer(
            "atomic_energies",
            torch.tensor(atomic_energies, dtype=torch.get_default_dtype()),
        )  # [n_elements, n_heads]

    def forward(
        self, x: torch.Tensor  # one-hot of elements [..., n_elements]
    ) -> torch.Tensor:  # [..., ]
        return torch.matmul(x, torch.atleast_2d(self.atomic_energies).T)

    def __repr__(self):
        formatted_energies = ", ".join(
            [
                "[" + ", ".join([f"{x:.4f}" for x in group]) + "]"
                for group in torch.atleast_2d(self.atomic_energies)
            ]
        )
        return f"{self.__class__.__name__}(energies=[{formatted_energies}])"


@compile_mode("script")
class RadialEmbeddingBlock(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        radial_type: str = "bessel",
        distance_transform: str = "None",
    ):
        super().__init__()
        if radial_type == "bessel":
            self.bessel_fn = BesselBasis(r_max=r_max, num_basis=num_bessel)
        elif radial_type == "gaussian":
            self.bessel_fn = GaussianBasis(r_max=r_max, num_basis=num_bessel)
        elif radial_type == "chebyshev":
            self.bessel_fn = ChebychevBasis(r_max=r_max, num_basis=num_bessel)
        if distance_transform == "Agnesi":
            self.distance_transform = AgnesiTransform()
        elif distance_transform == "Soft":
            self.distance_transform = SoftTransform()
        self.cutoff_fn = PolynomialCutoff(r_max=r_max, p=num_polynomial_cutoff)
        self.out_dim = num_bessel

    def forward(
        self,
        edge_lengths: torch.Tensor,  # [n_edges, 1]
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ):
        cutoff = self.cutoff_fn(edge_lengths)  # [n_edges, 1]
        if hasattr(self, "distance_transform"):
            edge_lengths = self.distance_transform(
                edge_lengths, node_attrs, edge_index, atomic_numbers
            )
        radial = self.bessel_fn(edge_lengths)  # [n_edges, n_basis]
        return radial * cutoff  # [n_edges, n_basis]


@compile_mode("script")
class EquivariantProductBasisBlock(torch.nn.Module):
    def __init__(
        self,
        node_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        correlation: int,
        use_sc: bool = True,
        num_elements: Optional[int] = None,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ) -> None:
        super().__init__()

        self.use_sc = use_sc
        self.symmetric_contractions = SymmetricContractionWrapper(
            irreps_in=node_feats_irreps,
            irreps_out=target_irreps,
            correlation=correlation,
            num_elements=num_elements,
            cueq_config=cueq_config,
        )
        # Update linear
        self.linear = Linear(
            target_irreps,
            target_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=cueq_config,
        )
        self.cueq_config = cueq_config

    def forward(
        self,
        node_feats: torch.Tensor,
        sc: Optional[torch.Tensor],
        node_attrs: torch.Tensor,
    ) -> torch.Tensor:
        use_cueq = False
        use_cueq_mul_ir = False
        if hasattr(self, "cueq_config"):
            if self.cueq_config is not None:
                if self.cueq_config.enabled and (
                    self.cueq_config.optimize_all or self.cueq_config.optimize_symmetric
                ):
                    use_cueq = True
                if self.cueq_config.layout_str == "mul_ir":
                    use_cueq_mul_ir = True
        if use_cueq:
            if use_cueq_mul_ir:
                node_feats = torch.transpose(node_feats, 1, 2)
            index_attrs = torch.nonzero(node_attrs)[:, 1].int()
            node_feats = self.symmetric_contractions(
                node_feats.flatten(1),
                index_attrs,
            )
        else:
            node_feats = self.symmetric_contractions(node_feats, node_attrs)
        if self.use_sc and sc is not None:
            return self.linear(node_feats) + sc
        return self.linear(node_feats)


@compile_mode("script")
class InteractionBlock(torch.nn.Module):
    def __init__(
        self,
        node_attrs_irreps: o3.Irreps,
        node_feats_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,
        edge_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        hidden_irreps: o3.Irreps,
        avg_num_neighbors: float,
        radial_MLP: Optional[List[int]] = None,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ) -> None:
        super().__init__()
        self.node_attrs_irreps = node_attrs_irreps
        self.node_feats_irreps = node_feats_irreps
        self.edge_attrs_irreps = edge_attrs_irreps
        self.edge_feats_irreps = edge_feats_irreps
        self.target_irreps = target_irreps
        self.hidden_irreps = hidden_irreps
        self.avg_num_neighbors = avg_num_neighbors
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        self.radial_MLP = radial_MLP
        self.cueq_config = cueq_config
        self._setup()

    @abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError

    def handle_lammps(
        self,
        node_feats: torch.Tensor,
        lammps_class: Optional[Any],
        lammps_natoms: Tuple[int, int],
        first_layer: bool,
    ) -> torch.Tensor:  # noqa: D401 – internal helper
        if lammps_class is None or first_layer or torch.jit.is_scripting():
            return node_feats
        _, n_total = lammps_natoms
        pad = torch.zeros(
            (n_total, node_feats.shape[1]),
            dtype=node_feats.dtype,
            device=node_feats.device,
        )
        node_feats = torch.cat((node_feats, pad), dim=0)
        node_feats = LAMMPS_MP.apply(node_feats, lammps_class)
        return node_feats

    def truncate_ghosts(
        self, tensor: torch.Tensor, n_real: Optional[int] = None
    ) -> torch.Tensor:
        """Truncate the tensor to only keep the real atoms in case of presence of ghost atoms during multi-GPU MD simulations."""
        return tensor[:n_real] if n_real is not None else tensor

    @abstractmethod
    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


nonlinearities = {1: torch.nn.functional.silu, -1: torch.tanh}


@compile_mode("script")
class RealAgnosticInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        if not hasattr(self, "cueq_config"):
            self.cueq_config = None
        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )

        # Linear
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        # Selector TensorProduct
        self.skip_tp = FullyConnectedTensorProduct(
            self.irreps_out,
            self.node_attrs_irreps,
            self.irreps_out,
            cueq_config=self.cueq_config,
        )
        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.cueq_config)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        lammps_natoms: Tuple[int, int] = (0, 0),
        lammps_class: Optional[Any] = None,
        first_layer: bool = False,
    ) -> Tuple[torch.Tensor, None]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        n_real = lammps_natoms[0] if lammps_class is not None else None
        node_feats = self.linear_up(node_feats)
        node_feats = self.handle_lammps(
            node_feats,
            lammps_class=lammps_class,
            lammps_natoms=lammps_natoms,
            first_layer=first_layer,
        )
        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.truncate_ghosts(message, n_real)
        node_attrs = self.truncate_ghosts(node_attrs, n_real)
        message = self.linear(message) / self.avg_num_neighbors
        message = self.skip_tp(message, node_attrs)
        return (
            self.reshape(message),
            None,
        )  # [n_nodes, channels, (lmax + 1)**2]


@compile_mode("script")
class RealAgnosticResidualInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        if not hasattr(self, "cueq_config"):
            self.cueq_config = None
        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,  # gate
        )

        # Linear
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        # Selector TensorProduct
        self.skip_tp = FullyConnectedTensorProduct(
            self.node_feats_irreps,
            self.node_attrs_irreps,
            self.hidden_irreps,
            cueq_config=self.cueq_config,
        )
        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.cueq_config)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        lammps_class: Optional[Any] = None,
        lammps_natoms: Tuple[int, int] = (0, 0),
        first_layer: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        n_real = lammps_natoms[0] if lammps_class is not None else None
        sc = self.skip_tp(node_feats, node_attrs)
        node_feats = self.linear_up(node_feats)
        node_feats = self.handle_lammps(
            node_feats,
            lammps_class=lammps_class,
            lammps_natoms=lammps_natoms,
            first_layer=first_layer,
        )
        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.truncate_ghosts(message, n_real)
        node_attrs = self.truncate_ghosts(node_attrs, n_real)
        sc = self.truncate_ghosts(sc, n_real)
        message = self.linear(message) / self.avg_num_neighbors
        return (
            self.reshape(message),
            sc,
        )  # [n_nodes, channels, (lmax + 1)**2]
    

def _extract_scalar_slice(irreps: o3.Irreps) -> Tuple[int, int]:
    """Return the (start, stop) index range covering all l=0 channels in
    a contiguous irreps tensor layout.  Only the *first* contiguous l=0
    block is returned; standard MACE configs always place scalars first.
 
    Raises ValueError if no l=0 channels are found.
    """
    idx = 0
    scalar_mul = 0
    start = None
    for mul, ir in irreps:
        dim = mul * ir.dim  # ir.dim == 2l+1
        if ir.l == 0:
            if start is None:
                start = idx
            scalar_mul += mul
        idx += dim
    if start is None:
        raise ValueError(
            f"Irreps {irreps} contains no l=0 (scalar) channels. "
            "Geometric attention requires at least some scalar features."
        )
    return start, start + scalar_mul

@compile_mode("script")
class RealAgnosticResidualGeometricAttentionInteractionBlock(InteractionBlock):
    def _setup(self, attention_dim: int = 16) -> None:
        if not hasattr(self, "cueq_config"):
            self.cueq_config = None

        ## ---- All original layers - unchanged ----

        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,  # gate
        )

        # Linear
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        # Selector TensorProduct
        self.skip_tp = FullyConnectedTensorProduct(
            self.node_feats_irreps,
            self.node_attrs_irreps,
            self.hidden_irreps,
            cueq_config=self.cueq_config,
        )
        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.cueq_config)

        # --- New: single-head geometric attention ---
        self.attention_dim = attention_dim

        # Get scale (l=0) components of node_feats in phi_ij so dot product is invariant scalar
        h_s, h_e = _extract_scalar_slice(self.node_feats_irreps)
        phi_s, phi_e = _extract_scalar_slice(irreps_mid)
        self._h_s: int = h_s
        self._h_e: int = h_e
        self._phi_s: int = phi_s
        self._phi_e: int = phi_e

        # Number of scalar channels  in h_i and in phi_ij
        h_scalar_dim = h_e - h_s
        phi_scalar_dim = phi_e - phi_s

        # Query and Key linear layers for attention
        self.W_Q = torch.nn.Linear(h_scalar_dim, attention_dim, bias=False)
        self.W_K = torch.nn.Linear(phi_scalar_dim, attention_dim, bias=False)

        self.attn_scale: float = attention_dim ** -0.5

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        lammps_class: Optional[Any] = None,
        lammps_natoms: Tuple[int, int] = (0, 0),
        first_layer: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        n_real = lammps_natoms[0] if lammps_class is not None else None
        sc = self.skip_tp(node_feats, node_attrs)

        node_feats = self.linear_up(node_feats)
        node_feats = self.handle_lammps(
            node_feats,
            lammps_class=lammps_class,
            lammps_natoms=lammps_natoms,
            first_layer=first_layer,
        )
        # per-edge pairwise embeddings phi_ij
        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]

        # --- Geometric attention (eqn: 5) ---

        # Query: scalar channels of receiver node features h_i
        h_scalars_i = node_feats[:, self._h_s:self._h_e]    # [n_nodes, h_scalar_dim]
        q = self.W_Q(h_scalars_i[receiver])                 # [n_edges, attention_dim]

        # Key: scalar channels of pairwise embedding phi_ij
        phi_scalars = mji[:, self._phi_s:self._phi_e]       # [n_edges, phi_scalar_dim]
        k = self.W_K(phi_scalars)                           # [n_edges, attention_dim]

        # Equivariant inner product - raw_score_ij = <W_Q h_i, W_K phi_ij>
        raw_scores = (q * k).sum(dim=-1) * self.attn_scale  # [n_edges]
        

        # Softmax normalized *per reciver node* (not globally)
        alpha = scatter_softmax(raw_scores, receiver, num_nodes=num_nodes)  # [n_edges]

        if not self.training:
            self.last_alpha = alpha.detach()
            self.last_receiver = receiver.detach()
            self.last_sender = sender.detach()

        # Eqn. 6: weight messages by attention coefficients
        mji_weighted = mji * alpha.unsqueeze(-1)             # [n_edges, irreps_mid_dim]

        message = scatter_sum(
            src=mji_weighted,
            index=receiver,
            dim=0,
            dim_size=num_nodes,
        )  # [n_nodes, irreps_mid_dim]

        message = self.truncate_ghosts(message, n_real)
        node_attrs = self.truncate_ghosts(node_attrs, n_real)
        sc = self.truncate_ghosts(sc, n_real)
 
        message = self.linear(message) / self.avg_num_neighbors
        return (
            self.reshape(message),
            sc,
        )  # [n_nodes, channels, (lmax + 1)**2]


def _common_irrep_types(irreps_a: o3.Irreps, irreps_b: o3.Irreps) -> List[o3.Irrep]:
    """Distinct (l, parity) irrep types present in both `irreps_a` and `irreps_b`,
    in the order they first appear in `irreps_a`.

    An equivariant linear map can only connect matching irrep types, so the
    attention query/key projections below may only be built out of irrep
    types that exist on both sides of the map.
    """
    types_b = {ir for _, ir in irreps_b}
    seen = set()
    common: List[o3.Irrep] = []
    for _, ir in irreps_a:
        if ir in types_b and ir not in seen:
            seen.add(ir)
            common.append(ir)
    if not common:
        raise ValueError(
            f"No shared irrep types between {irreps_a} and {irreps_b}; cannot "
            "build equivariant attention query/key projections."
        )
    return common


@compile_mode("script")
class RealAgnosticResidualMultiHeadAttentionInteractionBlock(InteractionBlock):
    """Geometric multi-head self-attention interaction block.

    Follows Norwood, Schaaf, Batatia, Csanyi & Bhowmik, "Enhancing the local
    expressivity of geometric graph neural networks" (NeurIPS 2023 ML4PS
    workshop), Eqns. 5-6, more closely than
    `RealAgnosticResidualGeometricAttentionInteractionBlock`:

    - Eq. 5 scores attention on the *full* equivariant node features h_i and
      pairwise embedding phi_ij (all l, not just their l=0/scalar slices).
      Query/key projections W_Q, W_K are *equivariant* linear maps (not plain
      `nn.Linear` restricted to invariant channels), and the similarity score
      is an equivariant inner product: summing the elementwise product of two
      tensors transforming under the same irreps is O(3)-invariant, since
      O(3) irreps are orthogonal representations (D^l(R)^T D^l(R) = I).
    - Eq. 6 uses multiple parallel attention heads mu, each producing its own
      attention-weighted aggregate of phi_ij, concatenated (oplus) along the
      channel axis before the shared self-interaction weight W (`self.linear`)
      is applied.

    W_Q and W_K are implemented as single equivariant `Linear` layers whose
    output multiplicity is `attention_dim * num_heads`; splitting that
    multiplicity into `num_heads` contiguous groups is mathematically
    equivalent to having `num_heads` independently-parameterised W_Q^mu,
    W_K^mu, since every output channel is already an independently learned
    linear combination of the input.
    """

    def _setup(self, attention_dim: int = 16, num_heads: int = 4) -> None:
        if not hasattr(self, "cueq_config"):
            self.cueq_config = None

        # ---- Same base layers as RealAgnosticResidualInteractionBlock ----
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
        )
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )

        self.skip_tp = FullyConnectedTensorProduct(
            self.node_feats_irreps,
            self.node_attrs_irreps,
            self.hidden_irreps,
            cueq_config=self.cueq_config,
        )

        # ---- Multi-head geometric attention (paper Eq. 5-6) ----
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self._n_mid = irreps_mid.dim

        common_irs = _common_irrep_types(self.node_feats_irreps, irreps_mid)
        attn_irreps = o3.Irreps([(attention_dim * num_heads, ir) for ir in common_irs])

        # Equivariant query/key projections (Eq. 5's W_Q, W_K), all heads at once.
        self.W_Q = Linear(
            self.node_feats_irreps,
            attn_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        self.W_K = Linear(
            irreps_mid,
            attn_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        # Groups the flat Q/K output into [n, mul_total, sum_of_ir_dims] so the
        # mul axis (= num_heads * attention_dim) can be split by head below.
        self.reshape_attn = reshape_irreps(attn_irreps, cueq_config=self.cueq_config)
        self._attn_feat_dim = sum(ir.dim for ir in common_irs)
        # NOTE: paper's Eq. 5 has no explicit 1/sqrt(d) term before the softmax;
        # this scaling is added (standard scaled-dot-product-attention practice)
        # to stop logits from growing with per-head dimensionality and saturating
        # the softmax. Scale by the full per-head vector size (channels x angular
        # components), matching the dimensionality actually being dotted below.
        self.attn_scale: float = (attention_dim * self._attn_feat_dim) ** -0.5

        # Pre-declared (rather than only assigned inside `if not self.training`)
        # so TorchScript knows these debug attributes exist ahead of time.
        self.last_alpha = torch.zeros(1, num_heads)
        self.last_receiver = torch.zeros(1, dtype=torch.long)
        self.last_sender = torch.zeros(1, dtype=torch.long)

        # Eq. 6's self-interaction weight W, now over head-concatenated
        # pairwise embeddings (irreps_mid repeated once per head).
        irreps_multi_head = irreps_mid
        for _ in range(num_heads - 1):
            irreps_multi_head = irreps_multi_head + irreps_mid
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_multi_head,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.cueq_config)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        lammps_class: Optional[Any] = None,
        lammps_natoms: Tuple[int, int] = (0, 0),
        first_layer: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        num_edges = edge_index.shape[1]
        n_real = lammps_natoms[0] if lammps_class is not None else None
        sc = self.skip_tp(node_feats, node_attrs)

        node_feats = self.linear_up(node_feats)
        node_feats = self.handle_lammps(
            node_feats,
            lammps_class=lammps_class,
            lammps_natoms=lammps_natoms,
            first_layer=first_layer,
        )

        # Pairwise embedding phi_ij (paper Eq. 1)
        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps_mid]

        # --- Multi-head geometric attention (Eq. 5) ---
        q = self.reshape_attn(self.W_Q(node_feats)).reshape(
            num_nodes, self.num_heads, self.attention_dim, self._attn_feat_dim
        )
        k = self.reshape_attn(self.W_K(mji)).reshape(
            num_edges, self.num_heads, self.attention_dim, self._attn_feat_dim
        )

        # Equivariant inner product <W_Q h_i, W_K phi_ij>: summing across the
        # (channel, m) axes of matching irreps is O(3)-invariant.
        raw_scores = (
            q[receiver] * k
        ).sum(dim=[-1, -2]) * self.attn_scale  # [n_edges, num_heads]

        alpha = scatter_softmax(raw_scores, receiver, num_nodes=num_nodes)  # [n_edges, num_heads]

        if not self.training:
            self.last_alpha = alpha.detach()
            self.last_receiver = receiver.detach()
            self.last_sender = sender.detach()

        # Eq. 6: weight phi_ij per head, then concatenate heads (oplus) along
        # the channel axis before the shared self-interaction weight W.
        mji_per_head = mji.unsqueeze(1) * alpha.unsqueeze(-1)  # [n_edges, num_heads, irreps_mid]
        mji_multi_head = mji_per_head.reshape(num_edges, self.num_heads * self._n_mid)

        message = scatter_sum(
            src=mji_multi_head, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, num_heads * irreps_mid]

        message = self.truncate_ghosts(message, n_real)
        node_attrs = self.truncate_ghosts(node_attrs, n_real)
        sc = self.truncate_ghosts(sc, n_real)

        message = self.linear(message) / self.avg_num_neighbors
        return (
            self.reshape(message),
            sc,
        )  # [n_nodes, channels, (lmax + 1)**2]


@compile_mode("script")
class RealAgnosticAttentionGateResidualInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        if not hasattr(self, "cueq_config"):
            self.cueq_config = None
        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,  # gate
        )

        # ==== attention gate ====
        # Attention gate from invariant edge features
        # self.attn_mlp = nn.FullyConnectedNet(
        #     [input_dim] + self.radial_MLP + [1],
        #     torch.nn.functional.silu,
        # )

        # # NEW: gate input = edge_feats + sender node_attrs + receiver node_attrs
        # edge_input_dim = self.edge_feats_irreps.num_irreps
        # node_attr_dim = self.node_attrs_irreps.num_irreps
        # attn_input_dim = edge_input_dim + 2 * node_attr_dim

        # self.attn_mlp = nn.FullyConnectedNet(
        #     [attn_input_dim] + self.radial_MLP + [1],
        #     torch.nn.functional.silu,
        # )
              
        # # ====== 3rd attempt =====
        # scalar indices from node_feats irreps
        scalar_idx_list = []
        start = 0
        for mul, ir in self.node_feats_irreps:
            ir_dim = ir.dim
            block_dim = mul * ir_dim

            if ir.l == 0:
                # For l=0, ir_dim = 1, so this block is just mul scalar channels
                scalar_idx_list.extend(range(start, start + block_dim))

            start += block_dim

        self.scalar_index = torch.tensor(scalar_idx_list, dtype=torch.long)
        self.num_scalar_feats = len(scalar_idx_list)

        edge_input_dim = self.edge_feats_irreps.num_irreps
        node_attr_dim = self.node_attrs_irreps.num_irreps
        attn_input_dim = edge_input_dim + 2 * node_attr_dim + 2 * self.num_scalar_feats

        self.attn_mlp = nn.FullyConnectedNet(
            [attn_input_dim] + self.radial_MLP + [1],
            torch.nn.functional.silu,
        )


        # Linear
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        # Selector TensorProduct
        self.skip_tp = FullyConnectedTensorProduct(
            self.node_feats_irreps,
            self.node_attrs_irreps,
            self.hidden_irreps,
            cueq_config=self.cueq_config,
        )
        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.cueq_config)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        lammps_class: Optional[Any] = None,
        lammps_natoms: Tuple[int, int] = (0, 0),
        first_layer: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        n_real = lammps_natoms[0] if lammps_class is not None else None
        sc = self.skip_tp(node_feats, node_attrs)
        node_feats = self.linear_up(node_feats)
        node_feats = self.handle_lammps(
            node_feats,
            lammps_class=lammps_class,
            lammps_natoms=lammps_natoms,
            first_layer=first_layer,
        )
        tp_weights = self.conv_tp_weights(edge_feats)

        # Raw equivairiant edge messages
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]

        # ==== attention gate ====
        # # Attention gate from invariant edge features
        # attn_logits = self.attn_mlp(edge_feats)  # [n_edges, 1]
        # attn = 1.0 + 0.1 * torch.tanh(attn_logits)  # [n_edges, 1], in range [0.9, 1.1]

        # # For plotting
        # self.last_attn = attn.detach()
        # self.last_attn_edge_index = edge_index.detach()

        # # reweight edge messages
        # mji = mji * attn  # [n_edges, irreps]
        # ==== end of attention gate ====

        # # NEW: gate input includes species info from both endpoints
        # attn_input = torch.cat([edge_feats, node_attrs[sender],node_attrs[receiver],],dim=-1,)
        # attn_logits = self.attn_mlp(attn_input)              # [n_edges, 1]
        # attn = 1.0 + 0.1 * torch.tanh(attn_logits)           # [n_edges, 1]
        # # Save for inspection
        # self.last_attn = attn.detach()
        # self.last_attn_edge_index = edge_index.detach()
        # # Reweight edge messages
        # mji = mji * attn 

        ## ====== 3rd attempt: gate input includes scalar features from both endpoints ======
        node_scalar = node_feats[:, self.scalar_index]
        attn_input = torch.cat(
            [
                edge_feats,
                node_attrs[sender],
                node_attrs[receiver],
                node_scalar[sender],
                node_scalar[receiver],
            ],
            dim=-1,
        )
        attn_logits = self.attn_mlp(attn_input)
        attn = 1.0 + 0.1 * torch.tanh(attn_logits)

        self.last_attn = attn.detach()
        self.last_attn_edge_index = edge_index.detach()
        
        mji = mji * attn
         

        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]

        message = self.truncate_ghosts(message, n_real)
        node_attrs = self.truncate_ghosts(node_attrs, n_real)
        sc = self.truncate_ghosts(sc, n_real)

        message = self.linear(message) / self.avg_num_neighbors
        return (
            self.reshape(message),
            sc,
        )  # [n_nodes, channels, (lmax + 1)**2]


@compile_mode("script")
class RealAgnosticFullAttentionResidualInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        if not hasattr(self, "cueq_config"):
            self.cueq_config = None
        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,  # gate
        )

        # Linear
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        # Selector TensorProduct
        self.skip_tp = FullyConnectedTensorProduct(
            self.node_feats_irreps,
            self.node_attrs_irreps,
            self.hidden_irreps,
            cueq_config=self.cueq_config,
        )
        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.cueq_config)

        # Scalar hidden feature extraction
        scalar_idx_list = []
        start = 0
        for mul, ir in self.node_feats_irreps:
            ir_dim = ir.dim
            block_dim = mul * ir_dim

            if ir.l == 0:
                scalar_idx_list.extend(range(start, start + block_dim))

            start += block_dim

        self.scalar_index = torch.tensor(scalar_idx_list, dtype=torch.long)
        self.num_scalar_feats = len(scalar_idx_list)

        # ----------------------------
        # Full attention pieces
        # ----------------------------
        # Small attention dimension for stability
        self.attn_dim = 32
        edge_input_dim = self.edge_feats_irreps.num_irreps

        # Query: receiver scalar hidden features -> attention space
        self.q_proj = Linear(self.num_scalar_feats, self.attn_dim)

        # Key: sender scalar hidden features + edge features -> attention space
        self.k_proj = Linear(self.num_scalar_feats + edge_input_dim, self.attn_dim)

        # Optional debug attrs
        self.last_attn = torch.zeros(1, 1)
        self.last_attn_edge_index = torch.zeros(2, 1, dtype=torch.long)
        self.last_scores = torch.zeros(1)


    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        lammps_class: Optional[Any] = None,
        lammps_natoms: Tuple[int, int] = (0, 0),
        first_layer: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        n_real = lammps_natoms[0] if lammps_class is not None else None
        sc = self.skip_tp(node_feats, node_attrs)
        node_feats = self.linear_up(node_feats)
        node_feats = self.handle_lammps(
            node_feats,
            lammps_class=lammps_class,
            lammps_natoms=lammps_natoms,
            first_layer=first_layer,
        )
        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]

        # ----------------------------
        # Full scalar attention
        # ----------------------------
        # scalar hidden node features only
        node_scalar = node_feats[:, self.scalar_index]  # [n_nodes, n_scalar]

        # query from receiver
        q = self.q_proj(node_scalar[receiver])  # [n_edges, attn_dim]

        # key from sender + edge
        k_input = torch.cat(
            [
                node_scalar[sender],
                edge_feats,
            ],
            dim=-1,
        )  # [n_edges, n_scalar + edge_dim]

        k = self.k_proj(k_input)  # [n_edges, attn_dim]

        # dot-product scores
        scores = (q * k).sum(dim=-1) / math.sqrt(float(self.attn_dim))  # [n_edges]

        # softmax over incoming edges per receiver node
        alpha = scatter_softmax(scores, receiver)  # [n_edges]

        # save for inspection
        self.last_scores = scores.detach()
        self.last_attn = alpha.unsqueeze(-1).detach()
        self.last_attn_edge_index = edge_index.detach()

        # reweight equivariant messages by scalar attention
        mji = mji * alpha.unsqueeze(-1)

        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.truncate_ghosts(message, n_real)
        node_attrs = self.truncate_ghosts(node_attrs, n_real)
        sc = self.truncate_ghosts(sc, n_real)
        message = self.linear(message) / self.avg_num_neighbors
        return (
            self.reshape(message),
            sc,
        )  # [n_nodes, channels, (lmax + 1)**2]

@compile_mode("script")
class RealAgnosticLocalAttentionResidualInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        self.linear_up = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = o3.TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,  # gate
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = self.target_irreps
        self.linear = o3.Linear(
            irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True
        )

        # Selector TensorProduct
        self.skip_tp = o3.FullyConnectedTensorProduct(
            self.node_feats_irreps, self.node_attrs_irreps, self.hidden_irreps
        )
        self.reshape = reshape_irreps(self.irreps_out)

        # === for local attention ===
        # Attention: Query and Key Projections     #equation 5
        self.WQ = o3.Linear(self.node_feats_irreps, self.node_feats_irreps, internal_weights=True, shared_weights=True,) # WQ
        self.WK = o3.Linear(irreps_mid, self.node_feats_irreps, internal_weights=True, shared_weights=True,)    # WK

        #self.local_atten_conv_tp = o3.ElementwiseTensorProduct(self.node_feats_irreps, self.node_feats_irreps, ["0e",])
        self.local_atten_conv_tp = o3.FullTensorProduct(self.node_feats_irreps, self.node_feats_irreps, ["0e",])

    def forward(
        self, node_attrs: torch.Tensor, node_feats: torch.Tensor, edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        
        def dbg(name, x):
            x_det = x.detach()
            print(
                f"{name:20s} "
                f"min={x_det.min().item(): .3e} "
                f"max={x_det.max().item(): .3e} "
                f"mean|x|={x_det.abs().mean().item(): .3e} "
            )

        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        sc = self.skip_tp(node_feats, node_attrs)

        #dbg("node_feats_in", node_feats)
        node_feats = self.linear_up(node_feats)
        #dbg("node_feats_up", node_feats)

        tp_weights = self.conv_tp_weights(edge_feats)
        #dbg("tp_weights", tp_weights)


        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        #dbg("mji", mji)

        # evaluate key and query
        # receiver here is number is of length n_edges and node_feats[receiver] is also of shape n_edges
        # TODO: make it generalize to multi-head (different mu)
        query = self.WQ(node_feats[receiver])  # [n_edges, irreps]
        #dbg("query", query)
        key = self.WK(mji)  # [n_edges, irreps]
        #dbg("key", key)
        alpha_ji_num_l0 = self.local_atten_conv_tp(query, key) # [n_edges, scalers]
        #dbg("alpha_ji_num_l0", alpha_ji_num_l0)
        #alpha_ji_num= torch.sum(alpha_ji_num_l0, dim=1)  # [n_edges, 1]
        alpha_ji_num = torch.mean(alpha_ji_num_l0, dim=1)   # mean, not sum
        #alpha_ji_num = 5.0 * torch.tanh(alpha_ji_num / 5.0)  # scale and tanh for stability, can be tuned
        #dbg("alpha_ji_num", alpha_ji_num)
        #alpha_ji = torch.softmax(alpha_ji_num, dim=0) # [n_edges,]]
        alpha_ji = scatter_softmax(alpha_ji_num, receiver)  # [n_edges]
        #dbg("alpha_ji", alpha_ji)
        

        self.last_attn = alpha_ji.detach()
        self.last_attn_edge_index = edge_index.detach()
        

        # === equation 6 ===
        #mji = mji * alpha_ji.unsqueeze(-1) # Element-wise multiplication, equation 6, poolong
        mji = mji * ( alpha_ji.unsqueeze(-1))
        
        # Aggregate messages
        message = scatter_sum(mji, receiver, dim=0, dim_size=num_nodes)  #  A function , summation over all neighbors
        message = self.linear(message) / self.avg_num_neighbors # equation 3
        return (
            self.reshape(message),
            sc,
        )  # [n_nodes, channels, (lmax + 1)**2] 

@compile_mode("script")
class RealAgnosticDensityInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        if not hasattr(self, "cueq_config"):
            self.cueq_config = None
        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )

        # Linear
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        # Selector TensorProduct
        self.skip_tp = FullyConnectedTensorProduct(
            self.irreps_out,
            self.node_attrs_irreps,
            self.irreps_out,
            cueq_config=self.cueq_config,
        )

        # Density normalization
        self.density_fn = nn.FullyConnectedNet(
            [input_dim]
            + [
                1,
            ],
            torch.nn.functional.silu,
        )
        # Reshape
        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.cueq_config)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        lammps_class: Optional[Any] = None,
        lammps_natoms: Tuple[int, int] = (0, 0),
        first_layer: bool = False,
    ) -> Tuple[torch.Tensor, None]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        n_real = lammps_natoms[0] if lammps_class is not None else None
        node_feats = self.linear_up(node_feats)
        node_feats = self.handle_lammps(
            node_feats,
            lammps_class=lammps_class,
            lammps_natoms=lammps_natoms,
            first_layer=first_layer,
        )
        tp_weights = self.conv_tp_weights(edge_feats)
        edge_density = torch.tanh(self.density_fn(edge_feats) ** 2)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        density = scatter_sum(
            src=edge_density, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, 1]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.truncate_ghosts(message, n_real)
        node_attrs = self.truncate_ghosts(node_attrs, n_real)
        density = self.truncate_ghosts(density, n_real)
        message = self.linear(message) / (density + 1)
        message = self.skip_tp(message, node_attrs)
        return (
            self.reshape(message),
            None,
        )  # [n_nodes, channels, (lmax + 1)**2]


@compile_mode("script")
class RealAgnosticDensityResidualInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        if not hasattr(self, "cueq_config"):
            self.cueq_config = None

        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,  # gate
        )

        # Linear
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        # Selector TensorProduct
        self.skip_tp = FullyConnectedTensorProduct(
            self.node_feats_irreps,
            self.node_attrs_irreps,
            self.hidden_irreps,
            cueq_config=self.cueq_config,
        )

        # Density normalization
        self.density_fn = nn.FullyConnectedNet(
            [input_dim]
            + [
                1,
            ],
            torch.nn.functional.silu,
        )

        # Reshape
        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.cueq_config)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        lammps_class: Optional[Any] = None,
        lammps_natoms: Tuple[int, int] = (0, 0),
        first_layer: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        n_real = lammps_natoms[0] if lammps_class is not None else None
        sc = self.skip_tp(node_feats, node_attrs)
        node_feats = self.linear_up(node_feats)
        node_feats = self.handle_lammps(
            node_feats,
            lammps_class=lammps_class,
            lammps_natoms=lammps_natoms,
            first_layer=first_layer,
        )
        tp_weights = self.conv_tp_weights(edge_feats)
        edge_density = torch.tanh(self.density_fn(edge_feats) ** 2)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        density = scatter_sum(
            src=edge_density, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, 1]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.truncate_ghosts(message, n_real)
        node_attrs = self.truncate_ghosts(node_attrs, n_real)
        density = self.truncate_ghosts(density, n_real)
        sc = self.truncate_ghosts(sc, n_real)
        message = self.linear(message) / (density + 1)
        return (
            self.reshape(message),
            sc,
        )  # [n_nodes, channels, (lmax + 1)**2]


@compile_mode("script")
class RealAgnosticAttResidualInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        if not hasattr(self, "cueq_config"):
            self.cueq_config = None
        self.node_feats_down_irreps = o3.Irreps("64x0e")
        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
        )

        # Convolution weights
        self.linear_down = Linear(
            self.node_feats_irreps,
            self.node_feats_down_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        input_dim = (
            self.edge_feats_irreps.num_irreps
            + 2 * self.node_feats_down_irreps.num_irreps
        )
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + 3 * [256] + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )

        # Linear
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.cueq_config)

        # Skip connection.
        self.skip_linear = Linear(
            self.node_feats_irreps, self.hidden_irreps, cueq_config=self.cueq_config
        )

    # pylint: disable=unused-argument
    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        lammps_class: Optional[Any] = None,
        lammps_natoms: Tuple[int, int] = (0, 0),
        first_layer: bool = False,
    ) -> Tuple[torch.Tensor, None]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        sc = self.skip_linear(node_feats)
        node_feats_up = self.linear_up(node_feats)
        node_feats_down = self.linear_down(node_feats)
        augmented_edge_feats = torch.cat(
            [
                edge_feats,
                node_feats_down[sender],
                node_feats_down[receiver],
            ],
            dim=-1,
        )
        tp_weights = self.conv_tp_weights(augmented_edge_feats)
        mji = self.conv_tp(
            node_feats_up[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.linear(message) / self.avg_num_neighbors
        return (
            self.reshape(message),
            sc,
        )  # [n_nodes, channels, (lmax + 1)**2]


@compile_mode("script")
class ScaleShiftBlock(torch.nn.Module):
    def __init__(self, scale: float, shift: float):
        super().__init__()
        self.register_buffer(
            "scale",
            torch.tensor(scale, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "shift",
            torch.tensor(shift, dtype=torch.get_default_dtype()),
        )

    def forward(self, x: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
        return (
            torch.atleast_1d(self.scale)[head] * x + torch.atleast_1d(self.shift)[head]
        )

    def __repr__(self):
        formatted_scale = (
            ", ".join([f"{x:.4f}" for x in self.scale])
            if self.scale.numel() > 1
            else f"{self.scale.item():.4f}"
        )
        formatted_shift = (
            ", ".join([f"{x:.4f}" for x in self.shift])
            if self.shift.numel() > 1
            else f"{self.shift.item():.4f}"
        )
        return f"{self.__class__.__name__}(scale={formatted_scale}, shift={formatted_shift})"
