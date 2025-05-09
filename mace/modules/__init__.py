from typing import Callable, Dict, Optional, Type

import torch

from .blocks import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearDipoleReadoutBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearDipoleReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    RealAgnosticAttResidualInteractionBlock,
    RealAgnosticDensityInteractionBlock,
    RealAgnosticDensityResidualInteractionBlock,
    RealAgnosticInteractionBlock,
    RealAgnosticResidualInteractionBlock,
<<<<<<< HEAD
    RealAgnosticDensityInteractionBlock,
    RealAgnosticDensityResidualInteractionBlock,
    RealAgnosticLocalAttentionInteractionBlock,
    RealAgnosticLocalAttentionResidualInteractionBlock,
    ResidualElementDependentInteractionBlock,
=======
>>>>>>> upstream/main
    ScaleShiftBlock,
)
from .loss import (
    DipoleSingleLoss,
    UniversalLoss,
    WeightedEnergyForcesDipoleLoss,
    WeightedEnergyForcesL1L2Loss,
    WeightedEnergyForcesLoss,
    WeightedEnergyForcesStressLoss,
    WeightedEnergyForcesVirialsLoss,
    WeightedForcesLoss,
    WeightedHuberEnergyForcesStressLoss,
    AtomicTargetsLoss
)
<<<<<<< HEAD
from .models import (
    MACE,
    AtomicDipolesMACE,
    AtomicTargetsMACE,
    BOTNet,
    EnergyDipolesMACE,
    ScaleShiftBOTNet,
    ScaleShiftMACE,
)
=======
from .models import MACE, AtomicDipolesMACE, EnergyDipolesMACE, ScaleShiftMACE
>>>>>>> upstream/main
from .radial import BesselBasis, GaussianBasis, PolynomialCutoff, ZBLBasis
from .symmetric_contraction import SymmetricContraction
from .utils import (
    compute_avg_num_neighbors,
    compute_fixed_charge_dipole,
    compute_mean_rms_energy_forces,
    compute_mean_std_atomic_inter_energy,
    compute_mean_std_atomic_targets,
    compute_rms_dipoles,
    compute_statistics,
)

interaction_classes: Dict[str, Type[InteractionBlock]] = {
    "RealAgnosticResidualInteractionBlock": RealAgnosticResidualInteractionBlock,
    "RealAgnosticDensityInteractionBlock": RealAgnosticDensityInteractionBlock,
    "RealAgnosticDensityResidualInteractionBlock": RealAgnosticDensityResidualInteractionBlock,
    "RealAgnosticLocalAttentionInteractionBlock": RealAgnosticLocalAttentionInteractionBlock,
    "RealAgnosticLocalAttentionResidualInteractionBlock": RealAgnosticLocalAttentionResidualInteractionBlock,
    "RealAgnosticAttResidualInteractionBlock": RealAgnosticAttResidualInteractionBlock,
    "RealAgnosticInteractionBlock": RealAgnosticInteractionBlock,
    "RealAgnosticDensityInteractionBlock": RealAgnosticDensityInteractionBlock,
    "RealAgnosticDensityResidualInteractionBlock": RealAgnosticDensityResidualInteractionBlock,
}

scaling_classes: Dict[str, Callable] = {
    "std_scaling": compute_mean_std_atomic_inter_energy,
    "rms_forces_scaling": compute_mean_rms_energy_forces,
    "rms_dipoles_scaling": compute_rms_dipoles,
    "atomic_targets_std_scaling": compute_mean_std_atomic_targets,
}

gate_dict: Dict[str, Optional[Callable]] = {
    "abs": torch.abs,
    "tanh": torch.tanh,
    "silu": torch.nn.functional.silu,
    "None": None,
}

__all__ = [
    "AtomicEnergiesBlock",
    "RadialEmbeddingBlock",
    "ZBLBasis",
    "LinearNodeEmbeddingBlock",
    "LinearReadoutBlock",
    "EquivariantProductBasisBlock",
    "ScaleShiftBlock",
    "LinearDipoleReadoutBlock",
    "NonLinearDipoleReadoutBlock",
    "InteractionBlock",
    "NonLinearReadoutBlock",
    "PolynomialCutoff",
    "BesselBasis",
    "GaussianBasis",
    "MACE",
    "ScaleShiftMACE",
    "AtomicDipolesMACE",
    "AtomicTargetsMACE",
    "EnergyDipolesMACE",
    "WeightedEnergyForcesLoss",
    "WeightedForcesLoss",
    "WeightedEnergyForcesVirialsLoss",
    "WeightedEnergyForcesStressLoss",
    "DipoleSingleLoss",
    "WeightedEnergyForcesDipoleLoss",
    "WeightedHuberEnergyForcesStressLoss",
    "UniversalLoss",
<<<<<<< HEAD
    "AtomicTargetsLoss",
=======
    "WeightedEnergyForcesL1L2Loss",
>>>>>>> upstream/main
    "SymmetricContraction",
    "interaction_classes",
    "compute_mean_std_atomic_inter_energy",
    "compute_mean_std_atomic_targets",
    "compute_avg_num_neighbors",
    "compute_statistics",
    "compute_fixed_charge_dipole",
]
