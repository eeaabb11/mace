"""Sanity check: alpha=0 forward pass == forward pass with vecs manually zeroed."""
import numpy as np
import torch

# e3nn 0.4.4 uses torch.load without weights_only=False; patch before importing e3nn
import torch.serialization
torch.serialization.add_safe_globals([slice])

from e3nn import o3

from mace import data, modules
from mace.tools import AtomicNumberTable, torch_geometric
from mace.data import Configuration, AtomicData

torch.set_default_dtype(torch.float64)


def _make_model(staged_delta_training: bool):
    z_table = AtomicNumberTable([1, 8])
    atomic_energies = np.array([0.0, 0.0])
    vec_cls = modules.interaction_classes["VectorialRealAgnosticDensityInteractionBlock"]
    return modules.VectorialAtomicTargetsSolidHarmonicsSelfVecMACE(
        r_max=5.0,
        num_bessel=4,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=vec_cls,
        interaction_cls_first=vec_cls,
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps("16x0e"),
        MLP_irreps=o3.Irreps("16x0e"),
        atomic_energies=atomic_energies,
        avg_num_neighbors=3.0,
        atomic_numbers=z_table.zs,
        correlation=2,
        gate=modules.gate_dict["silu"],
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
        v_max=[5.0],
        max_v_ell=2,
        num_vec_radial_basis=4,
        contraction_cls="SymmetricContraction",
        contraction_cls_first="SymmetricContraction",
        staged_delta_training=staged_delta_training,
    )


def _make_batch():
    z_table = AtomicNumberTable([1, 8])
    rng = np.random.default_rng(42)
    config = Configuration(
        atomic_numbers=np.array([8, 1, 1]),
        positions=np.array([[0.0, -2.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        properties={
            "vecs": rng.standard_normal((3, 3)).astype(np.float64),
            "atomic_targets": np.array([-0.1, 0.2, 0.05]),
        },
        property_weights={"vecs": 1.0, "atomic_targets": 1.0},
    )
    atomic_data = AtomicData.from_config(config, z_table=z_table, cutoff=5.0)
    loader = torch_geometric.dataloader.DataLoader([atomic_data], batch_size=1)
    batch = next(iter(loader))
    return batch.to_dict()


def test_alpha_zero_matches_zeroed_vecs():
    model_staged = _make_model(staged_delta_training=True)
    model_staged.eval()

    # Confirm buffer starts at 0
    assert model_staged.alpha.item() == 0.0, "alpha should be 0.0 when staged_delta_training=True"

    batch = _make_batch()

    # Forward with alpha=0 (vecs pass through but get multiplied by 0)
    with torch.no_grad():
        out_alpha0 = model_staged(batch)["atomic_targets"]

    # Forward with alpha=1 but vecs manually zeroed
    model_normal = _make_model(staged_delta_training=False)
    model_normal.load_state_dict(model_staged.state_dict())
    model_normal.eval()
    assert model_normal.alpha.item() == 0.0  # loaded from staged model

    batch_zeroed = dict(batch)
    batch_zeroed["vecs"] = torch.zeros_like(batch["vecs"])
    model_normal.set_alpha(1.0)

    with torch.no_grad():
        out_zeroed = model_normal(batch_zeroed)["atomic_targets"]

    assert torch.allclose(out_alpha0, out_zeroed, atol=1e-10), (
        f"Mismatch: alpha=0 output {out_alpha0} != zeroed-vecs output {out_zeroed}"
    )
    print(f"PASS: alpha=0 output == zeroed-vecs output: {out_alpha0.tolist()}")


def test_alpha_one_differs_from_zero():
    """Smoke test: alpha=1 with real vecs should give different output than alpha=0."""
    model = _make_model(staged_delta_training=False)
    model.eval()
    batch = _make_batch()

    with torch.no_grad():
        model.set_alpha(0.0)
        out_zero = model(batch)["atomic_targets"]
        model.set_alpha(1.0)
        out_one = model(batch)["atomic_targets"]

    assert not torch.allclose(out_zero, out_one, atol=1e-10), (
        "alpha=0 and alpha=1 should differ when vecs are non-zero"
    )
    print(f"PASS: alpha=0 {out_zero.tolist()} != alpha=1 {out_one.tolist()}")


if __name__ == "__main__":
    test_alpha_zero_matches_zeroed_vecs()
    test_alpha_one_differs_from_zero()
    print("All sanity checks passed.")
