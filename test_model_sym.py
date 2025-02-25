import numpy as np
import torch
import torch.nn.functional
from e3nn import o3
from e3nn.util import jit
from scipy.spatial.transform import Rotation as R

from mace import data, modules, tools
from mace.tools import torch_geometric

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)
table = tools.AtomicNumberTable([1, 8])
atomic_energies = np.array([0.0, 0.0], dtype=float)

model_config = dict(
    r_max=5,
    num_bessel=8,
    num_polynomial_cutoff=5,
    max_ell=3,
    interaction_cls_first=modules.interaction_classes[
        #"RealAgnosticLocalAttentionInteractionBlock"
        "RealAgnosticLocalAttentionInteractionBlock"
    ],
    interaction_cls=modules.interaction_classes[
        "RealAgnosticLocalAttentionResidualInteractionBlock"
    ],
    num_interactions=2,
    num_elements=2,
    hidden_irreps=o3.Irreps("128x0e+128x1o"),
    MLP_irreps=o3.Irreps("16x0e"),
    gate=torch.nn.functional.silu,
    atomic_energies=atomic_energies,
    avg_num_neighbors=8,
    atomic_numbers=table.zs,
    correlation=3,
    radial_type="bessel",
    atomic_inter_scale=[1.0, 1.0],
    atomic_inter_shift=[0.0, 0.1],
)
model = modules.ScaleShiftMACE(**model_config)

for i in range(20):    
    config = data.Configuration(
        atomic_numbers=np.array([8, 1, 1]),
        positions=np.array(
            [
                [0.0, -2.0 * np.random.randn(), 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        forces=np.array(
            [
                [0.0, -1.3, 0.0],
                [1.0, 0.2, 0.0],
                [0.0, 1.1, 0.3],
            ]
        ),
        energy=-1.5,
        charges=np.array([-2.0, 1.0, 1.0]),
        dipole=np.array([-1.5, 1.5, 2.0]),
    )
    # Created the rotated environment
    rot = R.from_euler("z", np.random.rand() * 180, degrees=True).as_matrix()
    positions_rotated = np.array(rot @ config.positions.T).T
    config_rotated = data.Configuration(
        atomic_numbers=np.array([8, 1, 1]),
        positions=positions_rotated,
        forces=np.array(
            [
                [0.0, -1.3, 0.0],
                [1.0, 0.2, 0.0],
                [0.0, 1.1, 0.3],
            ]
        ),
        energy=-1.5,
        charges=np.array([-2.0, 1.0, 1.0]),
        dipole=np.array([-1.5, 1.5, 2.0]),
    )

    atomic_data = data.AtomicData.from_config(config, z_table=table, cutoff=3.0)
    atomic_data2 = data.AtomicData.from_config(
        config_rotated, z_table=table, cutoff=3.0
    )

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[atomic_data, atomic_data2],
        batch_size=4,
        shuffle=True,
        drop_last=False,
    )
    batch = next(iter(data_loader))
    output1 = model(batch.to_dict(), training=True)
    print(output1["energy"][0])
    print(output1["energy"][1])
    assert torch.allclose(output1["energy"][0], output1["energy"][1])