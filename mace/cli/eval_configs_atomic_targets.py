###########################################################################################
# Script for evaluating configurations contained in an xyz file with a trained model
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import argparse

import ase.io
import numpy as np
import torch

from mace import data
from mace.tools import torch_geometric, torch_tools, utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--configs", help="path to XYZ configurations", required=True)
    parser.add_argument("--model", help="path to model", required=True)
    parser.add_argument("--output", help="output path", required=True)
    parser.add_argument(
        "--device",
        help="select device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
    )
    parser.add_argument(
        "--default_dtype",
        help="set default dtype",
        type=str,
        choices=["float32", "float64"],
        default="float64",
    )
    parser.add_argument("--batch_size", help="batch size", type=int, default=64)
    parser.add_argument(
        "--info_prefix",
        help="prefix for atomic target keys",
        type=str,
        default="MACE_",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


def run(args: argparse.Namespace) -> None:
    torch_tools.set_default_dtype(args.default_dtype)
    device = torch_tools.init_device(args.device)

    # Load model
    model = torch.load(f=args.model, map_location=args.device)
    model = model.to(args.device)

    for param in model.parameters():
        param.requires_grad = False

    # Load data and prepare input
    atoms_list = ase.io.read(args.configs, index=":")
    configs = [data.config_from_atoms(atoms) for atoms in atoms_list]

    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(
                config, z_table=z_table, cutoff=float(model.r_max)
            )
            for config in configs
        ],
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Collect data
    atomic_targets_collection = []

    for batch in data_loader:
        batch = batch.to(device)
        output = model(batch.to_dict())

        # Atomic Targets
        atomic_targets = np.split(
            torch_tools.to_numpy(output["atomic_targets"]),  # Add this
            indices_or_sections=batch.ptr[1:],
            axis=0,
        )
        atomic_targets_collection.append(atomic_targets[:-1])  # Drop the last empty section


    atomic_targets_list = [
    targets for targets_list in atomic_targets_collection for targets in targets_list
    ]  

    assert len(atoms_list) == len(atomic_targets_list)

    # Store data in atoms objects
    for i, (atoms, atomic_targets) in enumerate(zip(atoms_list, atomic_targets_list)):
        atoms.calc = None  # crucial
        atoms.arrays[args.info_prefix + "atomic_targets"] = atomic_targets

    # Write atoms to output path
    ase.io.write(args.output, images=atoms_list, format="extxyz")

    print("Evaluation Complete! :)))))))")


if __name__ == "__main__":
    main()
