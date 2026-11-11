#!/usr/bin/env python
import argparse

import torch
import h5py
import nugraph as ng

Data = ng.data.H5DataModule

def configure():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Checkpoint file for trained NuGraph2 model')
    parser.add_argument('--outfile', type=str, required=True,
                        help='Output TorchScript file name (full path)')
    parser.add_argument('--infile', type=str, default=None,
                        help='Input .gnn.h5 file used to retrieve feature normalization constants')

    parser = Data.add_data_args(parser)

    return parser.parse_args()

def compile(args):
    
    # Load normalization factors when extended hit features are enabled.
    if args.featext is True:
        try:
            with h5py.File(args.infile, "r") as f:
                normf = {
                    p: torch.tensor(f[f"norm/{p}"][()], dtype=torch.float32)
                    for p in ("u", "v", "y")
                }
        except (TypeError, OSError, KeyError) as e:
            raise ValueError(
                "--featext requires a valid --inputfile containing norm/u, norm/v and norm/y"
            ) from e
    # print ('norm ', normf)

    print('using checkpoint =', args.checkpoint)
    model = ng.models.NuGraph2.load_from_checkpoint(args.checkpoint, map_location='cpu')

    print('exporting TorchScript module...')
    scripted = model.export()

    print('output file =', args.outfile)
    torch.jit.save(scripted, args.outfile)
    print('done.')

if __name__ == '__main__':
    args = configure()
    compile(args)
