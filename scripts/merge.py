#!/usr/bin/env python
import argparse
import os
import glob
import tqdm
import h5py

from nugraph.data import H5DataModule

def configure():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=True,
                        help='HDF5 file pattern')
    return parser.parse_args()

def merge(args):

    # Open final output file
    with h5py.File(args.file, 'w', libver='latest') as fout:
        data_out = fout.create_group('dataset')

        # Loop over each input file to merge it in
        for fname in tqdm.tqdm(glob.glob(f'{args.file}*.h5')):
            with h5py.File(fname) as fin:

                # Loop over keys in input file
                for key in fin.keys():
                    data_in = fin[key]
                    # if it's the dataset group, loop over graphs and write those
                    if key == 'dataset':
                        for graph in data_in.keys():
                            fin.copy(data_in[graph], data_out, graph)
                            
                    # Otherwise it's metadata, so just copy it directly
                    else:
                        #print(f"File: {fname}")
                        #print(f"Key: {key}")
                        #print(f"Already in the output: {key in fout}")

                        # If the metadata is not already present in the output file,
                        # copy it from the input file.
                        if key not in fout:
                            fin.copy(data_in, fout, key)
                            
                        # The metadata already exists in the output file:
                        else:
                            import numpy as np
                            
                            # Compare the input value with the value already stored.
                            if not np.array_equal(data_in[()], fout[key][()]):
                                
                                # If the values differ, stop the merge and report 
                                # the metadata name and the input file causing the mismatch.
                                raise ValueError(
                                    f"Metadata '{key}' differs in file {fname}"
                                )

            # Delete temporary file once it's been merged
            os.remove(fname)

    # Prepare dataset
    H5DataModule.generate_samples(args.file)
    H5DataModule.generate_norm(args.file, 64)

if __name__ == '__main__':
    args = configure()
    merge(args)