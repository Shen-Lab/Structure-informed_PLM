#!/usr/bin/env python3
import os
import sys
import json
import lmdb
import pickle as pkl
import argparse
import subprocess
from pathlib import Path

import pandas as pd
import h5py
import numpy as np

from Bio import SeqIO


def build_lmdb_for_fasta_dict(fasta_dict, data_dir, map_size_gb = 15.0) -> str:
    """
    Create an LMDB at {data_dir}/zero_shot.lmdb with entries:
      {'seq_id': fasta_name, 'seq_primary': fasta_seq}
    """
    new_dir = (data_dir / f"zero_shot").resolve()
    new_dir.mkdir(parents=True, exist_ok=True)
    
    lmdb_path = new_dir / f"zero_shot_mut_all.lmdb"

    # compute LMDB map_size in bytes
    map_size = int((1024 * 1024 * 1024) * map_size_gb)

    
    data_list = []
    
    for name, seq in fasta_dict.items():
        entry = {
            'set_nm': name, 
            'seq_id': name,
            'wt_seq': seq, 
            'seq_len': len(seq),
            'mut_seq': seq,
            'mutants': ['M0M'],
            'mut_relative_idxs': [0],
            'fitness': 0
        }
        print('len(seq)', len(seq))
        data_list.append(entry)
    

    env = lmdb.open(str(lmdb_path), map_size=map_size)
    i = -1
    with env.begin(write=True) as txn:
        # Now the 'data_list' variable exists and can be used here
        for i, entry in enumerate(data_list):
            txn.put(str(i).encode(), pkl.dumps(entry))
        txn.put(b"num_examples", pkl.dumps(i + 1))
    env.close()

    
def run_embedding(model_dir: Path, data_dir: Path, embed_modelNm: str, batch_size: int) -> Path:
    """
    Calls the model script to generate embeddings and returns the HDF5 output path.
    The model script will stream each sequence embedding directly into this HDF5.
    """
    # Where the model should write results
    eval_dir = Path("./eval_results").resolve()
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    
    lmdb_path = data_dir / f"zero_shot.lmdb"
    

    script_dir = Path(__file__).resolve().parent
    main_py_path = script_dir / "model_scripts" / "main.py"
    

    # Build command (lean/low-RAM run)
    cmd = [
        sys.executable, str(main_py_path),
        "run_eval",
        "transformer",
        "mutation_fitness_UNsupervise_mutagenesis",    # "embed_seq",
        str(model_dir),
        "--batch_size", str(batch_size),
        "--data_dir", str(data_dir),
        "--metrics", "save_logits",
        "--mutgsis_set", "zero_shot",
        "--embed_modelNm", embed_modelNm,
        "--eval_save_dir", str(eval_dir),
    ]
    print(f"\n[RUN] {' '.join(cmd)}")

    if not main_py_path.exists():
        raise FileNotFoundError(f"The script to run was not found at: {main_py_path}")

    # Reduce BLAS threads to avoid RAM spikes
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    subprocess.run(cmd, check=True)




def main():
    p = argparse.ArgumentParser(description="Generate per-CSV HDF5 embedding files from ProteinGym substitution CSVs.")
    p.add_argument("--fasta_dir", default="./CAGI7/", type=str,
                   help="Directory containing ProteinGym substitution CSVs.")
    p.add_argument("--data_dir", default="./CAGI7/", type=str,
                   help="Directory used by the embedding script for LMDB/JSON (should be visible under ProteinEncoder-LM/).")
    p.add_argument("--model_dir", default="trained_model/pre-trained_models/RP15_B4", type=str,
                   help="Path to the saved model folder.")
    p.add_argument("--embed_name", default="RP15_B4", type=str,
                   help="Identifier for --embed_modelNm (appears in output JSON filename).")
    p.add_argument("--batch_size", default=2, type=int, help="Batch size for embedding.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing .h5 files if present.")
                   
    
    args = p.parse_args()

    fasta_dir = Path(args.fasta_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    
    
    Path("./eval_results").mkdir(parents=True, exist_ok=True)
    
    # fasta_dir
    assert fasta_dir.exists(), f"fasta_dir not found: {fasta_dir}"
    data_dir.mkdir(parents=True, exist_ok=True)

    fasta_files = sorted(fasta_dir.glob("*.fasta"))
    
    # build dict
    fasta_dict={}
    for fasta_path in fasta_files:
        print(f"\n=== Processing {fasta_path.name} ===")
        
        record = SeqIO.read(fasta_path, "fasta")
        
        fasta_name = fasta_path.stem
        fasta_seq = record.seq
        
        fasta_dict[fasta_name] = fasta_seq
    
    # build lmdb
    
    build_lmdb_for_fasta_dict(fasta_dict, data_dir=data_dir, map_size_gb=15.0)
    
    print(f"[OK] LMDB ready: {data_dir / f'zero_shot.lmdb'}")
    
    
    # Run embedding -> pickle
    run_embedding(
        model_dir=model_dir,
        data_dir=data_dir,
        embed_modelNm=args.embed_name,
        batch_size=args.batch_size,
    )
    print(f"[OK] pickle written")
        
    print("\nAll done.")


if __name__ == "__main__":
    main()
