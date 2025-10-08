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


def build_lmdb_for_csv(csv_path: Path, data_dir: Path, map_size_gb: float = 15.0) -> str:
    """
    Create an LMDB at {data_dir}/{csv_name}.lmdb with entries:
      {'seq_id': mutant, 'seq_primary': mutated_sequence}
    Returns the split name (csv_name without extension).
    """
    df = pd.read_csv(csv_path)
    required = {"mutant", "mutated_sequence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} is missing columns: {missing}")

    # Clean / normalize sequences
    df["mutated_sequence"] = df["mutated_sequence"].astype(str).str.upper().str.strip()
    df["mutant"] = df["mutant"].astype(str).str.strip()

    split = csv_path.stem
    lmdb_path = data_dir / f"{split}.lmdb"

    # compute LMDB map_size in bytes
    map_size = int((1024 * 1024 * 1024) * map_size_gb)

    # --- START: RESTORED CODE ---
    # This block was missing. It creates the data_list from the DataFrame.
    data_list = []
    for _, row in df.iterrows():
        entry = {
            "seq_id": row["mutant"],          # key: mutant identifier
            "seq_primary": row["mutated_sequence"], # value: amino acid sequence
        }
        data_list.append(entry)
    # --- END: RESTORED CODE ---

    '''
    # Correctly remove a stale LMDB path if it exists.
    if lmdb_path.exists():
        print(f"Found stale LMDB path. Removing {lmdb_path}...")
        if lmdb_path.is_dir():
            import shutil
            shutil.rmtree(lmdb_path)
        else:
            lmdb_path.unlink()
    '''
    
    if lmdb_path.exists():
        return split

    env = lmdb.open(str(lmdb_path), map_size=map_size)
    i = -1
    with env.begin(write=True) as txn:
        # Now the 'data_list' variable exists and can be used here
        for i, entry in enumerate(data_list):
            txn.put(str(i).encode(), pkl.dumps(entry))
        txn.put(b"num_examples", pkl.dumps(i + 1))
    env.close()

    return split
    
def run_embedding(model_dir: Path, data_dir: Path, split: str, embed_modelNm: str, batch_size: int) -> Path:
    """
    Calls the model script to generate embeddings and returns the HDF5 output path.
    The model script will stream each sequence embedding directly into this HDF5.
    """
    # Where the model should write results
    eval_dir = Path("./eval_results").resolve()
    eval_dir.mkdir(parents=True, exist_ok=True)
    h5_out = eval_dir / f"{split}.h5"
    
    # --- START: MODIFIED VERIFICATION LOGIC ---
    # Path to the source LMDB file, derived from the split name
    lmdb_path = data_dir / f"{split}.lmdb"

    if h5_out.exists():
        h5_count = 0
        lmdb_count = 0

        # Safely count entries in the existing HDF5 file
        try:
            with h5py.File(h5_out, 'r') as f:
                # The number of sequences is the number of top-level groups
                h5_count = len(f.keys())
        except Exception as e:
            print(f"⚠️ Warning: Could not read existing HDF5 file '{h5_out.name}'. Error: {e}")
            # Treat as incomplete, it will be deleted below

        # Safely count the expected number of entries from the LMDB file
        try:
            if not lmdb_path.exists():
                print(f"⚠️ Warning: LMDB file not found at {lmdb_path}. Cannot verify HDF5 count.")
            else:
                env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
                with env.begin() as txn:
                    num_examples_bytes = txn.get(b'num_examples')
                    if num_examples_bytes:
                        lmdb_count = pkl.loads(num_examples_bytes)
                env.close()
        except Exception as e:
            print(f"⚠️ Warning: Could not read LMDB file '{lmdb_path.name}'. Error: {e}")

        # Compare counts and decide whether to skip or regenerate
        if lmdb_count > 0 and h5_count == lmdb_count:
            print(f"✅ Verified: HDF5 file '{h5_out.name}' is complete with {h5_count} sequences. Skipping.")
            return h5_out
        else:
            print(f"ncomplete HDF5 file found: '{h5_out.name}' has {h5_count} entries, but LMDB expects {lmdb_count}.")
            print("continue")
    # --- END: MODIFIED VERIFICATION LOGIC ---
    

    script_dir = Path(__file__).resolve().parent
    main_py_path = script_dir / "model_scripts" / "main.py"

    # Build command (lean/low-RAM run)
    cmd = [
        sys.executable, str(main_py_path),
        "run_eval",
        "transformer",
        "embed_seq",
        str(model_dir),
        "--batch_size", str(batch_size),
        "--num_workers", "0",         # avoid forking extra workers
        "--repr_layers", "-1",        # only last layer
        "--data_dir", str(data_dir),
        "--metrics", "save_embedding",
        "--split", split,
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

    if not h5_out.exists():
        raise FileNotFoundError(f"Expected HDF5 not found: {h5_out}")
    return h5_out



def run_embedding_old(model_dir: Path, data_dir: Path, split: str, embed_modelNm: str, batch_size: int) -> Path:
    """
    Calls the provided script to generate embeddings and returns the JSON output path.
    """
    # Output JSON the model should create
    out_json = data_dir / f"embedding_{split}_{embed_modelNm}.json"
    
    script_dir = Path(__file__).resolve().parent
    main_py_path = script_dir / "model_scripts" / "main.py"

    # Build command
    cmd = [
        sys.executable, str(main_py_path),
        "run_eval",
        "transformer",
        "embed_seq",
        str(model_dir),
        "--batch_size", str(batch_size),
        "--data_dir", str(data_dir),
        "--metrics", "save_embedding",
        "--split", split,
        "--embed_modelNm", embed_modelNm,
    ]
    print(f"\n[RUN] {' '.join(cmd)}")
    
    if not main_py_path.exists():
        raise FileNotFoundError(f"The script to run was not found at: {main_py_path}")
        
    subprocess.run(cmd, check=True)
    if not out_json.exists():
        raise FileNotFoundError(f"Expected output not found: {out_json}")
    return out_json


def json_to_h5(json_path: Path, h5_path: Path, seq_lookup: dict, overwrite: bool = False):
    """
    Convert {seq_id: embedding_flat_list} JSON into HDF5:
      - one group per `seq_id` (i.e., mutant)
      - dataset 'embedding' with shape (L, 768)
      - attribute 'sequence' with the original mutated_sequence
    """
    if h5_path.exists():
        if not overwrite:
            print(f"[SKIP] {h5_path.name} exists (use --overwrite to replace).")
            return
        else:
            h5_path.unlink()

    with open(json_path, "r") as f:
        emb_dict = json.load(f)

    with h5py.File(h5_path, "w") as h5f:
        for seq_id, emb_list in emb_dict.items():
            arr = np.asarray(emb_list, dtype=np.float32)
            # Expect flat length L*768 → reshape to (L, 768)
            if arr.size % 768 != 0:
                raise ValueError(
                    f"{json_path.name}: embedding for '{seq_id}' has size {arr.size}, not divisible by 768."
                )
            L = arr.size // 768
            arr = arr.reshape(L, 768)

            grp = h5f.create_group(seq_id)
            dset = grp.create_dataset("embedding", data=arr, compression="gzip")
            # Add original sequence (if available)
            seq = seq_lookup.get(seq_id)
            if seq is not None:
                grp.attrs["sequence"] = np.string_(seq)

    print(f"[OK] Wrote {h5_path.name} with {len(emb_dict)} entries.")


def main():
    p = argparse.ArgumentParser(description="Generate per-CSV HDF5 embedding files from ProteinGym substitution CSVs.")
    p.add_argument("--csv_dir", default="../dataset/ProteinGym/substitution/", type=str,
                   help="Directory containing ProteinGym substitution CSVs.")
    p.add_argument("--data_dir", default="../dataset/ProteinGym/substitution/", type=str,
                   help="Directory used by the embedding script for LMDB/JSON (should be visible under ProteinEncoder-LM/).")
    p.add_argument("--model_dir", default="trained_model/pre-trained_models/RP15_B4", type=str,
                   help="Path to the saved model folder.")
    p.add_argument("--embed_name", default="RP15_B4", type=str,
                   help="Identifier for --embed_modelNm (appears in output JSON filename).")
    p.add_argument("--batch_size", default=4, type=int, help="Batch size for embedding.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing .h5 files if present.")
    p.add_argument("--only_csv", nargs="*", default=None,
                   help="Process only these CSV basenames (without path). Example: A0A140D2T1_ZIKV_Sourisseau_2019.csv")
                   
    # --- START: NEW ARGUMENTS ---
    p.add_argument("--index", type=int, default=0,
                   help="Index of the subset to process (0-based). Used with --total_len.")
    p.add_argument("--total_len", type=int, default=0,
                   help="Total number of subsets to divide the CSV files into. If 0, all files are processed.")
    # --- END: NEW ARGUMENTS ---
    
    args = p.parse_args()

    csv_dir = Path(args.csv_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    
    # --- FIX: Add this line to create the directory ---
    Path("./eval_results").mkdir(parents=True, exist_ok=True)
    # ----------------------------------------------------

    assert csv_dir.exists(), f"csv_dir not found: {csv_dir}"
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(csv_dir.glob("*.csv"))
    if args.only_csv:
        only = set(args.only_csv)
        csv_files = [p for p in csv_files if p.name in only]
        if not csv_files:
            print(f"No matching CSVs for filter: {only}")
            return

    print(f"[INFO] Found {len(csv_files)} CSV files under {csv_dir}")
    
    # --- START: NEW SUBSETTING LOGIC ---
    if args.total_len > 0:
        print(f"[INFO] Subsetting enabled. Dividing files into {args.total_len} chunks.")
        
        # Validate the index to prevent errors
        if not 0 <= args.index < args.total_len:
            print(f"[ERROR] Invalid index {args.index}. Must be between 0 and {args.total_len - 1}.")
            return  # Exit the script

        # Use np.array_split for robust, even splitting of files
        subsets = np.array_split(csv_files, args.total_len)
        csv_files_to_process = subsets[args.index]
        
        if len(csv_files_to_process) == 0:
            print(f"[INFO] Subset {args.index + 1}/{args.total_len} is empty. Nothing to do. Exiting.")
            return

        print(f"[INFO] This job will process subset {args.index + 1}/{args.total_len}, which contains {len(csv_files_to_process)} files.")
    else:
        # If total_len is 0 (default), process all files
        csv_files_to_process = csv_files
    # --- END: NEW SUBSETTING LOGIC ---

    for csv_path in csv_files_to_process:
        print(f"\n=== Processing {csv_path.name} ===")
        # Build LMDB and keep a lookup mutant -> sequence for later HDF5 attributes
        df = pd.read_csv(csv_path, usecols=["mutant", "mutated_sequence"])
        df["mutant"] = df["mutant"].astype(str).str.strip()
        df["mutated_sequence"] = df["mutated_sequence"].astype(str).str.upper().str.strip()
        seq_lookup = dict(zip(df["mutant"], df["mutated_sequence"]))

        split = build_lmdb_for_csv(csv_path, data_dir=data_dir, map_size_gb=15.0)
        print(f"[OK] LMDB ready: {data_dir / (split + '.lmdb')}")
        
        
        # Run embedding -> streamed HDF5
        h5_path = run_embedding(
            model_dir=model_dir,
            data_dir=data_dir,
            split=split,
            embed_modelNm=args.embed_name,
            batch_size=args.batch_size,
        )
        print(f"[OK] HDF5 written: {h5_path}")
        
        
        ## Run embedding
        #out_json = run_embedding(
        #    model_dir=model_dir,
        #    data_dir=data_dir,
        #    split=split,
        #    embed_modelNm=args.embed_name,
        #    batch_size=args.batch_size,
        #)
        #print(f"[OK] Embeddings JSON: {out_json}")

        # Convert to H5
        #h5_path = csv_path.with_suffix(".h5")
        #json_to_h5(out_json, h5_path, seq_lookup=seq_lookup, overwrite=args.overwrite)

    print("\nAll done.")


if __name__ == "__main__":
    main()
