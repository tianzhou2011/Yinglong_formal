import glob
import json
import os
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
import lit_gpt.packed_dataset as packed_dataset


filenames_sample = [
    'dataset_1.json',
    #...
    'dataset_m.json',
]




def prepare_sample(
    source_path: Path, 
    checkpoint_dir: Path, 
    destination_path: Path, 
    chunk_size: int, 
    match: str = ""
    
) -> None:
    

    destination_path.mkdir(parents=True, exist_ok=True)
    for name in filenames_sample:
        if match and match not in name:
            continue

        filepath = source_path / name
                    
        prefix, _ = os.path.splitext(name)

        builder = packed_dataset.PackedDatasetBuilder(
            outdir=destination_path,
            prefix=prefix,
            chunk_size=chunk_size,
            dtype=np.float32,
            vocab_size=1145,
        )

        print(f"Processing {name}")

        with open(filepath, encoding="utf-8") as f:
            for row in tqdm(f):
                text = json.loads(row)["text"]
                text_ids = [float(x) for x in text.split(' ') if x != '' ]
                builder.add_array(np.array(text_ids, dtype=builder.dtype))

        builder.write_reminder()
        
        

def prepare_merge(
    source_path: Path, 
    checkpoint_dir: Path, 
    destination_path: Path, 
    chunk_size: int, 
    match: str = ""
    
) -> None:
    
    """Prepare the "Red Pajama" dataset using the original tokenizer."""
    destination_path.mkdir(parents=True, exist_ok=True)
    
    builder = packed_dataset.PackedDatasetBuilder(
            outdir=destination_path,
            prefix='era_5_14_all',
            chunk_size=chunk_size,
            dtype=np.float32,
            vocab_size=1145,
        )



    for name in filenames_sample:
        if match and match not in name:
            continue

        filepath = source_path / name 
        prefix, _ = os.path.splitext(name)
        print(f"Processing {name}")

        with open(filepath, encoding="utf-8") as f:
            for row in tqdm(f):
                text = json.loads(row)["text"]
                text_ids = [float(x) for x in text.split(' ') if x != '' ]
                builder.add_array(np.array(text_ids, dtype=builder.dtype))

    builder.write_reminder()


def prepare_full(
    source_path: Path, 
    checkpoint_dir: Path, 
    destination_path: Path, 
    chunk_size: int, 
    match: str = "",
    split: list = [1.0,0,0],
) -> None:

    destination_path.mkdir(parents=True, exist_ok=True)

    for set_name, pattern in filename_sets.items():
        if match and match not in set_name:
            continue

        # is_cc = set_name == "common_crawl"

        filenames = glob.glob(os.path.join(source_path, pattern), recursive=True)

        if not filenames:
            raise RuntimeError(
                f"No files matching {pattern} found at {source_path}."
            )

            
        builder = packed_dataset.PackedDatasetBuilder(
            outdir=destination_path,
            prefix=set_name,
            chunk_size=chunk_size,
            vocab_size=1145,
            dtype=np.float32,
        )

        for name in filenames:
            filepath = source_path / name

            print(f"Processing {name}")
            
            with open(filepath, encoding="utf-8") as f:
                for row in tqdm(f):
                    text = json.loads(row)["text"]
                    # text_ids = tokenizer.encode(text)
                    text_ids = [float(x) for x in text.split(' ')]
                    # print(test_ids)
                    builder.add_array(np.array(text_ids, dtype=builder.dtype))

        builder.write_reminder()


def prepare(
    source_path: Path = Path("data/RedPajama-Data-1T-Sample"),
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    destination_path: Path = Path("data/redpajama_sample"),
    sample: bool = True,
    match: str = "",
) -> None:
    
    """Prepare the "Red Pajama" dataset. We assume tokenizer has been trained."""
 
    block_size = 2048
    chunk_number = 2048
    prepare_fn = prepare_sample if sample else prepare_full
    prepare_fn(
        source_path=source_path,
        checkpoint_dir=checkpoint_dir,
        destination_path=destination_path,
        chunk_size=(block_size + 1) * chunk_number, 
        match=match,
    )


if __name__ == "__main__":
    prepare(source_path = Path("./"),
            checkpoint_dir = Path("./checkpoint"),
            destination_path = Path("./data_processed_new")
    )
    