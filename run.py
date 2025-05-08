import mllm

import os
import argparse
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser(description="MLLM filter parser")

    # required
    parser.add_argument("--token_or_env", "-t", help="Llama HF access token, or path to a .env file containing HF_TOKEN: <token>")
    parser.add_argument("--input_path", "-i", required=True, help="Path to metadata file")
    parser.add_argument("--prompt", "-p", required=True, help="Prompt for MLLM")
    parser.add_argument("--output_path", "-o", required=True, help="Path to save filtered metadata")

    # optional / defaults
    parser.add_argument("--threshold", "-th", type=float, default=0.5, help="Confidence required for MLLM to predict 'true' (Must follow constrain: 0 < threshold < 1)")
    parser.add_argument("--id_column", "-id", type=str, default="0", help="Name of metadata column with ids, or index of column with ids")
    parser.add_argument("--caption_column", "-cap", type=str, default="1", help="Name of metadata column with captions, or index of column with captions")
    parser.add_argument("--image_column", "-img", type=str, default="2", help="Name of metadata column with image paths, or index of column with image paths")
    parser.add_argument("--save_every", "-s", type=int, help="How often to save the filtered dataset. If not provided, then dataset will only be saved at the end.")
    parser.add_argument("--has_header", "-h", action="store_true", help="If your dataset has a header row that needs to be skipped")

    # advanced
    parser.add_argument("--max_steps", "-m", type=int, default=10, help="Max number of classifier token generations before giving up and classifying as 'true'")

    return parser.parse_args()

def load_dataset(metadata, id_column, caption_column, image_column, has_header) -> pd.DataFrame:
   
    header = 0 if has_header else None
    df = pd.read_csv(metadata, delimiter="\t", header=header)
    
    # if no header, create default column names
    if not has_header:
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
    
    def _get_col(col):
        if isinstance(col, int):
            return df.columns[col]
        return col
    
    id_col = _get_col(id_column)
    caption_col = _get_col(caption_column)
    image_col = _get_col(image_column)
    
    df = df[[id_col, caption_col, image_col]]
    df.columns = ['id', 'caption', 'image_path']
    
    return df

def mllm_filter(args):
    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input path: {args.input_path} could not be found.")

    if os.path.exists(args.output_path):
        raise FileExistsError(f"Designated output_path: {args.output_path} already exists. Please delete it or provide a different output_path.")
    
    save_extension = args.output_path[-4:]
    if save_extension != ".tsv" or save_extension != ".csv":
        raise NameError(f"Provided output_path: {args.output_path} must either be a .tsv or .csv file.")
    delim = "\t" if save_extension == ".tsv" else ","

    if args.threshold > 1 or args.threshold < 0:
        raise ValueError(f"Threshold value of {args.threshold} is invalid. Must follow constrain: 0 < threshold < 1.")
    
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    metadata = load_dataset(args.input_path, args.id_column, args.caption_column, args.image_column, args.has_header)
    model, processor = mllm.get_model(args.token_or_env)
    model.eval()
    mllm.filter(model, processor, metadata, args.prompt, args.output_path, delim, args.threshold, args.save_every)

if __name__ == "__main__":
    # remove any samples with corrupted images before running this.
    args = get_args()
    mllm_filter(args)