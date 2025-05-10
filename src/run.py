import src.filter as filter

import os
import argparse
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser(description="MLLM filter parser")

    # required
    parser.add_argument("--token_or_env", "-t", required=True, help="Llama HF access token, or path to a .env file containing HF_TOKEN: <token>")
    parser.add_argument("--input_path", "-i", required=True, help="Path to dataset file")
    parser.add_argument("--prompt", "-p", required=True, help="Prompt for MLLM")
    parser.add_argument("--output_path", "-o", required=True, help="Path to save filtered dataset")

    # optional / defaults
    parser.add_argument("--image_column", "-img", type=_str_or_int, default=None, help="Name of dataset column with image paths, or index of column with image paths")
    parser.add_argument("--caption_column", "-cap", type=_str_or_int, default=None, help="Name of dataset column with captions, or index of column with captions")
    parser.add_argument("--threshold", "-th", type=float, default=0.5, help="Confidence required for MLLM to predict 'true' (Must follow constraint: 0 < threshold < 1)")
    parser.add_argument("--save_every", "-s", type=int, help="How often to save the filtered dataset. If not provided, then dataset will only be saved at the end")
    parser.add_argument("--has_header", "-hd", action="store_true", help="If your dataset has a header row that needs to be skipped")
    parser.add_argument("--keep_corrupted", "-k", action="store_true", help="Will keep samples with corrupted/missing images")

    # advanced
    parser.add_argument("--max_steps", "-m", type=int, default=None, help="Max number of classifier token generations before giving up and classifying as 'true'. If None, it will always classify on the first try")
    parser.add_argument("--top_k", "-tk", type=int, default=1, help="A classifier token (1 or 0) must appear in the top_k predicted tokens, otherwise continue generating to get a more accurate classification. Only relevant if max_steps != None")

    return parser.parse_args()

def load_dataset(metadata, has_header, delim) -> pd.DataFrame:
    header = 0 if has_header else None
    df = pd.read_csv(metadata, delimiter=delim, header=header)
    if not has_header:
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
    return df

def save_dataset(metadata, output_path, has_header, delim):
    metadata.to_csv(output_path, sep=delim, index=False, header=has_header, encoding='utf-8')
    print(f"Saved filtered dataset to {output_path}")

def mllm_filter(args):
    # input validation
    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input path: {args.input_path} could not be found.")

    if os.path.exists(args.output_path):
        raise FileExistsError(f"Designated output_path: {args.output_path} already exists. Please delete it or provide a different output_path.")
    
    if not args.image_column and not args.caption_column:
        raise argparse.ArgumentError(f"No value was provided for both image column and caption column. Please provide at least one of these.")
    
    load_extension = args.input_path[-4:]
    if load_extension != ".tsv" and load_extension != ".csv":
        raise NameError(f"Provided input_path: {args.input_path} must either be a .tsv or .csv file.")
    load_delim = "\t" if load_extension == ".tsv" else ","
    
    save_extension = args.output_path[-4:]
    if save_extension != ".tsv" and save_extension != ".csv":
        raise NameError(f"Provided output_path: {args.output_path} must either be a .tsv or .csv file.")
    delim = "\t" if save_extension == ".tsv" else ","

    if args.threshold >= 1 or args.threshold <= 0:
        raise ValueError(f"Threshold value of {args.threshold} is invalid. Must follow constraint: 0 < threshold < 1.")
    
    if not args.has_header and (isinstance(args.caption_column, str) or isinstance(args.image_column, str)):
        raise ValueError(f"If has_header is false, then caption_column and/or image_column must be indices, not strings.")

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # load data
    metadata = load_dataset(args.input_path, args.has_header, load_delim)
    if isinstance(args.image_column, int):
        args.image_column = metadata.columns[args.image_column]
    if isinstance(args.caption_column, int):
        args.caption_column = metadata.columns[args.caption_column]

    # filter
    vision = args.image_column != None
    model, processor = filter.get_model(args.token_or_env, vision)
    model.eval()
    corrupted = 0
    if vision:
        results, corrupted = filter.vision_filter(model=model, 
                                        processor=processor, 
                                        metadata=metadata, 
                                        caption_column=args.caption_column, 
                                        image_column=args.image_column, 
                                        prompt=args.prompt, 
                                        output_path=args.output_path,
                                        has_header=args.has_header,
                                        delim=delim, 
                                        threshold=args.threshold, 
                                        save_every=args.save_every, 
                                        max_steps=args.max_steps, 
                                        topk=args.top_k,
                                        keep_corrupted=args.keep_corrupted)
    else:
        results = filter.text_filter(model=model,
                                tokenizer=processor,
                                metadata=metadata,
                                caption_column=args.caption_column,
                                prompt=args.prompt, 
                                output_path=args.output_path,
                                has_header=args.has_header,
                                delim=delim, 
                                threshold=args.threshold, 
                                save_every=args.save_every, 
                                max_steps=args.max_steps, 
                                topk=args.top_k)

    # print filter stats and save dataset
    filtered_dataset = metadata[results]
    if args.keep_corrupted:
        corrupted = 0
    filtered_count = len(metadata) - len(filtered_dataset) - corrupted
    print(f"\nFiltered out {filtered_count} samples.")
    if not args.keep_corrupted and corrupted:
        print(f"Removed an additional {corrupted} samples that were missing or corrupted.")
    save_dataset(filtered_dataset, args.output_path, args.has_header, delim)

def _str_or_int(value):
    # to allow column name or column index
    try:
        return int(value)
    except ValueError:
        return value

if __name__ == "__main__":
    # remove any samples with corrupted images before running this.
    args = get_args()
    # args.prompt = args.prompt.replace("\\n", "\n") # for debugging with launch.json
    mllm_filter(args)