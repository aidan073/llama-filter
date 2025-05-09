# MLLM Filtering

## Description
This script allows you to filter a dataset using Llama3.2-Vision with a single command.

## Installation

This project has only been tested for Python 3.10.12. To get started:

```
git clone https://github.com/aidan073/mllm-filter.git
pip install -r requirements.txt
```

## Usage
**Prompt requirements:** You MUST format the prompt so that it is asking llama to output a 1 (for true), and a 0 (for false). False samples will be filtered out, while true samples will be kept. I strongly suggest you include "only output the number and nothing else" in your prompt, to avoid skewing the logits towards other tokens such as the bot_token. If you pass in a caption column with the -cap flag, you should have {caption} in your prompt wherever the caption should go.

### Basic Usage
```
python -m src.run -t TOKEN_OR_ENV -i INPUT_PATH -p PROMPT -o OUTPUT_PATH -img IMAGE_COLUMN [-cap CAPTION_COLUMN] [-hd (if input data has a header)]
```
Example without caption:
```
python -m src.run -t path/to/.env -i path/to/metadata.tsv -p "If the image contains a dog output 1, else output 0." -o path/to/output.tsv -img image_path -hd
```
Example with caption:
```
python -m src.run -t path/to/.env -i path/to/metadata.tsv -p "Caption: {caption}\n\nOutput 1 if the caption matches the image, else output 0." -o path/to/output.tsv -img image_path -cap caption -hd
```

### Full Usage
```
python -m src.run -t TOKEN_OR_ENV -i INPUT_PATH -p PROMPT -o OUTPUT_PATH -img IMAGE_COLUMN [-cap CAPTION_COLUMN] [-hd (if input data has a header)] [-th THRESHOLD] [-s SAVE_EVERY] [-m MAX_STEPS] [-tk TOP_K]
```
Example:
```
python -m src.run -t path/to/.env -i path/to/metadata.tsv -p "Caption: {caption}\n\nOutput 1 if the caption matches the image, else output 0. Only output the number and no extra text." -o path/to/output.tsv -img img_p -cap caption -hd -th 0.6 -s 1000 -m 5 -tk 1
```

### Argument Explanations
**-t:** You must provide a hugging face token with access to llama 3.2. You can directly pass the token, or a .env containing HF_TOKEN variable.  
**-i:** Path to your dataset. Must be a .csv or .tsv file.  
**-p:** Your prompt for llama. You MUST format the prompt so that it is asking llama to output a 1 (for true), and a 0 (for false). If you pass in a caption column with the -cap flag, you should have {caption} in your prompt where the caption should go.  
**-o:** Path to save the filtered dataset. Must be a .csv or .tsv file.  
**-img:** Name of the image column in your dataset. Alternatively, you can also pass in an int representing the location of the column (0 indexed). This is ideal for no header.  
**-cap (optional):** Name of the caption column. If you do not need the captions in your prompt, you don't need the argument. If you do, then see "prompt requirements".  
**-hd (default=False):** Use this flag if your dataset has a header.  
**-th (default=0.5):** Confidence level required by Llama to give a 'True' classification. Must follow constraint 0 < threshold < 1. Larger value means more confidence required, smaller value means less confidence required.  
**-s (optional):** Save the filtered dataset every time this many new classifications have been made. Good safety feature for large datasets, in case of a crash.  
  
**Advanced args:**  
Llama often generates a bot_token, and sometimes other random tokens before outputing 1 or 0. This can skew the logit values for the 1 and 0 tokens, which could lead to less accurate classifications. For this reason, you can use the next two arguments to keep generating tokens for a sample until the max_steps are reached (-m), or until one of the top_k (-tk) tokens is 1 or 0. When the latter occurs, it will consider this a safe time to make the classification. When the former occurs, it gives up and labels that sample 1 (aka. "True").  
**-m (optional):** MAX_STEPS  
**-tk (optional/default=1):** TOP_K