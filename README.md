# Llama Filtering

## Description
This script allows you to filter a dataset using Llama with a single command.

## Installation

This project has only been tested for Python 3.10.12. To get started:

```
git clone https://github.com/aidan073/llama-filter.git
pip install -r requirements.txt
cd path/to/llama-filter
```
  
**Example Videos:**  
[Usage & Examples](https://www.youtube.com/watch?v=Vhy5E8jTCWs)  
[(extra) Threshold Argument Explanation](https://www.youtube.com/watch?v=hePC_rnJaWM)  
  
## Usage  
**Prompt requirements:** You MUST format the prompt so that it is asking llama to output a 1 (for true), and a 0 (for false). False samples will be filtered out, while true samples will be kept. I strongly suggest you include "only output the number and nothing else" in your prompt, to avoid skewing the logits towards other tokens such as the bot_token. If you pass in a caption column with the -cap flag, you should have {caption} in your prompt wherever the caption should go.  
  
**Filter Modes:**  
1. Image Only:  
If you use "-img" flag only (no -cap) → filters based on images using llama3.2-vision  
  
2. Image + Caption:  
If you use both "-img" and "-cap" flags → filters based on images & captions using llama3.2-vision  
  
3. Caption Only:  
If you use "-cap" flag only (no -img) → filters based on captions using llama3.1

### Basic Usage
```
python -m src.run -t TOKEN_OR_ENV -i INPUT_PATH -p PROMPT -o OUTPUT_PATH [-img IMAGE_PATH_COLUMN] [-cap CAPTION_COLUMN] [-hd (if input data has a header)]
```
1. Image Only filtering example:
```
python -m src.run -t path/to/.env -i path/to/dataset.tsv -p "If the image contains a dog output 1, else output 0." -o path/to/output.tsv -img image_path_column -hd
```
2. Image + Caption filtering example:
```
python -m src.run -t path/to/.env -i path/to/dataset.tsv -p "Caption: {caption}\n\nOutput 1 if the caption matches the image, else output 0." -o path/to/output.tsv -img image_path_column -cap caption_column -hd
```  
3. Caption Only filtering example:   
```
python -m src.run -t path/to/.env -i path/to/dataset.tsv -p "Caption: {caption}\n\nOutput 1 if this caption is related to math. Output 0 if this caption is not related to math." -o path/to/output.tsv -cap caption_column -hd
```

### Full Usage
```
python -m src.run -t TOKEN_OR_ENV -i INPUT_PATH -p PROMPT -o OUTPUT_PATH [-img IMAGE_PATH_COLUMN] [-cap CAPTION_COLUMN] [-hd (if input data has a header)] [-th THRESHOLD] [-s SAVE_EVERY] [-k KEEP_CORRUPTED] [-m MAX_STEPS] [-tk TOP_K]
```
Example:
```
python -m src.run -t path/to/.env -i path/to/dataset.tsv -p "Caption: {caption}\n\nOutput 1 if the caption matches the image, else output 0. Only output the number and no extra text." -o path/to/output.tsv -img image_path_column -cap caption_column -hd -th 0.6 -s 1000 -k -m 5 -tk 1
```

### Argument Explanations  
**Key args:**  
**-t:** You must provide a hugging face token with access to llama 3.1 and 3.2. You can directly pass the token, or a .env containing HF_TOKEN variable.  
**-i:** Path to your dataset. Must be a .csv or .tsv file.  
**-p:** Your prompt for llama. You MUST format the prompt so that it is asking llama to output a 1 (for true), and a 0 (for false). If you pass in a caption column with the -cap flag, you should have {caption} in your prompt where the caption should go.  
**-o:** Path to save the filtered dataset. Must be a .csv or .tsv file.  
**-img (optional):** Name/index of the image column in your dataset.  
**-cap (optional):** Name/index of the caption column in your dataset.  
**-hd:** Include this flag if your dataset has a header row.  
  
**Extra args:**  
**-th (default=0.5):** Confidence level required by Llama to give a 'True' classification. Must follow constraint 0 < threshold < 1. Larger value means more confidence required, smaller value means less confidence required.  
**-s (optional):** Save the filtered dataset every time this many new classifications have been made. Good safety feature for large datasets, in case of a crash.  
**-k (optional):** This flag causes samples with corrupted images to NOT get filtered out (by default they get filtered).  
  
**Advanced args:**  
Llama often generates a bot_token, and sometimes other random tokens before outputing 1 or 0. This can skew the logit values for the 1 and 0 tokens, which could lead to less accurate classifications. For this reason, you can use the next two arguments to keep generating tokens for a sample until the max_steps are reached (-m), or until one of the top_k (-tk) tokens is 1 or 0. When the latter occurs, it will consider this a safe time to make the classification. When the former occurs, it gives up and labels that sample 1 (aka. "True").  
**-m (optional):** MAX_STEPS  
**-tk (optional/default=1):** TOP_K