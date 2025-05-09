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

### Simple Usage
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
python -m src.run -t TOKEN_OR_ENV -i INPUT_PATH -p PROMPT -o OUTPUT_PATH -i IMAGE_COLUMN [-cap CAPTION_COLUMN] [-th THRESHOLD] [-hd (if input data has a header)] [-s SAVE_EVERY] [-m MAX_STEPS] [-tk TOP_K]
```
Example:
```
python -m src.run -t .env -i test_files/metadata.csv -p "Caption: {caption}\n\nOutput 1 if the caption matches the image, else output 0. Only output the number and no extra text." -o test_files/results.tsv -th 0.6 -cap caption -img img_p -s 2 -hd -m 5
```

### Argument Explanations