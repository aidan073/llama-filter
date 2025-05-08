import os
import torch
from tqdm import tqdm
from PIL import Image
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import MllamaForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

def get_model(token_or_env):

    MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    hf_token = token_or_env
    if(hf_token[-4:] == ".env"):
        load_dotenv(hf_token)
        hf_token = os.getenv("HF_TOKEN") 
    login(hf_token)

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = MllamaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # quantization has a large negative effect on perceived classification accuracy
    # BNB_CONFIG = BitsAndBytesConfig(load_in_8bit=True)
    # model = AutoModelForCausalLM.from_pretrained(
    #     MODEL_ID,
    #     quantization_config=BNB_CONFIG,
    #     device_map="auto",
    #     trust_remote_code=True,
    # )

    return model, processor

def filter(model, processor, metadata, prompt, output_path, threshold:int=0.5, save_every:int=None):
    """
    Filter metadata file using an MLLM
    """
    msg = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": None}
        ]}
    ]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    req_logit_diff = torch.log(torch.tensor(threshold / (1 - threshold))) # required logit difference to meet confidence threshold (inverse sigmoid)
    true_token_id = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.tokenize("1")[0])
    false_token_id = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.tokenize("0")[0])
    results = []
    since_last_save = 0
    samples = metadata.iterrows()
    for idx, sample in tqdm(samples, total=len(samples), desc="Classifying Samples"):
        msg[0]["content"][1]["text"] = prompt.format(text=sample.loc("caption"))
        input_text = processor.apply_chat_template(msg, add_generation_prompt=True)
        input_image = Image.open(sample.loc("image_path"))
        input = processor(input_image, input_text, add_special_tokens=False, truncation=True, return_tensors="pt").to(device)
        results.append(classify(model, input, req_logit_diff, true_token_id, false_token_id, topk=1))
        input_image.close()

        since_last_save += 1
        if save_every and since_last_save >= save_every:
            metadata[results].to_csv(output_path, sep='\t', index=False, encoding='utf-8')
            since_last_save = 0

    return results

def classify(model, input, req_logit_diff, id_1, id_0, max_steps=10, topk=1)->bool:
    """
    Classify until the model gives a clear answer or reaches max_steps. If max_steps is reached, false classification is assumed.
    **only works for 1 sample at a time currently**
    """
    device = input["input_ids"].device

    for _ in range(max_steps):
        with torch.no_grad():
            output = model(**input)
            logits = output.logits[:, -1, :] # shape: (1, vocab_size)

        topk_ids = torch.topk(logits, topk, dim=-1).indices[0].tolist()

        if id_1 in topk_ids or id_0 in topk_ids:
            target_logits = logits[:, [id_1, id_0]]
            difference = target_logits[:, 0] - target_logits[:, 1]
            prediction = difference >= req_logit_diff
            # logit_pred = torch.nn.functional.softmax(target_logits, dim=1)
            # prediction = True if logit_pred[0, 0] >= threshold else False

            return prediction.item()

        # Append the most likely token and update attention masks
        next_token_id = topk_ids[0]
        # print(processor.tokenizer.decode([next_token_id]))
        next_token_tensor = torch.tensor([[next_token_id]], device=device)
        input["input_ids"] = torch.cat([input["input_ids"], next_token_tensor], dim=1)
        next_attention_mask = torch.ones_like(next_token_tensor)
        input["attention_mask"] = torch.cat([input["attention_mask"], next_attention_mask], dim=1)
        next_cross_attention_mask = torch.tensor([[[[1, 1, 0, 0]]]], device=device)
        input["cross_attention_mask"] = torch.cat([input["cross_attention_mask"], next_cross_attention_mask], dim=1)
        "[[[[1,1,0,0]]],[[[1,1,0,0]]],...]"
        
    print(f"Reached classification attempt limit of {max_steps}")
    return True