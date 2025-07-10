import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
 
def load_model(model_name="llava-1.5-7b-hf", use_4bit=True, device_map="auto"):
    hf_repo = model_name
    if model_name == "llava-1.5-7b":
        hf_repo = "llava-hf/llava-1.5-7b-hf"
    elif model_name == "llava-1.5-13b":
        hf_repo = "liuhaotian/llava-v1.5-13b"
    elif model_name == "llava-1.6-7b":
        hf_repo = "liuhaotian/llava-v1.6-vicuna-7b"
    elif model_name == "llava-1.6-13b":
        hf_repo = "liuhaotian/llava-v1.6-vicuna-13b"
 
    load_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": device_map
    }
 
    if use_4bit:
        load_kwargs["load_in_4bit"] = True
    else:
        load_kwargs["low_cpu_mem_usage"] = True
 
    model = LlavaForConditionalGeneration.from_pretrained(hf_repo, **load_kwargs)
    processor = AutoProcessor.from_pretrained(hf_repo)
    return model, processor
 
def load_image(image_path):
    image = Image.open(image_path)
    return image.convert("RGB")
 
def format_prompt(question):
    question = question.strip()
    prompt = f"USER: <image>\n{question} ASSISTANT:"
    return prompt
 
def answer_question(model, processor, image, question, inputs=None):
    """
    Generate answer from the model for a given image and question.
    If `inputs` is provided (already preprocessed and on correct device), uses it directly.
    """
    if inputs is None:
        if isinstance(image, str):
            image = load_image(image)
        prompt = format_prompt(question)
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}  # ensure device alignment
 
    outputs = model.generate(**inputs, max_new_tokens=256)
    output_text = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
 
    if "ASSISTANT:" in output_text:
        answer = output_text.split("ASSISTANT:")[-1].strip()
    else:
        answer = output_text.strip()
 
    return answer