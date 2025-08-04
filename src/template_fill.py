from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import os

# Load base LLM and tokenizer from Hugging Face
def load_base_model():
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", rust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return model, tokenizer

# Load LoRA adapters and merge them with base model
def load_lora_model(base_model, lora_path):
    if not os.path.isdir(lora_path):
        raise FileNotFoundError(f"LoRA path not found: {lora_path}")
    # Load the PEFT adapters onto the existing model
    model = PeftModel.from_pretrained(base_model, lora_path)
    # Merge the base model and the adapters for inference
    model = model.merge_and_unload()
    return model

# Initialize pipline for merged model 
def init_llm(model, tokenizer, max_new_tokens=150):
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )

# Return system and user prompt in required format
def create_full_prompt(system_prompt, user_prompt):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

# Perform slot filling given context and system prompt
def slot_filling(llm_pipeline, context, system_prompt):
  full_prompt = create_full_prompt(system_prompt, context)
  output = llm_pipeline(full_prompt)
  return output[0]["generated_text"][2]["content"]