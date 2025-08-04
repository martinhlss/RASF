# Information Retrieval and LoRA-based Template Filling Models as part of Master Thesis

This repository contains the information retrieval models and LoRA-fine-tuned Llama 3.1 8B Instruct model for template filling. Together they form a retrieval-augmented slot filling model for prospectuses.

## Setup

Install required Python packages:
pip install -r requirements.txt

## Usage

Run the example pipeline:
python src/main.py

This runs document retrieval on sample data and returns an output using the fine-tuned LLM that aligns with the pre-defined template.

## Project Structure

- `src/`: Python scripts (pdf_conversion, retrieval, template_fill, main).
- `data/`: Sample of prospectuses saved with its embedded chunks saved as pickle file.
- `lora_weights/`: LoRA weights for LLama 3.1 8B Instruct model.
- `system_prompts/`: Instructions and few-shot examples. Together they constitute the system prompt for each information need.

## Notes

See [CREDITS.md](CREDITS.md) for third-party code and acknowledgments
Code was created by and with the assistance of generative AI
