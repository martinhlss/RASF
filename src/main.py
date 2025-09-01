import spacy
import sentence_transformers
import github.src.retrieval as retrieval
import github.src.template_fill as template_fill
import github.src.pdf_conversion as pdf_conversion

def main():
    
    # Parameters
    pdf_path = "data/pdfs/evaluation/prospectus_01.pdf"
    pickle_folder_path = "data/pickles/evaluation"
    query = "ISIN"
    instruction_path = "system_prompts/instructions/isin.txt"
    examples_path = "system_prompts/examples/isin.txt"
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    lora_path = "lora_weights"

    # Creating system prompt from txt files
    with open(instruction_path, "r", encoding="utf-8") as f:
        instructions = f.read()
    with open(examples_path, "r", encoding="utf-8") as f:
        examples = f.read()
    system_prompt = instructions.strip() + "\n" + examples.strip()

    # Loading tokenization and embedding model
    nlp = spacy.load("en_core_web_sm")
    embedding_model = sentence_transformers.SentenceTransformer("BAAI/bge-m3")

    # Loading query-independent data
    # return format: {"list_chunks": [], "list_tokenized_chunks": [], "list_vectors": []}
    general_data = pdf_conversion.loading_query_independent_data(pdf_path, pickle_folder_path, nlp, embedding_model)

    # Retrieving documents matching query
    # return format: [(chunk, weighted_score, bm25_score, dense_score), ...]
    ranked_chunks = retrieval.retrieval(general_data["list_chunks"], general_data["list_tokenized_chunks"],
                                    general_data["list_vectors"], query, nlp, embedding_model, top_n=5)

    # Creating user prompt for slot filling
    top_5_chunks = [chunk for chunk, _, _, _ in ranked_chunks[:5]]
    context = " . . . ".join(top_5_chunks)

    # Initializing LoRA fine-tuned LLM
    base_model, tokenizer = template_fill.load_base_model(model_id)
    merged_model = template_fill.load_lora_model(base_model, lora_path)
    llm = template_fill.init_llm_pipeline(merged_model, tokenizer)

    # Generating slot filling output
    output = template_fill.slot_filling(llm, context, system_prompt)
    print(output)

if __name__ == "__main__":
    main()
