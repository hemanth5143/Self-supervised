### Fine-Tuning LLAMA-2 Model with Bhagavad Gita Text

#### Objective:
The goal of this project is to fine-tune the LLAMA-2 model using the text extracted from the Bhagavad Gita (Spanish 1975 edition) to enhance the model's understanding and responses in the context of this specific text.

#### Tools and Libraries Used:
- **Transformers**: For tokenization and model handling.
- **Datasets**: For creating and managing the dataset.
- **PEFT**: For parameter-efficient fine-tuning using LoRA (Low-Rank Adaptation).
- **Hugging Face Hub**: For accessing and storing models.
- **PyPDF2**: For extracting text from the PDF file.
- **Torch**: For deep learning operations.
- **SentencePiece**: For tokenizer operations.
- **Accelerate, BitsAndBytes, TRL**: For efficient training.

#### Steps and Workflow:

1. **Install Required Libraries:**
   Install necessary Python libraries to facilitate text extraction, tokenization, and model training.

2. **Extract Text from PDF:**
   Use `PyPDF2` to extract the text from the Bhagavad Gita PDF file.

    ```python
    def extract_text_from_pdf(pdf_path):
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    ```

3. **Create a Dataset:**
   Convert the extracted text into a dataset using the `datasets` library.

    ```python
    corpus = extract_text_from_pdf(pdf_path)
    dataset = Dataset.from_dict({"text": [corpus]})
    ```

4. **Authenticate with Hugging Face Hub:**
   Log in to the Hugging Face Hub to access and manage models.

    ```python
    from huggingface_hub import login
    token = "your_huggingface_token"
    login(token)
    ```

5. **Load and Tokenize the Model:**
   Load the pre-trained LLAMA-2 model and tokenizer, and then tokenize the dataset.

    ```python
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', load_in_8bit=True)

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    ```

6. **Configure LoRA (Low-Rank Adaptation):**
   Set up LoRA for parameter-efficient fine-tuning.

    ```python
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
    )

    model = get_peft_model(model, lora_config)
    ```

7. **Set Up Training Arguments:**
   Define the training parameters, including batch size, learning rate, and evaluation strategy.

    ```python
    from transformers import TrainingArguments

    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
        learning_rate=2e-5,
        weight_decay=0.01,
        fp16=True,
        logging_steps=500,
        evaluation_strategy="steps",
        eval_steps=5000,
    )
    ```

8. **Prepare Data Collator:**
   Create a data collator for language modeling without masked language modeling (MLM).

    ```python
    from transformers import DataCollatorForLanguageModeling

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    ```

9. **Initialize and Train the Model:**
   Set up the `Trainer` and begin the training process.

    ```python
    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    ```

10. **Save the Fine-Tuned Model:**
    Save the trained model for future use.

    ```python
    trainer.save_model("./finetuned-llama2-bhagavad-gita")
    ```

#### Result:
The LLAMA-2 model is successfully fine-tuned using the text from the Bhagavad Gita. The fine-tuned model is saved locally for further use and deployment.

This project demonstrates a full pipeline for fine-tuning a language model using custom text data, leveraging advanced techniques like LoRA for efficient training.
