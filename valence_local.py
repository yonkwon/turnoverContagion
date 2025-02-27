import os
import pandas as pd
import datetime
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer

# Adapted Python code provided by
# Speer, A. B., Perrotta, J., Tenbrink, A. P., Wegmeyer, L. J., Delacruz, A. Y., & Bowker, J. (2023). Turning words into numbers: Assessing work attitudes using natural language processing. Journal of Applied Psychology, 108(6), 1027.

if __name__ == "__main__":

    # File paths - Update these to match your local file structure
    filepath_for_data = "df.csv"  # Update with your dataset path
    filepath_for_BERT = "BERT"  # Folder containing BERT model
    filepath_for_output = "Predicted_Scores"  # Output directory

    # Load dataset
    df = pd.read_csv(filepath_for_data)
    n_rows = len(df["Text"])
    batch_size = 8

    # Define construct prediction settings
    run_all_constructs = True
    number_of_constructs_to_predict = 0

    if run_all_constructs:
        number_of_constructs_to_predict = 25

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(examples["Text"], max_length=512, padding="max_length",
                         truncation=True)

    # Convert dataset to Hugging Face format and tokenize
    test = df[["Text"]].astype(str)
    test = Dataset.from_pandas(test)
    tokenized_test = test.map(tokenize_function, batched=True)

    # Prediction function
    def predict(construct_name_string):
        filepath = f"{filepath_for_BERT}/{construct_name_string}"
        model = AutoModelForSequenceClassification.from_pretrained(filepath, local_files_only=True)
        trainer = Trainer(model=model)
        # Uncomment if running on GPU
        # trainer.model = trainer.model.cuda()

        construct_Composite = trainer.predict(tokenized_test)
        construct_Composite = pd.DataFrame(construct_Composite.predictions)

        new_construct_name = f"Pred_{construct_name_string}"
        construct_Composite = construct_Composite.rename(columns={0: new_construct_name})

        os.makedirs(filepath_for_output, exist_ok=True)

        construct_filepath = f"{filepath_for_output}/{construct_name_string}_predicted_scores.csv"
        construct_Composite.to_csv(construct_filepath, index=False)
        print(f"Saved prediction results to: {construct_filepath}")

    # Construct list to predict
    construct_list = ["Turnover_Intentions"]

    if run_all_constructs:
        for construct in construct_list:
            predict(construct)
    else:
        secondary_construct_list_input = ""  # Provide a comma-separated list of constructs
        secondary_construct_list = secondary_construct_list_input.split(", ")
        for construct in secondary_construct_list:
            predict(construct)

    print("\nPredictions complete! Your predicted construct scores are saved in:", filepath_for_output)

    # Combine function
    def combine(construct_name_string):
        construct_filepath = f"{filepath_for_output}/{construct_name_string}_predicted_scores.csv"
        if not os.path.exists(construct_filepath):
            print(f"Warning: File not found - {construct_filepath}")
            return
        temp_df = pd.read_csv(construct_filepath)
        temp_var = temp_df.iloc[:, 0]
        new_construct_name_string = f"Pred_{construct_name_string}"
        final_df[new_construct_name_string] = temp_var
        return final_df

    # Combine predicted constructs into one CSV
    final_df = pd.DataFrame()
    if run_all_constructs:
        for construct in construct_list:
            combine(construct)
    else:
        for construct in secondary_construct_list:
            combine(construct)

    output_filepath = f"{filepath_for_output}/Combined_Constructs.csv"
    final_df.to_csv(output_filepath, index=False)
    print("Combined CSV saved at:", output_filepath)
