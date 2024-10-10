import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import Dataset
import pandas as pd


class PandasDataset(Dataset):
    def __init__(self, dataframe, input_col='input', output_col='output'):
        self.dataframe = dataframe
        self.input_col = input_col
        self.output_col = output_col

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        input_data = row[self.input_col]
        output_data = row[self.output_col]

        input_tensor = torch.tensor(input_data, dtype=torch.long)
        output_tensor = torch.tensor(output_data, dtype=torch.long)
        return input_tensor, output_tensor


def preprocess_dataset(dataset_name, split_ratio='train[:10%]', input_col='input', output_col='output', max_length=512, entries_per_dataset=64, num_datasets=10):
    # Load dataset
    dataset = load_dataset(dataset_name, split=split_ratio)
    dataset = dataset.remove_columns(['instruction', 'text'])

    # Limit the dataset to total required entries
    total_entries = entries_per_dataset * num_datasets
    if len(dataset) > total_entries:
        dataset = dataset.select(range(total_entries))

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad_token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Preprocess the entire dataset
    processed_data = {
        input_col: [tokenizer(example[input_col], padding='max_length', truncation=True, max_length=max_length)['input_ids'] for example in dataset],
        output_col: [tokenizer(example[output_col], padding='max_length', truncation=True, max_length=max_length)['input_ids'] for example in dataset]
    }

    # Convert to DataFrame
    df = pd.DataFrame(processed_data)

    # Split the dataset into `num_datasets` each with `entries_per_dataset` rows
    datasets = [df.iloc[i * entries_per_dataset:(i + 1) * entries_per_dataset] for i in range(num_datasets)]

    return datasets

# Example usage:
dataset_name = "tatsu-lab/alpaca"
processed_datasets = preprocess_dataset(dataset_name, entries_per_dataset=64, num_datasets=5)
processed_datasets.append(processed_datasets[1])