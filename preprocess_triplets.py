from datasets import load_dataset as hf_load_dataset
from swift.llm import register_dataset, DatasetMeta

def preprocess_my_dataset(dataset):
    """Preprocess the entire dataset for embedding task with triplets"""
    def process_example(example):
        # Format for embedding tasks according to swift docs
        query_text = f"{example['query']['title']}\n{example['query']['abstract']}"
        pos_text = f"{example['pos']['title']}\n{example['pos']['abstract']}"
        neg_text = f"{example['neg']['title']}\n{example['neg']['abstract']}"

        return {
            "messages": [{"role": "user", "content": query_text}],
            "positive_messages": [[{"role": "user", "content": pos_text}]],  # list-of-list, outer length must be 1
            "negative_messages": [[{"role": "user", "content": neg_text}]]   # list-of-list
        }

    # Remove original columns to keep only messages, positive_messages, negative_messages
    return dataset.map(process_example, remove_columns=dataset.column_names)

def load_specter_dataset(dataset_syntax, dataset_meta, **kwargs):
    """Custom load function for the specter dataset"""
    # Load both train and validation splits
    train_dataset = hf_load_dataset("allenai/scirepeval", "cite_prediction", split="train", streaming=True)
    val_dataset = hf_load_dataset("allenai/scirepeval", "cite_prediction", split="validation", streaming=True)
    
    # Limit for testing
    train_dataset = train_dataset.take(1600)
    val_dataset = val_dataset.take(200)  # 10-20% of training size
    
    def process_example(example):
        query_text = f"{example['query']['title']}\n{example['query']['abstract']}"
        pos_text = f"{example['pos']['title']}\n{example['pos']['abstract']}"
        neg_text = f"{example['neg']['title']}\n{example['neg']['abstract']}"
        
        return {
            "messages": [{"role": "user", "content": query_text}],
            "positive_messages": [[{"role": "user", "content": pos_text}]],
            "negative_messages": [[{"role": "user", "content": neg_text}]]
        }
    
    train_dataset = train_dataset.map(process_example, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(process_example, remove_columns=val_dataset.column_names)
    
    return train_dataset, val_dataset  # Return tuple for train and val

# Register the dataset with ms-swift
# NOTE: preprocess_func is set to None because we do the transformation in load_function
register_dataset(
    DatasetMeta(
        dataset_name='specter_triplets',
        load_function=load_specter_dataset,
        preprocess_func=None  # No additional preprocessing needed
    )
)

# Test code (only runs if this file is executed directly)
if __name__ == "__main__":
    dataset = hf_load_dataset("allenai/scirepeval", "cite_prediction", streaming=True)

    def process_single_example(example):
        system_prompt = "Given a scientific paper title and abstract, find other scientific papers which have a citation relationship with it."
        system_message = {"role": "system", "content": system_prompt}
        user_message = {"role": "user", "content": f"{example['query']['title']}\n{example['query']['abstract']}"}
        positive_messages = {"role": "user", "content": f"{example['pos']['title']}\n{example['pos']['abstract']}"}
        negative_messages = {"role": "user", "content": f"{example['neg']['title']}\n{example['neg']['abstract']}"}
        return {"messages": [system_message, user_message], "positive_messages": [positive_messages], "negative_messages": [negative_messages]}

    for i, example in enumerate(dataset['train'].take(3)):
        print(f"\n--- Example {i} ---")
        print("Raw:", example)

        processed = process_single_example(example)
        print("Processed:", processed)