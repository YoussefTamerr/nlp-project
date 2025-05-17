from datasets import load_dataset, Dataset

# Step 1: Stream and load first 1000 examples
ds = load_dataset("deepmind/narrativeqa", split="train", streaming=True)
samples = [example for _, example in zip(range(1000), ds)]

# Step 2: Save to disk as Hugging Face dataset
dataset_1000 = Dataset.from_list(samples)
dataset_1000.save_to_disk("narrativeqa_1000")