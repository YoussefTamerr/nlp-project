from datasets import load_from_disk


# Load previously saved dataset
samples = load_from_disk("narrativeqa_1000")

ds_train_context = [row['document']['summary']['text'] for row in samples]
ds_train_question = [row['question']['text'] for row in samples]
ds_train_answer = [row['answers'][0]['text'] for row in samples]

print("Loaded dataset with {} samples.".format(len(samples)))
print(ds_train_question)