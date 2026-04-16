from datasets import load_dataset

dataset = load_dataset(
    "mozilla-foundation/common_voice_13_0",
    "en",
    split="train",
    streaming=True
)

for i, sample in enumerate(dataset):
    print(sample["age"])
    if i == 5:
        break