from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):

        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)
        # return 5

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(images=item["image"], text=item["captions"], padding="max_length", return_tensors="pt")
        # remove batch dimension
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        labels = encoding["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        encoding["labels"] = labels
        return encoding


def rsicd_data_loader(ds_type, processor=None):
    dataset = load_dataset("arampacha/rsicd", split="{}".format(ds_type))
    if ds_type == 'test':
        return dataset
    else:
        dataset = ImageCaptioningDataset(dataset, processor)
        if ds_type == 'train':
            dataloader = DataLoader(dataset, shuffle=True, batch_size=2)
        else: # if valid
            dataloader = DataLoader(dataset, shuffle=False, batch_size=2)
    return dataloader


def levir_cc_dataloader(dataset, processor):
    pass