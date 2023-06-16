import torch
from torch.utils.data import Dataset, DataLoader


class BanglaSent_Dataset(Dataset):

    def __init__(self, comments, tokenizer, label, max_len):
        self.comments = comments
        self.tokenizer = tokenizer
        self.label = label
        self.max_len = max_len

    def __len__(self):
        return (len(self.comments))

    def __getitem__(self, index):
        comment = str(self.comments[index])
        label = self.label[index]
        tokens = self.tokenizer.encode_plus(comment,
                                            add_special_tokens=True,
                                            return_tensors='pt',
                                            truncation=True,
                                            max_length=self.max_len,
                                            padding='max_length',
                                            return_attention_mask=True)
        return {'input_ids': tokens['input_ids'].flatten(),
                'attention_mask': tokens['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)}


class TestDataset(Dataset):

    def __init__(self, comments, tokenizer, label, max_len):
        self.comments = comments
        self.label = label
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return (len(self.comments))

    def __getitem__(self, index):
        comment = str(self.comments[index])
        label = self.label[index]
        tokens = self.tokenizer.encode_plus(comment,
                                            add_special_tokens=True,
                                            return_tensors='pt',
                                            truncation=True,
                                            max_length=self.max_len,
                                            padding='max_length',
                                            return_attention_mask=True)
        return {'input_ids': tokens['input_ids'].flatten(),
                'attention_mask': tokens['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
                }


def Dataloader(df, tokenizer, max_len, batch_size):
    dt = BanglaSent_Dataset(comments=df.Comment.to_numpy(),
                            label=df.Label.to_numpy(),
                            tokenizer=tokenizer,
                            max_len=max_len)
    return DataLoader(dt, batch_size=batch_size, shuffle=True)


def TestDataloader(df, tokenizer, max_len, batch_size):
    dt = TestDataset(comments=df.Comment.to_numpy(),
                     label=df.Label.to_numpy(),
                     tokenizer=tokenizer,
                     max_len=max_len)
    return DataLoader(dt, batch_size=batch_size, shuffle=False)
