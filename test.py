import torch
import logging
import numpy as np
import pandas as pd

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import classification_report

from dataloader import TestDataloader


def test():
    target_list = []
    predictions = []
    model.eval()
    torch.set_grad_enabled(False)
    for step, data in enumerate(test_dataloader, 0):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        targets = data['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)

        logits, idx = torch.max(outputs[0], dim=1)
        preds = idx.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        target_list.extend(targets)
        predictions.extend(preds)

    return predictions, target_list


if __name__ == "__main__":

    # reproducability
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    logging.basicConfig(level=logging.ERROR)

    MAX_LEN = 200
    BATCH_SIZE = 16
    PRE_TRAINED_MODEL_NAME = 'csebuetnlp/banglabert'

    tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    df_test = pd.read_csv("/content/10K_Dataset/Final_Test.csv")
    test_dataloader = TestDataloader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

    model = AutoModelForSequenceClassification.from_pretrained(
        PRE_TRAINED_MODEL_NAME, num_labels=3)
    # set your saved model path here
    model_path = r'/content/drive/MyDrive/Saved Models/logs/banglabert-auto_5e-5_3eps/banglabert-auto_5e-5_3eps.pth'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    y_pred, y_true = test()
    print(classification_report(y_true, y_pred))
