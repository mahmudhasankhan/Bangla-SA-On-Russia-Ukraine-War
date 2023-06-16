from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
import torch

if __name__ == "__main__":

    # reproducability
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    PRE_TRAINED_MODEL_NAME = 'csebuetnlp/banglabert'
    tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    # load model
    model = AutoModelForSequenceClassification.from_pretrained(
        PRE_TRAINED_MODEL_NAME, num_labels=3)
    model_path = r'E:\BanglaSent-SavedModels\saved-models\xlm_roberta_base_final.pt'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model = model.to(device)

    # example texts

    text1 = 'আমরা যুদ্ধ চাই না, আমরা শান্তি চাই।'  # positive
    text2 = 'রাশিয়া ধ্বংস করে দাও ইউক্রেনকে'  # negative
    text3 = 'আল্লাহ ভালো জানেন কি হবে সামনে।'  # neutral

    # tokenize the input strings
    encoded_input = tokenizer(text3, return_tensors='pt')
    encoded_input = encoded_input.to(device)  # move them to gpu

    # Inference
    output = model(**encoded_input)
    _, pred = torch.max(output.logits, dim=1)  # get the prediction

    pred = pred.detach().cpu().numpy()  # from tensors to ndarray

    print(pred)

    if pred == 0:
        print('neutral')
    elif pred == 1:
        print('positive')
    else:
        print('negative')
