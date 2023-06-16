import os
import logging
import torch
import numpy as np
import pandas as pd
import time

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dataloader import Dataloader


# helper func
def calcuate_accu(big_idx, targets):
    n_correct = (big_idx == targets).sum().item()
    return n_correct


# train func
def train(epoch):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    torch.set_grad_enabled(True)
    for step, data in enumerate(train_dataloader, 0):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        targets = data['labels'].to(device)

        model.zero_grad()

        outputs = model(input_ids,
                        attention_mask=attention_mask,
                        labels=targets)

        loss = outputs[0]
        tr_loss = tr_loss + loss.item()
        big_val, big_idx = torch.max(outputs[1], dim=1)
        n_correct += calcuate_accu(big_idx, targets)
        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        if step % 100 == 0:
            loss_step = tr_loss/nb_tr_steps
            accu_step = (n_correct*100)/nb_tr_examples
            print(f"Training Loss per 100 steps: {loss_step}")
            print(f"Training Accuracy per 100 steps: {accu_step}")

        # zero the gradients before backpropagating
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Clip the norm of the gradients to 1.0
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # # When using GPU
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    print(
        f'The Total Accuracy for Epoch {epoch+1}: {(n_correct*100)/nb_tr_examples}')
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Training Loss Epoch at {epoch+1}: {epoch_loss}")
    print(f"Training Accuracy Epoch at {epoch+1}: {epoch_accu}")
    print('=' * 30)

    return epoch_loss, epoch_accu


# validation
def evaluation(epoch):
    val_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.eval()
    torch.set_grad_enabled(False)
    for step, data in enumerate(val_dataloader, 0):

        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        targets = data['labels'].to(device)
        outputs = model(
            input_ids, attention_mask=attention_mask, labels=targets)

        loss = outputs[0]
        val_loss = val_loss + loss.item()

        big_val, big_idx = torch.max(outputs.logits, dim=1)
        n_correct += calcuate_accu(big_idx, targets)
        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        if step % 50 == 0:
            loss_step = val_loss/nb_tr_steps
            accu_step = (n_correct*100)/nb_tr_examples
            print(f"Validation Loss per 50 steps: {loss_step}")
            print(f"Validation Accuracy per 50 steps: {accu_step}")

    print(
        f'The Total Accuracy for Epoch {epoch+1}: {(n_correct*100)/nb_tr_examples}')
    epoch_loss = val_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Validation Loss Epoch at {epoch+1}: {epoch_loss}")
    print(f"Validation Accuracy Epoch at {epoch+1}: {epoch_accu}")
    print('=' * 30)

    return epoch_loss, epoch_accu


if __name__ == "__main__":
    # reproducability
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    logging.basicConfig(level=logging.ERROR)

    # dir setup
    EXPERIMENT_NAME = "banglabert-auto_5e-5_3eps"
    ROOT_DIR = os.path.abspath("/content/drive/MyDrive/Saved Models")
    LOG_PATH = os.path.join(ROOT_DIR, "logs", EXPERIMENT_NAME)

    if not os.path.exists(os.path.join(ROOT_DIR, "logs")):
        os.mkdir(os.path.join(ROOT_DIR, "logs"))
    if not os.path.exists(LOG_PATH):
        os.mkdir(LOG_PATH)

    # Load Pre-Trained Model
    PRE_TRAINED_MODEL_NAME = 'csebuetnlp/banglabert'
    tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    # Hyper params
    MAX_LEN = 200
    BATCH_SIZE = 16
    EPOCHS = 3
    L_RATE = 5e-5
    WEIGHT_DECAY = 1e-2

    # instantiate dataloaders
    df_train = pd.read_csv("/content/10K_Dataset/Final_Train.csv")
    df_val = pd.read_csv("/content/10K_Dataset/Final_Val.csv")

    train_dataloader = Dataloader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    val_dataloader = Dataloader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)

    # instantiate model
    model = AutoModelForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME,
                                                               num_labels=3)
    model = model.to(device)

    # optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=L_RATE)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=220,
                                                num_training_steps=total_steps)

    # training

    start_time = time.time()

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    best_accuracy = 0

    for epoch in range(EPOCHS):

        training_loss, training_accuracy = train(epoch)
        train_loss.append(training_loss)
        train_acc.append(training_accuracy)
        validation_loss, validation_accuracy = evaluation(epoch)

        if validation_accuracy > best_accuracy:
            print("Saving model at accuracy={:.3f}".format(
                validation_accuracy))
            torch.save(model.state_dict(),
                       '{}/{}.pth'.format(LOG_PATH, EXPERIMENT_NAME))
            best_accuracy = validation_accuracy

        val_loss.append(validation_loss)
        val_acc.append(validation_accuracy)

        elapsed = (time.time() - start_time)/60
        print(f'Time elapsed: {elapsed:.2f} min')

    elapsed = (time.time() - start_time)/60
    print(f'Total Training Time: {elapsed:.2f} min')

    # save results
    train_loss = np.array(train_loss)
    np.savetxt("{}/{}_train_loss.txt".format(LOG_PATH,
               EXPERIMENT_NAME), train_loss, delimiter=",")
    val_loss = np.array(val_loss)
    np.savetxt("{}/{}_val_loss.txt".format(LOG_PATH,
               EXPERIMENT_NAME), val_loss, delimiter=",")

    train_acc = np.array(train_acc)
    np.savetxt("{}/{}_train_acc.txt".format(LOG_PATH,
               EXPERIMENT_NAME), train_acc, delimiter=",")
    val_acc = np.array(val_acc)
    np.savetxt("{}/{}_val_acc.txt".format(LOG_PATH,
               EXPERIMENT_NAME), val_acc, delimiter=",")
