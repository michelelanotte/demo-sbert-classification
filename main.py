from torch.optim import Adam
from tqdm import tqdm
from dataset import Dataset
import torch
from bertClassifier import *
import numpy as np
import math
import pandas as pd
from utils import preprocessing

def train(model, train_data, val_data, learning_rate, epochs):

    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    counter = 0
    best_validation_loss = math.inf
    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):

            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
            """print(output)
            print(train_label)
            print("________")"""
            
            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()
            
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            total_acc_val = 0
            total_loss_val = 0

        with torch.no_grad():
            for val_input, val_label in tqdm(val_dataloader):
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()
                
                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc
            
        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
            | Train Accuracy: {total_acc_train / len(train_data): .3f} \
            | Val Loss: {total_loss_val / len(val_data): .3f} \
            | Val Accuracy: {total_acc_val / len(val_data): .3f}')

        validation_loss = total_loss_val / len(val_data)    
        alpha = 0.95
        if validation_loss < alpha * best_validation_loss:
            counter = 0
            best_validation_loss = validation_loss
        else:
            counter += 1
            
            if counter == 3:
                break
        
 
    torch.save(model, 'model/sbert.pth')

"""----------------------------------MAIN-------------------------------------------"""
datapath = 'training_set.csv'
df = pd.read_csv(datapath)
df['text'] = preprocessing(df['text'])

np.random.seed(112)
df_train, df_val = np.split(df.sample(frac=1, random_state=42), 
                                     [int(.8*len(df))])

EPOCHS = 10
model = BertClassifier()
LR = 1e-6
train(model, df_train, df_val, LR, EPOCHS)



"""
model = torch.load('model/sbert.pth'))
"""

