import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def train(model, dataloader, epochs=50):

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in dataloader:
            optimizer.zero_grad()
            
            outputs = model(inputs.float(), labels)
            loss = criterion(outputs, labels.to(torch.float32))
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        if epoch % 10 == 0: 
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(dataloader)}")

    print("Training Finished")