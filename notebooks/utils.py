# The Pytorch Training Loop
# LIBRARIES
import torch 
from torch.utils.data import DataLoader 

def train_model(model, device, epochs, batch_size, trainset, testset,
                optimizer, loss_function, metric):
    # Transfer model to supported computing device.
    model.to(device)
    # Create dataloaders.
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    acc = False
    mae = False
    if(metric == 'acc'):
        acc = True
    elif(metric == 'mae'):
        mae = True
    else:
        print('Error: unsupported metric')
        return
        
    for i in range(epochs):
        model.train() # Set model in training mode.
        train_loss = 0.0
        if(acc):
            train_correct = 0
        if(mae):
            train_absolute_error = 0.0
        train_batches = 0
        for inputs, targets in trainloader:
            # Move data to supported computing device.
            inputs, targets = inputs.to(device), targets.to(device)
            # Zero the parameter gradients.
            optimizer.zero_grad()
            # Forward pass.
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            # Accumulate metrics.
            if(acc):
                _, indices = torch.max(outputs.data, 1)
                train_correct += (indices == targets).sum().item()
            if(mae):
                train_absolute_error += (targets - outputs.data).abs().sum().item()
            train_batches +=  1
            train_loss += loss.item()
            # Backward pass and update.
            loss.backward()
            optimizer.step()

        train_loss = train_loss / train_batches
        if(acc):
            train_acc = train_correct / (train_batches * batch_size)
        if(mae):
            train_mae = train_absolute_error / (train_batches * batch_size)

        # Evaluate the model on the test dataset.
        model.eval() # Set model in inference mode.
        test_loss = 0.0
        if(acc):
            test_correct = 0
        if(mae):
            test_absolute_error = 0.0
        test_batches = 0
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            if(acc):
                _, indices = torch.max(outputs.data, 1)
                test_correct += (indices == targets).sum().item()
            if(mae):
                test_absolute_error += (targets - outputs.data).abs().sum().item()
            test_batches +=  1
            test_loss += loss.item()
        test_loss = test_loss / test_batches
        if(acc):
            test_acc = test_correct / (test_batches * batch_size)
            print(f'Epoch {i+1}/{epochs} loss: {train_loss:.4f} - acc: {train_acc:0.4f} - val_loss: {test_loss:.4f} - val_acc: {test_acc:0.4f}')
            return_value = [train_acc, test_acc]
        if(mae):
            test_mae = test_absolute_error / (test_batches * batch_size)
            print(f'Epoch {i+1}/{epochs} loss: {train_loss:.4f} - mae: {train_mae:0.4f} - val_loss: {test_loss:.4f} - val_mae: {test_mae:0.4f}')
            return_value = [train_mae, test_mae]
    return return_value