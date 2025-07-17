def train_forcing(dataloader, model, loss_fn, optimizer,loss_dis, lamb=1., device='cuda'):  ### our method
    
    # Total size of dataset for reference
    size = 0
    
    # places your model into training mode
    model.train()
    
    # loss batch
    batch_loss = {}
    batch_accuracy = {}
    
    correct = 0
    _correct = 0
    
    
    
    # Gives X , y for each batch
    for batch, (X, X_noise, y) in enumerate(dataloader):
        
        # Converting device to cuda
        X, X_noise, y = X.to(device), X_noise.to(device), y.to(device)
        model.to(device)
        
        # Compute prediction error / loss
        # 1. Compute y_pred 
        # 2. Compute loss between y and y_pred using selectd loss function
        
        
        feature = model.forward_before_softmax(X)
        feature_noise = model.forward_before_softmax(X_noise)
        
        y_pred = model.forward_softmax(feature)
        
        loss_pred = loss_fn(y_pred, y)
        
        loss_d = loss_dis(feature, feature_noise, y, model.fc_layer[-1])
        # Backpropagation on optimizing for loss
        # 1. Sets gradients as 0 
        # 2. Compute the gradients using back_prop
        # 3. update the parameters using the gradients from step 2
        
        loss = loss_pred + lamb*loss_d
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _correct = (y_pred.argmax(1) == y).type(torch.float).sum().item()
        _batch_size = len(X)
        
        correct += _correct
        
        # Updating loss_batch and batch_accuracy
        batch_loss[batch] = loss.item()
        batch_accuracy[batch] = _correct/_batch_size
        
        size += _batch_size
        
        if batch % 100 == 0:
            loss,loss_pred, loss_d, current = loss.item(), loss_pred.item(), loss_d.item(), batch * len(X)
            print(f"loss: {loss:>7f}, loss_pred: {loss_pred:>7f}, loss_dis: {loss_d:>7f}  [{current:>5d}]")
    
    correct/=size
    print(f"Train Accuracy: {(100*correct):>0.1f}%")
    
    return batch_loss , batch_accuracy


def train_both(dataloader, model, loss_fn, optimizer, device='cuda'):
    
    # Total size of dataset for reference
    size = 0
    
    # places your model into training mode
    model.train()
    
    # loss batch
    batch_loss = {}
    batch_accuracy = {}
    
    correct = 0
    _correct = 0
    
    
    
    # Gives X , y for each batch
    for batch, (X, X_noise, y) in enumerate(dataloader):

        # Converting device to cuda
        X, X_noise, y = X.to(device), X_noise.to(device), y.to(device)
        X = torch.concatenate((X,X_noise))
        y = torch.concatenate((y,y))
        
        model.to(device)

        # Compute prediction error / loss
        # 1. Compute y_pred 
        # 2. Compute loss between y and y_pred using selectd loss function
        
        
        y_pred = model.forward(X)
        loss = loss_fn(y_pred, y)
        # Backpropagation on optimizing for loss
        # 1. Sets gradients as 0 
        # 2. Compute the gradients using back_prop
        # 3. update the parameters using the gradients from step 2
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _correct = (y_pred.argmax(1) == y).type(torch.float).sum().item()
        _batch_size = len(X)
        
        correct += _correct
        
        # Updating loss_batch and batch_accuracy
        batch_loss[batch] = loss.item()
        batch_accuracy[batch] = _correct/_batch_size
        
        size += _batch_size
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}, [{current:>5d}]")
    
    correct/=size
    print(f"Train Accuracy: {(100*correct):>0.1f}%")
    
    return batch_loss , batch_accuracy


def train_stability(dataloader, model, loss_fn, optimizer, lamb=1., device='cuda'):
    
    # Total size of dataset for reference
    size = 0
    
    # places your model into training mode
    model.train()
    
    # loss batch
    batch_loss = {}
    batch_accuracy = {}
    
    correct = 0
    _correct = 0
    
    
    
    # Gives X , y for each batch
    for batch, (X, X_noise, y) in enumerate(dataloader):
        
        # Converting device to cuda
        X, X_noise, y = X.to(device), X_noise.to(device), y.to(device)
        model.to(device)
        
        # Compute prediction error / loss
        # 1. Compute y_pred 
        # 2. Compute loss between y and y_pred using selectd loss function
        
        
        feature = model.forward_before_softmax(X)
        feature_noise = model.forward_before_softmax(X_noise)
        
        y_pred = model.forward_softmax(feature)
        
        loss_pred = loss_fn(y_pred, y)
        
        loss_d = torch.norm(feature-feature_noise,p=2,dim=1).mean()
        
        # Backpropagation on optimizing for loss
        # 1. Sets gradients as 0 
        # 2. Compute the gradients using back_prop
        # 3. update the parameters using the gradients from step 2
        
        loss = loss_pred + lamb*loss_d
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _correct = (y_pred.argmax(1) == y).type(torch.float).sum().item()
        _batch_size = len(X)
        
        correct += _correct
        
        # Updating loss_batch and batch_accuracy
        batch_loss[batch] = loss.item()
        batch_accuracy[batch] = _correct/_batch_size
        
        size += _batch_size
        
        if batch % 100 == 0:
            loss,loss_pred, loss_d, current = loss.item(), loss_pred.item(), loss_d.item(), batch * len(X)
            print(f"loss: {loss:>7f}, loss_pred: {loss_pred:>7f}, loss_stab: {loss_d:>7f}  [{current:>5d}]")
    
    correct/=size
    print(f"Train Accuracy: {(100*correct):>0.1f}%")
    
    return batch_loss , batch_accuracy

def train_no(dataloader, model, loss_fn, optimizer, device='cuda'):
    
    # Total size of dataset for reference
    size = 0
    
    # places your model into training mode
    model.train()
    
    # loss batch
    batch_loss = {}
    batch_accuracy = {}
    
    correct = 0
    _correct = 0
    
    
    
    # Gives X , y for each batch
    for batch, (X, y) in enumerate(dataloader):

        # Converting device to cuda
        X, y = X.to(device), y.to(device)
        
        model.to(device)

        # Compute prediction error / loss
        # 1. Compute y_pred 
        # 2. Compute loss between y and y_pred using selectd loss function
        
        
        y_pred = model.forward(X)
        loss = loss_fn(y_pred, y)
        # Backpropagation on optimizing for loss
        # 1. Sets gradients as 0 
        # 2. Compute the gradients using back_prop
        # 3. update the parameters using the gradients from step 2
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _correct = (y_pred.argmax(1) == y).type(torch.float).sum().item()
        _batch_size = len(X)
        
        correct += _correct
        
        # Updating loss_batch and batch_accuracy
        batch_loss[batch] = loss.item()
        batch_accuracy[batch] = _correct/_batch_size
        
        size += _batch_size
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}, [{current:>5d}]")
    
    correct/=size
    print(f"Train Accuracy: {(100*correct):>0.1f}%")
    
    return batch_loss , batch_accuracy


def accuracy_evaluation(dataloader, model, device='cuda'):
    
    # Total size of dataset for reference
    size = 0
    
    # Setting the model under evaluation mode.
    model.eval()

    correct = 0
    
    _correct = 0
    _batch_size = 0
    
    with torch.no_grad():
        
        # Gives X , y for each batch
        for batch , (X, y) in enumerate(dataloader):
            
            X, y = X.to(device), y.to(device)
            model.to(device)
            
            pred = model(X)
            _batch_size = len(X)   
            _correct = (pred.argmax(1) == y).type(torch.float).sum().item()
            correct += _correct            
            size += _batch_size
    
    ## Calculating Accuracy based on how many y match with y_pred
    correct /= size
    
    return correct

def copy_state_dict(model):
    old_state_dict = {}
    for key in model.state_dict():
        old_state_dict[key] = model.state_dict()[key].clone()
    return old_state_dict