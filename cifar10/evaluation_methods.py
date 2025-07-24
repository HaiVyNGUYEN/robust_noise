import torch
import numpy as np

def per_image_curvature_with_loss(model, data_loader, loss_fn, h=1e-2, K=20, device='cuda'):
    """
    Estimate ||H_i g_i||^2 per image using the loss gradient ∇_x l(f(x), y)
    
    Args:
        model: maps (B, C, H, W) → model output (e.g., logits)
        images: input batch (B, C, H, W)
        labels: ground truth labels (B,) or (B, num_classes)
        loss_fn: loss function like nn.CrossEntropyLoss(reduction='none')
        h: finite difference step size

    Returns:
        Tensor of shape (B,) with curvature estimate for each image
    """
    model.eval()
    model.to(device)
    list_curvature = []
    for batch, (images, labels) in enumerate(data_loader):
        if batch % 20 == 0:
            print("Processing batch ",batch)
        B = images.size(0)
        images = images.detach().clone().requires_grad_(True).to(device)
        labels = labels.to(device)
        preds = model(images)  # (B, ...)
        loss = loss_fn(preds, labels)  # (B,)
        grad = torch.autograd.grad(loss.sum(), images, create_graph=True)[0]  # (B, C, H, W)
    
        # Accumulate curvature estimates
        curvature_sum = torch.zeros(B, device=images.device)
    
        for _ in range(K):
            g = torch.randn_like(images)

            # Perturbed input
            x_perturbed = (images + h * g).detach().clone().requires_grad_(True)
            preds_perturbed = model(x_perturbed)
            loss_perturbed = loss_fn(preds_perturbed, labels)
            grad_perturbed = torch.autograd.grad(loss_perturbed.sum(), x_perturbed, create_graph=True)[0]

            diff = (grad_perturbed - grad) / h
            curvature_sum += diff.flatten(1).pow(2).sum(dim=1)
        curvature = curvature_sum / K  # (B,)
        curvature = list(curvature.detach().cpu().numpy())  # shape: (B,)
        list_curvature += curvature
        
    return np.array(list_curvature)

def noise_testing(model, length_data, noise_dataloader, num_sim=10, device='cuda'):
    """
    function to compute number of correct predictions over different random noisy corruption (for each input)
    """
    model.eval()
    model.to(device)
    correct_list = np.zeros((num_sim,length_data))
    for sim in range(num_sim):
        print("Noise simulation", sim)
        count = 0
        # Gives X , y for each batch
        for X, y in noise_dataloader:
            X, y = X.to(device), y.to(device)
            with torch.no_grad():
                pred = model(X) 
                correct_list[sim,count:count+len(X)] = \
                (pred.argmax(1) == y).type(torch.float).clone().detach().cpu().numpy()
            count += len(X)
    correct_list = correct_list.sum(axis=0)
    return correct_list