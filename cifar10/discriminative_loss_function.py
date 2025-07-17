from torch.nn.modules.loss import _Loss
import numpy as np
import torch

class DiscriminativeLoss_forcing(_Loss):

    def __init__(self, num_classes = 10, dim = 256, delta_var=0.5, delta_dist=3.0,
                 norm=2, alpha=1.0, beta=1.0, gamma=0.001, momentum=0.99, size_average=True):
        super(DiscriminativeLoss_forcing, self).__init__(size_average)
        self.dim = dim
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.norm = norm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.momentum = momentum
        self.init = [False]*num_classes
        self.mean_before = torch.zeros(num_classes,dim)
        assert self.norm in [1, 2]
        
    def cuda(self):
        self.mean_before = self.mean_before.cuda()
        
    def cpu(self):
        self.mean_before = self.mean_before.cpu()
        
    def forward(self, input, input_noise, target, fc):
        return self._discriminative_loss(input, input_noise, target, fc)

    def _discriminative_loss(self, input, input_noise, target, fc):
        c_means, c_means_current = self._cluster_means(input, target)
        l_var = self._variance_term(input, input_noise, target, c_means)
        l_dist = self._margin_term(c_means_current, target, fc)
        l_reg = self._regularization_term(c_means)
        loss = self.alpha * l_var + self.beta * l_dist + self.gamma * l_reg
        return loss

    def _cluster_means(self, input, target):
        means_current = []
        means = []
        labels = sorted(set(np.array(target.cpu())))
        for label in labels:
            input_group = input[np.where(np.array(target.cpu())==label)]
            mean_sample_current = input_group.mean(axis=0)
            if not self.init[label]:
                mean_sample = mean_sample_current
                self.init[label] = True
            else:
                mean_sample = (1-self.momentum)*mean_sample_current + self.momentum*self.mean_before[label]
            self.mean_before[label] = mean_sample.detach().clone()
            means.append(mean_sample)
            means_current.append(mean_sample_current)

        means = torch.stack(means)
        means_current = torch.stack(means_current)
        return means, means_current

    def _variance_term(self, input, input_noise, target, c_means):
        labels = sorted(set(np.array(target.cpu())))
        var_term = 0
        for i,label in enumerate(labels):
            input_group = input[np.where(np.array(target.cpu())==label)]
            input_group_noise = input_noise[np.where(np.array(target.cpu())==label)]
            input_group = torch.concatenate((input_group,input_group_noise))
            mean = c_means[i]
            input_group_mean = input_group - mean
            var = torch.clamp(torch.norm(input_group_mean, self.norm, 1)
                           - self.delta_var, min=0) ** 2
            var_term += torch.mean(var)
        var_term /= len(labels)
        return var_term

    def _margin_term(self, c_means, target, fc):
        labels = sorted(set(np.array(target.cpu())))
        margin_term = 0
        for i,label in enumerate(labels):
            mean = c_means[i].reshape(1,-1)
            z = fc(mean).reshape(-1)  #  (K,)
            index = self.create_index(label,len(z))
            z_d = z[index]-z[label]  # (K-1,)
            z_diff = torch.abs(z_d)
            sign = torch.sign(z_d)
            weight_diff = fc.weight[label]-fc.weight[index]
            norm_diff = torch.norm(weight_diff,dim=1,p=2).reshape(-1)
            margin = torch.div(z_diff,norm_diff)  # (K-1,)
            margin_term += torch.max(torch.clamp(self.delta_dist + torch.mul(margin,sign), min=0))
        margin_term /= len(labels)
        return margin_term

    def _regularization_term(self, c_means):
        reg_term = torch.mean(torch.norm(c_means, self.norm, 1))
        return reg_term
    
    def create_index(self, label,size):
        index = torch.ones(size, dtype=bool)
        index[label] = False
        return index