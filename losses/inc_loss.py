import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchpack.utils.config import configs 
from tqdm import tqdm 

class NoIncLoss:
    def __init__(self):
        pass 

    def adjust_weight(self, epoch):
        pass
    
    def __call__(self, *args, **kwargs):
        return torch.tensor(0, dtype = float, device = 'cuda')


class LwF:
    def __init__(self):
        self.weight = configs.train.loss.incremental.weight
        self.temperature = 2 
    
    def adjust_weight(self, epoch):
        pass 

    def __call__(self, old_rep, new_rep):
        log_p = torch.log_softmax(new_rep / self.temperature, dim=1)
        q = torch.softmax(old_rep / self.temperature, dim=1)
        res = torch.nn.functional.kl_div(log_p, q, reduction="batchmean")
        loss_incremental = self.weight * res
        return loss_incremental

class StructureAware:
    def __init__(self):
        self.orig_weight = configs.train.loss.incremental.weight 
        self.weight = configs.train.loss.incremental.weight 
        self.margin = configs.train.loss.incremental.margin 

        # Constraint relaxation
        gamma = configs.train.loss.incremental.gamma
        lin = torch.linspace(0, 1, configs.train.optimizer.epochs)
        exponential_factor = gamma*(lin - 0.5)
        self.weight_factors = 1 / (1 + exponential_factor.exp())


    def adjust_weight(self, epoch):
        if configs.train.loss.incremental.adjust_weight:
            self.weight = self.orig_weight * self.weight_factors[epoch - 1]
        else:
            pass 

    def __call__(self, old_rep, new_rep):
        with torch.no_grad():
            old_vec = old_rep.unsqueeze(0) - old_rep.unsqueeze(1) # B x D x D
            norm_old_vec = F.normalize(old_vec, p = 2, dim = 2)
            old_angles = torch.bmm(norm_old_vec, norm_old_vec.transpose(1,2)).view(-1)

        new_vec = new_rep.unsqueeze(0) - new_rep.unsqueeze(1)
        norm_new_vec = F.normalize(new_vec, p = 2, dim = 2)
        new_angles = torch.bmm(norm_new_vec, norm_new_vec.transpose(1,2)).view(-1)

        loss_incremental = F.smooth_l1_loss(new_angles, old_angles, reduction = 'none')
        loss_incremental = F.relu(loss_incremental - self.margin)

        # Remove 0 terms from loss which emerge due to margin 
        # Only do if there are any terms where inc. loss is not zero
        if torch.any(loss_incremental > 0):
            loss_incremental = loss_incremental[loss_incremental > 0]

        loss_incremental = self.weight * loss_incremental.mean()
        return loss_incremental

class EWC:
    def __init__(self):
        self.weight = configs.train.loss.incremental.weight

    def adjust_weight(self, epoch):
        pass 

    def get_fisher_matrix(self, model, dataloader, optimizer, loss_fn):
        fisher = {n: torch.zeros(p.shape).to('cuda') for n,p in model.named_parameters() if p.requires_grad}
        pbar = tqdm(desc = 'Getting Fisher Matrix', total = len(dataloader.dataset))
        for batch, positives_mask, negatives_mask in dataloader:
            batch = {e: batch[e].to('cuda') if e!= 'coords' else batch[e] for e in batch}

            embeddings = model(batch)
            loss, _, _, _ = loss_fn(embeddings, positives_mask, negatives_mask)
            optimizer.zero_grad()
            loss.backward()

            # Accumulate all gradients from loss with regularization
            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2) * len(embeddings)
            pbar.update(len(positives_mask))
        pbar.close()
        # Apply mean across all samples
        n_samples = len(dataloader.dataset)
        fisher = {n: (p / n_samples) for n, p in fisher.items()}
        old_parameters = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors
        return fisher, old_parameters


    def __call__(self, model_new, old_parameters, fisher_matrix):
        loss_incremental = torch.tensor(0, device = 'cuda', dtype = float)
        for n, new_param in model_new.named_parameters():
            if n in fisher_matrix.keys():
                old_param = old_parameters[n]
                loss_incremental += torch.sum(fisher_matrix[n] * (new_param - old_param).pow(2))

        loss_incremental = self.weight * loss_incremental
        return loss_incremental

