import torch
import torch.nn.functional as F 

import numpy as np 
from tqdm import tqdm 

from misc.utils import AverageMeter
from models.model_factory import model_factory
from datasets.dataset_utils import make_dataloader
from losses.loss_factory import make_pr_loss, make_inc_loss

from torchpack.utils.config import configs 





class TrainerIncremental:
    def __init__(self, logger, memory, old_environment_pickle, new_environment_pickle, pretrained_checkpoint, env_idx):
        # Initialise inputs
        self.debug = configs.debug 
        self.logger = logger 
        self.env_idx = env_idx
        self.epochs = configs.train.optimizer.epochs 

        # Set up meters and stat trackers 
        self.loss_total_meter = AverageMeter()
        self.loss_pr_meter = AverageMeter()
        self.loss_inc_meter = AverageMeter()
        self.num_triplets_meter = AverageMeter()
        self.non_zero_triplets_meter = AverageMeter()
        self.embedding_norm_meter = AverageMeter()

        # Make dataloader
        self.dataloader = make_dataloader(pickle_file = new_environment_pickle, memory = memory)

        # Build models and init from pretrained_checkpoint
        assert torch.cuda.is_available, 'CUDA not available.  Make sure CUDA is enabled and available for PyTorch'
        self.model_frozen = model_factory(ckpt = pretrained_checkpoint, device = 'cuda')
        self.model_new = model_factory(ckpt = pretrained_checkpoint, device = 'cuda')

        # Make optimizer 
        if configs.train.optimizer.weight_decay is None or configs.train.optimizer.weight_decay == 0:
            self.optimizer = torch.optim.Adam(self.meters(), lr=configs.train.optimizer.lr)
        else:
            self.optimizer = torch.optim.Adam(self.model_new.parameters(), lr=configs.train.optimizer.lr, weight_decay=configs.train.optimizer.weight_decay)

        # Scheduler
        if configs.train.optimizer.scheduler is None:
            self.scheduler = None
        else:
            if configs.train.optimizer.scheduler == 'CosineAnnealingLR':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=configs.train.optimizer.epochs+1,
                                                                    eta_min=configs.train.optimizer.min_lr)
            elif configs.train.optimizer.scheduler == 'MultiStepLR':
                if not isinstance(configs.train.optimizer.scheduler_milestones, list):
                    configs.train.optimizer.scheduler_milestones = [configs.train.optimizer.scheduler_milestones]
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, configs.train.optimizer.scheduler_milestones, gamma=0.1)
            else:
                raise NotImplementedError('Unsupported LR scheduler: {}'.format(configs.train.optimizer.scheduler))

        # Make loss functions 
        self.loss_fn = make_pr_loss()
        self.loss_fn_inc = make_inc_loss()
        if configs.train.loss.incremental.name == 'EWC':
            self.fisher_matrix, self.old_parameters = self.loss_fn_inc.get_fisher_matrix(
                                                        dataloader = self.dataloader,
                                                        model = self.model_new, 
                                                        optimizer = self.optimizer, 
                                                        loss_fn = self.loss_fn
                                                        )

    def before_epoch(self, epoch):        
        # Reset meters
        self.loss_total_meter.reset()
        self.loss_pr_meter.reset()
        self.loss_inc_meter.reset()
        self.num_triplets_meter.reset()
        self.non_zero_triplets_meter.reset()
        self.embedding_norm_meter.reset()
        
        # Adjust weight of incremental loss function if constraint relaxation enabled
        self.loss_fn_inc.adjust_weight(epoch)

    def training_step(self, batch, positives_mask, negatives_mask):
        
        # Prepare batch
        batch_stats = {}
        batch = {x: batch[x].to('cuda') if x!= 'coords' else batch[x] for x in batch}

        n_positives = torch.sum(positives_mask).item()
        n_negatives = torch.sum(negatives_mask).item()
        if n_positives == 0 or n_negatives == 0:
            # Skip a batch without positives or negatives
            print('WARNING: Skipping batch without positive or negative examples')
            return None 

        # Get embeddings and Loss
        self.optimizer.zero_grad()
        with torch.no_grad():
            embeddings_old = self.model_frozen(batch)
        embeddings_new = self.model_new(batch)

        loss_place_rec, num_triplets, non_zero_triplets, embedding_norm = self.loss_fn(embeddings_new, positives_mask, negatives_mask)
        if configs.train.loss.incremental.name != 'EWC':
            loss_incremental = self.loss_fn_inc(embeddings_old, embeddings_new)
        else:
            loss_incremental = self.loss_fn_inc(self.model_new, self.old_parameters, self.fisher_matrix)
        loss_total = loss_place_rec + loss_incremental

        # Backwards
        loss_total.backward()
        self.optimizer.step()
        torch.cuda.empty_cache() # Prevent excessive GPU memory consumption by SparseTensors

        # Stat tracking
        self.loss_total_meter.update(loss_total.item())
        self.loss_pr_meter.update(loss_place_rec.item())
        self.loss_inc_meter.update(loss_incremental.item())
        self.num_triplets_meter.update(num_triplets)
        self.non_zero_triplets_meter.update(non_zero_triplets)
        self.embedding_norm_meter.update(embedding_norm)

        return None 

    def after_epoch(self, epoch):
        # Scheduler 
        if self.scheduler is not None:
            self.scheduler.step()

        # Dynamic Batch Expansion
        if configs.train.batch_expansion_th is not None:
            ratio_non_zeros = self.non_zero_triplets_meter.avg / self.num_triplets_meter.avg 
            if ratio_non_zeros < configs.train.batch_expansion_th:
                self.dataloader.batch_sampler.expand_batch()

        # Tensorboard plotting 
        self.logger.add_scalar(f'Step_{self.env_idx}/Total_Loss', self.loss_total_meter.avg, epoch)
        self.logger.add_scalar(f'Step_{self.env_idx}/Place_Rec_Loss', self.loss_pr_meter.avg, epoch)
        self.logger.add_scalar(f'Step_{self.env_idx}/Incremental_Loss', self.loss_inc_meter.avg, epoch)
        self.logger.add_scalar(f'Step_{self.env_idx}/Non_Zero_Triplets', self.non_zero_triplets_meter.avg, epoch)
        self.logger.add_scalar(f'Step_{self.env_idx}/Embedding_Norm', self.embedding_norm_meter.avg, epoch)




    def train(self):
        for epoch in tqdm(range(1, self.epochs + 1)):
            self.before_epoch(epoch)
            for idx, (batch, positives_mask, negatives_mask) in enumerate(self.dataloader):
                self.training_step(batch, positives_mask, negatives_mask)
                if self.debug and idx > 2:
                    break 
            self.after_epoch(epoch)
            if self.debug and epoch > 2:
                break 

        return self.model_new
