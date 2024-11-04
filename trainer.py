import os
import time
nowtime = time.localtime()

import logging
logger = logging.getLogger(__name__)

import argparse
from collections import OrderedDict
import numpy as np
import random
import json
import yaml
import copy

import torch
from torch.cuda.amp import autocast
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from dataset import dataset_name2cls
from model import model_name2cls
from utils import init_distributed_mode, get_rank, is_main_process, concat_gather, MetricLogger, SmoothedValue
from metrics import name2metric

class Trainer(object):
    @staticmethod
    def add_arguments(parser):
        # path & data params
        parser.add_argument("--dataset_name", type=str, default="mut_eff")
        parser.add_argument("--dataset_path", type=str, default="./data/")
        parser.add_argument("--nshot", type=int, default=None)
        parser.add_argument("--exclude", type=str, nargs='*', default=[])
        parser.add_argument("--model_name", type=str, default="bert_esm")
        parser.add_argument("--model_config_path", type=str, default="./configs/bert_esm.yaml")
        parser.add_argument("--model_checkpoint", type=str, default=None)
        parser.add_argument("--save_path", type=str, default="./ckpts/fusion_ckpts/bert_esm")
        parser.add_argument("--log_path", type=str, default=f"./logs/{nowtime[1]}-{nowtime[2]}_{nowtime[3]}-{nowtime[4]}.logger.info")
        parser.add_argument("--resume", action="store_true")
        parser.add_argument("--resume_checkpoint", type=str, default="./ckpts/")
        parser.add_argument("--data_percent", type=float, default=1.0)

        # training params
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--epochs", type=int, default=30)
        parser.add_argument("--warmup_steps", type=int, default=5000)
        parser.add_argument("--batch_size", type=int, default=2)
        parser.add_argument("--num_workers", type=int, default=1)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--weight_decay", type=float, default=1e-2)
        parser.add_argument("--clip_grad_norm", type=bool, default=True)
        parser.add_argument("--save_epochs", type=int, default=1)
        parser.add_argument("--log_steps", type=int, default=10)
        parser.add_argument("--grad_accu_steps", type=int, default=1)

        # validation params
        parser.add_argument("--validate", action="store_true")
        parser.add_argument("--patience", type=int, default=5)
        parser.add_argument("--metric", type=str, default="spearmanr")
        parser.add_argument("--lower", action="store_true")

        # distributed params
        parser.add_argument("--distributed", action="store_true")
        parser.add_argument('--world_size', type=int, default=2, help='number of distributed processes')
        parser.add_argument('--local-rank', type=int, default=0)

        return parser

    def __init__(self, args):
        super().__init__()
        init_distributed_mode(args)

        self.args = args
        self.local_rank = get_rank()
        self.device = torch.device("cuda", self.local_rank)
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            #level=logging.INFO
            level=logging.INFO if is_main_process() else logging.ERROR,
        )
        logger.info("Rank: %d" % (self.local_rank))
        self._setup_seed(self.args.seed + self.local_rank)
        self._setup_data()
        self._setup_model()
        
    def _setup_seed(self, seed):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _setup_data(self):
        logger.info("Loading dataset...")
        self.train_dataset = dataset_name2cls[self.args.dataset_name](self.args.dataset_path, split="train", name=self.args.dataset_name, nshot=self.args.nshot, exclude=self.args.exclude)
        logger.info(f"Num Train Samples: {len(self.train_dataset)}")
        if hasattr(self.train_dataset, "get_example"):
            for i, example in enumerate(self.train_dataset.get_example()):
                if i >= 2:
                    break
                logger.info(example)

        if self.args.distributed:
            self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
        else:
            self.train_sampler = RandomSampler(self.train_dataset)
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            sampler=self.train_sampler,
            num_workers=self.args.num_workers,
        )


        if self.args.validate:
            self.valid_dataset = dataset_name2cls[self.args.dataset_name](self.args.dataset_path, split="valid", name=self.args.dataset_name, nshot=self.args.nshot, exclude=self.args.exclude)
            logger.info(f"Num Valid Samples: {len(self.valid_dataset)}")
            if self.args.distributed:
                self.valid_sampler = DistributedSampler(self.valid_dataset, seed=self.seed, shuffle=True)
            else:
                self.valid_sampler = RandomSampler(self.valid_dataset)

            self.valid_dataloader = DataLoader(
                self.valid_dataset,
                batch_size=self.args.batch_size,
                sampler=self.valid_sampler,
                num_workers=self.args.num_workers,
            )
            self.patience = 0

    def _setup_model(self):
        logger.info("Loading model...")
        model_cls = model_name2cls[self.args.model_name]
        model_cfg = yaml.load(open(self.args.model_config_path, "r"), Loader=yaml.Loader)
        model_cfg["device"] = self.device
        self.model = model_cls(**model_cfg).to(self.device)

        logger.info(f"Trainable params: {sum([p.numel() if p.requires_grad else 0 for p in self.model.parameters()])/1000000}M")
        logger.info(f"Total params: {sum([p.numel() for p in self.model.parameters()])/1000000}M")

        if self.args.model_checkpoint is not None:
            logger.info(f"Load model checkpoint from {self.args.model_checkpoint}")
            state_dict = torch.load(open(self.args.model_checkpoint, "rb"), map_location=torch.device("cpu"))
            # NOTE: change back to state_dict["model"]
            self.model.load_state_dict(state_dict["model"], strict=True)
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.schedular = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, 
            max_lr=self.args.lr,
            total_steps=int(self.args.epochs * len(self.train_dataloader)),
            epochs=self.args.epochs,
            pct_start=self.args.warmup_steps * 1.0 / self.args.epochs / len(self.train_dataloader),
            anneal_strategy='cos',
            final_div_factor=1e2
        )
        logger.info(f"Epochs = {self.args.epochs}, Dataloader Length = {len(self.train_dataloader)}, world size = {self.args.world_size}")
    
        # continue training
        if self.args.resume:
            logger.info(f"resume from {self.args.resume_checkpoint}...")
            ckpt = torch.load(self.args.resume_checkpoint, map_location=torch.device("cpu"))
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.schedular.load_state_dict(ckpt["schedular"])
            self.start_epoch = ckpt["epoch"] + 1
            del ckpt
            logger.info("Load model successfully.")
        else:
            logger.info("Train from scratch")
            self.start_epoch = 0

        if self.args.distributed:
            logger.info("Parallizing model...")
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, 
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
                broadcast_buffers=False
            )
            self.model_without_ddp = self.model.module
        else:
            self.model_without_ddp = self.model

    def train(self):
        self.all_steps = 0
        logger.info("Start training!")
        if self.args.validate:
            best_result = -1e7
        for epoch in range(self.start_epoch, self.args.epochs):
            logger.info(f"Epoch {epoch + 1} / {self.args.epochs}")
            if self.args.distributed:
                self.train_sampler.set_epoch(epoch)
            
            train_stats = self.train_epoch(epoch)

            if self.args.validate:
                result = self.validate_epoch()
                if self.args.lower ^ (result > best_result):
                    best_result = result
                    self.patience = 0
                    logger.info(f"Best ckpt at epoch {epoch}...")
                    save_dict = {
                        "model": copy.deepcopy(self.model_without_ddp.state_dict()),
                    }
                else:
                    self.patience += 1
                    logger.info(f"Remaining patience: {self.args.patience - self.patience}/{self.args.patience}")
                    if self.patience >= self.args.patience:
                        break
            elif is_main_process() and (epoch + 1) % self.args.save_epochs == 0:
                logger.info(f"Saving ckpt at epoch {epoch}...")
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch,}
                save_dict = {
                    "epoch": epoch,
                    "model": self.model_without_ddp.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "schedular": self.schedular.state_dict(),
                }
                if not os.path.exists(self.args.save_path):
                    os.mkdir(self.args.save_path)
                torch.save(save_dict, os.path.join(self.args.save_path, "checkpoint_%d.pth" % epoch))
                logger.info(json.dumps(log_stats))
            
            if self.args.distributed:
                dist.barrier()
        if self.args.validate:
            if not os.path.exists(self.args.save_path):
                os.mkdir(self.args.save_path)
            torch.save(save_dict, os.path.join(self.args.save_path, "best_model.pth"))

    def train_epoch(self, epoch):
        metric_logger = MetricLogger(self.args, delimiter="  ")
        metric_logger.add_meter('lr', SmoothedValue(window_size=50, fmt='{value:.6f}'))
        for k in self.model_without_ddp.loss_names:
            metric_logger.add_meter(k, SmoothedValue(window_size=200, fmt='{avg:.4f}'))
        header = 'Train Epoch: [{}]'.format(epoch)

        self.model.train()
        for i, data in enumerate(metric_logger.log_every(self.train_dataloader, 5, header)):
            with autocast(dtype=torch.bfloat16):
                if not hasattr(self.model_without_ddp, "forward_fn"):
                    loss, output = self.model(*data)
                else:
                    loss, output = self.model_without_ddp.forward_fn(*data)

            loss /= self.args.grad_accu_steps
            loss.backward()
            if self.args.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(list(self.model.parameters()), max_norm=10)
            if (i + 1) % self.args.grad_accu_steps == 0:
                self.all_steps += 1
                if self.all_steps in [5000, 20000] and is_main_process():
                    logger.info(f"Best ckpt at step {self.all_steps}...")
                    save_dict = {
                        "model": copy.deepcopy(self.model_without_ddp.state_dict()),
                    }
                    torch.save(save_dict, os.path.join(self.args.save_path, "step_%dK.pth" % (self.all_steps // 1000)))
                    if self.args.distributed:
                        dist.barrier()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.schedular.step()

            metric_logger.update(lr = self.optimizer.param_groups[0]["lr"])
            metric_logger.update(**output)

        metric_logger.synchronize_between_processes()
        logger.info(f"Averaged stats: {metric_logger.global_avg()}")
        return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

    def validate_epoch(self):
        logger.info("Validating...")
        self.model.eval()
        all_preds, all_gts = [], []
        for i, data in enumerate(self.valid_dataloader):
            with torch.no_grad():
                with autocast(dtype=torch.bfloat16):
                    if not hasattr(self.model_without_ddp, "validate_fn"):
                        preds, _ = self.model(*data)
                        gt = torch.zeros(1).to(self.device)
                    else:
                        if len(data) == 2:
                            preds, gt = self.model_without_ddp.validate_fn(data[0], [self.train_dataset.wild_type], [self.train_dataset.prompt], data[1])
                        else:
                            preds, gt = self.model_without_ddp.validate_fn(*data)
                all_preds.append(preds.cpu())
                all_gts.append(gt.cpu())
            
            if i % 50 == 0:
                logger.info(f"Validation step {i}/{len(self.valid_dataloader)}")
        all_preds = torch.cat(all_preds, dim=0)
        all_gts = torch.cat(all_gts, dim=0)
        all_preds = concat_gather(all_preds)
        all_gts = concat_gather(all_gts)
        if self.args.dataset_name == "mixfitness":
            scores = []
            for i in range(self.valid_dataset.n_datasets):
                st, ed = i * self.valid_dataset.nshot, (i + 1) * self.valid_dataset.nshot
                scores.append(name2metric[self.args.metric](all_gts[st:ed].numpy(), all_preds[st:ed].numpy()))
            print(scores)
            score = np.mean(scores)
        else:
            score = name2metric[self.args.metric](all_gts.numpy(), all_preds.numpy())
        logger.info(f"Validation result: {self.args.metric} = {score}")
        return score