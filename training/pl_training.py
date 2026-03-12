import json
import sys

# setting path
sys.path.append('../')

import argparse
from typing import Dict, Optional, Any
from collections import defaultdict
import datasets
import torch
import torch.nn
from torch.distributed import ReduceOp
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel, AutoConfig

from adapter_fusion import AdapterFactory
from bert_pals import BertPalsEncoder
from mtl_datasets import ClassificationDataset, multi_collate, MultiLabelClassificationDataset, IRDataset, \
    CustomChainDataset, TripletDataset, RegressionDataset
from schedulers import InverseSquareRootSchedule, InverseSquareRootScheduleConfig
from strategies import BatchingStrategy
from tasks import TaskFamily, load_tasks

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, BatchSizeFinder
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from lightning_fabric.utilities.distributed import _sync_ddp_if_available as sync_ddp_if_available
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS, STEP_OUTPUT
from pytorch_lightning.callbacks import BatchSizeFinder

pl.seed_everything(42, workers=True)


def init_weights(modules):
    for module in modules:
        module.linear.weight.data.normal_(mean=0.0, std=0.02)
        if module.linear.bias is not None:
            module.linear.bias.data.zero_()


pl_to_split_map = {"fit": "train", "validate": "dev", "test": "test", "predict": "test"}


class SciRepTrain(pl.LightningModule):
    def __init__(self, batch_size: int, init_lr: float, peak_lr: float, tokenizer: str, model: str, warmup_steps: int,
                 log_dir: str,
                 use_ctrl_tokens=False,
                 task_dict: Dict[str, TaskFamily] = None,
                 pals_cfg: str = None, adapter_type: str = None, max_len: int = 512, load_adapters_as=None, use_prompts=False, use_last_token=False, use_cosine_schedule=False):
        super().__init__()
        self.task_dict = load_tasks() if not task_dict else task_dict
        print(self.task_dict.keys())
        self.heads = torch.nn.ModuleDict(
            {t.name: t.head for t in self.task_dict.values() if t.head}
        )
        self.init_loss = None
        self.task_idx = {t: i for i, t in enumerate(self.task_dict)}
        self.loss_wt = torch.ones(len(self.task_dict)).float()
        init_weights(self.heads.values())
        self.warmup_steps = warmup_steps
        self.multi_train = None
        self.multi_test = None
        self.multi_val = None
        self.pals = pals_cfg is not None
        self.adapters = adapter_type is not None
        self.use_ctrl_tokens = use_ctrl_tokens
        self.use_last_token = use_last_token
        spl_ctrl_tokens = set()
        for t in self.task_dict.values():
            if type(t.ctrl_token) == str:
                spl_ctrl_tokens.add(t.ctrl_token)
            else:
                spl_ctrl_tokens.update(t.ctrl_token.values())
        spl_ctrl_tokens = sorted(list(spl_ctrl_tokens))
        task_ids = spl_ctrl_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        if self.adapters:
            adapters_dir = f'{log_dir}/model/adapters/' if not load_adapters_as else load_adapters_as
            try:
                adapters_dir = json.loads(adapters_dir)
            except:
                pass
            self.encoder = AdapterFactory.get_adapter(model, task_ids,
                                                      adapter_type == "fusion", adapters_dir)
        else:
            self.encoder = AutoModel.from_pretrained(model, trust_remote_code=True, use_cache=False)
            self.encoder.gradient_checkpointing_enable()
            if self.pals:
                self.encoder = BertPalsEncoder(f"bert_pals_config/{pals_cfg}", task_ids, self.encoder)
        if self.use_ctrl_tokens:
            print("Using Control Tokens", spl_ctrl_tokens)
            special_tokens_dict = {'additional_special_tokens': spl_ctrl_tokens}
            num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
            self.encoder.resize_token_embeddings(len(self.tokenizer))
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.peak_lr = peak_lr
        self.max_len = max_len
        self.save_hyperparameters(ignore=["task_dict"])
        self.use_prompts = use_prompts
        self.val_losses = []
        self.task_val_losses = defaultdict(list)
        self.use_cosine_schedule = use_cosine_schedule

    def forward(self, input_ids, attention_mask=None, token_idx=0, task_id=None):
        if not self.pals:
            embedding = self.encoder(input_ids, attention_mask=attention_mask) if not self.adapters else self.encoder(
                input_ids,
                attention_mask=attention_mask,
                task_id=task_id)
            if self.use_last_token:
                # Last token pooling (matching Qwen3-Embedding)
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = embedding.last_hidden_state.shape[0]
                return embedding.last_hidden_state[
                    torch.arange(batch_size, device=embedding.last_hidden_state.device),
                    sequence_lengths,
                ]
            else:
                # Original CLS/position-based pooling
                return embedding.last_hidden_state[:, token_idx, :]
        else:
            embedding = self.encoder(input_ids, attention_mask=attention_mask, task_id=task_id)
            return embedding[:, token_idx, :]

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if
                           p.requires_grad and not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in self.named_parameters() if
                           p.requires_grad and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=self.init_lr, eps=1e-8
        )

        self.opt = optimizer

        # Calculate total training steps first (needed for all schedulers)
        try:
            # Try using trainer's estimate (works for non-streaming datasets)
            estimated_steps = self.trainer.estimated_stepping_batches
            self.logger.log_hyperparams({"estimated_stepping_batches": float(estimated_steps) if estimated_steps else 0.0})
            # If estimated_stepping_batches is 0 or None, fall back to manual calculation
            if not estimated_steps or estimated_steps <= 0:
                raise RuntimeError("estimated_stepping_batches is 0 or None")
            total_steps = estimated_steps
        except (AttributeError, RuntimeError):
            # Fallback for streaming datasets: calculate from sample sizes
            # total_steps = sum(sample_sizes) / (batch_size * grad_accum * num_gpus) * epochs
            total_samples = sum(
                task.sample_size.get('train', task.sample_size) if isinstance(task.sample_size, dict) else task.sample_size
                for task in self.task_dict.values()
            )
            steps_per_epoch = total_samples // (self.batch_size * self.trainer.accumulate_grad_batches * self.trainer.num_devices)
            total_steps = steps_per_epoch * self.trainer.max_epochs

        # Calculate warmup steps (supports both absolute number and proportion)
        # If warmup_steps < 1, treat as proportion; otherwise treat as absolute steps
        if self.warmup_steps < 1:
            warmup_steps = int(self.warmup_steps * total_steps)
        else:
            warmup_steps = int(self.warmup_steps)

        # Log scheduler configuration
        

        if self.pals or self.adapters:
            # Linear schedule uses optimizer's lr (init_lr), peak_lr not used
            scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
            self.logger.log_hyperparams(dict(scheduler_type="linear", num_warmup_steps=warmup_steps, num_training_steps=total_steps))
        elif self.use_cosine_schedule:
            # Use cosine schedule for instruction-aware training (Qwen3-Embedding best practice)
            # Cosine schedule uses optimizer's lr (init_lr), peak_lr not used
            # Warmup: 0 → init_lr, then cosine decay: init_lr → ~0.1 * init_lr
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
            self.logger.log_hyperparams(dict(scheduler_type="cosine", num_warmup_steps=warmup_steps, num_training_steps=total_steps))
        else:
            # InverseSquareRootSchedule uses both init_lr and peak_lr
            scheduler_config = InverseSquareRootScheduleConfig(warmup_updates=warmup_steps,
                                                               warmup_init_lr=self.init_lr,
                                                               lr=self.peak_lr)
            scheduler = InverseSquareRootSchedule(scheduler_config, optimizer)
            self.logger.log_hyperparams(dict(scheduler_type="inverse_square_root", warmup_updates=warmup_steps, warmup_init_lr=self.init_lr, lr=self.peak_lr))

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1}
        }

    def calc_loss(self, train_batch, batch_idx):
        losses, loss_per_task = [], torch.zeros(len(self.task_dict)).cuda()
        scl = torch.tensor(0.0)
        for name, batch in train_batch.items():
            task = self.task_dict[name]
            idx = 0 if not self.use_ctrl_tokens else 1
            task_id = task.ctrl_token
            if task.type not in set(["classification", "regression"]):
                query, pos, neg = batch[0][0], batch[0][1], batch[0][2]
                query_ctrl = cand_ctrl = task_id
                if type(task_id) == dict:
                    query_ctrl = task_id["query"]
                    cand_ctrl = task_id["candidates"]
                query_emb, pos_emb, neg_emb = (
                    self(
                        query["input_ids"],
                        attention_mask=query["attention_mask"],
                        token_idx=idx,
                        task_id=query_ctrl
                    ),
                    self(
                        pos["input_ids"],
                        attention_mask=pos["attention_mask"],
                        token_idx=idx,
                        task_id=cand_ctrl
                    ),
                    self(
                        neg["input_ids"],
                        attention_mask=neg["attention_mask"],
                        token_idx=idx,
                        task_id=cand_ctrl
                    ),
                )
                curr_loss = task.loss(query_emb, pos_emb, neg_emb)
            else:
                x, y = batch[0], batch[1]
                encoding = self(
                    x["input_ids"],
                    attention_mask=x["attention_mask"],
                    token_idx=idx,
                    task_id=task_id
                )
                logits = self.heads[name](encoding)
                if task.type == "regression":
                    logits = logits.squeeze()
                curr_loss = task.loss(logits, y)
                if task.multi_label:
                    curr_loss = torch.mean(curr_loss, dim=1)
                elif task.contrastive_loss:
                    scl = task.contrastive_loss(encoding, y, self.heads[name].num_labels)
                    curr_loss = 0.1 * curr_loss + 0.9 * scl
            loss_per_task[self.task_idx[name]] = torch.mean(curr_loss)
        return loss_per_task

    def training_step(self, train_batch, batch_idx):
        try:
            loss_per_task = self.calc_loss(train_batch, batch_idx)
            loss = torch.sum(loss_per_task)
            self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=self.batch_size, sync_dist=True)
            self.log("lr", self.lr_schedulers().get_last_lr()[-1], on_step=True, on_epoch=False, prog_bar=True, logger=True)
            return {"loss": loss}
        except Exception as e:
            import traceback
            print(f"[RANK {self.trainer.global_rank}] Exception in training_step at batch {batch_idx}: {e}")
            print(traceback.format_exc())
            raise

    def validation_step(self, train_batch, batch_idx) -> Optional[STEP_OUTPUT]:
        try:
            loss_per_task = self.calc_loss(train_batch, batch_idx)
            # loss_per_task = torch.mul(self.loss_wt.cuda(), loss_per_task)
            loss = torch.sum(loss_per_task)
            dist_loss_per_task = loss_per_task.clone().data
            dist_loss_per_task = sync_ddp_if_available(dist_loss_per_task, reduce_op=ReduceOp.SUM)
            for task in self.task_dict:
                self.task_val_losses[task].append(dist_loss_per_task[self.task_idx[task]].detach())
                self.log(f"val_loss_{task}", dist_loss_per_task[self.task_idx[task]], on_step=True, on_epoch=True,
                         prog_bar=False,
                         batch_size=self.batch_size, rank_zero_only=True)
            self.log("val_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
            self.val_losses.append(loss.detach())
            #self.log("avg_val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=self.batch_size)
            return {"val_loss": loss}
        except Exception as e:
            import traceback
            print(f"[RANK {self.trainer.global_rank}] Exception in validation_step at batch {batch_idx}: {e}")
            print(traceback.format_exc())
            raise

    def on_validation_epoch_end(self):
    
        self.log("avg_val_loss", torch.stack(self.val_losses).mean(), sync_dist=True) 
        for task, losses in self.task_val_losses.items():
            self.log(f"avg_val_loss_{task}", torch.stack(losses).mean(), sync_dist=True)
        self.val_losses = []
        self.task_val_losses = defaultdict(list)

    def load_data(self, split) -> CustomChainDataset:
        hf_split = "validation" if split == "dev" else "train"
        dataset_list = []
        task_dataset_map = {"classification": ClassificationDataset, "regression": RegressionDataset, "ir": IRDataset}
        for t_name, task in self.task_dict.items():
            data_file = {hf_split: task.data_files[split]} if task.data_files else None
            dataset_name = (task.dataset, hf_split)
            data_src = data_file if data_file else dataset_name
            op_token = task.ctrl_token if self.use_ctrl_tokens else None
            instr_prompt = task.instr_prompt if self.use_prompts else None
            if type(data_src) == dict:
                data = datasets.load_dataset("json", data_files=data_src, streaming=True)[
                    next(iter(data_src.keys()))]
            else:
                data = datasets.load_dataset(**data_src[0], split=data_src[1], streaming=True)
            kwargs = {"data": data, "ctrl_token": op_token, "instr_prompt": instr_prompt, "max_len": self.max_len, "task_name": t_name,
                      "tokenizer": self.tokenizer, "fields": task.input_fields,
                      "sample_size": task.sample_size[split] if type(task.sample_size) == dict else task.sample_size}

            if task.type == "classification":
                kwargs.update({"label_field": task.labels_field, "labels": task.labels})
            elif task.type == "regression":
                kwargs.update({"label_field": task.labels_field})
            if task.multi_label:
                dataset_list.append(MultiLabelClassificationDataset(**kwargs))
            else:
                dataset_list.append(task_dataset_map.get(task.type, TripletDataset)(**kwargs))
        multi_dataset = CustomChainDataset(dataset_list, batch_size=self.batch_size,
                                           device_rank=self.trainer.global_rank, num_devices=self.trainer.world_size,
                                           batching_strategy=BatchingStrategy.MIXED_PROPORTIONAL)
        if split == "train":
            self.multi_train = multi_dataset
        elif split == "dev":
            self.multi_val = multi_dataset

    def setup(self, stage: Optional[str] = None) -> None:
        self.load_data("train")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.multi_train, batch_size=self.batch_size, collate_fn=multi_collate, num_workers=1,
                          pin_memory=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        self.load_data("dev")
        return DataLoader(self.multi_val, batch_size=self.batch_size, collate_fn=multi_collate, num_workers=1)

    @rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        try:
            logger = self.logger
            log_dir = f'{logger.save_dir}/{logger.name}/{logger.version}/checkpoints'
            self.tokenizer.save_pretrained(f'{log_dir}/tokenizer/')
            self.tokenizer.save_vocabulary(f'{log_dir}/tokenizer/')
            self.encoder.save_pretrained(f'{log_dir}/model')
        except:
            print("Exception encountered while saving, try agin from checkpoint")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks-config', help='path to the task config file', default="sample_data/tasks_config.json")
    parser.add_argument('model', help='HuggingFace model to be used')
    parser.add_argument('--tokenizer', help='HuggingFace tokenizer to be used (same as model name if not supplied)',
                        default=None)
    parser.add_argument('--output', help='dir to save checkpoints and finetuned model', default="./lightning_logs/")

    parser.add_argument('version', help='experiment version')
    parser.add_argument('--pals-config', default=None, help='path to config file for PALS architecture')
    parser.add_argument('--adapter-type', default=None, help='type of adapter architecture (single/fusion)')
    parser.add_argument('--adapters-chkpt', default=None,
                        help='Adapters to be loaded either from a directory path or a dictionary of pretrained huggingface adapters with id')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--peak-lr', type=float, default=5e-5, help='initial learning rate')
    parser.add_argument('--warmup', type=float, default=700, help='warmup steps (absolute number if >= 1, proportion of total if < 1, e.g., 0.03 for 3%%)')
    parser.add_argument('--epochs', type=int, default=2, help='number of epochs')
    parser.add_argument('--grad-accum', type=int, default=8, help='grad accumulation steps')
    parser.add_argument('--ctrl-tokens', action='store_true', default=False, help='use control codes for tasks')
    parser.add_argument('--instr-prompts', action='store_true', default=False, help='use instruction prompts for tasks')
    parser.add_argument('--gpu', type=int, default=None, help='number of gpus')
    parser.add_argument('--max-len', type=int, default=512, help='max sequence length')
    parser.add_argument('--val-check-interval', type=float, default=1.0, help='validation loop interval')
    parser.add_argument('--checkpoint', default=None, help='resume from checkpoint path')
    parser.add_argument('--fast-dev-run', default=False, action='store_true', help='Do a quick testing run, not a full finetuning.')
    parser.add_argument('--find-batch-size', default=False, action='store_true', help='Search for optimal batch size')
    parser.add_argument('--limit-train-batches', default=None, type=int, help='Number of training batches to limit to.')
    parser.add_argument('--limit-val-batches', default=None, type=int, help='Number of validation batches to limit to.')
    parser.add_argument('--checkpoint-n-steps', default=100, type=int, help='How often to save checkpoints in number of effective batches.')
    parser.add_argument('--use-last-token', default=False, action='store_true', help='Whether to use last token for pooling.')
    parser.add_argument('--use-cosine-schedule', default=False, action='store_true', help='Whether to use cosine decay for the learning rate scheduler. Defaults to inverse square root scheduler if False and not adapters or pals.')
    parser.add_argument('--training-strategy', default=None, type=str, help='Training strategy to use.')

    args = parser.parse_args()
    mconfig = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    tasks_dict = load_tasks(args.tasks_config, mconfig.hidden_size)
    log_dir = args.output
    logger = TensorBoardLogger(
        save_dir=log_dir,
        version=args.version,
        name='full_run',
    )

    # second part of the path shouldn't be f-string
    filepath = f'{log_dir}/{logger.name}/{logger.version}/checkpoints/'
    checkpoint_callback_val_loss = ModelCheckpoint(
        dirpath=filepath,
        filename='ep-{epoch}_st-{step}_avg_val_loss-{avg_val_loss:.3f}',
        save_top_k=4,
        verbose=True,
        monitor='avg_val_loss',  # monitors metrics logged by self.log.
        mode='min',
        save_on_exception=False,
        save_last=True
    )
    checkpoint_callback_steps = ModelCheckpoint(
        dirpath=filepath,
        filename='ep-{epoch}_st-{step}',
        verbose=True,
        every_n_train_steps=args.checkpoint_n_steps
    )

    if not args.training_strategy:
        training_strategy="ddp_find_unused_parameters_true" if args.gpu > 1 else "auto"
    else:
        training_strategy = args.training_strategy
    
    model = SciRepTrain(batch_size=args.batch_size, init_lr=args.lr,
                        peak_lr=args.peak_lr,
                        tokenizer=args.tokenizer if args.tokenizer else args.model,
                        model=args.model,
                        warmup_steps=args.warmup,
                        use_ctrl_tokens=args.ctrl_tokens, task_dict=tasks_dict, pals_cfg=args.pals_config,
                        adapter_type=args.adapter_type, log_dir=filepath, max_len=args.max_len,
                        load_adapters_as=args.adapters_chkpt, use_prompts=args.instr_prompts, use_last_token=args.use_last_token, 
                        use_cosine_schedule=args.use_cosine_schedule)

    hparams = {"accelerator": "gpu" if args.gpu else "cpu", "devices": args.gpu if args.gpu else "auto",
               "val_check_interval": args.val_check_interval, "num_sanity_val_steps": 4,
               "max_epochs": args.epochs,
               "accumulate_grad_batches": args.grad_accum}
    
    callbacks = [checkpoint_callback_val_loss, checkpoint_callback_steps]
    
    if args.find_batch_size:
        batch_size_finder = BatchSizeFinder(init_val=1)
        callbacks.append(batch_size_finder)

    trainer = pl.Trainer(logger=logger,
                         strategy=training_strategy,
                         enable_checkpointing=True,
                         callbacks=callbacks,
                         precision="bf16-mixed",
                         fast_dev_run=args.fast_dev_run,
                         log_every_n_steps=5,
                         limit_train_batches=args.limit_train_batches,
                         limit_val_batches=args.limit_val_batches,
                         **hparams)
    logger.log_hyperparams(hparams)
    logger.log_hyperparams({"tasks": {k: str(v) for k, v in tasks_dict.items()}})
    trainer.fit(model, ckpt_path=args.checkpoint)
