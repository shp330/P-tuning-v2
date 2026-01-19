import logging
import os
import random
import sys

from typing import Any, Dict, List, Optional, OrderedDict, Tuple, Union
import math
import random
import time
import warnings
import collections

from tqdm.auto import tqdm
from transformers import TrainingArguments, PretrainedConfig, __version__
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.deepspeed import deepspeed_init
from transformers.trainer import TRAINER_STATE_NAME
from transformers.trainer_callback import TrainerState
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.trainer_utils import (
    HPSearchBackend,
    ShardedDDPOption,
    TrainOutput,
    get_last_checkpoint,
    set_seed,
    speed_metrics,
)
from transformers.file_utils import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    is_torch_tpu_available,
)

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from training.trainer_base import BaseTrainer, logger


class ExponentialTrainer(BaseTrainer):
    """
    Notes:
        该方法的核心作用是实现端到端的模型训练流程，封装了从训练前的环境准备、 checkpoint 恢复、优化器配置，
        到训练中的批次迭代、梯度计算、参数更新，再到训练后的模型保存、指标统计等全流程逻辑，无需用户手动编写繁琐的训练循环，
        只需配置好训练参数（TrainingArguments）和相关组件即可完成模型训练。

        支持：断点续训、超参数搜索、混合精度训练、分布式训练（DDP/Deepspeed）、梯度累积、梯度裁剪、训练过程日志记录与模型保存等。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = None

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        if self.lr_scheduler is None:
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95, verbose=True)
        return self.lr_scheduler


    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Main training entry point.
        Args:
            resume_from_checkpoint (:obj:`str` or :obj:`bool`, `optional`):
                If a :obj:`str`, local path to a saved checkpoint as saved by a previous instance of
                :class:`~transformers.Trainer`. If a :obj:`bool` and equals `True`, load the last checkpoint in
                `args.output_dir` as saved by a previous instance of :class:`~transformers.Trainer`. If present,
                training will resume from the model/optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
                超参数搜索相关的 trial 实例
            ignore_keys_for_eval (:obj:`List[str]`, `optional`)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
                评估时需要忽略的模型输出键值
            kwargs:
                Additional keyword arguments used to hide deprecated arguments
                兼容已废弃参数的关键字参数
        """
        resume_from_checkpoint = None if not resume_from_checkpoint else resume_from_checkpoint
        # 一. 训练前准备（环境、设备、内存）
        #
        # 1. 内存追踪启动
        # 启动内存追踪器，用于记录训练过程中的内存占用情况，后续会将内存指标加入训练结果中，方便排查 OOM（内存溢出）问题。
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        # 2. 提取训练参数与训练状态标记
        # args 包含所有训练配置（批次大小、学习率、训练轮数等）
        args: TrainingArguments = self.args
        # 标记当前处于训练状态，避免训练过程中触发其他非训练逻辑（如评估的特殊处理）
        self.is_in_train = True

        # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # the following is a workaround:
        # 3. 全精度评估特殊处理（非训练时的设备配置）
        # 如果配置了 fp16_full_eval（全精度评估）且不进行训练（仅评估），将模型移动到指定设备（GPU/CPU/TPU），确保评估正常运行
        if args.fp16_full_eval and not args.do_train:
            self._move_model_to_device(self.model, args.device)

        # 二. 兼容废弃参数 model_path
        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")

        # 三. 超参数搜索初始化
        #
        # 如果是超参数搜索场景（传入了 trial），进行超参数搜索的初始化工作：
        # 配置超参数搜索后端（如 optuna、sigopt），根据 trial 调整训练参数（如学习率、批次大小等）
        # 重置随机种子（保证超参数搜索的可复现性）： 需要先运行，因为该方法可能修改随机种子，后续模型初始化、数据加载都依赖正确的种子。
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # 四. 模型重新初始化（可选）
        model_reloaded = False
        # 如果用户提供了 model_init 方法（自定义模型初始化逻辑），则执行以下操作：
        # 设置随机种子，保证模型初始化的可复现性
        # 调用 call_model_init() 执行自定义初始化逻辑，生成新的模型实例并赋值给 self.model
        # 标记 model_reloaded = True，后续需要对新模型进行设备迁移、包装等处理
        # 置空优化器和学习率调度器，因为模型重新初始化后，原有优化器 / 调度器不再适配，需要后续重新创建
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # 五. 断点续训：加载模型 checkpoint（模型权重、配置等信息）
        # Load potential model checkpoint
        # 1. 处理布尔类型的 resume_from_checkpoint：找最新的有效 checkpoint 路径，未找到 checkpoint，则中断训练
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        # 2. 验证 checkpoint 有效性
        if resume_from_checkpoint is not None:
            # 如果不存在模型权重文件则报错
            if not os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
                raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

            logger.info(f"Loading model from {resume_from_checkpoint}).")
            # 3. 加载 checkpoint 配置并检查版本兼容性
            if os.path.isfile(os.path.join(resume_from_checkpoint, CONFIG_NAME)):
                config = PretrainedConfig.from_json_file(os.path.join(resume_from_checkpoint, CONFIG_NAME))
                checkpoint_version = config.transformers_version
                if checkpoint_version is not None and checkpoint_version != __version__:
                    logger.warn(
                        f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                        f"Transformers but your current version is {__version__}. This is not recommended and could "
                        "yield to errors or unwanted behaviors."
                    )

            # 4. 加载模型权重（非 DeepSpeed 场景）
            if args.deepspeed:
                # will be resumed in deepspeed_init
                pass
            else:
                # 加载权重文件，指定 map_location="cpu" 避免直接加载到 GPU 导致 OOM
                # We load the model state dict on the CPU to avoid an OOM error.
                state_dict = torch.load(os.path.join(resume_from_checkpoint, WEIGHTS_NAME), map_location="cpu")

                # 将权重加载到 self.model 中（自动处理权重与模型的匹配、多卡兼容等）
                # If the model is on the GPU, it still works!
                self._load_state_dict_in_model(state_dict)

                # 删除 state_dict 释放内存，避免内存占用过高
                # release memory
                del state_dict

        # 六. 重新初始化模型后的后续处理
        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            # 如果配置了 place_model_on_device（默认开启），将新模型移动到指定设备
            if self.place_model_on_device:
                self._move_model_to_device(self.model, args.device)
            # 更新 self.model_wrapped（包装后的模型，用于分布式训练、混合精度等），使其指向新初始化的模型
            self.model_wrapped = self.model

        # 七. 训练数据加载与训练步数计算
        # Keeping track whether we can can len() on the dataset or not
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)

        # 训练数据加载器完成以下工作：
        #   - 数据批次划分、乱序
        #   - 分布式训练下的采样器配置
        #   - 数据预处理（基于用户定义的 collate_fn）
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # 计算训练参数（批次大小、步数、轮数）
        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        # 先计算总训练批次大小，包含三个维度的放大：
        #   - train_batch_size：单设备训练批次大小
        #   - gradient_accumulation_steps：梯度累积步数（模拟大批次训练）
        #   - world_size：分布式训练的设备数量（多卡 / 多节点）
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size
        if train_dataset_is_sized: # 可获取数据集长度的场景
            # len(train_dataloader) 返回的是 训练数据加载器（DataLoader）中一个 epoch 包含的 batch 数量
            # 计算每轮训练的更新步数 num_update_steps_per_epoch（数据加载器长度 ÷ 梯度累积步数）：
            #   总共 len(train_dataloader) 批数据，每 args.gradient_accumulation_steps 批更新一次，两者相除后得到每个轮次需要更新多少次
            num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            # 优先遵循 max_steps（最大训练步数）配置，其次遵循 num_train_epochs（训练轮数）配置
            if args.max_steps > 0:
                # max_steps: 总的优化步骤
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training datalaoder has a smaller size but it's
                # the best we can do.
                # 计算总训练样本数 num_train_samples，用于后续速度指标统计
                num_train_samples = args.max_steps * total_train_batch_size
            else: # 未配置最大步数，以训练轮数为准
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = len(self.train_dataset) * args.num_train_epochs
        else:   # 不可获取数据集长度的场景（如迭代式数据集）
            # see __init__. max_steps is set when the dataset has no __len__
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_train_samples = args.max_steps * total_train_batch_size

        # 八. 混合精度 / 分布式训练特殊配置
        # 1.下溢 / 上溢调试配置
        #   如果开启了 underflow_overflow 调试选项，用于检测训练过程中的梯度下溢 / 上溢，
        #   但该功能不支持数据并行（DP），仅支持分布式数据并行（DDP）。
        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa
        # 2. DeepSpeed 初始化（分布式训练优化）
        # delay_optimizer_creation：标记是否延迟创建优化器（分片 DDP 场景需要）
        delay_optimizer_creation = self.sharded_ddp is not None and self.sharded_ddp != ShardedDDPOption.SIMPLE
        # DeepSpeed 场景：调用 deepspeed_init() 完成 DeepSpeed 引擎初始化，加载模型、优化器、调度器，并更新相关实例属性
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        # 非 DeepSpeed 且不延迟创建：调用 create_optimizer_and_scheduler() 创建优化器（如 AdamW）和学习率调度器（如线性衰减）
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # 九. 训练状态初始化与模型包装
        # 1. 初始化训练状态: TrainerState 用于记录训练过程中的状态信息（全局步数、轮数、最佳模型 checkpoint 等），
        #    此处初始化并标记是否为超参数搜索场景。
        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # 2. 开启梯度检查点（可选）
        # Activate gradient checkpointing if needed
        # 如果配置了 gradient_checkpointing（梯度检查点），开启该功能以节省内存（牺牲少量计算速度，适用于大模型训练）
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # 3. 模型包装（分布式 / 混合精度兼容）
        model = self._wrap_model(self.model_wrapped)

        # 调用 _wrap_model() 对模型进行包装，支持以下功能：
        #   - 分布式数据并行（DDP）
        #   - 混合精度训练（AMP）
        #   - 模型分片（Sharded DDP）
        # 包装后更新 self.model_wrapped，后续训练均使用包装后的模型。
        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # 4. 延迟创建优化器（分片 DDP 场景）: 如果之前标记了延迟创建优化器，此处完成优化器和调度器的创建。
        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # 5. 加载优化器与调度器的 checkpoint（断点续训）
        # 断点续训场景下，加载之前保存的优化器和学习率调度器状态，保证训练的连续性（避免重新初始化优化器导致训练震荡
        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # 十、训练日志打印（初始化信息）
        # Train!
        num_examples = (
            self.num_examples(train_dataloader) if train_dataset_is_sized else total_train_batch_size * args.max_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        # 十一. 断点续训：加载训练状态与跳过已训练步数
        # 1. 初始化训练时间、已训练轮数、已训练步数等训练状态变量，为断点续训和后续指标统计做准备
        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0 # 已训练的轮数
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # 2. 加载之前的训练状态（断点续训）
        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            # global_step 更新步数
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            # ignore_data_skip：忽略数据的跳过：如果不忽略，则跳过恢复训练后的首个轮次已训练过的批次数据；
            #   忽略则从首个批次的首批次数据开始训练：这将导致模型在已见过的数据训练
            if not args.ignore_data_skip:
                # 当前轮已训练的步数
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            # 日志打印断点续训信息
            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                # 将跳过前 {epochs_trained} 个轮次（epochs），以及在第一个轮次中跳过前 {steps_trained_in_current_epoch} 个批次（batches），
                # 如果此过程耗时较长，您可以在启动命令中添加 --ignore_data_skip 标志，但这样会导致模型在已经见过的数据上继续训练。
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # 3. 更新回调处理器: 更新回调处理器的相关引用（模型、优化器等）
        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader

        # 同步训练状态参数，保证回调函数（如日志、保存 checkpoint）能获取正确的训练信息
        # 用于在超参数搜索（hyperparameter search） 过程中记录当前 trial（试验）的名称和参数
        # trial：表示一次超参数搜索的试验（例如 Optuna、Ray Tune、SigOpt 中的一个 trial）
        # hp_name：一个可选的回调函数，用于自定义 trial 的名称（比如根据超参数生成有意义的名字）。
        #   trial_name 作用：让日志、checkpoint 文件名等能包含有意义的 trial 标识（如 "lr=2e-5_bs=16"）
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        if trial is not None:
            # hp_search_backend：当前使用的超参数搜索后端（如 OPTUNA, RAY, SIGOPT 等）。
            # 不同超参搜索库的 trial 对象结构不同：
            #   - SigOpt：超参数存储在 trial.assignments 属性中（是一个字典）。
            #   - Optuna / Ray Tune：trial 本身就是一个类似字典的对象（可以直接当作参数字典使用）。
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            # hp_params(assignments)：将 trial 的超参数赋值（assignments）转换为标准字典格式。
            #   hp_params 会标准化超参数字典（例如过滤掉非标量值、确保 key 是字符串等），使其适合记录到日志或状态中
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None

        # 如果状态已经被保存，这里应该是一样的；但为了安全起见，以防训练参数发生了变化，最好在加载之后再设置此项。
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # 十二、训练循环（轮数循环 + 批次循环）：负责完成梯度计算、更新、日志记录等逻辑。
        # tr_loss：用于累积批次损失，使用 tensor 类型避免 TPU 训练中的同步问题【避免在 TPU（或分布式训练）中过早触发设备间同步】
        # 背景知识：在 TPU、多 GPU（DDP）等分布式训练环境中，不同设备（如多个 TPU core）是并行计算的。
        #         如果在训练循环中频繁调用 .item()（例如 loss.item()），PyTorch 会强制将张量从设备复制到 CPU，这会阻塞并同步所有设备，
        #         造成性能下降。特别是在 TPU 上，.item() 会触发昂贵的 host-device 同步，严重影响吞吐。

        # 1. 初始化损失变量
        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        # 创建一个值为 0.0 的标量张量；将其放到当前设备（CPU/GPU /TPU）上；它仍然是一个 device-side 张量，没有调用 .item()，所以不会触发同步
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        # _total_loss_scalar：用于存储损失的标量总和，后续计算平均训练损失
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        # 初始化模型梯度，将所有参数的梯度置为 0
        model.zero_grad()

        # 2. 开始训练，触发回调
        # 调用所有注册 on_train_begin 的回调函数，执行训练开始前的自定义逻辑（如初始化日志工具、自定义指标）。
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # 3. 跳过已训练的整轮（断点续训）
        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        # 跳过前 epochs_trained 个 epoch，以使数据加载器的随机状态处于正确的位置。
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                # 我们只需要开始一次迭代，即可触发采样器的随机化。
                # We just need to begin an iteration to create the randomization of the sampler.
                for _ in train_dataloader:
                    break

        # 4. 轮数循环（外循环）
        for epoch in range(epochs_trained, num_train_epochs):
            # 分布式训练采样器轮数更新
            # 分布式训练下更新采样器的轮数，保证每轮数据乱序的一致性
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # 如果需要，在每个 epoch 开始时重置 past mems（历史记忆）状态。
            # Reset the past mems state at the beginning of each epoch if necessary.
            ## Transformer 自回归模型（如 GPT、XLNet 等），在每个训练 epoch 开始时清除模型的历史缓存（past key/values 或 memory states），
            #    以避免跨 epoch 的状态污染。
            #
            ## args.past_index 是一个整数标志，表示模型输出中是否包含 past 状态，以及它在返回元组中的位置。
            #    - 如果 past_index >= 0，说明模型启用了 past 缓存功能；
            #    - 如果 past_index == -1，则表示未使用。
            ## self._past 用于在训练过程中累积或传递上一个 batch 的 past 状态（例如在长序列训练中实现 truncated BPTT）。
            #    但在标准 epoch-based 训练中，不同 epoch 之间不应共享历史状态，否则会导致数据泄露或训练不稳定。
            #
            # 为什么要重置：每个 epoch 应该从“干净”的状态开始；
            #   如果不重置，上一个 epoch 最后一个 batch 的 past 状态可能会错误地影响当前 epoch 的第一个 batch，破坏训练的独立性；
            #   特别是在使用【梯度检查点（gradient checkpointing）】 或【长序列建模】时，显式管理 past 状态很重要。
            if args.past_index >= 0:
                self._past = None

            # max_steps: 最大更新步数
            # steps_in_epoch：每轮的步数（批次数）
            # 总批次数 = args.max_steps(更新的次数) * args.gradient_accumulation_steps(每次更新多少批次)
            steps_in_epoch = (
                len(epoch_iterator) if train_dataset_is_sized else args.max_steps * args.gradient_accumulation_steps
            )

            # 调用轮开始回调函数 on_epoch_begin
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            step = -1 # DIFF ADD
            # 批次循环（内循环）：迭代当前轮的所有训练批次
            for step, inputs in enumerate(epoch_iterator):
                # (1) 跳过已训练批次（断点续训）
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                # (2) step 开始回调
                # 每 gradient_accumulation_steps 步调用一次步骤开始回调函数，执行自定义逻辑（如日志记录、学习率调整）
                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                # （3）单批次训练步骤（计算损失与梯度）
                # 在分布式训练（如 DDP）中配合梯度累积时，优化通信开销: 减少 DDP（DistributedDataParallel） 中不必要的梯度同步
                # 在 PyTorch 的 DistributedDataParallel (DDP) 中：
                #   默认情况下，每次调用 loss.backward() 都会触发跨 GPU 的梯度同步（all-reduce）。
                #   但在 梯度累积（Gradient Accumulation） 场景下，我们希望 只在累积的最后一步同步梯度，中间步骤不通信，以节省带宽和时间。
                if (
                        ((step + 1) % args.gradient_accumulation_steps != 0) # 当前不是梯度累积周期的最后一步
                        and args.local_rank != -1 # 处于分布式训练环境（DDP）
                        and args._no_sync_in_gradient_accumulation # 启用了“梯度累积期间不同步”选项
                ):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    # 梯度累积过程中，关闭 DDP 的梯度同步（model.no_sync()），避免不必要的通信开销，提升训练效率
                    with model.no_sync(): # 暂停梯度同步
                        # 调用 training_step() 完成单批次训练：前向传播计算损失、反向传播计算梯度
                        tr_loss_step = self.training_step(model, inputs)
                else:
                    tr_loss_step = self.training_step(model, inputs)

                # （4）损失累积与异常处理
                # 损失为 NaN/Inf：用之前的平均损失填充，避免训练中断
                #   - 防止因单个 batch 出现 NaN/Inf 导致训练崩溃或日志失真。
                #   - 保持 tr_loss 的连续性，使得日志中的平均损失（如每 500 步打印一次）不会因异常值而跳变
                #   - 不中断训练流程（注意：这里只影响日志损失，不影响梯度！梯度仍会因 NaN 而失效）
                if (
                        args.logging_nan_inf_filter # 是否启用 NaN/Inf 过滤（默认 True）。
                        and not is_torch_tpu_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # 用“历史平均损失”代替当前的 NaN/Inf 损失，让 tr_loss 继续平滑增长。
                    # if loss is nan or inf simply add the average of previous logged losses
                    #   - self.state.global_step：当前全局训练步数。
                    #   - self._globalstep_last_logged：上次记录日志时的 global_step。
                    #   - tr_loss / (1 + global_step - last_logged): 从上次日志以来的平均损失：
                    #     1 + ... 是为了防止除零（当第一次记录时，global_step == _globalstep_last_logged）。
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    # 累积单批次损失到 tr_loss
                    tr_loss += tr_loss_step

                # 累加当前的浮点运算次数（FLOPs）
                # self.floating_point_ops(inputs) 根据输入 inputs 和当前模型结构，估算本次前向+反向传播的 FLOPs
                # 这是估算值，不是硬件实际计数，但对比较模型效率很有用
                self.current_flos += float(self.floating_point_ops(inputs))

                # 在使用 DeepSpeed 时，每一步都调用优化器更新
                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                # （5）优化器步骤（梯度更新）:已达梯度累积批次数或该轮次总批次数小于梯度累积数并且已训练到最后一批
                if (step + 1) % args.gradient_accumulation_steps == 0 or steps_in_epoch == (step + 1):
                    # 梯度裁剪：每 gradient_accumulation_steps 步执行一次梯度裁剪（防止梯度爆炸）
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping
                        # AMP（自动混合精度）下需先 unscale_ 梯度，再裁剪。
                        if self.use_amp:
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        # 优先使用 优化器或模型自带的裁剪方法（如 FSDP、Sharded Optimizer），否则回退到 nn.utils.clip_grad_norm_
                        if hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if self.deepspeed:
                        pass  # called outside the loop
                    elif is_torch_tpu_available():
                        xm.optimizer_step(self.optimizer)
                    elif self.use_amp: # AMP 使用 scaler.step() + scaler.update()，并检查是否因 overflow 跳过更新
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        # 只有当 scale_before <= scale_after 时，才认为优化器真正运行了（未跳过）
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()

                    # 学习率调度器更新: 只在 epoch 结束时更新 LR scheduler！
                    if optimizer_was_run and not self.deepspeed and (step + 1) == steps_in_epoch: # DIFF Add condition: and (step + 1) == steps_in_epoch
                        self.lr_scheduler.step()

                    # 梯度清零：梯度清零，准备下一批次训练
                    model.zero_grad()
                    # 更新全局步数
                    self.state.global_step += 1
                    # 更新当前轮的进度（小数形式，如 1.5 表示第 2 轮的 50% 进度）
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch

                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                    # （6）日志、保存、评估：打印训练日志（损失/学习率等）、保存模型 checkpoint（按步数/轮数）、执行中间评估（验证集性能）
                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0: # DIFF BEGIN: FROM transformers.train()
                logger.warning(
                    f"There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True # DIFF END
            # 6. 轮数结束处理
            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            # 检查用户是否在 TrainingArguments 中启用了 debug="tpu_metrics_debug"
            # DebugOption.TPU_METRICS_DEBUG 是一个枚举值，表示“启用 TPU/XLA 调试指标”。
            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                # 检查当前环境是否支持 PyTorch/XLA（是否在 TPU 上运行，且已安装 torch_xla）。
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break


        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()

            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )

            best_model_path = os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)
            if os.path.exists(best_model_path):
                # We load the model state dict on the CPU to avoid an OOM error.
                state_dict = torch.load(best_model_path, map_location="cpu")
                # If the model is on the GPU, it still works!
                self._load_state_dict_in_model(state_dict)
            else:
                logger.warn(
                    f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                    "on multiple nodes, you should activate `--save_on_each_node`."
                )

            if self.deepspeed:
                self.deepspeed.load_checkpoint(
                    self.state.best_model_checkpoint, load_optimizer_states=False, load_lr_scheduler_states=False
                )

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step
        # 计算速度、浮点运算量等指标
        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)
        
        self.log(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        
        return TrainOutput(self.state.global_step, train_loss, metrics)
