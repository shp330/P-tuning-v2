import logging
import os
from typing import Dict, OrderedDict

from transformers import Trainer

logger = logging.getLogger(__name__)

_default_log_level = logging.INFO
logger.setLevel(_default_log_level)

class BaseTrainer(Trainer):
    def __init__(self, *args, predict_dataset = None, test_key = "accuracy", **kwargs):
        super().__init__(*args, **kwargs)
        self.predict_dataset = predict_dataset
        self.test_key = test_key
        self.best_metrics = OrderedDict({
            "best_epoch": 0,
            f"best_eval_{self.test_key}": 0,
        })

    def log_best_metrics(self):
        self.log_metrics("best", self.best_metrics)
        self.save_metrics("best", self.best_metrics, combined=False)

      

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            logs: Dict[str, float] = {}

            # 收集分布式训练中的损失值（处理多卡 / 分布式场景下的损失聚合），再取均值并转换为 Python 标量（item()），
            # 得到 tr_loss_scalar（训练损失标量）。
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            # 简洁的损失张量重置方式，通过张量自身相减，将训练损失张量（tr_loss）的值置为 0，为下一个日志周期的损失累积做准备。
            tr_loss -= tr_loss

            # 用当前累积的损失标量，除以从上一次日志记录到现在的训练步数，得到平均步损失并保留 4 位小数。
            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            # 记录当前学习率
            logs["learning_rate"] = self._get_learning_rate()

            # 更新总损失统计
            self._total_loss_scalar += tr_loss_scalar
            # 更新上一次日志记录的步数
            self._globalstep_last_logged = self.state.global_step
            # 存储计算量（FLOPs）
            self.store_flos()

            # 将日志信息（损失、学习率）正式上报 / 输出
            self.log(logs)

        eval_metrics = None
        if self.control.should_evaluate:
            # 执行模型评估：调用 self.evaluate() 方法，传入需要忽略的评估指标键，返回评估结果字典 eval_metrics。
            eval_metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            # 超参数搜索上报：将评估结果上报给超参数搜索工具（如 Optuna），用于超参数优化决策。
            self._report_to_hp_search(trial, epoch, eval_metrics)

            # 对比当前 epoch 的评估指标（self.test_key 是核心评估指标，如准确率accuracy、F1 值等）与历史最优指标，如果当前指标更优（此处为大于，适用于准确率等 "越高越好" 的指标）：
            #   更新历史最优指标字典 self.best_metrics，记录当前最优 epoch 和最优评估指标值。
            if eval_metrics["eval_"+self.test_key] > self.best_metrics["best_eval_"+self.test_key]:
                self.best_metrics["best_epoch"] = epoch
                self.best_metrics["best_eval_"+self.test_key] = eval_metrics["eval_"+self.test_key]

                if self.predict_dataset is not None:
                    # 若存在测试集（predict_dataset），进一步在测试集上执行预测（self.predict()），
                    #   并将测试集上的最优指标存入 self.best_metrics（支持单个测试集和多个测试集（字典形式）两种场景）
                    if isinstance(self.predict_dataset, dict):
                        for dataset_name, dataset in self.predict_dataset.items():
                            _, _, test_metrics = self.predict(dataset, metric_key_prefix="test")
                            self.best_metrics[f"best_test_{dataset_name}_{self.test_key}"] = test_metrics["test_"+self.test_key]
                    else:
                        _, _, test_metrics = self.predict(self.predict_dataset, metric_key_prefix="test")
                        self.best_metrics["best_test_"+self.test_key] = test_metrics["test_"+self.test_key]

            # 打印当前 epoch 的历史最优结果汇总，遍历 self.best_metrics 输出所有最优指标（如最优 epoch、最优评估指标、最优测试指标），
            #   并将最优指标字典上报 / 输出
            logger.info(f"***** Epoch {epoch}: Best results *****")
            for key, value in self.best_metrics.items():
                logger.info(f"{key} = {value}")
            self.log(self.best_metrics)

        # self.control.should_save：由控制器判断是否到达保存时机，通常是评估指标最优时、每个 epoch 结束后或按固定步数间隔
        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=eval_metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
