# -*- coding: utf-8 -*-
"""Ref:https://github.com/RUCAIBox/RecBole/blob/master/recbole/trainer/trainer.py"""

# standard library
import os
from logging import getLogger
from time import time
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.optim as optim
import pytorch_warmup as warmup
from tqdm import tqdm

# internal
import sys

sys.path.append('../')
from evaluation.metrics import eiu_cut_off, calculate_global_metrics
from utils.helpers import *
from utils.logger import set_color
from utils.wandblogger import *


class AbstractTrainer(ABC):
    r"""Trainer Class is used to manage the training and evaluation processes of recommender system models.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, config, model):
        self.config = config
        self.model = model

    @abstractmethod
    def fit(self, train_data):
        r"""Train the model based on the train data.
        """
        raise NotImplementedError('Method [next] should be implemented.')

    @abstractmethod
    def evaluate(self, eval_data):
        r"""Evaluate the model based on the eval data.
        """
        raise NotImplementedError('Method [next] should be implemented.')


class Trainer(AbstractTrainer):
    def __init__(self, config, model):
        super().__init__(config, model)
        self.logger = getLogger()
        self.wandblogger = WandbLogger(config)

        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.weight_decay = config['weight_decay']

        self.gd_accum_steps = config['gd_accum']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        # self.clip_grad_norm = config['clip_grad_norm']
        self.valid_metric = config['valid_metric'].lower()
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.test_batch_size = config['eval_batch_size']
        self.gpu_available = torch.cuda.is_available() and config['device'] == 'cuda'
        self.device = config['device']

        self.checkpoint_dir = config['checkpoint_dir']
        ensure_dir(self.checkpoint_dir)
        self.saved_model_file = config['saved_model_file']

        self.start_epoch = 0
        self.cur_step = 0
        self.best_train_loss = np.inf
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result = None
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()

    def _build_optimizer(self, **kwargs):
        r"""Init the Optimizer
        Args:
            params (torch.nn.Parameter, optional): The parameters to be optimized.
                Defaults to ``self.model.parameters()``.
            learner (str, optional): The name of used optimizer. Defaults to ``self.learner``.
            learning_rate (float, optional): Learning rate. Defaults to ``self.learning_rate``.
            weight_decay (float, optional): The L2 regularization weight. Defaults to ``self.weight_decay``.
        Returns:
            torch.optim: the optimizer
        """
        params = kwargs.pop('params', self.model.parameters())
        learner = kwargs.pop('learner', self.learner)
        learning_rate = kwargs.pop('learning_rate', self.learning_rate)

        if learner.lower() == 'adam':
            if self.weight_decay > 0:
                optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=self.weight_decay)
            else:
                optimizer = optim.Adam(params, lr=learning_rate)
        elif learner.lower() == 'sgd':
            optimizer = optim.SGD(params, lr=learning_rate)
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        r"""Train the model in an epoch
        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.
            show_progress (bool): Show the progress of training epoch. Defaults to ``False``.
        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, it will return a
            tuple which includes the sum of loss in each part.
        """
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                # desc=set_color(f"Train {epoch_idx:>5}", 'pink'),
                desc=f"Train {epoch_idx:>5}"
            ) if show_progress else train_data
        )
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            losses = loss_func(interaction)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)
            loss.backward()
            # if self.clip_grad_norm:
            #     clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            # if self.gpu_available and show_progress:
            #     iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
        return total_loss

    def _valid_epoch(self, valid_data, show_progress=False):
        r"""Valid the model with valid data
        Args:
            valid_data (DataLoader): the valid data.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.
        Returns:
            float: valid score
            dict: valid result
        """
        valid_result = self.evaluate(valid_data, load_best_model=False, show_progress=show_progress)
        valid_score = calculate_valid_score(valid_result, self.valid_metric)
        return valid_score, valid_result

    def _save_checkpoint(self, epoch, verbose=True, **kwargs):
        r"""Store the model parameters information and training information.
        Args:
            epoch (int): the current epoch id
        """
        saved_model_file = kwargs.pop('saved_model_file', self.saved_model_file)
        state = {
            'config': self.config,
            'epoch': epoch,
            'cur_step': self.cur_step,
            'best_train_loss': self.best_train_loss,
            'best_valid_score': self.best_valid_score,
            'state_dict': self.model.state_dict(),
            'other_parameter': self.model.other_parameter(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, saved_model_file)
        if verbose:
            # self.logger.info(set_color('Saving current', 'blue') + f': {saved_model_file}')
            print(set_color('Saving current', 'blue') + f': {saved_model_file}')

    def resume_checkpoint(self, resume_file):
        r"""Load the model parameters information and training information.
        Args:
            resume_file (file): the checkpoint file
        """
        resume_file = str(resume_file)
        self.saved_model_file = resume_file
        checkpoint = torch.load(resume_file)
        self.start_epoch = checkpoint['epoch'] + 1
        self.cur_step = checkpoint['cur_step']
        self.best_valid_score = checkpoint['best_valid_score']

        # load architecture params from checkpoint
        if checkpoint['config']['model'].lower() != self.config['model'].lower():
            self.logger.warning(
                'Architecture configuration given in config file is different from that of checkpoint. '
                'This may yield an exception while state_dict is being loaded.'
            )
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.load_other_parameter(checkpoint.get('other_parameter'))

        # load optimizer state from checkpoint only when optimizer type is not changed
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        message_output = 'Checkpoint loaded. Resume training from epoch {}'.format(self.start_epoch)
        self.logger.info(message_output)

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        des = self.config['loss_decimal_place'] or 4
        train_loss_output = (set_color('epoch %d training', 'green') + ' [' + set_color('time', 'blue') +
                             ': %.2fs, ') % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            des = (set_color('train_loss%d', 'blue') + ': %.' + str(des) + 'f')
            train_loss_output += ', '.join(des % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            des = '%.' + str(des) + 'f'
            train_loss_output += set_color('train loss', 'blue') + ': ' + des % losses
        return train_loss_output + ']'

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        r"""Train the model based on the train data and the valid data.
        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True
            show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
            callback_fn (callable): Optional callback function executed at end of epoch.
                                    Includes (epoch_idx, valid_score) input arguments.
        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1, verbose=verbose)

        # self.eval_collector.data_collect(train_data)
        # if self.config['train_neg_sample_args'].get('dynamic', 'none') != 'none':
        #     train_data.get_model(self.model)
        valid_step = 0

        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)
            # self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
            self.wandblogger.log_metrics({'epoch': epoch_idx, 'train_loss': train_loss, 'train_step': epoch_idx},
                                         head='train')

            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data, show_progress=show_progress)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger
                )
                valid_end_time = time()
                valid_score_output = (set_color("epoch %d evaluating", 'green') + " [" + set_color("time", 'blue')
                                      + ": %.2fs, " + set_color("valid_score", 'blue') + ": %f]") % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = set_color('valid result', 'blue') + ': \n' + dict2str(valid_result)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                # self.tensorboard.add_scalar('Vaild_score', valid_score, epoch_idx)
                self.wandblogger.log_metrics({**valid_result, 'valid_step': valid_step}, head='valid')

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx, verbose=verbose)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break

                valid_step += 1

        # self._add_hparam_to_tensorboard(self.best_valid_score)
        return self.best_valid_score, self.best_valid_result

    @torch.no_grad()
    def evaluate(self, eval_data, eval_func=None, load_best_model=True, model_file=None, show_progress=False):
        r"""Evaluate the model based on the eval data.
        Args:
            eval_data (DataLoader): the eval data
            load_best_model (bool, optional): whether load the best model in the training process, default: True.
                                              It should be set True, if users want to test the model after training.
            model_file (str, optional): the saved model file, default: None. If users want to test the previously
                                        trained model file, they can set this parameter.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.
        Returns:
            collections.OrderedDict: eval result, key is the eval metric and value in the corresponding metric value.
        """
        if not eval_data:
            return

        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.load_other_parameter(checkpoint.get('other_parameter'))
            message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
            self.logger.info(message_output)

        self.model.eval()

        if eval_func is None:
            self.logger.error('No evaluation function defined')
            return

        result = eval_func(eval_data)  # dictionary

        self.wandblogger.log_eval_metrics(result, head='eval')

        return result

    def _load_checkpoint(self, checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.load_other_parameter(checkpoint.get('other_parameter'))
        message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
        self.logger.info(message_output)


class MOOTrainer(Trainer):

    def _train_epoch(self, R, epoch_idx, loss_func=None):
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        self.optimizer.zero_grad(set_to_none=True)
        eloss, iloss, uloss, pconst = loss_func(R.to(self.config['device']))
        loss = eloss * self.model.weights[0] + iloss * self.model.weights[1] + uloss * self.model.weights[2] + pconst * \
               self.model.weights[3]
        loss.backward()
        self.optimizer.step()
        if self.gpu_available:
            print(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
        return eloss, iloss, uloss, pconst, loss

    def _valid_epoch(self, valid_data, show_progress=False):
        r"""Valid the model with valid data
        Args:
            valid_data (DataLoader): the valid data.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.
        Returns:
            float: valid score
            dict: valid result
        """
        valid_result = self.evaluate(valid_data, load_best_model=False, show_progress=show_progress)
        if not self.valid_metric:
            valid_score = -valid_result['envy'] - valid_result['inferiority'] + valid_result['utility']
        else:
            valid_score = valid_result[self.valid_metric]
        return valid_score, valid_result

    @torch.no_grad()
    def evaluate(self, eval_data, eval_func=eiu_cut_off, agg=True, load_best_model=True, model_file=None,
                 show_progress=False, extra_eval=calculate_global_metrics):
        res = {}
        res['model_path'] = ''
        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file
            self._load_checkpoint(checkpoint_file)
            res['model_path'] = checkpoint_file.split('/')[-1].split('.')[0]

        self.model.eval()
        p = self.model()
        # TODO move to cpu to save memory -> worked, but too slow?
        e, i, u = eval_func(R=eval_data.to(self.config['eval_device']), Pi=p.detach().to(self.config['eval_device']),
                            k=self.model.k, agg=False)  # always return the matrices
        e_mean = e.sum(-1).mean().item()
        i_mean = i.sum(-1).mean().item()
        u_mean = u.mean().item()
        if agg:
            res.update({'envy': e_mean, 'inferiority': i_mean, 'e+i': e_mean + i_mean, 'utility': u_mean,
                        'u-i': u_mean - i_mean})
        else:
            res.update({'envy': e, 'inferiority': i, 'utility': u})

        if load_best_model:  # log best evaluation scores
            e_std = e.sum(-1).std().item()
            i_std = i.sum(-1).std().item()
            u_std = u.std().item()
            log_dict = {'envy': e_mean, 'inferiority': i_mean, 'e+i': e_mean + i_mean, 'utility': u_mean,
                        'envy_std': e_std, 'inferiority_std': i_std, 'utility_std': u_std}
            if extra_eval is not None:
                extra_res = extra_eval(p.detach().to(self.config['eval_device']),
                                       eval_data.to(self.config['eval_device']), k=self.model.k)
                res.update(extra_res)
                log_dict.update(extra_res)
            self.wandblogger.log_metrics(log_dict, head='eval')
        return res

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        if self.wandblogger.log_wandb:
            self.wandblogger._wandb.watch(self.model, log='all', log_freq=5)
        valid_step = 0

        for epoch_idx in range(self.start_epoch, self.epochs):

            training_start_time = time()
            train_loss_tuple = self._train_epoch(train_data, epoch_idx)
            train_loss_tuple = tuple(per_loss.item() for per_loss in train_loss_tuple)
            eloss, iloss, uloss, pconst, loss = train_loss_tuple

            training_end_time = time()

            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss_tuple)

            print(train_loss_output)

            self.wandblogger.log_metrics(
                {'epoch': epoch_idx, 'eloss': eloss, 'iloss': iloss, 'uloss': uloss, 'pconst': pconst, 'loss': loss})

            # eval on cut-off rec
            if self.eval_step <= 0 or valid_data is None:
                if saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data, show_progress=show_progress)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger
                )

                valid_end_time = time()
                valid_score_output = (set_color("epoch %d evaluating", 'green') + " [" + set_color("time", 'blue')
                                      + ": %.2fs, " + set_color("valid_score", 'blue') + ": %f]") % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = set_color('valid result', 'blue') + ': \n' + dict2str(valid_result)

                print(valid_score_output)
                print(valid_result_output)

                self.wandblogger.log_metrics({**valid_result, 'valid_score': valid_score, 'epoch': valid_step},
                                             head='valid')

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx, verbose=verbose)
                    self.best_valid_result = valid_result

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        # self.logger.info(stop_output)
                        print(stop_output)
                    return
                valid_step += 1

        return


class MOObatchTrainer(MOOTrainer):
    def _train_epoch(self, data_tuple, epoch_idx, loss_func=None):
        R, idx_loader = data_tuple
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        self.optimizer.zero_grad(set_to_none=True)
        for i, batch_idx in enumerate(idx_loader):
            batch_idx = batch_idx.to(self.device)
            losses = loss_func(R, batch_idx)
            loss = losses[0] * self.model.weights[0] + losses[1] * self.model.weights[1] + \
                   losses[2] * self.model.weights[2] + losses[3] * self.model.weights[3]

            loss_tuple = tuple(per_loss.item() for per_loss in losses + (loss,))
            total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            loss.backward()
            if epoch_idx == 0:  # wait for the first epoch
                if ((i + 1) % self.gd_accum_steps == 0) or (i + 1 == len(idx_loader)):  # gradient accumulation
                    self.optimizer.step()
                    self.optimizer.zero_grad(
                        set_to_none=True)
            else:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

        total_loss = tuple([loss / (i + 1) for loss in total_loss])  # average person
        if self.gpu_available:
            print(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
        return total_loss

    def fit(self, train_data_tuple, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        if self.wandblogger.log_wandb:
            self.wandblogger._wandb.watch(self.model, log='all', log_freq=5)
        valid_step = 0

        for epoch_idx in range(self.start_epoch, self.epochs):

            training_start_time = time()

            train_loss_tuple = self._train_epoch(train_data_tuple, epoch_idx)
            eloss, iloss, uloss, pconst, loss = train_loss_tuple

            training_end_time = time()

            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss_tuple)

            # self.logger.info(train_loss_output) # TODO
            print(train_loss_output)

            self.wandblogger.log_metrics(
                {'epoch': epoch_idx, 'eloss': eloss, 'iloss': iloss, 'uloss': uloss, 'pconst': pconst, 'loss': loss})

            # eval on cut-off rec
            if self.eval_step <= 0 or valid_data is None:
                if saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data, show_progress=show_progress)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger
                )

                valid_end_time = time()
                valid_score_output = (set_color("epoch %d evaluating", 'green') + " [" + set_color("time", 'blue')
                                      + ": %.2fs, " + set_color("valid_score", 'blue') + ": %f]") % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = set_color('valid result', 'blue') + ': \n' + dict2str(valid_result)

                print(valid_score_output)
                print(valid_result_output)

                self.wandblogger.log_metrics({**valid_result, 'valid_score': valid_score, 'epoch': valid_step},
                                             head='valid')

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx, verbose=verbose)
                    self.best_valid_result = valid_result

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        print(stop_output)
                    return
                valid_step += 1

        return


class MOOsampleTrainer(MOObatchTrainer):
    def _train_epoch(self, data_tuple, sample_mode, epoch_idx, loss_func=None):
        def _common_loss_helper(R_, P_, eloss_scaler, iloss_scaler, uloss_scaler, loss_func, total_loss, renorm=False):
            losses = loss_func(R_, P_, renorm=renorm)
            if self.config['dynamic_scale'] and epoch_idx > self.config['warmup']:
                tmp_ = min(self.config['scale_max'],
                           np.round(abs(losses[2].detach().cpu()) / losses[1].detach().cpu(), decimals=-1),
                           np.round(losses[0].detach().cpu() / losses[1].detach().cpu(), decimals=-1))
                if tmp_ > 0.:
                    print(tmp_)
                    iloss_scaler *= tmp_
            loss = losses[0] * self.model.weights[0] * eloss_scaler + losses[1] * iloss_scaler * self.model.weights[1] + \
                   losses[2] * uloss_scaler * self.model.weights[2] + losses[3] * self.model.weights[3]
            loss_tuple = tuple(per_loss.item() for per_loss in losses + (loss,))
            total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            loss.backward()
            self.optimizer.step()
            if self.config.get('lr_warmup'):
                with self.warmup_scheduler.dampening():
                    self.lr_scheduler.step()
            self.wandblogger.log_metrics({'lr': self.optimizer.param_groups[0]['lr']})
            self.optimizer.zero_grad(set_to_none=True)
            return total_loss

        R, u_idx_loader, u_idx_loader1, i_idx_loader = data_tuple
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        self.optimizer.zero_grad(set_to_none=True)

        eloss_scaler = iloss_scaler = uloss_scaler = 1.
        if self.config.get('mixture'):
            u_idx_iter = iter(u_idx_loader)
            for i, sample_idx in enumerate(u_idx_loader1):
                u_idx_out = next(u_idx_iter)
                sample_idx = torch.concat((sample_idx, u_idx_out))
                P = self.model()
                R_, P_ = R[sample_idx, :].to(self.device), P[sample_idx, :]
                total_loss = _common_loss_helper(R_, P_, eloss_scaler, iloss_scaler, uloss_scaler, loss_func,
                                                 total_loss)
            total_loss = tuple([loss / (i + 1) for loss in total_loss])  # average person
            del u_idx_iter  # if not use all, just ignore; delete it to avoid error msg
        elif sample_mode == 'u':
            if not self.config['cluster']:
                for i, sample_idx in enumerate(u_idx_loader):
                    # sample_idx = sample_idx.to(self.device)
                    P = self.model()
                    R_, P_ = R[sample_idx, :].to(self.device), P[sample_idx, :]
                    total_loss = _common_loss_helper(R_, P_, eloss_scaler, iloss_scaler, uloss_scaler, loss_func,
                                                     total_loss)
                total_loss = tuple([loss / (i + 1) for loss in total_loss])  # average person
            else:
                idx_loader = u_idx_loader1
                print(epoch_idx, len(idx_loader))
                for i, sample_idx in enumerate(idx_loader):
                    P = self.model()
                    R_, P_ = R[sample_idx, :].to(self.device), P[sample_idx, :]
                    total_loss = _common_loss_helper(R_, P_, eloss_scaler, iloss_scaler, uloss_scaler, loss_func,
                                                     total_loss)
                total_loss = tuple([loss / (i + 1) for loss in total_loss])  # average person
        elif sample_mode == 'i':
            eloss_scaler = 0.005  # TODO don't understand, set this value from empirical observation
            for i, sample_idx in enumerate(i_idx_loader):
                P = self.model()
                R_, P_ = R[:, sample_idx].to(self.device), P[:, sample_idx]
                total_loss = _common_loss_helper(R_, P_, eloss_scaler, iloss_scaler, uloss_scaler, loss_func,
                                                 total_loss)
        elif sample_mode == 'both':
            for i, u_sample_idx in enumerate(u_idx_loader):
                u_sample_idx = u_sample_idx.to(self.device)
                for j, i_sample_idx in enumerate(i_idx_loader):
                    i_sample_idx = i_sample_idx.to(self.device)
                    P = self.model()
                    total_loss = _common_loss_helper(R[u_sample_idx, :][:, i_sample_idx].to(self.device),
                                                     P[u_sample_idx, :][:, i_sample_idx], eloss_scaler, iloss_scaler,
                                                     uloss_scaler, loss_func, total_loss, renorm=False)

            total_loss = tuple([loss / (i + 1) for loss in total_loss])  # average person
        elif sample_mode == 'alter':
            idx_loader = i_idx_loader if epoch_idx % 2 else u_idx_loader
            for i, sample_idx in enumerate(idx_loader):
                P = self.model()
                if epoch_idx % 2:
                    R_, P_ = R[:, sample_idx].to(self.device), P[:, sample_idx].to(self.device)
                    renorm = False
                else:
                    R_, P_ = R[sample_idx, :].to(self.device), P[sample_idx, :].to(self.device)
                    renorm = False
                total_loss = _common_loss_helper(R_, P_, eloss_scaler, iloss_scaler, uloss_scaler, loss_func,
                                                 total_loss, renorm=renorm)
            if not epoch_idx % 2:
                total_loss = tuple([loss / (i + 1) for loss in total_loss])  # average person
        else:
            raise NotImplementedError

        if self.gpu_available:
            print(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
        return total_loss

    def fit(self, train_data_tuple, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        if self.wandblogger.log_wandb:
            self.wandblogger._wandb.watch(self.model, log='all', log_freq=5)

        if self.config.get('lr_warmup'):
            num_steps = self.config['m'] * self.config['epochs']
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_steps)
            self.warmup_scheduler = warmup.UntunedLinearWarmup(self.optimizer)

        valid_step = 0

        # logic for different sample modes
        sample_mode = self.config['sample_mode']

        update_flag = stop_flag = False
        for epoch_idx in range(self.start_epoch, self.epochs):
            training_start_time = time()
            train_loss_tuple = self._train_epoch(train_data_tuple, sample_mode, epoch_idx)

            eloss, iloss, uloss, pconst, loss = train_loss_tuple
            # early stopping based on training loss
            if self.config['stopping'] == 'loss':
                self.best_train_loss, self.cur_step, stop_flag, update_flag = early_stopping(
                    iloss + uloss,
                    self.best_train_loss,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=False
                )
            training_end_time = time()

            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss_tuple)

            print(train_loss_output)

            self.wandblogger.log_metrics(
                {'epoch': epoch_idx, 'eloss': eloss, 'iloss': iloss, 'uloss': uloss, 'pconst': pconst, 'loss': loss})

            if update_flag:
                if saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose)

            if stop_flag:
                stop_output = 'Finished training, best loss in epoch %d' % \
                              (epoch_idx - self.cur_step * self.eval_step)
                if verbose:
                    print(stop_output)
                return

            # eval on cut-off rec
            if self.eval_step <= 0 or valid_data is None:  # Dont use eval step
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data, show_progress=show_progress)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger
                )

                valid_end_time = time()
                valid_score_output = (set_color("epoch %d evaluating", 'green') + " [" + set_color("time", 'blue')
                                      + ": %.2fs, " + set_color("valid_score", 'blue') + ": %f]") % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = set_color('valid result', 'blue') + ': \n' + dict2str(valid_result)

                print(valid_score_output)
                print(valid_result_output)

                self.wandblogger.log_metrics({**valid_result, 'valid_score': valid_score, 'epoch': valid_step},
                                             head='valid')

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx, verbose=verbose)
                    self.best_valid_result = valid_result

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        print(stop_output)
                    return
                valid_step += 1

        return
