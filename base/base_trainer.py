import os
import math
import json
import logging
import datetime
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from utils.util import ensure_dir


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self,
                 model,
                 loss,
                 metrics,
                 optimizer,
                 resume,
                 config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data_type = torch.float32

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer

        self.epochs = config['trainer']['epochs']
        self.save_freq = config['trainer']['save_freq']
        self.verbosity = config['trainer']['verbosity']
        self.eval_freq = config['trainer']['eval_freq']
        self.metric_freq = config['trainer']['metric_freq']
        self.early_stopping = config['trainer']['early_stopping']

        # configuration to monitor model performance and save best
        self.monitor = config['trainer']['monitor']
        self.monitor_mode = config['trainer']['monitor_mode']
        assert self.monitor_mode in ['min', 'max', 'off']
        self.monitor_best = math.inf if self.monitor_mode == 'min' else -math.inf
        self.monitor_best_se = 0
        self.start_epoch = 1
        self.best_epoch = 0

        # setup directory for checkpoint saving
        start_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
        self.checkpoint_dir = os.path.join(config['trainer']['save_dir'], config['name'], start_time)
        # setup visualization writer instance
        writer_dir = os.path.join(config['visualization']['log_dir'], config['name'], start_time)
        self.writer = SummaryWriter(writer_dir)
        if hasattr(self.loss, "set_writer"):
            self.loss.set_writer(self.writer)

        # Save configuration file into checkpoint directory:
        ensure_dir(self.checkpoint_dir)
        config_save_path = os.path.join(self.checkpoint_dir, 'config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(config, handle, indent=4, sort_keys=False)

        if resume:
            self._resume_checkpoint(resume)
    
    def _prepare_device(self, n_gpu_use):
        """ 
        setup GPU device if available, move model into configured device
        """ 
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            msg = "Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(n_gpu_use, n_gpu)
            self.logger.warning(msg)
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def train(self):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch, self.epochs + 1):

            result = self._train_epoch(epoch)
            
            # save logged informations into log dict
            log = {'epoch': epoch}
            for key, value in result.items():
                if key == 'metrics':
                    log.update({mtr.__name__ : value[i] for i, mtr in enumerate(self.metrics)})
                elif key == 'val_metrics':
                    log.update({'val_' + mtr.__name__ : value[i] for i, mtr in enumerate(self.metrics)})
                elif key == 'val_metrics_se':
                    log.update({'val_' + mtr.__name__ + "_se" : value[i] for i, mtr in enumerate(self.metrics)})
                else:
                    log[key] = value

            # print logged informations to the screen
            if self.verbosity >= 1:
                for key, value in log.items():
                    print('    {:25.20s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            self.latest = None
            if epoch % self.save_freq == 0:
                self._save_checkpoint(epoch, save_best=best)
            if self.monitor_mode != 'off':
                try:
                    if np.isnan(log[self.monitor]):
                        return self.monitor_best, log[self.monitor]
                    self.latest = log[self.monitor]
                    if  (self.monitor_mode == 'min' and log[self.monitor] < self.monitor_best) or\
                        (self.monitor_mode == 'max' and log[self.monitor] > self.monitor_best):
                        self.monitor_best = log[self.monitor]
                        self.monitor_best_se = 0 #log[self.monitor + "_se"]
                        best = True
                        self._save_checkpoint(epoch, save_best=best)
                        self.best_epoch = epoch
                    if self.early_stopping != 0:
                        if epoch - self.best_epoch >= self.early_stopping:
                            return self.monitor_best, log[self.monitor], self.monitor_best_se
                except KeyError as e:
                    msg = "Warning: Can\'t recognize metric" \
                            + "for performance monitoring. model_best checkpoint won\'t be updated. {}".format(e)
                    self.logger.warning(msg)
        return self.monitor_best, self.latest, self.monitor_best_se

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.monitor_best,
            'config': self.config
        }
        filename = os.path.join(self.checkpoint_dir, 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: {} ...".format('model_best.pth'))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.monitor_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning('Warning: Architecture configuration given in config file is different from that of checkpoint. ' + \
                                'This may yield an exception while state_dict is being loaded.')
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed. 
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning('Warning: Optimizer type given in config file is different from that of checkpoint. ' + \
                                'Optimizer parameters not being resumed.')
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
    
        self.train_logger = checkpoint['logger']
        self.logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))
