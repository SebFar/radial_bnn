import numpy as np
import torch
from base import BaseTrainer
from model.metric import _mi


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self,
                 model,
                 loss,
                 metrics,
                 optimizer,
                 resume,
                 config,
                 data_loader,
                 valid_data_loader=None):
        super(Trainer, self).__init__(model,
                                      loss,
                                      metrics,
                                      optimizer,
                                      resume,
                                      config)
        self.config = config
        self.data_loader = data_loader

        if 'variational_train_samples' in config['trainer']:
            self.variational_train_samples = config['trainer']['variational_train_samples']
        else:
            self.variational_train_samples = 1

        if 'variational_eval_samples' in config['trainer']:
            self.variational_eval_samples = config['trainer']['variational_eval_samples']
        else:
            self.variational_eval_samples = 1


        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None

        self.pretrain_epochs = None
        if 'pretrain_epochs' in config['trainer']:
            self.pretrain_epochs = config['trainer']['pretrain_epochs']
            print("We will pretrain for {} epochs.".format(self.pretrain_epochs))

    def _eval_metrics(self, output, target, verbose=True):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            if verbose:
                self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        num_batches = len(self.data_loader)
        if hasattr(self.loss, 'set_num_batches'):
            self.loss.set_num_batches(num_batches)

        epoch_variational_samples = self.variational_train_samples
        if self.pretrain_epochs is not None:
            if epoch <= self.pretrain_epochs:
                layer_counter = self.model.pretraining(True)
                print("This epoch is with pretraining - point estimates and NLL loss only. {} layers affected".format(layer_counter))
                epoch_variational_samples = 1
            elif epoch == self.pretrain_epochs + 1:
                print("Turning off pretraining now.")
                self.model.pretraining(False)

        total_metrics = np.zeros(len(self.metrics))
        total_loss = 0

        for batch_idx, (data, target) in enumerate(self.data_loader):
            self.optimizer.zero_grad()
            output = self._predict(data, epoch_variational_samples)
            target = target.to(self.device)
            nll_loss, kl_term = self.loss.compute_loss(output, target)
            batch_loss = nll_loss + kl_term
            batch_loss.backward()
            self.optimizer.step()

            total_loss += batch_loss.item()

            if self.verbosity >= 2:
                batch_metrics = self._eval_metrics(output.detach(),
                                                   target.detach())  # Note dims are N x ___ x Var Samples
                total_metrics += batch_metrics / num_batches
                self.writer.add_scalar('loss', batch_loss.detach())

            if self.verbosity >= 2 and batch_idx % self.metric_freq == 0:

                self.logger.info('Train Epoch: {:4d} [{:3d}/{:3d} ({:2.0f}%)] Loss: {:10.2f} Running Acc: {:.3f}'.format(
                    epoch,
                    batch_idx,
                    num_batches,
                    100.0 * batch_idx / num_batches,
                    batch_loss,
                    total_metrics[0] * (num_batches / (batch_idx + 1))))

        log = {
            'loss': total_loss / num_batches,
            'metrics': (total_metrics).tolist()
        }

        if self.do_validation and epoch % self.eval_freq == 0:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()

        total_kl = 0.0
        total_nll = 0.0
        num_batches = len(self.valid_data_loader)
        all_val_metrics = np.zeros((num_batches, len(self.metrics)))
        if hasattr(self.data_loader.sampler, "indices"):
            print("Evaluating on datapoints: ", len(self.valid_data_loader.sampler.indices))
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                output = self._predict(data, self.variational_eval_samples)

                target = target.to(self.device)

                nll_loss, kl_term  = self.loss.compute_loss(output, target)
                total_nll += nll_loss
                total_kl += kl_term

                batch_eval_metrics = self._eval_metrics(output.detach(), target.detach(), verbose=False)
                all_val_metrics[batch_idx, :] = batch_eval_metrics

            total_val_metrics = np.mean(all_val_metrics, axis=0)
            total_val_metric_se = np.std(all_val_metrics, axis=0) / np.sqrt(num_batches)
            total_val_loss = total_nll + total_kl
            self.writer.add_scalar('val_loss', (total_val_loss))
            self.writer.add_scalar('val_kl', (total_kl))
            self.writer.add_scalar('val_nll', (total_nll))
            for i, metric in enumerate(self.metrics):
                self.writer.add_scalar('val_{}'.format(metric.__name__), total_val_metrics[i])

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics).tolist(),
            'val_metrics_se': (total_val_metric_se).tolist()
        }

    def _predict(self, data, variational_samples=None):
        """
        Predicts the class based on input data and the number of variational samples to take
        :param data: Tensor [examples, channels, height, width]
        :param variational_samples: integer
        :return: Tensor [examples, samples, class]
        """
        data = data.type(self.data_type)
        if len(data.shape) == 1:
            # If we have 1-d input data, we need the batch-size to be the first dimension.
            data = torch.unsqueeze(data, 1)

        data = data.to(self.device)
        data = data.unsqueeze(1)
        data = data.expand((-1, variational_samples, -1, -1, -1))
        output = self.model(data)
        return output

    def _estimate(self, data, variational_samples=20):
        """
        Estimates the mean and uncertainty for each example passed in the data.
        :param data: Tensor [examples, channels, height, width]
        :return: prediction, uncertainty
        prediction and uncertainty are both a np.array of length [examples] (where prediction is the probability that
        the class is equal to 1.
        """

        with torch.no_grad():
            data = torch.Tensor(data)
            data = data.to(self.device)
            output = self._predict(data.permute((0,3,1,2)), variational_samples=20)
            predictions = torch.mean(torch.exp(output), dim=1)[:, 1]
            uncertainties = _mi(output, None)  # This is [examples]
            return predictions, uncertainties
