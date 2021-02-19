import os
from opts import parse_opts
from model import generate_model
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mobilenetv2 import get_fine_tuning_parameters
from models import mobilenetv2

def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.
    Args:
        outputs: (np.ndarray) output of the model
        labels: (np.ndarray) [0, 1, ..., num_classes-1]
    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels)/float(labels.size)


def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.
    Args:
        outputs: (Variable) dimension batch_size x 6 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]
    Returns:
        loss (Variable): cross entropy loss for all images in the batch
    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    return nn.CrossEntropyLoss()(outputs, labels)


def loss_fn_kd(outputs, labels, model_outputs, opt):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = opt.alpha
    T = opt.temperature
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(model_output/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss
     
def train_kd(epoch, data_loader, teacher, student, optimizer, opt, epoch_logger, batch_logger):
    print('train kd at epoch {}'.format(epoch))

    teacher.eval()
    student.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            targets = targets.cuda()

        inputs = Variable(inputs)
        targets = Variable(targets)
        KD_loss = loss_fn_kd(teacher(inputs), targets, student(inputs), opt)
        losses.update(KD_loss.data, inputs.size(0))

        prec1, prec5 = calculate_accuracy(student(inputs).data, targets.data, topk=(1,1))
        top1.update(prec1, inputs.size(0))
        # top5.update(prec5, inputs.size(0))

        optimizer.zero_grad()
        KD_loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val.item(),
            'prec1': top1.val.item(),
            'prec5': top5.val.item(),
            'lr': optimizer.param_groups[0]['lr']
        })
        if i % 10 ==0:
            print('Epoch: [{0}][{1}/{2}]\t lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {KD_loss.val:.4f} ({KD_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
                  'Prec@5 {top5.val:.5f} ({top5.avg:.5f})'.format(
                      epoch,
                      i,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      top1=top1,
                      top5=top5,
                      lr=optimizer.param_groups[0]['lr']))

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg.item(),
        'prec1': top1.avg.item(),
        'prec5': top5.avg.item(),
        'lr': optimizer.param_groups[0]['lr']
    })
    
