# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import random
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score,multilabel_confusion_matrix,precision_score
import utils
import csv
import vision_transformer as vits
import pandas as pd



def misc_measures(confusion_matrix):
    
    acc = []
    sensitivity = []
    specificity = []
    G = []
    
    for i in range(1,confusion_matrix.shape[0]):
        cm1=confusion_matrix[i]

        acc.append(1.*(cm1[0,0]+cm1[1,1])/np.sum(cm1))
        sensitivity_ = 1.*cm1[1,1]/(cm1[1,0]+cm1[1,1])
        sensitivity.append(sensitivity_)
        specificity_ = 1.*cm1[0,0]/(cm1[0,1]+cm1[0,0])
        specificity.append(specificity_)
        G.append(np.sqrt(sensitivity_*specificity_))
        
    acc = np.array(acc).mean()
    sensitivity = np.array(sensitivity).mean()
    specificity = np.array(specificity).mean()
    G = np.array(G).mean()
    
    return acc, sensitivity, specificity, G





def eval_linear(args):
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    args.output_dir = args.task
    # ============ building network ... ============
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        embed_dim = model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
    # if the network is a XCiT
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
        embed_dim = model.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch]()
        embed_dim = model.fc.weight.shape[1]
        model.fc = nn.Identity()
    else:
        print(f"Unknow architecture: {args.arch}")
        sys.exit(1)
    model.cuda()
    model.train()
    # load weights to evaluate
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    print(f"Model {args.arch} built.")

    linear_classifier = LinearClassifier(embed_dim, num_labels=args.num_labels)
    linear_classifier = linear_classifier.cuda()
    linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.gpu])

    # ============ preparing data ... ============
    val_transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_val = datasets.ImageFolder(os.path.join(args.data_path, "val"), transform=val_transform)
    dataset_test = datasets.ImageFolder(os.path.join(args.data_path, "test"), transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if args.evaluate:
        utils.load_pretrained_linear_weights(linear_classifier, args.arch, args.patch_size)
        test_stats, auc_roc = validate_network(test_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens, args.task, epoch,mode='test',num_class=args.num_labels)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return
    
    

    train_transform = pth_transforms.Compose([
        pth_transforms.RandomResizedCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, "train"), transform=train_transform)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # set optimizer
    optimizer = torch.optim.SGD(
        linear_classifier.parameters(),
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256., # linear scaling rule
        momentum=0.9,
        weight_decay=0, # we do not apply weight decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": 0.}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=linear_classifier,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]

    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)

        train_stats = train(model, linear_classifier, optimizer, train_loader, epoch, args.n_last_blocks, args.avgpool_patchtokens)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            val_stats, auc_roc = validate_network(val_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens, args.task, epoch,mode='val',num_class=args.num_labels)
            print(f"Accuracy at epoch {epoch} of the network on the {len(dataset_val)} test images: {val_stats['acc1']:.1f}%")
            if auc_roc>best_acc:
                best_acc = auc_roc
                print('save the best checkpoint@@@')

                log_stats = {**{k: v for k, v in log_stats.items()},
                             **{f'test_{k}': v for k, v in val_stats.items()}}
                
                if utils.is_main_process():
                    with (Path(args.output_dir) / "log.txt").open("a") as f:
                        f.write(json.dumps(log_stats) + "\n")
                    save_dict = {
                        "epoch": epoch + 1,
                        'state_dict_encoder':model.state_dict(),
                        "state_dict_linear": linear_classifier.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "best_acc": best_acc,
                    }
                    torch.save(save_dict, os.path.join(args.output_dir, "checkpoint_best.pth.tar"))

        if epoch%1==0:
            test_stats, auc_roc = validate_network(test_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens, args.task, epoch,mode='test',num_class=args.num_labels)
            print(f"Accuracy of the network on the {len(dataset_test)} test images: {test_stats['acc1']:.1f}%")

            print(f'Max accuracy: {best_acc:.2f}%')        
    
    # after training, test the results with the best checkpoints
    
    state_dict_best = torch.load(os.path.join(args.output_dir, "checkpoint_best.pth.tar"), map_location="cpu")
    
    model.load_state_dict(state_dict_best['state_dict_encoder'], strict=True)
    linear_classifier.load_state_dict(state_dict_best['state_dict_linear'], strict=True)
    test_stats, auc_roc = validate_network(test_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens, args.task, epoch,mode='final_test',num_class=args.num_labels)
    
    

def train(model, linear_classifier, optimizer, loader, epoch, n, avgpool):
    linear_classifier.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for (inp, target) in metric_logger.log_every(loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        # cancel the no_grad() here and just output = model(inp)?
        if "vit" in args.arch:
            intermediate_output = model.get_intermediate_layers(inp, n)
            output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
            if avgpool:
                output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                output = output.reshape(output.shape[0], -1)
        else:
            output = model(inp)
        #print('@@@@@@@@@@',output.requires_grad)    
            
        output = linear_classifier(output)

        # compute cross entropy loss
        loss = nn.CrossEntropyLoss()(output, target)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # log 
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network(val_loader, model, linear_classifier, n, avgpool, task, epoch,mode,num_class):
    linear_classifier.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    if not os.path.exists(task):
        os.makedirs(task)
        
    prediction_decode_list = []
    prediction_list = []
    true_label_decode_list = []
    filename_list = []
    true_label_onehot_list = []
    pred_onehot = []
    
    for inp, target in metric_logger.log_every(val_loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        true_label=F.one_hot(target.to(torch.int64), num_classes=num_class)
        
        # forward
        with torch.no_grad():
            if "vit" in args.arch:
                intermediate_output = model.get_intermediate_layers(inp, n)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if avgpool:
                    output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)
            else:
                output = model(inp)
        output = linear_classifier(output)
        loss = nn.CrossEntropyLoss()(output, target)

        prediction_softmax = nn.Softmax(dim=1)(output)
        _,prediction_decode = torch.max(prediction_softmax, 1)
        _,true_label_decode = torch.max(true_label, 1)
        
        output_onehot = F.one_hot(prediction_decode.to(torch.int64), num_classes=num_class)


        prediction_decode_list.extend(prediction_decode.cpu().detach().numpy())
        true_label_decode_list.extend(true_label_decode.cpu().detach().numpy())
        true_label_onehot_list.extend(true_label.cpu().detach().numpy())
        #filename_list.extend(filename)
        
        prediction_list.extend(prediction_softmax.cpu().detach().numpy())
        pred_onehot.extend(output_onehot.detach().cpu().numpy())
            
            
        if linear_classifier.module.num_labels >= 5:
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        else:
            acc1, = utils.accuracy(output, target, topk=(1,))

        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
    # gather the stats from all processes
    true_label_decode_list = np.array(true_label_decode_list)
    prediction_decode_list = np.array(prediction_decode_list)
    confusion_matrix = multilabel_confusion_matrix(true_label_decode_list, prediction_decode_list,labels=[i for i in range(num_class)])
    acc, sensitivity, specificity, G = misc_measures(confusion_matrix)
    
    auc_roc = roc_auc_score(true_label_onehot_list, prediction_list,multi_class='ovr',average='macro')
    auc_pr = average_precision_score(true_label_onehot_list, prediction_list,average='macro') 
    F1 = f1_score(true_label_onehot_list, pred_onehot, zero_division=0, average='macro')  
    precision = precision_score(true_label_onehot_list, pred_onehot, zero_division=0, average='macro')       
            
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))

    
    print('Sklearn {} Metrics - Acc: {:.4f} AUC-roc: {:.4f} AUC-pr: {:.4f} F1-score: {:.4f}'.format(mode, acc, auc_roc, auc_pr, F1)) 
    
    if mode == 'final_test':
        dataset_name = os.path.basename(os.path.normpath(args.data_path))
        # Define the global CSV file path for final test results
        results_path = f'final_test_metrics_{dataset_name}.csv'

        # Check if the file exists to write the header
        file_exists = os.path.isfile(results_path)
        
        with open(results_path, mode='a', newline='', encoding='utf8') as cfa:
            wf = csv.writer(cfa)
            # Write the header only if the file is new
            if not file_exists:
                wf.writerow(['Seed', 'Task', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'AUC-ROC', 'AUC-PR', 'F1-Score', 'Loss'])
            wf.writerow([args.seed, task, acc, sensitivity, specificity, precision, auc_roc, auc_pr, F1, metric_logger.loss.global_avg])
    else:
        results_path = task+ '_metrics_{}.csv'.format(mode)
        with open(results_path,mode='a',newline='',encoding='utf8') as cfa:
            wf = csv.writer(cfa)
            data2=[[acc,sensitivity,specificity,precision,auc_roc,auc_pr,F1,metric_logger.loss]]
            for i in data2:
                wf.writerow(i)
    
    Data4stage2 = pd.DataFrame({ 'Label':true_label_decode_list, 'softmax_0':np.array(prediction_list)[:,0],'softmax_1':np.array(prediction_list)[:,1],'Prediction': prediction_decode_list})
    Data4stage2.to_csv(task+ 'results_{}.csv'.format(epoch), index = False, encoding='utf8')
    
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, auc_roc






class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=2):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
    parser.add_argument('--task', default='.', type=str, help='task name')
    
    args = parser.parse_args()
    eval_linear(args)

