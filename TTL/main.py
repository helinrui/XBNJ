from pickletools import optimize
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
import pandas as pd
from pandas.io.stata import precision_loss_doc
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import lr_scheduler, SGD, Adam, AdamW
import torch
from sklearn.metrics import precision_recall_curve, auc
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
import time
import random
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm
import sys
sys.path.extend("./utils/")
from utils.models import densenet121, densenet201, resnet50, clip_resnet50, bit_resnet50, freeze_resnet50, freeze_densenet201, cbr_larget, alexnet, efficientnet, layer_wise_freeze_resnet50
from utils.HAM import HAM
from utils.BIMCV import BIMCV
from utils.XBNJ import XBNJ
from utils.settings import parse_opts
from utils.slim_resnet import slim_resnet50
from utils.layer_wise_slim_resnet import layer_wise_slim_resnet50
from torch.utils.tensorboard import SummaryWriter
from utils.slim_densenet import slim_densenet201

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

resnet50_model_url = "https://download.pytorch.org/models/resnet50-0676ba61.pth"
densenet201_model_url = "https://download.pytorch.org/models/densenet201-c1103571.pth"


def plot(dict_loss, dict_acc, args):
    epochs = len(dict_loss['train'])
    plt.figure()
    plt.subplot(211)
    plt.plot(range(epochs), dict_loss['train'], label='train_loss')
    plt.plot(range(epochs), dict_loss['val'], label='val_loss')
    plt.legend()
    plt.subplot(212)
    plt.plot(range(epochs), dict_acc['train'], label='train_acc')
    plt.plot(range(epochs), dict_acc['val'], label='val_acc')
    plt.legend()
    plt.title(args.saved_path.split("/")[-1])
    plt.savefig(os.path.join(args.saved_path, "train_stat.png"))
    
    

def evaluate_single(model, valloader, criterion, args):
    model.eval()
    PRED = []
    LABELS = []
    LOSS = 0
    m = nn.Softmax(dim=1)
    total_run_time = 0
    
    counter = 0
    model.to(args.device)
    for data, label in tqdm(valloader, desc="Validation", leave=False):
        input = data.to(args.device)
        target = label.to(args.device).long()
        
        output = m(model(input))
        loss = criterion(output, target)

        LABELS.extend(label.detach().cpu().numpy())
        PRED.extend(output.detach().cpu().numpy())
        LOSS += loss.detach().cpu().numpy() * data.shape[0]
    
    LABELS = np.asarray(LABELS)
    PRED = np.asarray(PRED)
    
    TP = [0] * args.classes
    FP = [0] * args.classes
    
    PRED_LABEL = np.argmax(PRED, axis=1)
    acc = np.mean(PRED_LABEL==LABELS)
    # LOSS = LOSS / valloader.dataset.__len__()
    LOSS = LOSS / len(valloader.dataset)

    if args.test:
        rocs = []
        prcs = []
        for i in range(LABELS.shape[0]):
            if LABELS[i] == PRED_LABEL[i]:
                TP[LABELS[i]] += 1
            else:
                FP[LABELS[i]] += 1

        for i in range(PRED.shape[1]):
            tmp_labels = (LABELS==i).astype(int)
            roc = roc_auc_score(tmp_labels, PRED[:, i])
            p, r, t = precision_recall_curve(tmp_labels, PRED[:, i])
            prc = auc(r, p)
            rocs.append(roc)
            prcs.append(prc)

        
            
        Precision = [TP[i]/(TP[i] + FP[i]) for i in range(args.classes)]
        for p, r in zip(prcs, rocs):
            print("{} {}".format(p, r))
    
    return acc, LOSS


def evaluate_multi(model, valloader, criterion, args):
    model.eval()
    LABELS = []
    PRED = []
    LOSS = 0
    for data, label in valloader:
        input = data.to(args.device)
        target = label.to(args.device)
        output = model(input)
        loss = criterion(output, target)
        LOSS += loss.detach().cpu().numpy() * data.shape[0]

    PRED.extend(output.detach().cpu().numpy())
    LABELS.extend(label.detach().cpu().numpy())

    PRED = np.asarray(PRED)
    LABELS = np.asarray(LABELS)

    acc = []
    for i in range(LABELS.shape[1]):
        tmp_pred = (PRED[:, i] > 0).astype(int)
        tmp_labels = LABELS[:, i]
        acc.append(np.sum(tmp_pred == tmp_labels)/LABELS.shape[0])
    acc = np.mean(acc)

    prc = []
    for i in range(LABELS.shape[1]):
        p, r, t = precision_recall_curve(LABELS[:, i], PRED[:, i])
        prc.append(auc(r, p))
    print(prc)
    print("accuracy: ", acc)
    prc = np.mean(prc)
    LOSS = LOSS / valloader.dataset.__len__()
    return prc, LOSS



def evaluate(model, valloader, criterion, args):
    if args.dataset == "CheXpert" and (args.target == "all" or args.target == 'low' or args.target == 'high'):
        return evaluate_multi(model, valloader, criterion, args)
    elif args.dataset == "XBNJ" or args.dataset == "BIMCV" or (args.dataset == "CheXpert" and args.target != "all"):
        return evaluate_single(model, valloader, criterion, args)
    else:
        exit("evaluation not supported yet")

        
def getWeights(labels, args):
    new_labels = labels.detach().cpu().numpy()
    count = [np.sum(new_labels == i) for i in range(args.classes)]
    count = torch.Tensor(1/count)
    print(count)
    print(labels)
    return count

def train(model, trainloader, valloader, args):
    # training configs
    # 将模型移动到主 GPU 上，这里选择第一个 GPU（索引为 3）作为主 GPU
    model = model.to(args.device)
    # # 将模型放到多个 GPU 上进行并行计算
    # model = nn.DataParallel(model, device_ids=args.device)


    if args.data_parallel:
        model = nn.DataParallel(model, device_ids=[2,3,4,5])

    if "freeze" in args.model:
        modules=list(model.children())[:-1]
        base=nn.Sequential(*modules)
        fc = list(model.children())[-1]

        optimizer = Adam(
        [
            {"params": fc.parameters(), "lr": args.lr},
            {"params": base.parameters()},
        ],
        lr = args.lr/args.ptl_decay)


    else:
        modules=list(model.children())[:-1]
        base=nn.Sequential(*modules)
        fc = list(model.children())[-1]

        optimizer = Adam(model.parameters(), lr = args.lr)


    if args.resume:  # 如果要从之前的模型继续训练
        checkpoint = torch.load(args.resume_path)
        # if isinstance(model, nn.DataParallel):
        #     model = model.module  # 获取 DataParallel 对象中的原始模型对象
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, mode='min')
    criterion = args.criterion
    max_acc = 0
    min_loss = 1e8
    
    hist_loss = {"train":[], "val":[]}
    hist_acc = {"train":[], "val":[]}

    iter_count = 0
    # # 创建一个Tensorboard的SummaryWriter对象
    # log_dir = './tensorboard/test1'
    # writer = SummaryWriter(log_dir=log_dir)
    # Start Training
    log_file_path = os.path.join(args.saved_path, "train_log_diff2.txt")
    with open(log_file_path, 'a') as log_file:
        log_file.write("Epoch\tTrain Loss\tTrain Acc\tVal Loss\tVal Acc\n")

    for epoch in tqdm(range(start_epoch, args.max_epoch), desc="Epochs", leave=True):
        model.train()
        for data, label in tqdm(trainloader, desc="Training", leave=False):
            iter_count += 1
            input = data.to(args.device)
            target = label.to(args.device)

            # 检查输入和目标形状是否匹配
            # input_shape = input.size()
            # target_shape = target.size()
            # print("Input Shape:", input_shape, "Target Shape:", target_shape)

            # for BCE loss only
            if args.target != 'all' and args.target != 'low' and args.target != 'high':
                target = target.long()


            optimizer.zero_grad()
            output = model(input)

            # 打印输出张量形状
            # output_shape = output.size()
            # print("Output Shape:", output_shape)


            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()


            if args.eval_iter > 0 and (iter_count % args.eval_iter) == 0:
                train_acc, train_loss = evaluate(model, trainloader, criterion, args=args)
                val_acc, val_loss = evaluate(model, valloader, criterion, args=args)
                log_config = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc, "lr": optimizer.param_groups[0]['lr']}
                print("epoch{epoch}: train: loss:{train_loss} \t acc:{train_acc} | test: loss:{val_loss} \t acc:{val_acc} \t lr:{lr}".format(**log_config))
                scheduler.step(val_loss)

                with open(log_file_path, 'a') as log_file:
                    log_file.write(f"{epoch}\t{train_loss:.4f}\t{train_acc:.4f}\t{val_loss:.4f}\t{val_acc:.4f}\n")
                
                if min_loss > val_loss:
                    min_loss = val_loss
                    save_file_name = os.path.join(args.saved_path, "best.pt")
                    # torch.save(model, save_file_name)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                    }, save_file_name)

                if optimizer.param_groups[0]['lr'] < 1e-8:
                    break


        if args.eval_iter <= 0:
            train_acc, train_loss = evaluate(model, trainloader, criterion, args=args)
            val_acc, val_loss = evaluate(model, valloader, criterion, args=args)
            log_config = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc, "lr": optimizer.param_groups[0]['lr']}
            print("epoch{epoch}: train: loss:{train_loss} \t acc:{train_acc} | test: loss:{val_loss} \t acc:{val_acc} \t lr:{lr}".format(**log_config))
            scheduler.step(val_loss)

            with open(log_file_path, 'a') as log_file:
                log_file.write(f"{epoch}\t{train_loss:.4f}\t{train_acc:.4f}\t{val_loss:.4f}\t{val_acc:.4f}\n")
            
            if min_loss > val_loss:
                min_loss = val_loss
                save_file_name = os.path.join(args.saved_path, "best.pt")
                # torch.save(model, save_file_name)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, save_file_name)

            if optimizer.param_groups[0]['lr'] < 1e-8:
                break

        # # 记录训练和验证的损失和准确率
        # writer.add_scalar('Train Loss', train_loss, epoch)
        # writer.add_scalar('Train Accuracy', train_acc, epoch)
        # writer.add_scalar('Validation Loss', val_loss, epoch)
        # writer.add_scalar('Validation Accuracy', val_acc, epoch)
        # save for plot
        hist_loss['train'].append(train_loss)
        hist_loss['val'].append(val_loss)
        hist_acc['train'].append(train_acc)
        hist_acc['val'].append(val_acc)
    #
    # # 将列表数据转换为 numpy array 或 torch tensor
    # train_loss_np = np.array(hist_loss['train'])
    # val_loss_np = np.array(hist_loss['val'])
    # train_acc_np = np.array(hist_acc['train'])
    # val_acc_np = np.array(hist_acc['val'])

    # # 转换为标量（0 维张量）
    # train_loss_tensor = torch.tensor(np.mean(train_loss_np), dtype=torch.float32)
    # val_loss_tensor = torch.tensor(np.mean(val_loss_np), dtype=torch.float32)
    # train_acc_tensor = torch.tensor(np.mean(train_acc_np), dtype=torch.float32)
    # val_acc_tensor = torch.tensor(np.mean(val_acc_np), dtype=torch.float32)
    #
    # hparam_dict = {"lr": optimizer.param_groups[0]['lr']}
    # metric_dict = {
    #     "train_loss_history": train_loss_tensor,
    #     "val_loss_history": val_loss_tensor,
    #     "train_acc_history": train_acc_tensor,
    #     "val_acc_history": val_acc_tensor
    # }
    #
    # writer.add_hparams(hparam_dict, metric_dict)
    #
    #
    # writer.close()

    
    # save the final model
    save_file_name = os.path.join(args.saved_path, "final.pt")
    # torch.save(model, save_file_name)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_file_name)
    plot(hist_loss, hist_acc, args)
        

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)




if __name__ == "__main__":
    set_random_seed(1)
    torch.cuda.empty_cache()
    args = parse_opts()
    mode = "noisy"
    if not args.test:

        if args.dataset == "HAM":
            if args.sub == 100:
                train_df = pd.read_csv(os.path.join(args.root_path, str(args.exp), "train.csv"))
                val_df = pd.read_csv(os.path.join(args.root_path, str(args.exp), "val.csv"))
            else:
                train_df = pd.read_csv(os.path.join(args.root_path, str(args.exp), "train_{}.csv".format(args.sub/100)))
                val_df = pd.read_csv(os.path.join(args.root_path, str(args.exp), "val_{}.csv".format(args.sub/100)))          
            
            train_ds = HAM(train_df, root_dir=args.root_path+"jpgs/", mode='train', args=args)
            train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pin_memory, num_workers=args.num_workers, drop_last=True)
            val_ds = HAM(val_df, root_dir=args.root_path+"jpgs/", mode='val', args=args)
            val_dl = DataLoader(val_ds, batch_size=args.bs, shuffle=False, pin_memory=args.pin_memory, num_workers=args.num_workers)     

        elif args.dataset == "BIMCV":
            
            if args.sub == 100:
                train_df = pd.read_csv(os.path.join(args.root_path, mode,  str(args.exp), "train.csv"))
                val_df = pd.read_csv(os.path.join(args.root_path, mode, str(args.exp), "val.csv"))
            else:
                print("train with sub set")
                train_df = pd.read_csv(os.path.join(args.root_path, mode,  str(args.exp), "train_{}.csv".format(args.sub/100)))
                val_df = pd.read_csv(os.path.join(args.root_path, mode, str(args.exp), "val_{}.csv".format(args.sub/100)))

            train_ds = BIMCV(train_df, root_dir=args.root_path+"crop/", mode='train', args=args)
            train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pin_memory, num_workers=args.num_workers)
            
            
            val_ds = BIMCV(val_df, root_dir=args.root_path+"crop/", mode='val', args=args)
            val_dl = DataLoader(val_ds, batch_size=args.bs, shuffle=False, pin_memory=args.pin_memory, num_workers=args.num_workers)   
        
        elif args.dataset == "IMGNET":
            imagenet_data = torchvision.datasets.ImageNet('path/to/imagenet_root/')

        elif args.dataset == "XBNJ":
            root_dir = r'/home/helinrui/slns/TTL/data'

            train_dataset = XBNJ(root_dir=root_dir, mode='train', args=args)
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True,
                                          pin_memory=args.pin_memory, num_workers=args.num_workers)

            val_dataset = XBNJ(root_dir=root_dir, mode='val', args=args)
            val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.bs, shuffle=False, pin_memory=args.pin_memory, num_workers=args.num_workers)

        
        if args.model == "densenet121":
            model = densenet121(pretrained=args.pretrained, trunc=args.trunc, classes=args.classes)
        elif args.model == "densenet201":
            model = densenet201(pretrained=args.pretrained, trunc=args.trunc, classes=args.classes)
        elif args.model == "resnet50":
            model = resnet50(pretrained=args.pretrained, trunc=args.trunc, classes=args.classes, args=args)
        elif args.model == "resnet50_FAT":
            model = resnet50(pretrained=args.pretrained, trunc=args.trunc, classes=args.classes, args=args, freeze=True)
        elif args.model == "clip_resnet50":
            model, preprocess = clip_resnet50(trunc=args.trunc, classes=args.classes)
            args.preprocess = preprocess
        elif args.model == "bit_resnet50":
            model, preprocess = bit_resnet50(trunc=args.trunc, classes=args.classes)
            args.preprocess = preprocess
        elif args.model == "slim_resnet50":
            model = slim_resnet50(shrink_coefficient=args.slim_factor, load_up_to=args.slim_from, num_classes=args.classes)
            state_dict = load_state_dict_from_url(resnet50_model_url, progress=True)
            model.load_up_to_block(state_dict)
        elif args.model == "layer_wise_slim_resnet50":
            model = layer_wise_slim_resnet50(shrink_coefficient=args.slim_factor, load_up_to=args.slim_from, num_classes=args.classes, pretrained=True)
        elif args.model == "slim_densenet201":
            model = slim_densenet201(shrink_coefficient=args.slim_factor, load_up_to=args.slim_from, num_classes=args.classes)
            state_dict = load_state_dict_from_url(densenet201_model_url, progress=True)
            model.load_up_to_block(state_dict)
        elif args.model == "freeze_resnet50":
            model = freeze_resnet50(finetune_from=args.finetune_from, classes=args.classes)
        elif args.model == "layer_wise_freeze_resnet50":
            model = net = layer_wise_freeze_resnet50(finetune_from=args.finetune_from, classes=args.classes)
        elif args.model == "freeze_densenet201":
            model = freeze_densenet201(finetune_from=args.finetune_from, classes=args.classes)
        elif args.model == "cbr_larget":
            model = cbr_larget(pretrained=args.pretrained, classes=args.classes)
        elif args.model == 'alexnet':
            model = alexnet(pretrained=args.pretrained, classes=args.classes)
        elif args.model == 'layerttl_resnet50':
            model = resnet50(pretrained=args.pretrained, trunc=args.trunc, layer_wise=True, classes=args.classes, args=args)
        elif args.model == "efficientnet":
            model = efficientnet(num_classes=args.classes, pretrained=args.pretrained, trunc=args.trunc)
        else:
            exit("model not found")
        
        print("training with ", args.model)
        train(model=model, trainloader=train_dataloader, valloader=val_dataloader, args=args)
    else:
        args.pin_memory = True
        if args.dataset == "HAM":
            test_df = pd.read_csv(os.path.join(args.root_path, str(args.exp), "test.csv"))
            test_ds = HAM(test_df, root_dir=args.root_path+"jpgs/", mode='val', args=args)
            test_dl = DataLoader(test_ds, batch_size=args.bs, shuffle=False, pin_memory=args.pin_memory, num_workers=args.num_workers)
        
        elif args.dataset == "BIMCV":
            test_df = pd.read_csv(os.path.join(args.root_path, mode, str(args.exp), "test.csv"))
            test_ds = BIMCV(test_df, root_dir=args.root_path+"crop/", mode='val', args=args)
            test_dl = DataLoader(test_ds, batch_size=args.bs, shuffle=False, pin_memory=args.pin_memory, num_workers=args.num_workers)   

        if args.model == "clip_resnet50":
            _, preprocess = clip_resnet50(trunc=args.trunc, classes=args.classes)
            args.preprocess = preprocess
        elif args.model == "bit_resnet50":
            _, preprocess = bit_resnet50(trunc=args.trunc, classes=args.classes)
            args.preprocess = preprocess
        else:
            pass

        print(os.path.join(args.saved_path, "best.pt"))
        model = torch.load(os.path.join(args.saved_path, "best.pt"))
        try:
            model = model.module.to(args.device)
        except:
            model = model.to(args.device)
        # model = model.module
        criterion = nn.CrossEntropyLoss(reduction='mean')


        acc, _ = evaluate(model, test_dl, criterion, args)
        
        print("top one accuracy:", acc)
