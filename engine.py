import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms
import torch.nn.functional as F
device = torch.device('cuda', 1)
def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    step,
                    logger, 
                    config,
                    writer):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train()
    transform = transforms.Compose([
        transforms.Resize(224),
    ])
    loss_list = []
    ce_loss = CrossEntropyLoss()
    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()
        images = data['image'].to(device, dtype=torch.float)
        targets = data['mask'].to(device, dtype=torch.float32)
        #out1,out2,out3 = model(images)
        output,output2,output3= model(images)
        loss2 = structure_loss(output, targets)
        loss1 = structure_loss(output2, targets)
        loss3 = structure_loss(output3, targets)
        #loss0 = criterion(out, targets)
        loss = loss2+loss1+loss3
        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step() 
    return step


def val_one_epoch(test_loader,
                    model,
                    criterion, 
                    epoch, 
                    logger,
                    config):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    transform = transforms.Compose([
        transforms.Resize(224),
    ])
    ce_loss = CrossEntropyLoss()
    loss_list = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            img = data['image'].to(device, dtype=torch.float)
            msk = data['mask'].to(device, dtype=torch.float32)
            #gt_pre, out = model(img)
            output,output2,output3 = model(img)
            loss1 = structure_loss(output, msk)
            loss2 = structure_loss(output2, msk)
            loss3 = structure_loss(output3, msk)
            loss = loss1+loss2
            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(output) is tuple:
                output = output[0]
            output = output.squeeze(1).cpu().detach().numpy()
            preds.append(output)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        log_info = f'val epoch: {epoch},loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)
    
    return np.mean(loss_list)
    #return f1_or_dsc


def test_one_epoch(test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    test_data_name=None):
    # switch to evaluate mode
    model.eval()
    ce_loss = CrossEntropyLoss()
    preds = []
    gts = []
    loss_list = []
    transform = transforms.Compose([
        transforms.Resize(224),
    ])
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img = data['image'].to(device, dtype=torch.float)
            msk = data['mask'].to(device, dtype=torch.float32)
            #gt_pre, out = model(img)
            output,output2,output3 = model(img)

            loss = structure_loss(output, msk)

            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(output) is tuple:
                output = output[0]
            output = output.squeeze(1).cpu().detach().numpy()
            preds.append(output)
            if i % config.save_interval == 0:
                save_imgs(img, msk, output, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)
