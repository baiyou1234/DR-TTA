from unet3d import UNet3d, TAdaBN3D
from autoaugment import LearnableImageNetPolicy
import torch
import os
from metrics import *
from augmodel import Augmentmodel
import numpy as np
import argparse
import nibabel as nb
from combined_loss import CombinedLoss
from monai.inferers import sliding_window_inference
from tqdm import tqdm
from utils_brats_all import get_data_loader, parse_config, set_random
import monai.losses as losses
from src_loss import SRC_Loss
from monai.transforms import Compose, RandAffine, RandGaussianSmooth
import csv
dice_loss = losses.DiceLoss()

def test(config, upl_model, test_loader, exp_name):
    device = torch.device('cuda:{}'.format(config['train']['gpu']))
    for data_loader in [test_loader]:
        all_batch_dice = []
        all_batch_assd = []
        all_batch_hd95 = []
        hd95_list_wt = []
        hd95_list_co = []
        hd95_list_ec = [] 
        IoU_list_wt = []
        IoU_list_co = []
        IoU_list_ec = []
        RVE_list_wt = []
        RVE_list_co = []
        RVE_list_ec = []
        sen_list_wt = []
        sen_list_co = []
        sen_list_ec = []
        ppv_list_wt = []
        ppv_list_co = []
        ppv_list_ec = []
        output_result = []
        dice1_val = 0.0
        dice2_val = 0.0
        dice3_val = 0.0
        show = 0
        with torch.no_grad():
            upl_model.eval()
            skip_num = 0
            val_loss = 0
            dice1_val = 0
            dice2_val = 0
            dice3_val = 0
            RVE1_val = 0
            RVE2_val = 0
            RVE3_val = 0
            batch_count = 0
            num = 0
            for it, (image, label, xt_name, lab_Imag) in enumerate(test_loader):
                image = image.to(device, non_blocking=True)
                label = label.long().to(device, non_blocking=True).squeeze(1)
                blocks, latent_A = upl_model.enc(image)
                aux_seg_1 = upl_model.aux_dec1(latent_A, blocks).squeeze(1).to(device)
                dice1, dice2, dice3 = cal_dice(aux_seg_1, label) 
                hd95_ec, hd95_co, hd95_wt = cal_hd95(aux_seg_1, label)
                IoU_ec, IoU_co, IoU_wt = IoU(aux_seg_1, label)
                RVE_ec, RVE_co, RVE_wt = cal_RVE(aux_seg_1, label)
                sen_ec, sen_co, sen_wt = cal_sensitivity(aux_seg_1, label)
                ppv_ec, ppv_co, ppv_wt = cal_ppv(aux_seg_1, label)
                dice1_val += dice1
                dice2_val += dice2
                dice3_val += dice3
                RVE1_val += RVE_wt
                RVE2_val += RVE_co
                RVE3_val += RVE_ec
                batch_count += 1
                hd95mean = (hd95_wt + hd95_co + hd95_ec) / 3
                IoUmean = (IoU_wt + IoU_co + IoU_ec) / 3
                senmean = (sen_wt + sen_co + sen_ec) / 3
                ppvmean = (ppv_wt + ppv_co + ppv_ec) / 3
                dicemean = (dice1_val + dice2_val + dice3_val) / 3/batch_count
                RVEmean = (RVE1_val + RVE2_val + RVE3_val) / 3/batch_count
                hd95_list_wt.append(hd95_wt)
                hd95_list_co.append(hd95_co)
                hd95_list_ec.append(hd95_ec)
                IoU_list_wt.append(IoU_wt)
                IoU_list_co.append(IoU_co)
                IoU_list_ec.append(IoU_ec)
                sen_list_wt.append(sen_wt)
                sen_list_co.append(sen_co)
                sen_list_ec.append(sen_ec)
                ppv_list_wt.append(ppv_wt)
                ppv_list_co.append(ppv_co)
                ppv_list_ec.append(ppv_ec)
                print(f"dice:[{dicemean}] hd95:[{hd95mean}] IoU:[{IoUmean}] RVE:[{RVEmean}] sen:[{senmean}] ppv:[{ppvmean}]")

        avg_dice1_val = dice1_val / (len(test_loader)-skip_num)
        avg_dice2_val = dice2_val / (len(test_loader)-skip_num)
        avg_dice3_val = dice3_val / (len(test_loader)-skip_num)
        avg_hd95_wt = np.nanmean(hd95_list_wt)
        avg_hd95_co = np.nanmean(hd95_list_co)
        avg_hd95_ec = np.nanmean(hd95_list_ec)
        avg_IoU_wt = np.nanmean(IoU_list_wt)
        avg_IoU_co = np.nanmean(IoU_list_co)
        avg_IoU_ec = np.nanmean(IoU_list_ec)
        avg_RVE_wt = RVE1_val / (len(test_loader)-skip_num)
        avg_RVE_co = RVE2_val / (len(test_loader)-skip_num)
        avg_RVE_ec = RVE3_val / (len(test_loader)-skip_num)
        avg_sen_wt = np.nanmean(sen_list_wt)
        avg_sen_co = np.nanmean(sen_list_co)
        avg_sen_ec = np.nanmean(sen_list_ec)
        avg_ppv_wt = np.nanmean(ppv_list_wt)
        avg_ppv_co = np.nanmean(ppv_list_co)
        avg_ppv_ec = np.nanmean(ppv_list_ec)
        dicemean = (avg_dice1_val + avg_dice2_val + avg_dice3_val) / 3
        output_result.append(f"ET : {avg_dice1_val}")
        output_result.append(f"TC : {avg_dice2_val}")
        output_result.append(f"WT : {avg_dice3_val}")
        output_result.append(f"HD95_ET : {avg_hd95_ec}")
        output_result.append(f"HD95_TC : {avg_hd95_co}")
        output_result.append(f"HD95_WT : {avg_hd95_wt}")
        output_result.append(f"IoU_ET : {avg_IoU_ec}")
        output_result.append(f"IoU_TC : {avg_IoU_co}")
        output_result.append(f"IoU_WT : {avg_IoU_wt}")
        output_result.append(f"RVE_ET : {avg_RVE_ec}")
        output_result.append(f"RVE_TC : {avg_RVE_co}")
        output_result.append(f"RVE_WT : {avg_RVE_wt}")
        output_result.append(f"sen_ET : {avg_sen_ec}")
        output_result.append(f"sen_TC : {avg_sen_co}")
        output_result.append(f"sen_WT : {avg_sen_wt}")
        output_result.append(f"ppv_ET : {avg_ppv_ec}")
        output_result.append(f"ppv_TC : {avg_ppv_co}")
        output_result.append(f"ppv_WT : {avg_ppv_wt}")
        results_dir = f"results_debug/ffn"
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, 'result_tar.txt'), 'a') as file:
                for line in output_result:
                    file.write(line + "\n")
        return dicemean

def momentum_update_key_encoder(model, momentum_model):
    """
    Momentum update of the key encoder
    """
    # encoder_q -> encoder_k
    for param_q, param_k in zip(
        model.parameters(), momentum_model.parameters()
    ):
        param_k.data = param_k.data * 0.95 + param_q.data * 0.05
    return momentum_model

def train(config,train_loader,test_loader,source_model):
    show = 0
    print("train")
    # load exp_name
    exp_name = config['train']['exp_name']
    dataset = config['train']['dataset']
    device = torch.device('cuda:{}'.format(config['train']['gpu']))
    upl_model = UNet3d(config).to(device)
    momentum_model = UNet3d(config).to(device)
    print('source_model_created')
    checkpoint = torch.load(source_model)
    print('source_model_loaded')
    upl_model.load_state_dict(checkpoint)
    momentum_model.load_state_dict(checkpoint)
    aug_model = Augmentmodel(upl_model).to(device)
    aug_momentum_model = Augmentmodel(momentum_model).to(device)
    #test(config,upl_model,test_loader,exp_name=exp_name)
    #raise ValueError
    for module in aug_model.modules():
        if isinstance(module, TAdaBN3D) or isinstance(module, LearnableImageNetPolicy):
            module.train()  # TAdaBN3D 保持训练模式
            for param in module.parameters():
                param.requires_grad = True  # 允许参数更新
        else:
            module.eval()   # 非 TAdaBN3D 设为评估模式
            for param in module.parameters():
                param.requires_grad = False  # 冻结参数
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, aug_model.parameters()),lr=config['train']['lr'])
    num_epochs = 10
    best_dice = 0.
    aug_num = 1
    output_dir = "validation_results_ssa"
    os.makedirs(output_dir,exist_ok=True)
    train_flag = True
    for epoch in range(num_epochs):
        aug_model = aug_model.to(device)
        print(epoch)
        if train_flag:
            pesudo_labels = []
            with torch.no_grad():  # 确保在评估模式下不计算梯度
                for i, (B, B_label, _, _) in enumerate(train_loader):
                    B = B.to(device)
                    pesudo_label = aug_momentum_model(B, aug = 0)
                    pesudo_labels.append(pesudo_label.detach())

            aug_model.train()
            for module in aug_model.modules():
                if isinstance(module, TAdaBN3D) or isinstance(module, LearnableImageNetPolicy):
                    module.eval()

            for i, (B, B_label, _, _) in enumerate(train_loader):

                B = B.to(device)
                if test_ref == 0:
                    ref = B
                    test_ref = 1
                optimizer.zero_grad()  # 清空梯度
                for j in range(aug_num):
                    show += 1
                    out, weight, policy = aug_model(B, aug = 1)
                    weight = weight.to(device)
                    total_loss = torch.tensor(0.0).to(device)
                    for k in range(out.size(0)):  # 使用不同的变量名来避免冲突
                        input_i = out[k].unsqueeze(0)
                        loss_pesudo_label = SRC_Loss(pesudo_labels[i])
                       
                        if loss_pesudo_label >= 20:
                            print('pass')
                        else:
                            loss_dice = dice_loss(input_i, pesudo_labels[i]) 
                            loss_pred = SRC_Loss(input_i)
                            loss_pred = loss_pred.to(device) * 100
                            weighted_loss_i = (loss_dice + loss_pred) * weight[k]
                            print(weighted_loss_i)
                            total_loss += weighted_loss_i
                    if total_loss != 0:
                        total_loss.backward()  # 反向传播
                optimizer.step()  # 更新参数
                   
        # # valid for target domain
        if (epoch+1) % 1 == 0:
            current_dice = test(config,upl_model,test_loader,exp_name=exp_name)
            if (current_dice) > best_dice:
                best_dice = current_dice
                model_dir = "/data/birth/cyf/output/wyh_output/tta/save_model_final_lr_ffn_best_2/" + str(exp_name )
                os.makedirs(model_dir, exist_ok=True)
                best_epoch = '{}/model-{}-{}-{}.pth'.format(model_dir, 'best', str(epoch), np.round(best_dice,3))
                torch.save(upl_model.state_dict(), best_epoch)
            model_dir = "/data/birth/cyf/output/wyh_output/tta/save_model_final_lr_ssa_ffn_2/" + str(exp_name )
            os.makedirs(model_dir, exist_ok=True)
            best_epoch = '{}/model-{}.pth'.format(model_dir, str(epoch))
            torch.save(upl_model.state_dict(), best_epoch)   
        momentum_model = momentum_update_key_encoder(upl_model,momentum_model)
    if train_flag and (epoch+1) % 10 == 0:
        torch.save(upl_model.state_dict(), '{}/model-{}.pth'.format(model_dir, 'latest'))

    upl_model.load_state_dict(torch.load(best_epoch,map_location='cpu'),strict=True)
    upl_model.eval()
    print("test")
    test(config,upl_model,test_loader,exp_name=exp_name)
    
    

def mian():
    # load config
    parser = argparse.ArgumentParser(description='config file')
    parser.add_argument('--config', type=str, default="./config/train_target.cfg",
                        help='Path to the configuration file')
    args = parser.parse_args()
    config = args.config
    config = parse_config(config)

    source_model = '/data/birth/cyf/output/wyh_output/tta/okkkk-7-me-model/model-nobatch-110.pth'
    batch_train = 1
    batch_test = 1
    num_workers = 0
    source_root = '/data/birth/cyf/shared_data/BraTS2024'
    target_root = '/data/birth/cyf/shared_data/BraTS-SSA'
    train_path = 'train'
    test_path = 'test'
    mode = 'target_to_target'
    img = 'all'
    train_loader,test_loader = get_data_loader(source_root,target_root,
                                               train_path,test_path,
                                               batch_train,batch_test,
                                               nw = num_workers,
                                               img=img,mode=mode)
    print("数据加载完成")
    

    train(config,train_loader,test_loader,source_model)
        
if __name__ == '__main__':
    set_random()
    torch.manual_seed(0.95)
    torch.cuda.manual_seed(0.95) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True 
    
    mian()