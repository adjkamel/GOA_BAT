import os
import glob
import torch
import random
import logging
import numpy as np
from tqdm.auto  import tqdm
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from common_3dhp.opt import opts
from common_3dhp.utils import * 
from common_3dhp.camera import get_uvd2xyz
from common_3dhp.load_data_3dhp_mae import Fusion
from common_3dhp.h36m_dataset import Human36mDataset
#from model.block.refine import refine
from model_3dhp import GOA_TF
#from model.stmo_pretrain import Model_MAE

from thop import clever_format
from thop.profile import profile
import scipy.io as scio

opt = opts().parse()

 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:"+opt.gpu)





def train(opt, actions, train_loader, model, optimizer, epoch):
    return step('train', opt, actions, train_loader, model, optimizer, epoch,trts='train')

def val(opt, actions, val_loader, model):
    with torch.no_grad():
        return step('test', opt, actions, val_loader, model,trts='test')

def step(split, opt, actions, dataLoader, model, optimizer=None, epoch=None, trts='train'):
    model_trans = model['trans']
    #model_refine = model['refine']
    #model_MAE = model['MAE']

    if split == 'train':
        model_trans.train()
       # model_refine.train()
        #model_MAE.train()
    else:
        model_trans.eval()
        #model_refine.eval()
        #model_MAE.eval()

    loss_all = {'loss': AccumLoss()}
    error_sum = AccumLoss()
    error_sum_test = AccumLoss()

    action_error_sum = define_error_list(actions)
    action_error_sum_post_out = define_error_list(actions)
    action_error_sum_MAE = define_error_list(actions)

    joints_left = [5, 6, 7, 11, 12, 13]
    joints_right = [2, 3, 4, 8, 9, 10]

    data_inference = {}

    for i, data in enumerate(tqdm(dataLoader, desc=trts ,ncols=60)):


        if split == "train":
            batch_cam, gt_3D, input_2D, seq, subject, scale, bb_box, cam_ind = data
        else:
            batch_cam, gt_3D, input_2D, seq, scale, bb_box = data

        [input_2D, gt_3D, batch_cam, scale, bb_box] = get_varialbe(split,
                                                                    [input_2D, gt_3D, batch_cam, scale, bb_box])

        N = input_2D.size(0)

        out_target = gt_3D.clone().view(N, -1, opt.out_joints, opt.out_channels)
        out_target[:, :, 14] = 0
        gt_3D = gt_3D.view(N, -1, opt.out_joints, opt.out_channels).type(torch.cuda.FloatTensor)

        if out_target.size(1) > 1:
            out_target_single = out_target[:, opt.pad].unsqueeze(1)
            gt_3D_single = gt_3D[:, opt.pad].unsqueeze(1)
        else:
            out_target_single = out_target
            gt_3D_single = gt_3D

        if opt.test_augmentation and split =='test':
            input_2D, output_3D, output_3D_VTE = input_augmentation(input_2D, model_trans, joints_left, joints_right)
        else:
            input_2D = input_2D.view(N, -1, opt.n_joints, opt.in_channels, 1).permute(0, 3, 1, 2, 4).type(torch.cuda.FloatTensor)
            output_3D, output_3D_VTE = model_trans(input_2D.to(device))

        # output_3D_VTE = output_3D_VTE.permute(0, 2, 3, 4, 1).contiguous().view(N, -1, opt.out_joints, opt.out_channels)
        output_3D = output_3D.permute(0, 2, 3, 4, 1).contiguous().view(N, -1, opt.out_joints, opt.out_channels)
        output_3D=output_3D.to(device)

        # output_3D_VTE = output_3D_VTE * scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, output_3D_VTE.size(1),opt.out_joints, opt.out_channels)
        output_3D = output_3D * scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, output_3D.size(1),opt.out_joints, opt.out_channels).to(device)
        output_3D_single = output_3D

        if split == 'train':
            # pred_out = output_3D_VTE
            pred_out = output_3D_single

        if split == 'test':
            pred_out = output_3D_single

        input_2D = input_2D.permute(0, 2, 3, 1, 4).view(N, -1, opt.n_joints ,2)

        if opt.refine:
            pred_uv = input_2D
            uvd = torch.cat((pred_uv[:, opt.pad, :, :].unsqueeze(1), output_3D_single[:, :, :, 2].unsqueeze(-1)), -1)
            xyz = get_uvd2xyz(uvd, gt_3D_single, batch_cam)
            xyz[:, :, 0, :] = 0
            post_out = model_refine(output_3D_single, xyz)
            loss = mpjpe_cal(post_out, out_target_single)
        else:
            if split == 'train':
                loss = mpjpe_cal(output_3D_single.to(device), out_target_single.to(device))
            elif split == 'test':
                loss = mpjpe_cal(pred_out.to(device), out_target.to(device)) + mpjpe_cal(output_3D_single.to(device), out_target_single.to(device))

        loss_all['loss'].update(loss.detach().cpu().numpy() * N, N)

        if split == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not opt.MAE:

                if opt.refine:
                    post_out[:,:,14,:] = 0
                    joint_error = mpjpe_cal(post_out, out_target_single).item()
                else:
                    pred_out[:,:,14,:] = 0
                    joint_error = mpjpe_cal(pred_out.to(device), out_target_single.to(device)).item()
                    # joint_error = mpjpe_cal(pred_out, out_target).item()

                error_sum.update(joint_error*N, N)

        elif split == 'test':
            
            pred_out[:, :, 14, :] = 0
            #action_error_sum = test_calculation(pred_out, out_target, action, action_error_sum, opt.dataset, subject)
            joint_error_test = mpjpe_cal(pred_out.to(device), out_target.to(device)).item()
            out = pred_out
            # if opt.refine:
            #     post_out[:, :, 14, :] = 0
            #     action_error_sum_post_out = test_calculation(post_out, out_target, action, action_error_sum_post_out, opt.dataset, subject)

            if opt.train == 0:
                for seq_cnt in range(len(seq)):
                    seq_name = seq[seq_cnt]
                    if seq_name in data_inference:
                        data_inference[seq_name] = np.concatenate(
                            (data_inference[seq_name], out[seq_cnt].permute(2, 1, 0).cpu().numpy()), axis=2)
                    else:
                        data_inference[seq_name] = out[seq_cnt].permute(2, 1, 0).cpu().numpy()

            error_sum_test.update(joint_error_test * N, N)

    if split == 'train':
      
        return loss_all['loss'].avg, error_sum.avg
    elif split == 'test':
       

        if opt.train == 0:
            for seq_name in data_inference.keys():
                data_inference[seq_name] = data_inference[seq_name][:, :, None, :]
            mat_path = os.path.join(opt.checkpoint, 'inference_data.mat')
            scio.savemat(mat_path, data_inference)

        return error_sum_test.avg

def input_augmentation_MAE(input_2D, model_trans, joints_left, joints_right, mask, spatial_mask=None):
    N, _, T, J, C = input_2D.shape

    input_2D_flip = input_2D[:, 1].view(N, T, J, C, 1).permute(0, 3, 1, 2, 4)
    input_2D_non_flip = input_2D[:, 0].view(N, T, J, C, 1).permute(0, 3, 1, 2, 4)

    output_2D_flip = model_trans(input_2D_flip, mask, spatial_mask)

    output_2D_flip[:, 0] *= -1

    output_2D_flip[:, :, :, joints_left + joints_right] = output_2D_flip[:, :, :, joints_right + joints_left]

    output_2D_non_flip = model_trans(input_2D_non_flip, mask, spatial_mask)

    output_2D = (output_2D_non_flip + output_2D_flip) / 2

    input_2D = input_2D_non_flip

    return input_2D, output_2D

def input_augmentation(input_2D, model_trans, joints_left, joints_right):
    N, _, T, J, C = input_2D.shape 

    input_2D_flip = input_2D[:, 1].view(N, T, J, C, 1).permute(0, 3, 1, 2, 4)   
    input_2D_non_flip = input_2D[:, 0].view(N, T, J, C, 1).permute(0, 3, 1, 2, 4) 

    output_3D_flip, output_3D_flip_VTE = model_trans(input_2D_flip.to(device))

    # output_3D_flip_VTE[:, 0] *= -1
    output_3D_flip[:, 0] *= -1

    # output_3D_flip_VTE[:, :, :, joints_left + joints_right] = output_3D_flip_VTE[:, :, :, joints_right + joints_left]
    output_3D_flip[:, :, :, joints_left + joints_right] = output_3D_flip[:, :, :, joints_right + joints_left]

    output_3D_non_flip, output_3D_non_flip_VTE = model_trans(input_2D_non_flip.to(device))

    # output_3D_VTE = (output_3D_non_flip_VTE + output_3D_flip_VTE) / 2
    output_3D = (output_3D_non_flip + output_3D_flip) / 2

    input_2D = input_2D_non_flip

    return input_2D, output_3D, None
    # return input_2D, output_3D, output_3D_VTE

if __name__ == '__main__':
    opt.manualSeed = 1

    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if opt.train == 1:
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                            filename=os.path.join(opt.checkpoint, 'train.log'), level=logging.INFO)
            
    root_path = opt.root_path
    dataset_path = root_path + 'data_3d_' + opt.dataset + '.npz'

    #dataset = Human36mDataset(dataset_path, opt)
    actions = define_actions(opt.actions)

    if opt.train:
        #train_data = Fusion(opt=opt, train=True, root_path=root_path)
        train_data = Fusion(opt=opt, train=True, root_path=root_path, MAE=opt.MAE)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batchSize,
                                                       shuffle=True, num_workers=int(opt.workers), pin_memory=True)
    if opt.test:
        #test_data = Fusion(opt=opt, train=False,root_path =root_path)
        test_data = Fusion(opt=opt, train=False, root_path=root_path, MAE=opt.MAE)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize,
                                                      shuffle=False, num_workers=int(opt.workers), pin_memory=True)
                                                      

    opt.out_joints = 17

    model = {}
    model['trans'] = GOA_TF(opt).to(device)


    model_params = 0
    for parameter in model['trans'].parameters():
        model_params += parameter.numel()
    print('INFO: Trainable parameter count:', model_params)

    model_dict = model['trans'].state_dict()
    if opt.reload == 1:

        no_refine_path = opt.previous_dir

        pre_dict = torch.load(no_refine_path)
        for name, key in model_dict.items():
            model_dict[name] = pre_dict[name]
        model['trans'].load_state_dict(model_dict, strict=True)

    all_param = []
    lr = opt.lr
    for i_model in model:
        all_param += list(model[i_model].parameters())
    optimizer_all = optim.Adam(all_param, lr=opt.lr, amsgrad=True)

    for epoch in range(1, opt.nepoch):
        if opt.train == 1:
            if not opt.MAE:
                loss, mpjpe = train(opt, actions, train_dataloader, model, optimizer_all, epoch)
            else:
                loss = train(opt, actions, train_dataloader, model, optimizer_all, epoch)
        if opt.test == 1:
           
            p1 = val(opt, actions, test_dataloader, model)
            data_threshold = p1

            if opt.train and data_threshold < opt.previous_best_threshold:
                
                opt.previous_name = save_model(opt.previous_name, opt.checkpoint, epoch, data_threshold, model['trans'], 'no_refine')

                if opt.refine:
                    opt.previous_refine_name = save_model(opt.previous_refine_name, opt.checkpoint, epoch,
                                                            data_threshold, model['refine'], 'refine')
                opt.previous_best_threshold = data_threshold

            if opt.train == 0:
                print('p1: %.2f' % (p1))
                break
            else:
                
                logging.info('epoch: %d, lr: %.7f, loss: %.4f, MPJPE: %.2f, p1: %.2f' % (epoch, lr, loss, mpjpe, p1))
                print('e: %d, lr: %.7f, loss: %.4f, M: %.2f, p1: %.2f' % (epoch, lr, loss, mpjpe, p1))

        if epoch % opt.large_decay_epoch == 0: 
            for param_group in optimizer_all.param_groups:
                param_group['lr'] *= opt.lr_decay_large
                lr *= opt.lr_decay_large
        else:
            for param_group in optimizer_all.param_groups:
                param_group['lr'] *= opt.lr_decay
                lr *= opt.lr_decay








