import os
import sys
from tqdm import tqdm
import shutil
import argparse
import logging
import numpy as np
from skimage import io
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from networks.transformer_backbone import MyNet_fusion
from dataloaders.hybrid_sparse import Hybrid as MyDataset
from dataloaders.hybrid_sparse import  ToTensor

from utils import norm, trunc

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/train/')
parser.add_argument('--test_path', type=str, default='../data/test/')
parser.add_argument('--phase', type=str, default='test', help='Name of phase')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--exp', type=str, default='msl_model', help='model_name')
parser.add_argument('--seed', type=int, default=1337, help='random seed')

args = parser.parse_args()
train_data_path = args.root_path
test_data_path = args.test_path
snapshot_path = "../model/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    network = MyNet_fusion().cuda()

    db_test = MyDataset(base_dir=test_data_path,
                        split='test',
                        transform=transforms.Compose([
                            ToTensor()]))
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    if args.phase == 'test':

        save_mode_path = os.path.join(snapshot_path, 'iter_100000.pth')
        print('load weights from ' + save_mode_path)
        checkpoint = torch.load(save_mode_path)
        network.load_state_dict(checkpoint['network'])
        network.eval()
        cnt = 0
        save_path = snapshot_path + '/result_case/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for sampled_batch in tqdm(testloader, ncols=70):
            print('processing ' + str(cnt) + ' image')
            ct_in, ct, mri_in, mri = sampled_batch['ct_in'].cuda(), sampled_batch['ct'].cuda(), \
                                     sampled_batch['mri_in'].cuda(), sampled_batch['mri'].cuda()

            ct_img = (ct.data.cpu().numpy()[0, 0] * 255).astype(np.uint8)
            io.imsave(save_path + str(cnt) + '_ct.png', norm(ct_img,0,0.8))

            ct_in_img = (ct_in.data.cpu().numpy()[0, 0] * 255).astype(np.uint8)
            io.imsave(save_path + str(cnt) + '_ct_in.png', norm(ct_in_img,0,0.8))

            mri_img = (mri.data.cpu().numpy()[0, 0] * 255).astype(np.uint8)
            io.imsave(save_path + str(cnt) + '_mri.png', norm(mri_img,0,0.8))

            mri_in_img = (mri_in.data.cpu().numpy()[0, 0] * 255).astype(np.uint8)
            io.imsave(save_path + str(cnt) + '_mri_in.png', norm(mri_in_img,0,0.8))

            ii = 0
            for idx, lam in enumerate([0,0.3,0.5,0.7,1.0]):
                domainness = [torch.tensor(lam).cuda().float().reshape((1, 1))]
                with torch.no_grad():

                    fusion_out = network(ct_in, mri_in, domainness)[0][0]
                fusion_out[fusion_out < 0.0] = 0.0

                fusion_img = (fusion_out.data.cpu().numpy()[0, 0] * 255).astype(np.uint8)

                io.imsave(save_path + str(cnt) + '_fusion_' + str(lam) + '.png', norm(fusion_img,0,0.8))
                ii = ii + 1

            if cnt > 10:
                break

    elif args.phase == "diff":
        save_mode_path = os.path.join(snapshot_path, 'iter_100000.pth')
        print('load weights from ' + save_mode_path)
        checkpoint = torch.load(save_mode_path)
        network.load_state_dict(checkpoint['network'])
        network.eval()
        cnt = 0
        save_path = snapshot_path + '/result_case_diff/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for sampled_batch in tqdm(testloader, ncols=70):
            print('processing ' + str(cnt) + ' image')
            ct_in, ct, mri_in, mri = sampled_batch['ct_in'].cuda(), sampled_batch['ct'].cuda(), \
                                     sampled_batch['mri_in'].cuda(), sampled_batch['mri'].cuda()

            for idx, lam in enumerate([0, 0.3, 0.5, 0.7, 1]):
                domainness = [torch.tensor(lam).cuda().float().reshape((1, 1))]
                with torch.no_grad():
                    fusion_out = network(ct_in, mri_in, domainness)[0][0]
                fusion_out[fusion_out < 0.0] = 0.0
                fusion_img = (fusion_out.data.cpu().numpy()[0, 0] * 255).astype(np.uint8)

                diff_ct = fusion_out.data.cpu().numpy()[0, 0] - ct.data.cpu().numpy()[0, 0]
                diff_ct = (trunc(diff_ct*255 +135)).astype(np.uint8)

                diff_mri = fusion_out.data.cpu().numpy()[0, 0] - mri.data.cpu().numpy()[0, 0]
                diff_mri = (trunc(diff_mri*255 +135)).astype(np.uint8)

                io.imsave(save_path + 'diff_' + str(cnt) + '_'+ str(lam) + '_ct.png', diff_ct)
                io.imsave(save_path + 'diff_' + str(cnt) + '_'+ str(lam) + '_mri.png', diff_mri)

            cnt = cnt + 1
            if cnt > 3:
                break
