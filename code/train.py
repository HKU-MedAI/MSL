import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from networks.transformer_backbone import MyNet_fusion
from dataloaders.hybrid_sparse import Hybrid as MyDataset
from dataloaders.hybrid_sparse import RandomPadCrop, ToTensor

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/train/')
parser.add_argument('--test_path', type=str, default='../data/test/')
parser.add_argument('--phase', type=str, default='train', help='Name of phase')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--exp', type=str, default='msl_model', help='model_name')
parser.add_argument('--max_iterations', type=int, default=100000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.0002, help='maximum epoch numaber to train')
parser.add_argument('--seed', type=int, default=1337, help='random seed')

args = parser.parse_args()
train_data_path = args.root_path
test_data_path = args.test_path
snapshot_path = "../model/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr

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

    db_train = MyDataset(base_dir=train_data_path,
                         split='train',
                         transform=transforms.Compose([
                             RandomPadCrop(),
                             ToTensor()]))
    db_test = MyDataset(base_dir=test_data_path,
                        split='test',
                        transform=transforms.Compose([
                            ToTensor()]))

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    fixtrainloader = DataLoader(db_train, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    if args.phase == 'train':
        network.train()

        params = list(network.parameters())
        optimizer1 = optim.Adam(params, lr=base_lr, betas=(0.5, 0.999), weight_decay=1e-4)
        scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=10000, gamma=0.5)

        writer = SummaryWriter(snapshot_path + '/log')

        iter_num = 0
        max_epoch = max_iterations // len(trainloader) + 1

        for epoch_num in tqdm(range(max_epoch), ncols=70):
            time1 = time.time()
            for i_batch, sampled_batch in enumerate(trainloader):
                time2 = time.time()
                ct_in, ct, mri_in, mri = sampled_batch['ct_in'].cuda(), sampled_batch['ct'].cuda(), \
                                         sampled_batch['mri_in'].cuda(), sampled_batch['mri'].cuda()

                outputs, outputs_com, (feat_ct_ct, feat_mri_ct), (feat_ct_mri, feat_mri_mri) = network(ct_in, mri_in, [torch.tensor(0.0).cuda().reshape(1, 1).repeat(ct_in.shape[0], 1),
                                                   torch.tensor(1.0).cuda().reshape(1, 1).repeat(ct_in.shape[0], 1)])

                ct_out, mri_out = outputs
                ct_loss = F.l1_loss(ct_out, ct)
                mri_loss = F.l1_loss(mri_out, mri)

                ct_out_com, mri_out_com = outputs_com
                ct_loss_com = F.l1_loss(ct_out_com, ct)
                mri_loss_com = F.l1_loss(mri_out_com, mri)

                feature_loss = F.l1_loss(feat_ct_ct, feat_mri_ct) + F.l1_loss(feat_ct_mri, feat_mri_mri)
                
                domainness = [torch.tensor(0.5).cuda().float().reshape((1, 1))]
                fusion_out  = network(ct_in, mri_in, domainness)[0][0]
                f_loss = 0.5*F.mse_loss(fusion_out, ct)+0.5*F.mse_loss(fusion_out, mri)

                loss = ct_loss + mri_loss + 1.0*f_loss + 0.5* feature_loss + ct_loss_com + mri_loss_com
                optimizer1.zero_grad()
                loss.backward()
                optimizer1.step()
                scheduler1.step()

                # summary
                iter_num = iter_num + 1
                writer.add_scalar('lr', scheduler1.get_lr(), iter_num)
                writer.add_scalar('loss/loss', loss, iter_num)
                writer.add_scalar('loss/ct_loss', ct_loss, iter_num)
                writer.add_scalar('loss/mri_loss', mri_loss, iter_num)
                writer.add_scalar('loss/f_loss', f_loss, iter_num)

                writer.add_scalar('loss/ct_loss_com', ct_loss_com, iter_num)
                writer.add_scalar('loss/mri_loss_com', mri_loss_com, iter_num)
                writer.add_scalar('loss/feature_loss', feature_loss, iter_num)

                logging.info('iteration %d : ct_loss : %f mri_loss: %f ' % (iter_num, ct_loss.item(), mri_loss.item()))

                if iter_num % 2000 == 0:
                    grid_image = make_grid(ct, nrow=4, normalize=False)
                    writer.add_image('img/ct', grid_image, iter_num)

                    grid_image = make_grid(ct_in, nrow=4, normalize=False)
                    writer.add_image('img/ct_in', grid_image, iter_num)

                    ct_out[ct_out<0.0]=0.0
                    grid_image = make_grid(ct_out, nrow=4, normalize=False)
                    writer.add_image('img/ct_out', grid_image, iter_num)

                    grid_image = make_grid(mri, nrow=4, normalize=False)
                    writer.add_image('img/mri', grid_image, iter_num)

                    grid_image = make_grid(mri_in, nrow=4, normalize=False)
                    writer.add_image('img/mri_in', grid_image, iter_num)

                    mri_out[mri_out < 0.0] = 0.0
                    grid_image = make_grid(mri_out, nrow=4, normalize=False)
                    writer.add_image('img/mri_out', grid_image, iter_num)

                    network.eval()
                    domainness = [torch.tensor(0.5).cuda().float().reshape((1, 1))]
                    with torch.no_grad():
                        fusion_out = network(ct_in, mri_in, domainness)[0][0]
                        fusion_out[fusion_out < 0.0] = 0.0
                    grid_image = make_grid(fusion_out, nrow=4, normalize=False)
                    writer.add_image('img/fusion_out', grid_image, iter_num)
                    network.train()
    
                if iter_num % 20000 == 0:
                    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                    torch.save({'network': network.state_dict()}, save_mode_path)
                    logging.info("save model to {}".format(save_mode_path))

                if iter_num > max_iterations:
                    break
                time1 = time.time()
            if iter_num > max_iterations:
                break
        save_mode_path = os.path.join(snapshot_path, 'iter_' + str(max_iterations) + '.pth')
        torch.save({'network': network.state_dict()},
                   save_mode_path)
        logging.info("save model to {}".format(save_mode_path))
        writer.close()
