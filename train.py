"""
Training code for Adversarial patch training
"""

#import patch_config
import sys
import os

import time
from datetime import datetime
import argparse
import numpy as np
import scipy
import scipy.interpolate
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt
from easydict import EasyDict

from generator import *
from load_data import *
from tps import *
# from transformers import DeformableDetrForObjectDetection

import torch
import torch.nn as nn
from torch import autograd
from torch.nn import parameter
from torch.autograd import Variable, Function
from torchvision import transforms
import torchvision
from tensorboardX import SummaryWriter
import pytorch3d as p3d
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.renderer import (
    cameras,
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights, 
    DirectionalLights, 
    AmbientLights,
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,
    BlendParams,
    TexturesUV
)

# add path for demo utils functions 
sys.path.append(os.path.abspath(''))


from arch.yolov3_models import YOLOv3Darknet
from yolo2.darknet import Darknet
from color_util import *
from train_util import *
import pytorch3d_modify as p3dmd
import mesh_utils as MU


class PatchTrainer(object):
    def __init__(self, args):
        self.args = args
        if args.device is not None:
            device = torch.device(args.device)
            torch.cuda.set_device(device)
        else:
            device = None
        self.device = device
        self.img_size = 416
        self.DATA_DIR = "./data"

        if args.arch == "rcnn":
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval().to(device)
        elif args.arch == "yolov3":
            self.model = YOLOv3Darknet().eval().to(device)
            self.model.load_darknet_weights('arch/weights/yolov3.weights')
        elif args.arch == "detr":
            self.model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True).eval().to(
                device)
        # elif args.arch == "deformable-detr":
            # self.model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr").eval().to(device)
        elif args.arch == "yolov2":
            self.model = Darknet('yolo2/cfg/yolov2.cfg').eval().to(device)
            self.model.load_weights('yolo2/yolov2.weights')
        elif args.arch == "mask_rcnn":
            self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).eval().to(device)
        else:
            raise NotImplementedError

        for p in self.model.parameters():
            p.requires_grad = False

        self.batch_size = args.batch_size

        self.patch_transformer = PatchTransformer().to(device)
        if args.arch == "rcnn":
            self.prob_extractor = MaxProbExtractor(0, 80).to(device)
        elif args.arch == "yolov3":
            self.prob_extractor = YOLOv3MaxProbExtractor(0, 80, self.model, self.img_size).to(device)
        elif args.arch == "detr":
            self.prob_extractor = DetrMaxProbExtractor(0, 80, self.img_size).to(device)
        elif args.arch == "deformable-detr":
            self.prob_extractor = DeformableDetrProbExtractor(0,80,self.img_size).to(device)
        self.tv_loss = TotalVariation()

        self.alpha = args.alpha
        self.azim = torch.zeros(self.batch_size)
        self.blend_params = None

        self.sampler_probs = torch.ones([36]).to(device)
        self.loss_history = torch.ones(36).to(device)
        self.num_history = torch.ones(36).to(device)

        if not args.gen_test:
            # self.train_loader = self.get_loader('/content/drive/MyDrive/shared_dataset/advcat/data/background', True)
            args.train_dir = os.path.join("data", args.train_dir)
            self.train_loader = self.get_loader(args.train_dir, True)
            num_imgs = len(self.train_loader.dataset)
            print(f'One train epoch has {num_imgs} images - {num_imgs % 4} dropped images')
            self.epoch_length = len(self.train_loader)
        elif args.gen_test:
            # self.test_loader = self.get_loader('/content/drive/MyDrive/shared_dataset/advcat/data/background_test', True)
            args.test_dir = os.path.join("data", args.test_dir)
            self.test_loader = self.get_loader(args.test_dir, False)
            num_imgs = len(self.test_loader.dataset)
            print(f'One test epoch has {num_imgs} images')

        color_transform = ColorTransform('color_transform_dim6.npz')
        self.color_transform = color_transform.to(device)

        self.fig_size_H = 340
        self.fig_size_W = 864

        self.fig_size_H_t = 484
        self.fig_size_W_t = 700

        resolution = 4
        h, w, h_t, w_t = int(self.fig_size_H / resolution), int(self.fig_size_W / resolution), int(self.fig_size_H_t / resolution), int(self.fig_size_W_t / resolution)
        self.h, self.w, self.h_t, self.w_t = h, w, h_t, w_t

        # Set paths
        obj_filename_man = os.path.join(self.DATA_DIR, "Archive/Man_join/man.obj")
        obj_filename_tshirt = os.path.join(self.DATA_DIR, "Archive/tshirt_join/tshirt.obj")
        obj_filename_trouser = os.path.join(self.DATA_DIR, "Archive/trouser_join/trouser.obj")

        # self.colors = torch.load("./data/army_colors_9.pth").float().to(device)
        self.colors = torch.load(os.path.join("./data", args.color_pth)).float().to(device)
        num_colors = len(self.colors)
        self.coordinates = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)), -1).to(device)
        self.coordinates_t = torch.stack(torch.meshgrid(torch.arange(h_t), torch.arange(w_t)), -1).to(device)
        self.tshirt_point = torch.rand([num_colors, args.num_points_tshirt, 3], requires_grad=True, device=device)
        self.trouser_point = torch.rand([num_colors, args.num_points_trouser, 3], requires_grad=True, device=device)
        self.mesh_man = load_objs_as_meshes([obj_filename_man], device=device) # Returns new Meshes object
        self.mesh_tshirt = load_objs_as_meshes([obj_filename_tshirt], device=device)
        self.mesh_trouser = load_objs_as_meshes([obj_filename_trouser], device=device)

        self.faces = self.mesh_tshirt.textures.faces_uvs_padded()
        self.verts_uv = self.mesh_tshirt.textures.verts_uvs_padded()
        self.faces_uvs_tshirt = self.mesh_tshirt.textures.faces_uvs_list()[0]

        self.faces_trouser = self.mesh_trouser.textures.faces_uvs_padded()
        self.verts_uv_trouser = self.mesh_trouser.textures.verts_uvs_padded()
        self.faces_uvs_trouser = self.mesh_trouser.textures.faces_uvs_list()[0]
        self.optimizer = torch.optim.Adam([self.tshirt_point, self.trouser_point], lr=args.lr)

        if args.seed_type in ['fixed', 'random']:
            self.seeds_tshirt = torch.zeros(size=[h, w, num_colors], device=device).uniform_()
            self.seeds_trouser = torch.zeros(size=[h_t, w_t, num_colors], device=device).uniform_()
            self.optimizer_seed = torch.optim.SGD([torch.zeros(1, device=device).requires_grad_()], lr=args.lr_seed)
        else:
            self.seeds_tshirt_train = torch.zeros(size=[h, w, num_colors], device=device).uniform_(args.clamp_shift,
                                                                                              1 - args.clamp_shift).requires_grad_()  # NOTE when not fixed we use uniform
            self.seeds_trouser_train = torch.zeros(size=[h_t, w_t, num_colors], device=device).uniform_(args.clamp_shift,
                                                                                                   1 - args.clamp_shift).requires_grad_()

            self.seeds_tshirt_fixed = torch.zeros(size=[h, w, num_colors], device=device).uniform_()
            self.seeds_trouser_fixed = torch.zeros(size=[h_t, w_t, num_colors], device=device).uniform_()

            if args.seed_opt == 'sgd':
                self.optimizer_seed = torch.optim.SGD([self.seeds_tshirt_train, self.seeds_trouser_train], lr=args.lr_seed)
            elif args.seed_opt == 'adam':
                self.optimizer_seed = torch.optim.Adam([self.seeds_tshirt_train, self.seeds_trouser_train], lr=args.lr_seed)
            else:
                raise ValueError

        k = 3
        k2 = k * k
        self.camouflage_kernel = nn.Conv2d(num_colors, num_colors, k, 1, int(k / 2)).to(device)
        self.camouflage_kernel.weight.data.fill_(0)
        self.camouflage_kernel.bias.data.fill_(0)
        for i in range(num_colors):
            self.camouflage_kernel.weight[i, i, :, :].data.fill_(1 / k2)

        self.expand_kernel = nn.ConvTranspose2d(3, 3, resolution, stride=resolution, padding=0).to(device)
        self.expand_kernel.weight.data.fill_(0)
        self.expand_kernel.bias.data.fill_(0)
        for i in range(3):
            self.expand_kernel.weight[i, i, :, :].data.fill_(1)

        selected_tshirt = torch.cat([torch.arange(27), torch.arange(28, 31), torch.arange(32, 43)])
        self.tshirt_locations_infos = EasyDict({
            'nparts': 3,
            'centers': [[7.5, 0], [-7.5, 0], [0, 0]],
            'Rs': [1.5, 1.5, 15.0],
            'ntfs': [6, 6, 8],
            'ntws': [6, 6, 8],
            'radius_fixed': [[1.0], [1.0], [0.5]],
            'radius_wrap': [[0.5], [0.5], [1.0]],
            'signs': [-1, -1, 1],
            'selected': selected_tshirt,
        })

        self.trouser_locations_infos = EasyDict({
            'nparts': 2,
            'centers': [[3.43, 0], [-3.43, 0]],
            'Rs': [3.3] * 2,
            'ntfs': [20] * 2,
            'ntws': [12] * 2,
            'radius_fixed': [[1.2]] * 2,
            'radius_wrap': [[0.4]] * 2,
            'signs': [1, 1],
            'selected': None,
        })

        self.initialize_tps2d()
        self.initialize_tps3d()

    def get_loader(self, img_dir, shuffle=True):
        loader = torch.utils.data.DataLoader(InriaDataset(img_dir, self.img_size, shuffle=shuffle),
                                             batch_size=self.batch_size,
                                             shuffle=shuffle,
                                             drop_last=True,
                                             num_workers=2) # originally worker=4
        return loader

    def init_tensorboard(self, name=None):
        time_str = time.strftime("%m_%d-%H_%M")
        print("Created TensorBoard")
        return SummaryWriter(f'tensorboards/tb-{time_str}')

    def sample_cameras(self, theta=None, elev=None):
        if theta is not None:
            if isinstance(theta, float) or isinstance(theta, int):
                self.azim = torch.zeros(self.batch_size).fill_(theta)
            elif isinstance(theta, torch.Tensor):
                self.azim = theta.clone()
            elif isinstance(theta, np.ndarray):
                self.azip = torch.from_numpy(theta)
            else:
                raise ValueError
        else:
            if self.alpha > 0:
                exp = (self.alpha * self.sampler_probs).softmax(0)
                azim = torch.multinomial(exp, self.batch_size, replacement=True)
                self.azim_inds = azim
                azim = azim.to(exp)
                self.azim = (azim + azim.new(size=azim.shape).uniform_() - 0.5) * 360 / len(exp)
            else:
                self.azim_inds = None 
                self.azim = (torch.zeros(self.batch_size).uniform_() - 0.5) * 360
        if elev is not None:
            elev = torch.zeros(self.batch_size).fill_(elev)
        else:
            elev = 10 + 8 * torch.zeros(self.batch_size).uniform_(-1, 1)
        R, T = look_at_view_transform(dist=2.5, elev=elev, azim=self.azim)
        self.cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=45)
        return

    def sample_lights(self, r=None):
        if r is None:
            r = np.random.rand()
        theta = np.random.rand() * 2 * math.pi
        if r < 0.33:
            self.lights = AmbientLights(device=self.device)
        elif r < 0.67:
            self.lights = DirectionalLights(device=self.device, direction=[[np.sin(theta), 0.0, np.cos(theta)]])
        else:
            self.lights = PointLights(device=self.device, location=[[np.sin(theta) * 3, 0.0, np.cos(theta) * 3]])
        return

    def initialize_tps2d(self):
        locations_tshirt_ori = torch.load(os.path.join(self.DATA_DIR, 'Archive/tshirt_join/projections/part_all_2p5.pt'), map_location='cpu').to(self.device)
        self.infos_tshirt = MU.get_map_kernel(locations_tshirt_ori, self.faces_uvs_tshirt)

        locations_trouser_ori = torch.load(os.path.join(self.DATA_DIR, 'Archive/trouser_join/projections/part_all_off3p4.pt'), map_location='cpu').to(self.device)
        self.infos_trouser = MU.get_map_kernel(locations_trouser_ori, self.faces_uvs_trouser)

        target_control_points = p3dmd.get_points(self.tshirt_locations_infos, wrap=False).squeeze(0).cpu()
        tps2d_tshirt = TPSGridGen(None, target_control_points, locations_tshirt_ori.cpu())
        tps2d_tshirt.to(self.device)
        self.tps2d_tshirt = tps2d_tshirt

        target_control_points = p3dmd.get_points(self.trouser_locations_infos, wrap=False).squeeze(0).cpu()
        tps2d_trouser = TPSGridGen(None, target_control_points, locations_trouser_ori.cpu())
        tps2d_trouser.to(self.device)
        self.tps2d_trouser = tps2d_trouser
        return

    def initialize_tps3d(self):
        xmin, ymin, zmin = (-0.28170400857925415, -0.7323740124702454, -0.15313300490379333)
        xmax, ymax, zmax = (0.28170400857925415, 0.5564370155334473, 0.0938199982047081)
        xnum, ynum, znum = [5, 8, 5]
        max_range = (torch.Tensor([xmax, ymax, zmax]) - torch.Tensor([xmin, ymin, zmin])) / torch.Tensor(
            [xnum, ynum, znum])
        self.max_range = (max_range * self.args.tps3d_range).tolist()
        target_control_points = torch.tensor(list(itertools.product(
            torch.linspace(xmin, xmax, xnum),
            torch.linspace(ymin, ymax, ynum),
            torch.linspace(zmin, zmax, znum),
        )))
        mesh = MU.join_meshes([self.mesh_man, self.mesh_tshirt, self.mesh_trouser])

        tps3d = TPSGridGen(None, target_control_points, mesh.verts_packed().cpu())
        tps3d.to(self.device)
        self.tps3d = tps3d
        return

    def synthesis_image(self, img_batch, use_tps2d=True, use_tps3d=True):
        if use_tps2d:
            # tps_2d
            source_control_points_tshirt = p3dmd.get_points(self.tshirt_locations_infos, torch.pi / 180 * args.tps2d_range_t, args.tps2d_range_r,
                                                            bs=self.batch_size, random=True)
            locations_tshirt = self.tps2d_tshirt(source_control_points_tshirt.to(self.device))
            source_control_points_trouser = p3dmd.get_points(self.trouser_locations_infos, torch.pi / 180 * args.tps2d_range_t, args.tps2d_range_r,
                                                             bs=self.batch_size, random=True)
            locations_trouser = self.tps2d_trouser(source_control_points_trouser.to(self.device))
        else:
            locations_tshirt = locations_trouser = None

        if use_tps3d:
            # tps_3d
            source_coordinate = self.tps3d.tps_mesh(max_range=self.max_range, batch_size=self.batch_size).view(-1, 3)
        else:
            source_coordinate = None
        # render images
        images_predicted = p3dmd.view_mesh_wrapped([self.mesh_man, self.mesh_tshirt, self.mesh_trouser],
                                                   [None, locations_tshirt, locations_trouser],
                                                   [None, self.infos_tshirt, self.infos_trouser], source_coordinate,
                                                   cameras=self.cameras, lights=self.lights, image_size=800, fov=45,
                                                   max_faces_per_bin=30000, faces_per_pixel=3)
        adv_batch = images_predicted.permute(0, 3, 1, 2)
        p_img_batch, gt = self.patch_transformer(img_batch, adv_batch)
        return p_img_batch, gt

    def update_mesh(self, tau=0.3, type='gumbel'):
        # camouflage:
        prob_map = prob_fix_color(self.tshirt_point, self.coordinates, self.colors, self.h, self.w, blur=self.args.blur).unsqueeze(0) # For TSHIRT
        prob_trouser = prob_fix_color(self.trouser_point, self.coordinates_t, self.colors, self.h_t, self.w_t, blur=self.args.blur).unsqueeze(0) # For TROUSER
        prob_map = self.camouflage_kernel(prob_map) # kernel is Conv2DTranspose
        prob_trouser = self.camouflage_kernel(prob_trouser) # kernel is Conv2DTranspose
        prob_map = prob_map.squeeze(0).permute(1, 2, 0)
        prob_trouser = prob_trouser.squeeze(0).permute(1, 2, 0)

        gb_tshirt = -(-(self.seeds_tshirt + 5e-20).log() + 5e-20).log()
        gb_trouser = -(-(self.seeds_trouser + 5e-20).log() + 5e-20).log()

        tex = gumbel_color_fix_seed(prob_map, gb_tshirt, self.colors, tau=tau, type=type)
        tex_trouser = gumbel_color_fix_seed(prob_trouser, gb_trouser, self.colors, tau=tau, type=type)

        # fig = plt.figure()
        # plt.imshow(tex[0].detach().cpu().numpy())
        # plt.axis('off')
        # self.writer.add_figure('gumbel_color', fig)

        # fig = plt.figure()
        # plt.imshow(self.color_transform(tex.permute(0, 3, 1, 2))[0].detach().cpu().numpy())
        # plt.axis('off')
        # self.writer.add_figure('color_transform', fig)

        tex = tex.permute(0, 3, 1, 2)
        # tex = self.color_transform(tex)
        tex = self.expand_kernel(tex).permute(0, 2, 3, 1)

        tex_trouser = tex_trouser.permute(0, 3, 1, 2)
        # tex_trouser = self.color_transform(tex_trouser)
        tex_trouser = self.expand_kernel(tex_trouser).permute(0, 2, 3, 1)

        # fig = plt.figure()
        # plt.imshow(tex[0].detach().cpu().numpy())
        # plt.axis('off')
        # self.writer.add_figure('gumbel_tex2', fig)
        # exit()

        self.mesh_tshirt.textures = TexturesUV(maps=tex, faces_uvs=self.faces, verts_uvs=self.verts_uv)
        self.mesh_trouser.textures = TexturesUV(maps=tex_trouser, faces_uvs=self.faces_trouser, verts_uvs=self.verts_uv_trouser)

        return tex, tex_trouser

    def load_weights(self, save_path, epoch, best=False):
        if not os.path.exists(args.save_path):
            print("Loading path \"{}\"does not exist. Exiting...".format(args.save_path))
            exit()

        if best:
            save_path = os.path.join(save_path, "best")
        else:
            save_path = os.path.join(save_path, str(epoch))
        
        print("Loading weights from {}".format(save_path))

        path = save_path + '_circle_epoch.pth'
        self.tshirt_point.data = torch.load(path, map_location='cpu').to(self.device)

        path = save_path + '_color_epoch.pth'
        self.colors.data = torch.load(path, map_location='cpu').to(self.device)

        path = save_path + '_trouser_epoch.pth'
        self.trouser_point.data = torch.load(path, map_location='cpu').to(self.device)

        path = save_path + '_seed_tshirt_epoch.pth'
        self.seeds_tshirt = torch.load(path, map_location='cpu').to(self.device)

        path = save_path + '_seed_trouser_epoch.pth'
        self.seeds_trouser = torch.load(path, map_location='cpu').to(self.device)

        path = save_path + '_info.npz'
        if os.path.exists(path):
            x = np.load(path)
            self.loss_history = torch.from_numpy(x['loss_history']).to(self.device)
            self.num_history = torch.from_numpy(x['num_history']).to(self.device)

        print("Loaded saved weights")

    def train(self):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """
        self.writer = self.init_tensorboard()
        args = self.args

        checkpoint = args.checkpoint
        if checkpoint > 0:
            # loading trained checkpoint
            args.save_path = os.path.join(args.save_path, args.checkpoint_dir)
            self.load_weights(args.save_path, checkpoint, best=False)

            # Remove train folder to create new train folder
            args.save_path = args.save_path.rsplit('/', 1)[0]
        else:
            print("Starting new pattern training.")

        timestr = time.strftime("%m_%d-%H_%M")
        with open(os.path.join(args.save_path, timestr + ".txt"), "w") as info_f:
            info_f.write(str(args))
        args.save_path = os.path.join(args.save_path, timestr)
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

        # Where train images get stored
        sample_path = os.path.join(args.save_path, "train_samples")
        if not os.path.exists(sample_path):
            os.makedirs(sample_path)

        print("Starting training epochs...\n")
        checkpoint += 1
        best_det_loss = 1.0
        for epoch in range(checkpoint, checkpoint+args.nepoch):
            et0 = time.time()
            ep_det_loss = 0
            ep_loss = 0
            ep_mean_prob = 0
            ep_tv_loss = 0
            ep_ctrl_loss = 0
            ep_seed_loss = 0
            # ep_log_likelihood = 0
            eff_count = 0  # record how many images in this epoch are really in training so that we can calculate accurate loss
            self.sampler_probs = self.loss_history / self.num_history
            self.loss_history = self.loss_history / 2 + 1e-5
            self.num_history = self.num_history / 2 + 1e-5

            if args.anneal:
                tau = np.exp(-(epoch + 1) / args.nepoch * args.anneal_alpha) * args.anneal_init
            else:
                # Default
                tau = 0.3

            for i_batch, img_batch in tqdm(enumerate(self.train_loader),
                                            desc=f'Epoch {epoch}',
                                            total=len(self.train_loader)):
                img_batch = img_batch.to(self.device)
                # AG step
                self.optimizer.zero_grad()
                self.optimizer_seed.zero_grad()
                if i_batch % 10 == 0: # Used to be every 20 batches
                    self.sample_cameras()
                    self.sample_lights()

                tex, tex_trouser = self.update_mesh(tau=tau)
                p_img_batch, gt = self.synthesis_image(img_batch, not args.disable_tps2d, not args.disable_tps3d)
                output = self.model(p_img_batch)

                try:
                    det_loss, max_prob_list = self.prob_extractor(output, gt, loss_type=args.loss_type, iou_thresh=args.train_iou)
                    eff_count += 1
                except RuntimeError:  # current batch of imgs have no bbox be detected
                    continue

                if self.azim_inds is not None:
                    self.loss_history.index_put_([self.azim_inds], max_prob_list.detach(), accumulate=True)
                    self.num_history.index_put_([self.azim_inds], torch.ones_like(max_prob_list), accumulate=True)
                loss = 0
                tv_loss = torch.tensor([0])
                loss += det_loss
                if args.tv_loss > 0:
                    tv_loss = self.tv_loss(tex)
                    loss += tv_loss * args.tv_loss

                loss_c = ctrl_loss(self.tshirt_point, self.fig_size_H, self.fig_size_W)
                loss_c += ctrl_loss(self.trouser_point, self.fig_size_H_t, self.fig_size_W_t)
                # loss += args.ctrl * loss_c
                loss += 2 * loss_c

                if args.cdist != 0:
                    loss_seed = args.cdist * reg_dist(self.seeds_tshirt_train.flatten(), sample_num=args.rd_num)
                    loss_seed += args.cdist * reg_dist(self.seeds_trouser_train.flatten(), sample_num=args.rd_num)
                    loss += loss_seed
                else:
                    loss_seed = torch.zeros([], device=self.device)

                ep_mean_prob += max_prob_list.detach().cpu().mean().item()
                # ep_ctrl_loss += loss_c.item()
                ep_det_loss += det_loss.detach().cpu().item()
                ep_tv_loss += tv_loss.detach().cpu().item()
                ep_seed_loss += loss_seed.detach().cpu().item()
                ep_loss += loss.detach().cpu().item()
                loss.backward()
                self.optimizer.step()
                if args.seed_type == 'random':
                    self.seeds_tshirt.uniform_()
                    self.seeds_trouser.uniform_()
                elif args.seed_type != 'fixed':
                    self.seeds_tshirt_train.grad /= args.seed_temp
                    self.seeds_trouser_train.grad /= args.seed_temp
                    self.optimizer_seed.step()
                    self.seeds_tshirt_train.data.clamp_(args.clamp_shift, 1 - args.clamp_shift)
                    self.seeds_trouser_train.data.clamp_(args.clamp_shift, 1 - args.clamp_shift)
                    if args.seed_type == 'langevin':
                        beta = np.sqrt(2 * self.optimizer_seed.param_groups[0]['lr'])
                        for s in [self.seeds_tshirt_train, self.seeds_trouser_train]:
                            # assert beta and clamp_shift are both small
                            raw = s + s.new(s.shape).normal_() * beta
                            s.data = raw.clamp(args.clamp_shift, 1 - args.clamp_shift) * 2 - raw
                            s.data.clamp_(args.clamp_shift, 1 - args.clamp_shift)

                self.tshirt_point.data = self.tshirt_point.data.clamp(0, 1)
                self.trouser_point.data = self.trouser_point.data.clamp(0, 1)

                if i_batch % 100 == 0:
                #     iteration = self.epoch_length * epoch + i_batch
                #     self.writer.add_scalar('batch/total_loss', loss.detach().cpu().numpy(), iteration)
                #     self.writer.add_scalar('batch/tv_loss', tv_loss.detach().cpu().numpy(), iteration)
                #     self.writer.add_scalar('batch/det_loss', det_loss.detach().cpu().numpy(), iteration)
                #     self.writer.add_scalar('batch/ctrl_loss', loss_c.detach().cpu().numpy(), iteration)
                #     self.writer.add_scalar('batch/loss_seed', loss_seed.detach().cpu().numpy(), iteration)
                    torchvision.utils.save_image(
                        p_img_batch[0, :, :, :],
                        os.path.join(sample_path, '{}_{}.png'.format(epoch, i_batch)))


            et1 = time.time()
            ep_det_loss = ep_det_loss / eff_count
            ep_loss = ep_loss / eff_count
            ep_tv_loss = ep_tv_loss / eff_count
            ep_ctrl_loss = ep_ctrl_loss / eff_count
            ep_mean_prob = ep_mean_prob / eff_count
            ep_seed_loss = ep_seed_loss / eff_count
            if True:
                # print('\n  EPOCH NR: ', epoch),
                print('\nEPOCH LOSS: ', ep_loss)
                print('DET LOSS  : ', ep_det_loss)
                print('    MEAN PROB : ', ep_mean_prob)
                print('    TV LOSS   : ', ep_tv_loss)
                print('    CTRL LOSS : ', ep_ctrl_loss)
                print('    SEED LOSS : ', ep_seed_loss)
                print('    EPOCH TIME: ', et1 - et0)
                print('    LEARNING RATE', self.optimizer.param_groups[0]['lr'])

                torchvision.utils.save_image(
                    p_img_batch[0, :, :, :],
                    os.path.join(sample_path, '{}_{}.png'.format(epoch, 0)))

                self.writer.add_scalar('epoch/total_loss', ep_loss, epoch)
                # self.writer.add_scalar('epoch/tv_loss', ep_tv_loss, epoch)
                self.writer.add_scalar('epoch/det_loss', ep_det_loss, epoch)
                # self.writer.add_scalar('epoch/ctrl_loss', ep_ctrl_loss, epoch)
                # self.writer.add_scalar('epoch/seed_loss', ep_seed_loss, epoch)
                self.writer.add_scalar('epoch/lr', self.optimizer.param_groups[0]['lr'], epoch)

            # Save textures
            if True:
                fig = plt.figure()
                plt.imshow(tex[0].detach().cpu().numpy())
                plt.axis('off')
                self.writer.add_figure('maps_tshirt', fig, epoch)

                fig = plt.figure()
                plt.imshow(tex_trouser[0].detach().cpu().numpy())
                plt.axis('off')
                self.writer.add_figure('maps_trouser', fig, epoch)

            # Save checkpoint
            if True:
                # torchvision.utils.save_image(
                #     p_img_batch[0, :, :, :],
                #     os.path.join(args.save_path, '{}.png'.format(epoch)))

                path = args.save_path + '/' + str(epoch) + '_circle_epoch.pth'
                torch.save(self.tshirt_point, path)
                path = args.save_path + '/' + str(epoch) + '_color_epoch.pth'
                torch.save(self.colors, path)
                path = args.save_path + '/' + str(epoch) + '_trouser_epoch.pth'
                torch.save(self.trouser_point, path)
                path = args.save_path + '/' + str(epoch) + '_seed_tshirt_epoch.pth'
                torch.save(self.seeds_tshirt, path)
                path = args.save_path + '/' + str(epoch) + '_seed_trouser_epoch.pth'
                torch.save(self.seeds_trouser, path)
                path = args.save_path + '/' + str(epoch) + '_info.npz'
                np.savez(path, loss_history=self.loss_history.cpu().numpy(), num_history=self.num_history.cpu().numpy(), azim=self.azim.cpu().numpy())

            # Save pattern with BEST attack rate
            # if ep_det_loss < best_det_loss:
            #     print("Saving best detection loss...")
            #     best_det_loss = ep_det_loss

            #     # Save sample of training image
            #     torchvision.utils.save_image(
            #         p_img_batch[0, :, :, :],
            #         os.path.join(args.save_path, 'best_{}.png'.format(epoch)))

            #     path = args.save_path + '/' + 'best_circle_epoch.pth'
            #     torch.save(self.tshirt_point, path)
            #     path = args.save_path + '/' + 'best_color_epoch.pth'
            #     torch.save(self.colors, path)
            #     path = args.save_path + '/' + 'best_trouser_epoch.pth'
            #     torch.save(self.trouser_point, path)
            #     path = args.save_path + '/' + 'best_seed_tshirt_epoch.pth'
            #     torch.save(self.seeds_tshirt, path)
            #     path = args.save_path + '/' + 'best_seed_trouser_epoch.pth'
            #     torch.save(self.seeds_trouser, path)
            #     path = args.save_path + '/' + 'best_info.npz'
            #     np.savez(path, loss_history=self.loss_history.cpu().numpy(), num_history=self.num_history.cpu().numpy(), azim=self.azim.cpu().numpy())               

            print("")
            # Evaluate training
            # if (epoch + 1) % 300 == 0:
            #     self.update_mesh(type='determinate')
            #     for iou_thresh in [0.01, 0.1, 0.3, 0.5]:
            #         precision, recall, avg, confs, thetas = self.test(conf_thresh=0.01, iou_thresh=iou_thresh, angle_sample=37, use_tps2d=not args.disable_test_tps2d, use_tps3d=not args.disable_test_tps3d, mode=args.test_mode)
            #         info = [precision, recall, avg, confs]
            #         path = args.save_path + '/' + str(epoch) + 'test_results_tps'
            #         path = path + '_iou' + str(iou_thresh).replace('.', '') + '_' + args.test_mode
            #         path = path + '.npz'
            #         np.savez(path, thetas=thetas, info=info)

            # Learning rate decay
            if True:
                self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr'] / args.lr_decay
                self.optimizer_seed.param_groups[0]['lr'] = self.optimizer_seed.param_groups[0]['lr'] / args.lr_decay_seed
                # Decaying lr_decay (META-DECAY)
                args.lr_decay -= 0.15 # Start from 1.5 and count down?
                args.lr_decay = max(args.lr_decay, 1.1)


    def generate_test_images(self, angle_sample=37, use_tps2d=True, use_tps3d=True):
        args.save_path = os.path.join(args.save_path, args.checkpoint_dir)
        use_best = args.use_best
        self.load_weights(args.save_path, args.checkpoint, best=use_best)
        print("Loaded weights")
        self.update_mesh(type='determinate')

        print("Generating test images...")
        test_dir = os.path.join("test_images", args.checkpoint_dir)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        thetas_list = np.linspace(-180, 180, angle_sample)

        with torch.no_grad():
            for i_batch, img_batch in tqdm(enumerate(self.test_loader), total=len(self.test_loader), position=0):
                self.sample_lights()
                img_batch = img_batch.to(self.device)
                for it, theta in enumerate(thetas_list):
                    self.sample_cameras(theta=theta)
                    p_img_batch, gt = self.synthesis_image(img_batch, use_tps2d, use_tps3d)

                    # Save testing images for future 
                    for i in range(p_img_batch.size(0)):
                        img_name = "{}_{}_{}.png".format(i_batch, int(theta), i)
                        img_dir = os.path.join(test_dir, img_name)
                        torchvision.utils.save_image(p_img_batch[i, :, :, :], img_dir)
        return None, None, None, None, None        


if __name__ == '__main__':
    print('advcat version 2.0')
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--device', default='cuda:0', help='')
    parser.add_argument('--lr', type=float, default=0.001, help='')
    parser.add_argument('--lr_seed', type=float, default=0.01, help='')
    parser.add_argument('--nepoch', type=int, default=200, help='')
    parser.add_argument('--checkpoint', type=int, default=0, help='')
    parser.add_argument('--batch_size', type=int, default=4, help='')
    parser.add_argument('--save_path', default='results/', help='')
    parser.add_argument("--alpha", type=float, default=10, help='')
    parser.add_argument("--tv_loss", type=float, default=0, help='')
    parser.add_argument("--lr_decay", type=float, default=2, help='')
    parser.add_argument("--lr_decay_seed", type=float, default=2, help='')
    parser.add_argument("--blur", type=float, default=1, help='')
    parser.add_argument("--like", type=float, default=1, help='')
    parser.add_argument("--ctrl", type=float, default=1, help='')
    parser.add_argument("--num_points_tshirt", type=int, default=60, help='')
    parser.add_argument("--num_points_trouser", type=int, default=60, help='')
    parser.add_argument("--arch", type=str, default="yolov3")
    parser.add_argument("--cdist", type=float, default=0, help='')
    parser.add_argument("--seed_type", default='fixed', help='fixed, random, variable, langevin')
    parser.add_argument("--rd_num", type=int, default=200, help='')
    parser.add_argument("--clamp_shift", type=float, default=0, help='')
    parser.add_argument("--resample_type", default=None, help='')
    parser.add_argument("--seed_temp", type=float, default=1.0, help='')
    parser.add_argument("--seed_opt", default='adam', help='')
    parser.add_argument("--tps2d_range_t", type=float, default=50.0, help='')
    parser.add_argument("--tps2d_range_r", type=float, default=0.1, help='')
    parser.add_argument("--tps3d_range", type=float, default=0.15, help='')
    parser.add_argument("--disable_tps2d", default=False, action='store_true', help='')
    parser.add_argument("--disable_tps3d", default=False, action='store_true', help='')
    parser.add_argument("--disable_test_tps2d", default=False, action='store_true', help='')
    parser.add_argument("--disable_test_tps3d", default=False, action='store_true', help='')
    parser.add_argument("--seed_ratio", default=1.0, type=float, help='The ratio of trainable part when seed type is variable')
    parser.add_argument("--loss_type", default='max_iou', help='max_iou, max_conf, softplus_max, softplus_sum')
    parser.add_argument("--gen_test", default=False, action='store_true', help='')
    parser.add_argument("--test_iou", type=float, default=0.1, help='')
    parser.add_argument("--test_nms_thresh", type=float, default=1.0, help='')
    parser.add_argument("--test_mode", default='person', help='person, all')
    parser.add_argument("--test_suffix", default='', help='')
    parser.add_argument("--train_iou", type=float, default=0.01, help='')
    parser.add_argument("--anneal", default=False, action='store_true', help='')
    parser.add_argument("--anneal_init", type=float, default=5.0, help='')
    parser.add_argument("--anneal_alpha", type=float, default=3.0, help='')


    parser.add_argument("--use_best", default=False, help='Whether to load best weights')
    parser.add_argument("--checkpoint_dir", default='', help='Loading checkpoint')
    parser.add_argument("--color_pth", default="army_colors.pth", help='.pth file for pattern colors')
    parser.add_argument("--train_dir", default="background", help="folder name containing train background files")
    parser.add_argument("--test_dir", default="background_test", help="folder name containing test background files")


    args = parser.parse_args()
    assert args.seed_type in ['fixed', 'random', 'variable', 'langevin']

    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    os.environ['TZ'] = 'Asia/Seoul'
    time.tzset()

    print("Train info:", args)
    trainer = PatchTrainer(args)
    print("Created PatchTrainer...")
    if not args.gen_test:
        trainer.train()
    elif args.gen_test:
        trainer.generate_test_images(
            angle_sample=37,
            use_tps2d=not args.disable_test_tps2d, 
            use_tps3d=not args.disable_test_tps3d)


