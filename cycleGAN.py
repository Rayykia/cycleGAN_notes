"""
The cycle GAN structure.

Author: Rayykia
"""
import torch
from torch import Tensor
from torch.utils import data
from torch import nn

import numpy as np
import itertools
import matplotlib.pyplot as plt
from typing import Callable, Tuple

from BasicBlock import UnetGenerator, PatchDiscriminator, ResnetGenerator

from time import time


class cycleGAN(nn.Module):
    """Structure and training the cycle GAN.

    """
    def __init__(
            self, 
            gen_net: str,
            device: torch.device = torch.device('cuda'),
    ):
        """Initialize the cycle GAN.
        
        Attrs:
            gen_AB:
                the forward generator
            gen_BA:
                the backward generator
            dis_A:
                discriminator for the forward generator
            dis_B:
                discriminator for the backward generator
        """
        super().__init__()
        # Generators
        if gen_net == "unet":
            self.gen_AB = UnetGenerator().to(device)
            self.gen_BA = UnetGenerator().to(device)
        elif gen_net == "resnet":
            self.gen_AB = ResnetGenerator().to(device)
            self.gen_BA = ResnetGenerator().to(device)
        else:
            raise ValueError("Generator network not defined.")
        # Discriminators
        self.dis_A = PatchDiscriminator().to(device)
        self.dis_B = PatchDiscriminator().to(device)


    @staticmethod
    def _get_optim(
        gen_AB: nn.Module,
        gen_BA: nn.Module,
        dis_A: nn.Module,
        dis_B: nn.Module
    ) -> Tuple[torch.optim.Optimizer, ...]:
        """Get the optimizer for the generators and discriminators.
        
        Attrs:
            gen_AB:
                the forward generator
            gen_BA:
                the backward generator
            dis_A:
                discriminator for the forward generator
            dis_B:
                discriminator for the backward generator
        """
        gen_optim = torch.optim.Adam(
            itertools.chain(gen_AB.parameters(), gen_BA.parameters()),
            lr = 2e-4,
            betas = (0.5, 0.999)
        )
        dis_A_optim = torch.optim.Adam(
            dis_A.parameters(),
            lr = 2e-4,
            betas = (0.5, 0.999)
        )
        dis_B_optim = torch.optim.Adam(
            dis_B.parameters(),
            lr = 2e-4,
            betas = (0.5, 0.999)
        )
        return gen_optim, dis_A_optim, dis_B_optim
    
    @staticmethod
    def generate_images(
        model: Callable[...,Tensor], 
        test_input: Tensor,
        epoch: int,
        direction: str
    ) -> None:
        """Generate images during the training process."""
        prediction = model(test_input).permute(0, 2, 3, 1).detach().cpu().numpy()
        test_input = test_input.permute(0, 2, 3, 1).cpu().numpy()
        plt.figure(figsize=(100, 50))
        # display_list = [test_input[0], prediction[0]]
        title = ['Input Image', 'Generated Image']
        for i in range(4):
            plt.subplot(2, 4, i+1)
            plt.title(title[0])
            plt.imshow(test_input[i]*0.5+0.5)
            plt.axis('off')
        for i in range(4):
            plt.subplot(2, 4, i+5)
            plt.title(title[1])
            plt.imshow(prediction[i]*0.5+0.5)
            plt.axis('off')
        plt.savefig('./img/{}_epoch{}_.png'.format(direction, epoch))
        plt.close()

    @staticmethod
    def _get_loss() -> Tuple[Callable[..., Tensor], ...]:
        """Calculate the loss of the cycle GAN.
        1. adversarial loss: nn.BCELoss()
        2. cycle consistency loss: nn.L1Loss()
        3. identity loss: nn.L1Loss()
        """
        return nn.BCELoss(), nn.L1Loss(), nn.L1Loss()

    # def _epoch_fit(
    #         self, 
    #         loader_A: data.DataLoader, 
    #         loader_B: data.DataLoader,
    #         gen_optim: torch.optim.Optimizer,
    #         dis_A_optim: torch.optim.Optimizer,
    #         dis_B_optim: torch.optim.Optimizer,
    #         adv_loss, 
    #         cyc_con_loss, 
    #         id_loss,
    #         device: torch.device
    # ):
    #     """Train the generators and discriminators for one epoch."""
    #     D_epoch_loss, G_epoch_loss = 0, 0
    #     for step, real_A in enumerate(loader_A):
    #         real_B = next(iter(loader_B))
    #         real_A, real_B = real_A.to(device), real_B.to(device)

    #         # GAN training
    #         gen_optim.zero_grad()

            

    def fit(self,
            loader_A: data.DataLoader, 
            loader_B: data.DataLoader,
            epoches: int,
            device: torch.device = torch.device('cuda')
        ):
        """Train the cycle GAN for ``epoches``."""

        gen_optim, dis_A_optim, dis_B_optim = self._get_optim(
            self.gen_AB, self.gen_BA, self.dis_A, self.dis_B
        )
        dis_A_scheduler = torch.optim.lr_scheduler.StepLR(dis_A_optim, step_size=20, gamma = 0.7)
        dis_B_scheduler = torch.optim.lr_scheduler.StepLR(dis_B_optim, step_size=20, gamma = 0.7)

        adv_loss, cyc_con_loss, id_loss = self._get_loss()

        D_loss, G_loss = [], []
        test_batch_A = next(iter(loader_A))
        test_batch_B = next(iter(loader_B))
        test_input_A = (test_batch_A[:4]).to(device)
        test_input_B = (test_batch_B[:4]).to(device)
        count = len(loader_A)
        t_start = time()
        for epoch in range(epoches):
            D_epoch_loss, G_epoch_loss = 0, 0
            t_epoch_start = time()
            for step, real_A in enumerate(loader_A):
                real_B = next(iter(loader_B))
                real_A, real_B = real_A.to(device), real_B.to(device)

                # GAN training
                gen_optim.zero_grad()

                # identity loss
                same_B = self.gen_AB(real_B)
                identity_B_loss = id_loss(same_B, real_B)
                same_A = self.gen_BA(real_A)
                identity_A_loss = id_loss(same_A, real_A)

                # adversarial loss
                fake_B = self.gen_AB(real_A)
                pred_fake_B = self.dis_B(fake_B)
                adv_loss_AB = adv_loss(pred_fake_B,
                                       torch.ones_like(pred_fake_B, device=device))
                fake_A = self.gen_BA(real_B)
                pred_fake_A = self.dis_A(fake_A)
                adv_loss_BA = adv_loss(pred_fake_A,
                                       torch.ones_like(pred_fake_A, device=device))
                
                # cycle consistancy loss
                recoverd_A = self.gen_BA(fake_B)
                cyc_con_loss_ABA = cyc_con_loss(recoverd_A, real_A)
                recoverd_B = self.gen_AB(fake_A)
                cyc_con_loss_BAB = cyc_con_loss(recoverd_B, real_B)

                # generator training
                g_loss = (identity_A_loss* 0.5 + identity_B_loss * 0.5 + adv_loss_AB + adv_loss_BA
                          + cyc_con_loss_ABA + cyc_con_loss_BAB)
                # g_loss = (adv_loss_AB + adv_loss_BA
                #           + cyc_con_loss_ABA + cyc_con_loss_BAB)
                g_loss.backward()
                gen_optim.step()

                # dis_A training
                dis_A_optim.zero_grad()
                dis_A_real_output = self.dis_A(real_A)
                dis_A_real_loss = adv_loss(dis_A_real_output,
                                           torch.ones_like(dis_A_real_output, device=device))
                dis_A_fake_output = self.dis_A(fake_A.detach())
                dis_A_fake_loss = adv_loss(dis_A_fake_output,
                                           torch.zeros_like(dis_A_fake_output, device=device))
                dis_A_loss = (dis_A_real_loss + dis_A_fake_loss) * 0.5
                dis_A_loss.backward()
                dis_A_optim.step()

                # dis_B training
                dis_B_optim.zero_grad()
                dis_B_real_output = self.dis_B(real_B)
                dis_B_real_loss = adv_loss(dis_B_real_output,
                                           torch.ones_like(dis_B_real_output, device=device))
                dis_B_fake_output = self.dis_B(fake_B.detach())
                dis_B_fake_loss = adv_loss(dis_B_fake_output,
                                           torch.zeros_like(dis_B_fake_output, device=device))
                dis_B_loss = (dis_B_real_loss + dis_B_fake_loss) * 0.5
                dis_B_loss.backward()
                dis_B_optim.step()

                with torch.no_grad():
                    D_epoch_loss += (dis_A_loss + dis_B_loss).item()
                    G_epoch_loss += g_loss.item()
                    print("\r`{}` EPOCH {} IN PROGRESS {:2.1%}.".format(device, epoch, (step+1)/count), end="")

            dis_A_scheduler.step()
            dis_B_scheduler.step()
            with torch.no_grad():
                D_epoch_loss /= step
                G_epoch_loss /= step
                D_loss.append(D_epoch_loss)
                G_loss.append(G_epoch_loss)
                if epoch%10 == 0:
                    self.generate_images(self.gen_AB, test_input=test_input_A, direction="AB", epoch=epoch)
                    self.generate_images(self.gen_BA, test_input=test_input_B, direction="BA", epoch=epoch)
                print('\x1b[2K', end="\r")
                print("EPOCH {:3d} \t\t D_EPOCH_LOSS {:.5f} \t\t G_EPOCH_LOSS {:.5f} \t\t {:.3} s".format(
                    epoch, D_epoch_loss, G_epoch_loss, time()-t_epoch_start
                ))
        print("DONE! \t TOTAL TIME SPENT: {}s".format(time()-t_start))
        torch.save(self.gen_AB, './checkpoints/g_ab.pt') 


