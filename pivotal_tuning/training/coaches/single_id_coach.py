import os
import torch
from tqdm import tqdm
from configs import paths_config, hyperparameters, global_config
from training.coaches.base_coach import BaseCoach
from utils.log_utils import log_images_from_w, save_image_from_w, save_image
from editings.latent_editor_wrapper import LatentEditorWrapper
from criteria import l2_loss, msssim_loss
import numpy as np

import pdb


class SingleIDCoach(BaseCoach):

    def __init__(self, data_loader, use_wandb):
        super().__init__(data_loader, use_wandb)

    def train(self):
        self.seed = 0

        w_path_dir = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'
        self.w_path_dir = w_path_dir
        os.makedirs(w_path_dir, exist_ok=True)
        os.makedirs(f'{w_path_dir}/{paths_config.pti_results_keyword}', exist_ok=True)

        use_ball_holder = True

        self.images_dir = f'{w_path_dir}/inv_images/'
        os.makedirs(self.images_dir, exist_ok=True)

        for fname, image in tqdm(self.data_loader):
            torch.cuda.empty_cache()
            image_name = fname[0]

            self.restart_training()

            if self.image_counter >= hyperparameters.max_images_to_invert:
                break

            embedding_dir = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}'
            os.makedirs(embedding_dir, exist_ok=True)

            w_pivot = None

            if hyperparameters.use_saved_w_pivots:
                w_pivot = self.load_inversions_new(paths_config.last_w_dir, image_name,
                                                   hyperparameters.ft_steps)

            elif not hyperparameters.use_saved_w_pivots or w_pivot is None:
                w_pivot = self.calc_inversions(image, image_name)

            w_pivot = w_pivot.to(global_config.device)

            torch.save(w_pivot, f'{embedding_dir}/0.pt')
            log_images_counter = 0
            real_images_batch = image.to(global_config.device)

            self.eva_real(real_images_batch, image_name)
            self.eva_first_step(w_pivot, image_name)

            for i in tqdm(range(hyperparameters.max_pti_steps)):

                generated_images = self.forward(w_pivot)
                loss, l2_loss_val, loss_lpips = self.calc_loss(
                    generated_images, real_images_batch, image_name,
                        self.G, use_ball_holder, w_pivot)

                self.optimizer.zero_grad()

                if loss_lpips <= hyperparameters.LPIPS_value_threshold:
                    inv_images_path = f'{self.images_dir}/' \
                                      f'{image_name}_sec_inv_{i}.jpg'
                    save_image(generated_images, inv_images_path)
                    break

                loss.backward()
                self.optimizer.step()

                use_ball_holder = (global_config.training_step %
                                   hyperparameters.locality_regularization_interval
                                   == 0)

                if self.use_wandb and (log_images_counter %
                                       global_config.image_rec_result_log_snapshot
                                       == 0):
                    log_images_from_w([w_pivot], self.G, [image_name])

                global_config.training_step += 1
                log_images_counter += 1

                if i > 0 and (i % 1000 == 0 or i == hyperparameters.max_pti_steps - 1):
                    inv_images_path = f'{self.images_dir}/{image_name}_sec_inv_{i}.jpg'
                    save_image(generated_images, inv_images_path)

            self.eva_editing(w_pivot, image_name, real_images_batch)

            self.image_counter += 1


    def eva_real(self, real_images_batch, image_name):
        real_images_path = f'{self.images_dir}/{image_name}_real.jpg'
        save_image(real_images_batch, real_images_path)

    def eva_first_step(self, w_pivot, image_name):
        first_inv_images = self.forward(w_pivot)
        inv_images_path = f'{self.images_dir}/{image_name}_fst_inv.jpg'
        save_image_from_w(w_pivot, self.G, inv_images_path)

    def eva_editing(self, w_pivot, image_name, real_image):
        latent_editor = LatentEditorWrapper(self.G)
        latents_after_edit = latent_editor.get_single_interface_gan_edits(\
                                 w_pivot, [v for v in np.arange(-5,5,0.5)])
        for direction, factor_and_edit in latents_after_edit.items():
            for factor, latent in factor_and_edit.items():
                edited_image = self.G.synthesis(latent, noise_mode='const')
                image_path = f'{self.images_dir}/' \
                        f'{image_name}_{direction}_{factor:.1f}.jpg'
                save_image(edited_image, image_path)








