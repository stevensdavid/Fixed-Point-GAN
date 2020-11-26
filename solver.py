from model import Generator
from model import Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import transforms
from torchvision.models import resnet50
from matplotlib import colors
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from cv2 import cv2
from PIL import Image
import torch.nn as nn
from sklearn import metrics

class ResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = resnet50(pretrained=True)
        self.model.fc = nn.Linear(2048, num_classes)
        self.activation = nn.Sigmoid()
        # self.activation = nn.Identity()
        # Make AVG pooling input shape independent
        self.model.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.model(x)
        return self.activation(x)

    def predict(self, x):
        x = self.forward(x)
        return self.activation(x)

class Solver(object):
    """Solver for training and testing Fixed-Point GAN."""

    def __init__(self, data_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.data_loader = data_loader

        # Model configurations.
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        self.lambda_id = config.lambda_id

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs

        # Test configurations.
        self.test_iters = config.test_iters
        self.eval_resnet_id_name = config.eval_resnet_id_name
        self.eval_resnet_tilde_name = config.eval_resnet_tilde_name
        self.eval_dataset = config.eval_dataset


        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("device", self.device)

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()

        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        if self.dataset in ['CelebA', 'BRATS', 'PCam', 'Directory']:
            self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num) 

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
            
        self.G.to(self.device)
        self.D.to(self.device)

    def recreate_image(self, codebook, labels, w, h):
        """Recreate the (compressed) image from the code book & labels"""
        d = codebook.shape[1]
        image = np.zeros((w, h, d))
        label_idx = 0
        for i in range(w):
            for j in range(h):
                image[i][j] = codebook[labels[label_idx]]
                label_idx += 1
        return image

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        #print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))


    def restore_model_tilde_resnet(self, name):
        print('Loading the tilde resenet model')
        res_path = os.path.join(self.model_save_dir, name)
        print(res_path)
        self.resnet_tilde = ResNet(num_classes=1)
        self.resnet_tilde.load_state_dict(torch.load(res_path),strict=False)
        self.resnet_tilde.eval()

    def restore_model_id_resnet(self, name):
        print('Loading the id resenet model')
        res_path = os.path.join(self.model_save_dir, name)
        print(res_path)
        self.resnet_id = ResNet(num_classes=1)
        self.resnet_id.load_state_dict(torch.load(res_path),strict=False)
        self.resnet_id.eval()

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def norm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = x - 0.5
        return out.clamp_(-0.5, 5)


    def confusion_metrics (self, conf_matrix, name):
        
        # save confusion matrix and slice into four pieces    
        TP = conf_matrix[1][1]
        TN = conf_matrix[0][0]
        FP = conf_matrix[0][1]
        FN = conf_matrix[1][0]   

  

        metrics = []

        metrics.append(('{} {}'.format('True Positives:', TP)))
        metrics.append(('{} {}'.format('True Negatives:', TN)))
        metrics.append(('{} {}'.format('False Positives:', FP)))
        metrics.append(('{} {}'.format('False Negatives:', FN)))
        
        # calculate accuracy
        conf_accuracy = (float (TP+TN) / float(TP + TN + FP + FN))
        
        # calculate mis-classification
        conf_misclassification = 1- conf_accuracy
        
        # calculate the sensitivity
        conf_sensitivity = (TP / float(TP + FN))    # calculate the specificity
        conf_specificity = (TN / float(TN + FP))
        
        # calculate precision
        conf_precision = (TN / float(TN + FP))    # calculate f_1 score
        conf_f1 = 2 * ((conf_precision * conf_sensitivity) / (conf_precision + conf_sensitivity))    


        metrics.append('-'*50)
        metrics.append(f'Accuracy: {round(conf_accuracy,2)}')
        metrics.append(f'Mis-Classification: {round(conf_misclassification,2)}')
        metrics.append(f'Sensitivity: {round(conf_sensitivity,2)}')
        metrics.append(f'Specificity: {round(conf_specificity,2)}')
        metrics.append(f'Precision: {round(conf_precision,2)}')
        metrics.append(f'f_1 Score: {round(conf_f1,2)}')

        result_metrics_path = os.path.join(self.result_dir,  '{}-metrics.txt'.format(name))
        metrics = '\n'.join(map(str, metrics)) 

        f = open(result_metrics_path, "w")
        f.write(metrics)
        f.close()

        

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        if dataset in ['CelebA']:
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            if dataset in ['CelebA']:
                c_trg = c_org.clone()
                if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
                else:
                    c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
            elif dataset in ['BRATS', 'PCam']:
                c_trg = c_org.clone()
                c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
            elif dataset == 'Directory':
                c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def classification_loss(self, logit, target, dataset='CelebA'):
        """Compute binary or softmax cross entropy loss."""
        if dataset in ['CelebA', 'BRATS', 'PCam']:
            return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
        elif dataset == 'Directory':
            return F.cross_entropy(logit, target)

    def train(self):
        """Train Fixed-Point GAN within a single dataset."""
        # Set data loader.
        if self.dataset in ['CelebA', 'BRATS', 'PCam', 'Directory']:
            data_loader = self.data_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        x_fixed, c_org = next(data_iter)
        x_fixed = x_fixed.to(self.device)
        c_fixed_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real, label_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, label_org = next(data_iter)

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            if self.dataset in ['CelebA', 'BRATS', 'PCam']:
                c_org = label_org.clone()
                c_trg = label_trg.clone()
            elif self.dataset == 'Directory':
                c_org = self.label2onehot(label_org, self.c_dim)
                c_trg = self.label2onehot(label_trg, self.c_dim)

            x_real = x_real.to(self.device)           # Input images.
            c_org = c_org.to(self.device)             # Original domain labels.
            c_trg = c_trg.to(self.device)             # Target domain labels.
            label_org = label_org.to(self.device)     # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)     # Labels for computing classification loss.

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_cls = self.D(x_real)
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, label_org, self.dataset)

            # Compute loss with fake images.
            delta = self.G(x_real, c_trg)
            x_fake = torch.tanh(x_real + delta)
            out_src, out_cls = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()
            
            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            
            if (i+1) % self.n_critic == 0:
                # Original-to-target domain.
                delta = self.G(x_real, c_trg)
                x_fake = torch.tanh(x_real + delta)
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg, self.dataset)

                # Original-to-original domain.
                delta_id = self.G(x_real, c_org)
                x_fake_id = torch.tanh(x_real + delta_id)
                out_src_id, out_cls_id = self.D(x_fake_id)
                g_loss_fake_id = - torch.mean(out_src_id)
                g_loss_cls_id = self.classification_loss(out_cls_id, label_org, self.dataset)
                g_loss_id = torch.mean(torch.abs(x_real - torch.tanh(delta_id + x_real)))

                # Target-to-original domain.
                delta_reconst = self.G(x_fake, c_org)
                x_reconst = torch.tanh(x_fake + delta_reconst)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # Original-to-original domain.
                delta_reconst_id = self.G(x_fake_id, c_org)
                x_reconst_id = torch.tanh(x_fake_id + delta_reconst_id)
                g_loss_rec_id = torch.mean(torch.abs(x_real - x_reconst_id))

                # Backward and optimize.
                g_loss_same = g_loss_fake_id + self.lambda_rec * g_loss_rec_id + self.lambda_cls * g_loss_cls_id + self.lambda_id * g_loss_id
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls + g_loss_same

                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()
                loss['G/loss_fake_id'] = g_loss_fake_id.item()
                loss['G/loss_rec_id'] = g_loss_rec_id.item()
                loss['G/loss_cls_id'] = g_loss_cls_id.item()
                loss['G/loss_id'] = g_loss_id.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        print(tag, value, i+1)
                        #self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_fixed_list:
                        delta = self.G(x_fixed, c_fixed)
                        x_fake_list.append(torch.tanh(delta + x_fixed))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))


    def test(self):
        """Translate images using Fixed-Point GAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        # Set data loader.
        if self.dataset in ['CelebA', 'PCam', 'Directory']:
            data_loader = self.data_loader
        
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

                # Translate images.
                x_fake_list = [x_real]
                for c_trg in c_trg_list:
                    print("c_trg_tilde", x_real.size(), c_trg)
                    x_fake_list.append(torch.tanh(x_real + self.G(x_real, c_trg)))

                # Save the translated images.c
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))



    def test_brats(self):
        """Translate images using Fixed-Point GAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        # Set data loader.
        if self.dataset in ['BRATS', 'PCam']:
            data_loader = self.data_loader
        
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):
                x_real = x_real.to(self.device)

                c_trg = c_org.clone()
                c_trg[:, 0] = 0 # always to healthy              
                c_trg_list = [c_trg.to(self.device)]

                # Translate images.
                x_fake_list = [x_real]
                for c_trg in c_trg_list:
                    delta = self.G(x_real, c_trg)
                    delta_org = torch.abs(torch.tanh(delta + x_real) - x_real) - 1.0
                    delta_gray = np.mean(delta_org.data.cpu().numpy(), axis=1)
                    delta_gray_norm = []

                    loc = []
                    cls_mul = []

                    for indx in range(delta_gray.shape[0]):
                        temp = delta_gray[indx, :, :] + 1.0  
                        tempimg_th = np.percentile(temp, 99)
                        tempimg = np.float32(temp >= tempimg_th)
                        temploc = np.reshape(tempimg, (self.image_size*self.image_size, 1))

                        kmeans = KMeans(n_clusters=2, random_state=0).fit(temploc)
                        labels = kmeans.predict(temploc)

                        recreated_loc = self.recreate_image(kmeans.cluster_centers_, labels, self.image_size, self.image_size)
                        recreated_loc = ((recreated_loc - np.min(recreated_loc)) / (np.max(recreated_loc) - np.min(recreated_loc)))

                        loc.append(recreated_loc)
                        delta_gray_norm.append( tempimg )


                    loc = np.array(loc, dtype=np.float32)[:, :, :, 0]
                    delta_gray_norm = np.array(delta_gray_norm)

                    loc = (loc * 2.0) - 1.0
                    delta_gray_norm = (delta_gray_norm * 2.0) - 1.0

                    x_fake_list.append( torch.from_numpy(np.repeat(delta_gray[:, np.newaxis, :, :], 3, axis=1)).to(self.device) ) # difference map
                    x_fake_list.append( torch.from_numpy(np.repeat(delta_gray_norm[:, np.newaxis, :, :], 3, axis=1)).to(self.device) ) # localization thershold
                    x_fake_list.append( torch.from_numpy(np.repeat(loc[:, np.newaxis, :, :], 3, axis=1)).to(self.device) ) # localization kmeans
                    x_fake_list.append( torch.tanh(delta + x_real) ) # generated image

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))


    def test_pcam(self):

        """Translate images using Fixed-Point GAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        self.restore_model_id_resnet(self.eval_resnet_id_name)
        self.restore_model_tilde_resnet(self.eval_resnet_tilde_name)

        
        # Set data loader.
        if self.dataset in ['PCam', 'CelebA']:
            data_loader = self.data_loader

        y_test_id = torch.zeros(0)
        y_pred_id = torch.zeros(0)  

        y_test_tilde = torch.zeros(0)
        y_pred_tilde = torch.zeros(0)  

        input_image = None    
        
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):
                x_real = x_real.to(self.device)
                x_real_tilde = x_real.clone()
                x_real_tilde = x_real_tilde.to(self.device)
                x_real_id = x_real.clone()
                x_real_id = x_real_id.to(self.device)
                c_trg = c_org.clone()
                c_trg_list = [c_trg.to(self.device)]
              
                # Translate images.
                x_fake_list = [x_real]
                for c_trg in c_trg_list:
                                        
                    # settings
                    h, w = 0, 0        # for raster image
                    nrows, ncols = len(c_trg), 10  # array of sub-plots

                    my_dpi = 96
             
                    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(1078/my_dpi, (108 * nrows)/my_dpi), dpi=my_dpi)

                    def get_grey_image(image):
                        image = np.array(image) 
                        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        return gray_image

                                                
                    def get_prediction_image(image, v):
                        image = np.array(image) 
                        # Fill image with color
                        if v == 1:
                            image[:] = (0, 200, 0)
                        else:
                            image[:] = (200, 0, 0)
                        return image

                    c_trg_tilde = (~c_trg.bool()).float()
                    for index, c in enumerate(c_trg):
                        # Save generated Tilde image

                        c = c.bool()
                        c_tilde = ~c

                        c = int(c.item())
                        c_tilde = int(c_tilde.item())

                        input_image = self.denorm(x_real[index].data.cpu())

                        delta = self.G(x_real, c_trg)
                        generated_correct_class_image = torch.tanh(delta[index] + x_real[index])
                        generated_correct_class_image = self.denorm(generated_correct_class_image.data.cpu())
                        x_real_tilde[index] = self.norm(generated_correct_class_image.data.cpu())

                        delta_tilde = self.G(x_real, c_trg_tilde)
                        generated_tilde_class_image = torch.tanh(delta_tilde[index] + x_real[index])
                        generated_tilde_class_image = self.denorm(generated_tilde_class_image.data.cpu())
                        x_real_id[index] = self.norm(generated_tilde_class_image.data.cpu())

                        difference_real_generated_image = np.abs(input_image - generated_correct_class_image)
                        difference_real_generated_tilde_image = np.abs(input_image - generated_tilde_class_image)
                        difference_generated_image = np.abs(generated_correct_class_image - generated_tilde_class_image)

                        axi = ax.flat

                        ax_col_one = axi[index * ncols]
                        ax_col_two = axi[index * ncols+1]
                        ax_col_three = axi[index * ncols+2]
                        ax_col_four = axi[index * ncols+3]
                        ax_col_five = axi[index * ncols+4]
                        ax_col_six = axi[index * ncols+5]
                        ax_col_seven = axi[index * ncols+6]       
                        ax_col_eight = axi[index * ncols+7]

                        input_image = transforms.ToPILImage()(input_image).convert("RGB")
                        generated_correct_class_image = transforms.ToPILImage()(generated_correct_class_image).convert("RGB")
                        generated_tilde_class_image = transforms.ToPILImage()(generated_tilde_class_image).convert("RGB")
                        difference_real_generated_image = transforms.ToPILImage()(difference_real_generated_image).convert("RGB")
                        difference_real_generated_tilde_image = transforms.ToPILImage()(difference_real_generated_tilde_image).convert("RGB")
                        difference_generated_image = transforms.ToPILImage()(difference_generated_image).convert("RGB")

                        ax_col_one.imshow(input_image, aspect='equal')
                        ax_col_two.imshow(generated_correct_class_image, aspect='equal')
                        ax_col_three.imshow(generated_tilde_class_image, aspect='equal')
                        ax_col_four.imshow(get_grey_image(difference_real_generated_image), aspect='equal', cmap='jet')
                        ax_col_five.imshow(get_grey_image(difference_real_generated_tilde_image), aspect='equal', cmap='jet')
                        ax_col_six.imshow(get_grey_image(difference_generated_image), aspect='equal', cmap='jet')
                        ax_col_seven.imshow(difference_real_generated_image, aspect='equal')
                        ax_col_eight.imshow(difference_real_generated_tilde_image, aspect='equal')

                        ax_col_one.text(4,5, c, color='white', va="center", backgroundcolor='black')

                        ax_col_one.set_axis_off()
                        ax_col_two.set_axis_off()
                        ax_col_three.set_axis_off()
                        ax_col_four.set_axis_off()
                        ax_col_five.set_axis_off()
                        ax_col_six.set_axis_off()
                        ax_col_seven.set_axis_off()
                        ax_col_eight.set_axis_off()

                        #result_generated_path = os.path.join(self.result_dir,  'generated/{}'.format(c_tilde))
                        #if not os.path.exists(result_generated_path):
                            #os.makedirs(result_generated_path)  

                        #result_generated_path = os.path.join(result_generated_path, '{}_{}-images.png'.format(i+1, index+1))   
                        #generated_tilde_class_image.save(result_generated_path) 
                        #save_image(self.denorm(torch.tanh(delta_tilde[index] + x_real[index]).data.cpu()), result_generated_path, nrow=1, padding=0)





                    resnet_output_tilde = self.resnet_tilde(x_real_tilde.to("cpu")).to(self.device)
                    predictions_tilde = resnet_output_tilde >= 0.5
                    abs_diff = abs(predictions_tilde.to("cpu").float() - c_trg_tilde[:, :1].to("cpu").float())
                    for index, c in enumerate(c_trg[:, 0]):
                        axi = ax.flat      
                        ax_col_nine = axi[index * ncols+8]
                        if abs_diff[index].item() == 1:
                            ax_col_nine.imshow(get_prediction_image(input_image, 1), aspect='equal')
                        else:
                            ax_col_nine.imshow(get_prediction_image(input_image, 0), aspect='equal')
                        ax_col_nine.set_axis_off()

                    y_test_tilde = torch.cat([y_test_tilde, c_trg[:, :1].to("cpu").int()], 0)
                    y_pred_tilde = torch.cat([y_pred_tilde, predictions_tilde.to("cpu").int()], 0)
                    cm_tilde = metrics.confusion_matrix(y_test_tilde, y_pred_tilde)
                    self.confusion_metrics(cm_tilde, 'tilde')

                    resnet_output_id = self.resnet_id(x_real_id.to("cpu")).to(self.device)
                    predictions_id = resnet_output_id >= 0.5
                    abs_diff = abs(predictions_id.to("cpu").float() - c_trg[:, :1].to("cpu").float())
                    for index, c in enumerate(c_trg[:, 0]):
                        axi = ax.flat      
                        ax_col_ten = axi[index * ncols+9]
                        if abs_diff[index].item() == 1:
                            ax_col_ten.imshow(get_prediction_image(input_image, 1), aspect='equal')
                        else:
                            ax_col_ten.imshow(get_prediction_image(input_image, 0), aspect='equal')
                        ax_col_ten.set_axis_off()

                    y_test_id = torch.cat([y_test_id, c_trg_tilde[:, :1].to("cpu").int()], 0)
                    y_pred_id = torch.cat([y_pred_id, predictions_id.to("cpu").int()], 0)
                    cm_id = metrics.confusion_matrix(y_test_id, y_pred_id)
                    self.confusion_metrics(cm_id, 'id')
                    
                    print('Saving image {}'.format(i+1))
                    plt.tight_layout(True)
                    plt.gca().set_axis_off()
                    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
                    plt.margins(0,0)
                    result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                    plt.savefig(result_path, result_pathbbox_inches = 'tight', pad_inches = 0)
                    plt.close()
                    

                # Save the translated images.
               # x_concat = torch.cat(x_fake_list, dim=3)
                #result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
               # save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=1)
                #print('Saved real and fake images into {}...'.format(result_path))
            


    def test_celeba(self):

        """Translate images using Fixed-Point GAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        self.restore_model_id_resnet(self.eval_resnet_id_name)
        self.restore_model_tilde_resnet(self.eval_resnet_tilde_name)

        # Set data loader.
        if self.dataset in ['PCam', 'CelebA']:
            data_loader = self.data_loader

        y_test_tilde = torch.zeros(0)
        y_pred_tilde = torch.zeros(0)  
        y_test_id = torch.zeros(0)
        y_pred_id = torch.zeros(0)  

        input_image = None   
        
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):
                x_real = x_real.to(self.device)
                x_real_tilde = x_real.clone()
                x_real_tilde = x_real_tilde.to(self.device)
                x_real_id = x_real.clone()
                x_real_id = x_real_id.to(self.device)
                c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

                c_trg = c_org.clone()
                #c_trg[:, 0] = 0 # always to healthy
                
              
                # Translate images.
                x_fake_list = [x_real]
                for c_trg in c_trg_list:

                    c_trg_tilde = (~c_trg.bool()).float()

                    if self.eval_dataset == 'train':
                            resnet_output_tilde = self.resnet_tilde(x_real.to("cpu")).to(self.device)
                            predictions_tilde = resnet_output_tilde >= 0.5
                            y_test_tilde = torch.cat([y_test_tilde, c_trg_tilde[:, :1].to("cpu").int()], 0)
                            y_pred_tilde = torch.cat([y_pred_tilde, predictions_tilde.to("cpu").int()], 0)
                            cm_tilde = metrics.confusion_matrix(y_test_tilde, y_pred_tilde)
                            self.confusion_metrics(cm_tilde, 'train-tilde')

                            resnet_output_id = self.resnet_id(x_real.to("cpu")).to(self.device)
                            predictions_id = resnet_output_id >= 0.5
                            y_test_id = torch.cat([y_test_id, c_trg_tilde[:, :1].to("cpu").int()], 0)
                            y_pred_id = torch.cat([y_pred_id, predictions_id.to("cpu").int()], 0)
                            cm_id = metrics.confusion_matrix(y_test_id, y_pred_id)
                            self.confusion_metrics(cm_id, 'train-id')

                            print('{}%'.format(100* (i/len(data_loader))))

                            continue  
                                        
                    # settings
                    h, w = 0, 0        # for raster image
                    nrows, ncols = len(c_trg), 10  # array of sub-plots


                    my_dpi = 96
             
                    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(1078/my_dpi, (108 * nrows)/my_dpi), dpi=my_dpi)

                    def get_grey_image(image):
                        image = np.array(image) 
                        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        return gray_image

                                                
                    def get_prediction_image(image, v):
                        image = np.array(image) 
                        # Fill image with color
                        if v == 1:
                            image[:] = (0, 200, 0)
                        else:
                            image[:] = (200, 0, 0)
                        return image
   

                  

                    for index, c in enumerate(c_trg):
                        # Save generated Tilde image

                        

                        c = c.bool()
                        c_tilde = ~c

                        c = int(c.item())
                        c_tilde = int(c_tilde.item())

                        input_image = self.denorm(x_real[index].data.cpu())

                        delta = self.G(x_real, c_trg)
                        generated_correct_class_image = torch.tanh(delta[index] + x_real[index])
                        generated_correct_class_image = self.denorm(generated_correct_class_image.data.cpu())

                        x_real_tilde[index] = self.norm(generated_correct_class_image.data.cpu())


                        delta_tilde = self.G(x_real, c_trg_tilde)
                        generated_tilde_class_image = torch.tanh(delta_tilde[index] + x_real[index])
                        generated_tilde_class_image = self.denorm(generated_tilde_class_image.data.cpu())
                        x_real_id[index] = self.norm(generated_tilde_class_image.data.cpu())

                        difference_real_generated_image = np.abs(input_image - generated_correct_class_image)
                        difference_real_generated_tilde_image = np.abs(input_image - generated_tilde_class_image)
                        difference_generated_image = np.abs(generated_correct_class_image - generated_tilde_class_image)

                        axi = ax.flat

                        ax_col_one = axi[index * ncols]
                        ax_col_two = axi[index * ncols+1]
                        ax_col_three = axi[index * ncols+2]
                        ax_col_four = axi[index * ncols+3]
                        ax_col_five = axi[index * ncols+4]
                        ax_col_six = axi[index * ncols+5]
                        ax_col_seven = axi[index * ncols+6]       
                        ax_col_eight = axi[index * ncols+7]

                        input_image = transforms.ToPILImage()(input_image).convert("RGB")
                        generated_correct_class_image = transforms.ToPILImage()(generated_correct_class_image).convert("RGB")
                        generated_tilde_class_image = transforms.ToPILImage()(generated_tilde_class_image).convert("RGB")
                        difference_real_generated_image = transforms.ToPILImage()(difference_real_generated_image).convert("RGB")
                        difference_real_generated_tilde_image = transforms.ToPILImage()(difference_real_generated_tilde_image).convert("RGB")
                        difference_generated_image = transforms.ToPILImage()(difference_generated_image).convert("RGB")

                        ax_col_one.imshow(input_image, aspect='equal')
                        ax_col_two.imshow(generated_correct_class_image, aspect='equal')
                        ax_col_three.imshow(generated_tilde_class_image, aspect='equal')
                        ax_col_four.imshow(get_grey_image(difference_real_generated_image), aspect='equal', cmap='jet')
                        ax_col_five.imshow(get_grey_image(difference_real_generated_tilde_image), aspect='equal', cmap='jet')
                        ax_col_six.imshow(get_grey_image(difference_generated_image), aspect='equal', cmap='jet')
                        ax_col_seven.imshow(difference_real_generated_image, aspect='equal')
                        ax_col_eight.imshow(difference_real_generated_tilde_image, aspect='equal')

                        ax_col_one.text(4,5, c, color='white', va="center", backgroundcolor='black')

                        ax_col_one.set_axis_off()
                        ax_col_two.set_axis_off()
                        ax_col_three.set_axis_off()
                        ax_col_four.set_axis_off()
                        ax_col_five.set_axis_off()
                        ax_col_six.set_axis_off()
                        ax_col_seven.set_axis_off()
                        ax_col_eight.set_axis_off()

                        #result_generated_path = os.path.join(self.result_dir,  'generated/{}'.format(c_tilde))
                        #if not os.path.exists(result_generated_path):
                        #    os.makedirs(result_generated_path)  

                        #result_generated_path = os.path.join(result_generated_path, '{}_{}-images.png'.format(i+1, index+1))   
                        #generated_tilde_class_image.save(result_generated_path) 
                        #save_image(self.denorm(torch.tanh(delta_tilde[index] + x_real[index]).data.cpu()), result_generated_path, nrow=1, padding=0)

                    if input_image is not None:
                        resnet_output_tilde = self.resnet_tilde(x_real_tilde.to("cpu")).to(self.device)
                        predictions_tilde = resnet_output_tilde >= 0.5
                        abs_diff = abs(predictions_tilde.to("cpu").float() - c_trg_tilde[:, :1].to("cpu").float())
                        for index, c in enumerate(c_trg[:, 0]):
                            axi = ax.flat      
                            ax_col_nine = axi[index * ncols+8]
                            if abs_diff[index].item() == 1:
                                ax_col_nine.imshow(get_prediction_image(input_image, 1), aspect='equal')
                            else:
                                ax_col_nine.imshow(get_prediction_image(input_image, 0), aspect='equal')
                            ax_col_nine.set_axis_off()

                        y_test_tilde = torch.cat([y_test_tilde, c_trg[:, :1].to("cpu").int()], 0)
                        y_pred_tilde = torch.cat([y_pred_tilde, predictions_tilde.to("cpu").int()], 0)
                        cm_tilde = metrics.confusion_matrix(y_test_tilde, y_pred_tilde)
                        self.confusion_metrics(cm_tilde, 'tilde')

                        resnet_output_id = self.resnet_id(x_real_id.to("cpu")).to(self.device)
                        predictions_id = resnet_output_id >= 0.5
                        abs_diff = abs(predictions_id.to("cpu").float() - c_trg[:, :1].to("cpu").float())
                        for index, c in enumerate(c_trg[:, 0]):
                            axi = ax.flat      
                            ax_col_ten = axi[index * ncols+9]
                            if abs_diff[index].item() == 1:
                                ax_col_ten.imshow(get_prediction_image(input_image, 1), aspect='equal')
                            else:
                                ax_col_ten.imshow(get_prediction_image(input_image, 0), aspect='equal')
                            ax_col_ten.set_axis_off()

                        y_test_id = torch.cat([y_test_id, c_trg_tilde[:, :1].to("cpu").int()], 0)
                        y_pred_id = torch.cat([y_pred_id, predictions_id.to("cpu").int()], 0)
                        cm_id = metrics.confusion_matrix(y_test_id, y_pred_id)
                        self.confusion_metrics(cm_id, 'id')    



                        print('Saving image {}'.format(i+1))
                        plt.tight_layout(True)
                        plt.gca().set_axis_off()
                        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
                        plt.margins(0,0)
                        result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                        plt.savefig(result_path, result_pathbbox_inches = 'tight', pad_inches = 0)
                        
                        # plt.show()
                    plt.close()    


    def test_celeba_multi(self):

            """Translate images using Fixed-Point GAN trained on a single dataset."""
            # Load the trained generator.
            self.restore_model(self.test_iters)
            self.restore_model_id_resnet(self.eval_resnet_id_name)
            self.restore_model_tilde_resnet(self.eval_resnet_tilde_name)

            # Set data loader.
            if self.dataset in ['PCam', 'CelebA']:
                data_loader = self.data_loader  

            y_test_tilde = torch.zeros(0)
            y_pred_tilde = torch.zeros(0)
            y_test_id = torch.zeros(0)
            y_pred_id = torch.zeros(0)

            input_image = None   

            with torch.no_grad():
                for i, (x_real, c_org) in enumerate(data_loader):
                    x_real_tilde = x_real.clone()
                    x_real_tilde = x_real_tilde.to(self.device)
                    x_real_id = x_real.clone()
                    x_real_id = x_real_id.to(self.device)
                    x_real = x_real.to(self.device)
                    c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

                   
              
                    #print(pred)

                    # Translate images.
                    x_fake_list = [x_real]
                    for c_index, c_trg in enumerate(c_trg_list):                        
                
                        if c_index is not 0:
                            continue

                       

                        # Tilde glasses only.
                        c_trg_tilde = c_trg.clone()
                        for c_trg_index, c_trg_t in enumerate(c_trg_tilde):
                            for c_trg_index2, c_trg_2 in enumerate(c_trg_t):
                                if c_trg_index2 == 0:
                                    c_trg_tilde[c_trg_index][c_trg_index2] = (~c_trg_2.bool()).float()


                        # Tilde everything
                        #c_trg_tilde = (~c_trg.bool()).float()
            
                        if self.eval_dataset == 'train':
                            resnet_output_tilde = self.resnet_tilde(x_real.to("cpu")).to(self.device)
                            predictions_tilde = resnet_output_tilde >= 0.5
                            y_test_tilde = torch.cat([y_test_tilde, c_trg_tilde[:, :1].to("cpu").int()], 0)
                            y_pred_tilde = torch.cat([y_pred_tilde, predictions_tilde.to("cpu").int()], 0)
                            cm_tilde = metrics.confusion_matrix(y_test_tilde, y_pred_tilde)
                            self.confusion_metrics(cm_tilde, 'train-tilde')

                            resnet_output_id = self.resnet_id(x_real.to("cpu")).to(self.device)
                            predictions_id = resnet_output_id >= 0.5
                            y_test_id = torch.cat([y_test_id, c_trg_tilde[:, :1].to("cpu").int()], 0)
                            y_pred_id = torch.cat([y_pred_id, predictions_id.to("cpu").int()], 0)
                            cm_id = metrics.confusion_matrix(y_test_id, y_pred_id)
                            self.confusion_metrics(cm_id, 'train-id')

                            print('{}%'.format(100* (i/len(data_loader))))

                            continue  
                                    
                        # settings
                        h, w = 0, 0        # for raster image
                        nrows, ncols = len(c_trg), 10  # array of sub-plots

                        #print(len(c_trg))

                        my_dpi = 96
                
                        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(1078/my_dpi, (108 * nrows)/my_dpi), dpi=my_dpi)

                        def get_grey_image(image):
                            image = np.array(image) 
                            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                            return gray_image

                        def get_prediction_image(image, v):
                            image = np.array(image) 
                            # Fill image with color
                            if v == 1:
                                image[:] = (0, 200, 0)
                            else:
                                image[:] = (200, 0, 0)

                            return image    


                        for index, c in enumerate(c_trg[:, 0]):
                            # Save generated Tilde image

                            input_image = self.denorm(x_real[index].data.cpu())
                            
                            delta = self.G(x_real, c_trg)
                            delta_tilde = self.G(x_real, c_trg_tilde)

                            c = c.bool()
                            c_tilde = ~c

                            c = int(c.item())
                            c_tilde = int(c_tilde.item())


                            generated_correct_class_image = torch.tanh(delta[index] + x_real[index])
                            generated_correct_class_image = self.denorm(generated_correct_class_image.data.cpu())
                            x_real_tilde[index] = self.norm(generated_correct_class_image.data.cpu())


                            generated_tilde_class_image = torch.tanh(delta_tilde[index] + x_real[index])
                            generated_tilde_class_image = self.denorm(generated_tilde_class_image.data.cpu())
                            x_real_id[index] = self.norm(generated_tilde_class_image.data.cpu())

                            difference_real_generated_image = np.abs(input_image - generated_correct_class_image)
                            difference_real_generated_tilde_image = np.abs(input_image - generated_tilde_class_image)
                            difference_generated_image = np.abs(generated_correct_class_image - generated_tilde_class_image)


                            axi = ax.flat

                            ax_col_one = axi[index * ncols]
                            ax_col_two = axi[index * ncols+1]
                            ax_col_three = axi[index * ncols+2]
                            ax_col_four = axi[index * ncols+3]
                            ax_col_five = axi[index * ncols+4]
                            ax_col_six = axi[index * ncols+5]
                            ax_col_seven = axi[index * ncols+6]       
                            ax_col_eight = axi[index * ncols+7]

                            input_image = transforms.ToPILImage()(input_image).convert("RGB")
                            generated_correct_class_image = transforms.ToPILImage()(generated_correct_class_image).convert("RGB")
                            generated_tilde_class_image = transforms.ToPILImage()(generated_tilde_class_image).convert("RGB")
                            
                            difference_real_generated_image = transforms.ToPILImage()(difference_real_generated_image).convert("RGB")
                            difference_real_generated_tilde_image = transforms.ToPILImage()(difference_real_generated_tilde_image).convert("RGB")
                            difference_generated_image = transforms.ToPILImage()(difference_generated_image).convert("RGB")

                            ax_col_one.imshow(input_image, aspect='equal')
                            ax_col_two.imshow(generated_correct_class_image, aspect='equal')
                            ax_col_three.imshow(generated_tilde_class_image, aspect='equal')
                            ax_col_four.imshow(get_grey_image(difference_real_generated_image), aspect='equal', cmap='jet')
                            ax_col_five.imshow(get_grey_image(difference_real_generated_tilde_image), aspect='equal', cmap='jet')
                            ax_col_six.imshow(get_grey_image(difference_generated_image), aspect='equal', cmap='jet')
                            ax_col_seven.imshow(difference_real_generated_image, aspect='equal')
                            ax_col_eight.imshow(difference_real_generated_tilde_image, aspect='equal')

                            ax_col_one.text(4,5, c, color='white', va="center", backgroundcolor='black')

                            ax_col_one.set_axis_off()
                            ax_col_two.set_axis_off()
                            ax_col_three.set_axis_off()
                            ax_col_four.set_axis_off()
                            ax_col_five.set_axis_off()
                            ax_col_six.set_axis_off()
                            ax_col_seven.set_axis_off()
                            ax_col_eight.set_axis_off()

                            #result_generated_path = os.path.join(self.result_dir,  'generated/{}'.format(c_tilde))
                            #if not os.path.exists(result_generated_path):
                            #    os.makedirs(result_generated_path)  

                            #result_generated_path = os.path.join(result_generated_path, '{}_{}-images.png'.format(i+1, index+1))   
                            #generated_tilde_class_image.save(result_generated_path) 
                            #save_image(self.denorm(torch.tanh(delta_tilde[index] + x_real[index]).data.cpu()), result_generated_path, nrow=1, padding=0)

                        if input_image is not None:

                            resnet_output_tilde = self.resnet_tilde(x_real_tilde.to("cpu")).to(self.device)
                            predictions_tilde = resnet_output_tilde >= 0.5
                            abs_diff = abs(predictions_tilde.to("cpu").float() - c_trg_tilde[:, :1].to("cpu").float())
                            for index, c in enumerate(c_trg[:, 0]):
                                axi = ax.flat      
                                ax_col_nine = axi[index * ncols+8]
                                if abs_diff[index].item() == 1:
                                    ax_col_nine.imshow(get_prediction_image(input_image, 1), aspect='equal')
                                else:
                                    ax_col_nine.imshow(get_prediction_image(input_image, 0), aspect='equal')
                                ax_col_nine.set_axis_off()

                            y_test_tilde = torch.cat([y_test_tilde, c_trg[:, :1].to("cpu").int()], 0)
                            y_pred_tilde = torch.cat([y_pred_tilde, predictions_tilde.to("cpu").int()], 0)
                            cm_tilde = metrics.confusion_matrix(y_test_tilde, y_pred_tilde)
                            self.confusion_metrics(cm_tilde, 'tilde')

                            resnet_output_id = self.resnet_id(x_real_id.to("cpu")).to(self.device)
                            predictions_id = resnet_output_id >= 0.5
                            abs_diff = abs(predictions_id.to("cpu").float() - c_trg[:, :1].to("cpu").float())
                            for index, c in enumerate(c_trg[:, 0]):
                                axi = ax.flat      
                                ax_col_ten = axi[index * ncols+9]
                                if abs_diff[index].item() == 1:
                                    ax_col_ten.imshow(get_prediction_image(input_image, 1), aspect='equal')
                                else:
                                    ax_col_ten.imshow(get_prediction_image(input_image, 0), aspect='equal')
                                ax_col_ten.set_axis_off()

                            y_test_id = torch.cat([y_test_id, c_trg_tilde[:, :1].to("cpu").int()], 0)
                            y_pred_id = torch.cat([y_pred_id, predictions_id.to("cpu").int()], 0)
                            cm_id = metrics.confusion_matrix(y_test_id, y_pred_id)
                            self.confusion_metrics(cm_id, 'id')

                            print('Saving image {}'.format(i+1))
                            plt.tight_layout(True)
                            plt.gca().set_axis_off()
                            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
                            plt.margins(0,0)
                            result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                            plt.savefig(result_path, result_pathbbox_inches = 'tight', pad_inches = 0)
                        plt.close()
                            # plt.show()                    