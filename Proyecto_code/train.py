import torch
import argparse
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from models.generator import Generator
from models.discriminator import ImageDiscriminator
from models.discriminator import ObjectDiscriminator
from models.discriminator import add_sn
from coco_custom_mask import get_dataloader as get_dataloader_coco
# from data.vg_custom_mask import get_dataloader as get_dataloader_vg
from utils.model_saver import load_model, save_model, prepare_dir
from utils.data import imagenet_deprocess_batch
from utils.miscs import str2bool
import torch.backends.cudnn as cudnn
import yaml


def main(config):
    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    log_save_dir, model_save_dir, sample_save_dir, result_save_dir = prepare_dir(config['exp_name'])

    if config['dataset'] == 'vg': pass
        # data_loader, _ = get_dataloader_vg(batch_size=config['batch_size'], VG_DIR=config['vg_dir'])
    elif config['dataset'] == 'coco':
        data_loader, _ = get_dataloader_coco(batch_size=config['batch_size'], COCO_DIR=config['coco_dir'])
    vocab_num = data_loader.dataset.num_objects

    assert config['clstm_layers'] > 0

    netG = Generator(num_embeddings=vocab_num,
                     embedding_dim=config['embedding_dim'],
                     z_dim=config['z_dim'],
                     clstm_layers=config['clstm_layers']).to(device)
    netD_image = ImageDiscriminator(conv_dim=config['embedding_dim']).to(device)
    netD_object = ObjectDiscriminator(n_class=vocab_num).to(device)

    netD_image = add_sn(netD_image)
    netD_object = add_sn(netD_object)

    netG_optimizer = torch.optim.Adam(netG.parameters(), config['learning_rate'], [0.5, 0.999])
    netD_image_optimizer = torch.optim.Adam(netD_image.parameters(), config['learning_rate'], [0.5, 0.999])
    netD_object_optimizer = torch.optim.Adam(netD_object.parameters(), config['learning_rate'], [0.5, 0.999])

    start_iter = load_model(netG, model_dir=model_save_dir, appendix='netG', iter=config['resume_iter'])
    _ = load_model(netD_image, model_dir=model_save_dir, appendix='netD_image', iter=config['resume_iter'])
    _ = load_model(netD_object, model_dir=model_save_dir, appendix='netD_object', iter=config['resume_iter'])

    data_iter = iter(data_loader)

    if start_iter < config['niter']:

        if config['use_tensorboard']: writer = SummaryWriter(log_save_dir)

        for i in range(start_iter, config['niter']):
            try:
                batch = next(data_iter)
            except:
                data_iter = iter(data_loader)
                batch = next(data_iter)

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #
            imgs, objs, boxes, masks, obj_to_img = batch
            z = torch.randn(objs.size(0), config['z_dim'])
            imgs, objs, boxes, masks, obj_to_img, z = imgs.to(device), objs.to(device), boxes.to(device), \
                                                      masks.to(device), obj_to_img, z.to(device)

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Generate fake image
            output = netG(imgs, objs, boxes, masks, obj_to_img, z)
            crops_input, crops_input_rec, crops_rand, img_rec, img_rand, mu, logvar, z_rand_rec = output

            # Compute image adv loss with fake images.
            out_logits = netD_image(img_rec.detach())
            d_image_adv_loss_fake_rec = F.binary_cross_entropy_with_logits(out_logits, torch.full_like(out_logits, 0))

            out_logits = netD_image(img_rand.detach())
            d_image_adv_loss_fake_rand = F.binary_cross_entropy_with_logits(out_logits, torch.full_like(out_logits, 0))

            d_image_adv_loss_fake = 0.5 * d_image_adv_loss_fake_rec + 0.5 * d_image_adv_loss_fake_rand

            # Compute image src loss with real images rec.
            out_logits = netD_image(imgs)
            d_image_adv_loss_real = F.binary_cross_entropy_with_logits(out_logits, torch.full_like(out_logits, 1))

            # Compute object sn adv loss with fake rec crops
            out_logits, _ = netD_object(crops_input_rec.detach(), objs)
            g_object_adv_loss_rec = F.binary_cross_entropy_with_logits(out_logits, torch.full_like(out_logits, 0))

            # Compute object sn adv loss with fake rand crops
            out_logits, _ = netD_object(crops_rand.detach(), objs)
            d_object_adv_loss_fake_rand = F.binary_cross_entropy_with_logits(out_logits, torch.full_like(out_logits, 0))

            d_object_adv_loss_fake = 0.5 * g_object_adv_loss_rec + 0.5 * d_object_adv_loss_fake_rand

            # Compute object sn adv loss with real crops.
            out_logits_src, out_logits_cls = netD_object(crops_input.detach(), objs)
            d_object_adv_loss_real = F.binary_cross_entropy_with_logits(out_logits_src, torch.full_like(out_logits_src, 1))
            d_object_cls_loss_real = F.cross_entropy(out_logits_cls, objs)

            # Backward and optimizloe.
            d_loss = 0
            d_loss += config['lambda_img_adv'] * (d_image_adv_loss_fake + d_image_adv_loss_real)
            d_loss += config['lambda_obj_adv'] * (d_object_adv_loss_fake + d_object_adv_loss_real)
            d_loss += config['lambda_obj_cls'] * d_object_cls_loss_real

            netD_image.zero_grad()
            netD_object.zero_grad()

            d_loss.backward()

            netD_image_optimizer.step()
            netD_object_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss'] = d_loss.item()
            loss['D/image_adv_loss_real'] = d_image_adv_loss_real.item()
            loss['D/image_adv_loss_fake'] = d_image_adv_loss_fake.item()
            loss['D/object_adv_loss_real'] = d_object_adv_loss_real.item()
            loss['D/object_adv_loss_fake'] = d_object_adv_loss_fake.item()
            loss['D/object_cls_loss_real'] = d_object_cls_loss_real.item()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            # Generate fake image
            output = netG(imgs, objs, boxes, masks, obj_to_img, z)
            crops_input, crops_input_rec, crops_rand, img_rec, img_rand, mu, logvar, z_rand_rec = output

            # reconstruction loss of ae and img
            g_img_rec_loss = torch.abs(img_rec - imgs).mean()
            g_z_rec_loss = torch.abs(z_rand_rec - z).mean()

            # kl loss
            kl_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
            g_kl_loss = torch.sum(kl_element).mul_(-0.5)

            # Compute image adv loss with fake images.
            out_logits = netD_image(img_rec)
            g_image_adv_loss_fake_rec = F.binary_cross_entropy_with_logits(out_logits, torch.full_like(out_logits, 1))

            out_logits = netD_image(img_rand)
            g_image_adv_loss_fake_rand = F.binary_cross_entropy_with_logits(out_logits, torch.full_like(out_logits, 1))

            g_image_adv_loss_fake = 0.5 * g_image_adv_loss_fake_rec + 0.5 * g_image_adv_loss_fake_rand

            # Compute object adv loss with fake images.
            out_logits_src, out_logits_cls = netD_object(crops_input_rec, objs)
            g_object_adv_loss_rec = F.binary_cross_entropy_with_logits(out_logits_src, torch.full_like(out_logits_src, 1))
            g_object_cls_loss_rec = F.cross_entropy(out_logits_cls, objs)

            out_logits_src, out_logits_cls = netD_object(crops_rand, objs)
            g_object_adv_loss_rand = F.binary_cross_entropy_with_logits(out_logits_src, torch.full_like(out_logits_src, 1))
            g_object_cls_loss_rand = F.cross_entropy(out_logits_cls, objs)

            g_object_adv_loss = 0.5 * g_object_adv_loss_rec + 0.5 * g_object_adv_loss_rand
            g_object_cls_loss = 0.5 * g_object_cls_loss_rec + 0.5 * g_object_cls_loss_rand

            # Backward and optimize.
            g_loss = 0
            g_loss += config['lambda_img_rec'] * g_img_rec_loss
            g_loss += config['lambda_z_rec'] * g_z_rec_loss
            g_loss += config['lambda_img_adv'] * g_image_adv_loss_fake
            g_loss += config['lambda_obj_adv'] * g_object_adv_loss
            g_loss += config['lambda_obj_cls'] * g_object_cls_loss
            g_loss += config['lambda_kl'] * g_kl_loss

            netG.zero_grad()
            g_loss.backward()
            netG_optimizer.step()

            loss['G/loss'] = g_loss.item()
            loss['G/image_adv_loss'] = g_image_adv_loss_fake.item()
            loss['G/object_adv_loss'] = g_object_adv_loss.item()
            loss['G/object_cls_loss'] = g_object_cls_loss.item()
            loss['G/rec_img'] = g_img_rec_loss.item()
            loss['G/rec_z'] = g_z_rec_loss.item()
            loss['G/kl'] = g_kl_loss.item()

            # =================================================================================== #
            #                               4. Log                                                #
            # =================================================================================== #
            if (i + 1) % config['log_step'] == 0:
                log = 'iter [{:06d}/{:06d}]'.format(i+1, config['niter'])
                for tag, roi_value in loss.items():
                    log += ", {}: {:.4f}".format(tag, roi_value)
                print(log)

            if (i + 1) % config['tensorboard_step'] == 0 and config['use_tensorboard']:
                for tag, roi_value in loss.items():
                    writer.add_scalar(tag, roi_value, i+1)
                # print(imagenet_deprocess_batch(crops_input).shape)
                # print(imagenet_deprocess_batch(crops_input_rec).shape)
                # print(imagenet_deprocess_batch(crops_rand).shape)
                # print(imagenet_deprocess_batch(imgs).shape)
                # print(imagenet_deprocess_batch(img_rec).shape)
                # print(imagenet_deprocess_batch(img_rand).shape)
                writer.add_image('Result/crop_real', imagenet_deprocess_batch(crops_input).float() / 255, i + 1, dataformats='NCHW')
                writer.add_image('Result/crop_real_rec', imagenet_deprocess_batch(crops_input_rec).float() / 255, i + 1, dataformats='NCHW')
                writer.add_image('Result/crop_rand', imagenet_deprocess_batch(crops_rand).float() / 255, i + 1, dataformats='NCHW')
                writer.add_image('Result/img_real', imagenet_deprocess_batch(imgs).float() / 255, i + 1, dataformats='NCHW')
                writer.add_image('Result/img_real_rec', imagenet_deprocess_batch(img_rec).float() / 255, i + 1, dataformats='NCHW')
                writer.add_image('Result/img_fake_rand', imagenet_deprocess_batch(img_rand).float() / 255, i + 1, dataformats='NCHW')

            if (i + 1) % config['save_step'] == 0:
                save_model(netG, model_dir=model_save_dir, appendix='netG', iter=i + 1, save_num=5, save_step=config['save_step'])
                save_model(netD_image, model_dir=model_save_dir, appendix='netD_image', iter=i + 1, save_num=5, save_step=config['save_step'])
                save_model(netD_object, model_dir=model_save_dir, appendix='netD_object', iter=i + 1, save_num=5, save_step=config['save_step'])

        if config['use_tensorboard']: writer.close()


if __name__ == '__main__':
    with open('config.yaml','r') as f:
        config = yaml.safe_load(f)

    config['exp_name'] = 'layout2im_{}'.format(config['dataset'])
    print(config)
    main(config)
