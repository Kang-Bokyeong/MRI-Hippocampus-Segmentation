import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from Dataset import ImageDataset
from model import Model


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def main(args):
    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(in_size=3, out_size=1, hidden_size=256)
    model.apply(weights_init_normal)


    if args.checkpoint is not None:
        try:
            print("Load checkpoint, Train from {}".format(args.checkpoint))
            model.load_state_dict(torch.load(args.checkpoint), strict=False)  # resume checkpoint
        except Exception as e:
            print(e)

    # convert to 'cuda' or 'cpu'
    model.to(device)

    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.do_train:
        print('Train model...')

        # set dataset & dataloader
        train_dataset = ImageDataset(args.train_dir, args.seg_dir, target_input_size=args.img_size, train=True)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu)

        # Set optimizers
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        # Loss function
        criterion = torch.nn.BCELoss()
        criterion.to(device)

        model.train()

        for epoch in range(1, args.epochs + 1):
            lr = learning_rate_scheduler.get_lr()[0]  # set current learning rate to args

            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                img, label_img = batch[0], batch[1]
                image_output = model(img)

                # calculate loss
                loss = criterion(image_output, label_img).mean()

                # optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % 100 == 0:
                    print("[Epoch {}/{}] [Batch {}/{}] loss: {:.7f}, lr: {:.7f}".format(epoch, args.epochs, step, len(train_dataloader), loss.item(), lr))

            # save model
            if (epoch % args.checkpoint_save_interval) == 0:
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(), os.path.join(args.output_dir, "model_{}.pth".format(epoch)))

            # update learning rate
            learning_rate_scheduler.step()

    if args.do_predict:
        print('Evaluate model...')

        # set dataset & dataloader
        eval_dataset = ImageDataset(args.eval_dir, args.seg_dir, target_input_size=args.img_size)
        eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=1)

        # model to evaluate
        model.eval()

        for step, batch in enumerate(eval_dataloader):
            img, label_img = batch[0], batch[1]
            img, label_img = img.to(device), label_img.to(device)
            image_path_list, (top, bottom, right, left), (ori_w, ori_h) = batch[2], batch[3], batch[4]

            # infer model
            image_output = model(img)

            input_cpu = ((img.data.cpu().numpy() + 1.0) * 127.5).astype(np.uint8)
            label_cpu = (label_img.data.cpu().numpy() * 255).astype(np.uint8)
            output_cpu = (image_output.data.cpu().numpy() * 255).astype(np.uint8)

            for batch_index in range(output_cpu.shape[0]):
                input_image = input_cpu[batch_index][:3, :, :].transpose((1, 2, 0))
                output_image = np.squeeze(output_cpu[batch_index][0:1, :, :])
                label_image = np.squeeze(label_cpu[batch_index][0:1, :, :])

                # save mask
                output_ori_shape_image = output_image[top:args.img_size - bottom, left:args.img_size - right]
                out_filename = os.path.join(args.mask_dir, os.path.basename(image_path_list[batch_index]))
                Image.fromarray(output_ori_shape_image.astype(np.uint8)).resize((ori_w, ori_h)).save(out_filename)

                print('{} saved'.format(out_filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--train_dir", default='', type=str, help="The directory for training.")
    parser.add_argument("--eval_dir", default='', type=str, help="The directory for evaluation.")
    parser.add_argument("--seg_dir", default='', type=str, help="The directory for segmentation.")
    parser.add_argument("--output_dir", default='results', type=str, help="The output directory.")
    parser.add_argument("--lr", type=float, default=0.01, help="adam: learning rate")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--checkpoint", default=None, type=str, help="The directory of checkpoint file.")
    parser.add_argument("--checkpoint_save_interval", type=int, default=10, help="The checkpoint save interval.")
    parser.add_argument("--models_dir", type=str, default="results/models", help="The directory for models")
    args = parser.parse_args()

    # expanduser
    args.train_dir = os.path.expanduser(args.train_dir)
    args.eval_dir = os.path.expanduser(args.eval_dir)
    args.seg_dir = os.path.expanduser(args.seg_dir)
    args.output_dir = os.path.expanduser(args.output_dir)
    args.models_dir = os.path.expanduser(args.models_dir)


    args.mask_dir = os.path.join(args.output_dir, 'mask')

    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.mask_dir, exist_ok=True)

    # run main
    main(args)
