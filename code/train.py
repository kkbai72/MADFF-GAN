import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from tqdm import tqdm
from models.models import create_model
from util.visualizer import Visualizer
from torch.utils.tensorboard import SummaryWriter
import os
from util.metric import ImageMetrics


def get_config(config):
    import yaml
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.SafeLoader)

opt = TrainOptions().parse()
config = get_config(opt.config)
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

valdata_loader = CreateDataLoader(opt,True)
valdataset = valdata_loader.load_data()
print('#val images = %d' % len(valdata_loader))
metrics=ImageMetrics(opt.gpu_ids[0])

model = create_model(opt)
visualizer = Visualizer(opt)

writer = SummaryWriter(os.path.join(opt.checkpoints_dir,opt.name,"glo"))

total_steps = 0
start= int(opt.which_epoch) if opt.continue_train else 1


for epoch in range(start, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter=0
    iter_start_time = time.time()
    tbar = tqdm(dataset)
    for data in tbar:
        total_steps += data['A'].size(0)
        epoch_iter += data['A'].size(0)
        model.set_input(data)
        model.optimize_parameters(epoch)
        errors = model.get_current_errors(epoch)
        desc = 'Training  : Epoch %d, D_A  = %.5f, G_A  = %.5f, D_P = %.5f, G = %.5f, k= %.5f' % \
               (epoch, errors['D_A'], errors['G_A'], errors['D_P'],errors['G'],errors['k'])
        tbar.set_description(desc)
        visualizer.display_current_results(model.get_current_visuals(), epoch)

    t = (time.time() - iter_start_time)
    # visualizer.print_current_errors(epoch, epoch_iter, errors, t)
    # visualizer.display_current_results(model.get_current_visuals(), epoch)
    # print('saving the latest model (epoch %d, total_steps %d)' %
    #       (epoch, total_steps))
    model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        total_psnr=0
        total_ssim=0
        total_lpips=0
        for data in tqdm(valdataset):
            model.set_input(data)
            visuals = model.predict()
            B = visuals["real_B"]
            C = visuals["fake_B"]
            psnr,ssim,lpips = metrics(B,C)
            total_psnr += psnr
            total_ssim += ssim
            total_lpips += lpips
        total_psnr = total_psnr/len(valdataset)
        total_ssim = total_ssim/len(valdataset)
        total_lpips = total_lpips/len(valdataset)
        writer.add_scalar("val_psnr",total_psnr,epoch)
        writer.add_scalar("val_ssim",total_ssim,epoch)
        writer.add_scalar("val_lpips",total_lpips,epoch)
        print('epoch %d, iters %d,val psnr: %3f ssim: %3f lpips: %3f' %
              (epoch, total_steps, total_psnr, total_ssim, total_lpips))
        model.save(epoch)

    # print('End of epoch %d / %d \t Time Taken: %d sec' %
    #       (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    if opt.new_lr:
        if epoch == opt.niter:
            model.update_learning_rate()
        elif epoch == (opt.niter + 20):
            model.update_learning_rate()
        elif epoch == (opt.niter + 70):
            model.update_learning_rate()
        elif epoch == (opt.niter + 90):
            model.update_learning_rate()
            model.update_learning_rate()
            model.update_learning_rate()
            model.update_learning_rate()
    else:
        if epoch > opt.niter:
            model.update_learning_rate()
