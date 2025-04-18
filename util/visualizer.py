import numpy as np
import sys
import time
from . import util
from torch.utils.tensorboard import SummaryWriter

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError

class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt, datasize):
        self.opt = opt  # cache the option
        self.datasize = datasize
        self.display_id = opt.display_id
        self.name = opt.name
        self.port = opt.display_port
        self.saved = False

        now = int(round(time.time() * 1000))
        data_time = time.strftime('%Y-%m-%d %H_%M_%S',time.localtime(now/1000))
        self.writer = SummaryWriter('runs/{}/{}'.format(opt.name, data_time))

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def display_current_results(self, visuals, step):
        """Display current results on tensorboard; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        if self.display_id > 0:  # show images in the browser using tensorboard
            imgs = []
            labels = []
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                imgs.append(image_numpy)
                labels.append(label)
                # self.writer.add_image(label + ' ', image_numpy.transpose([2, 0, 1]), global_step=step)
            cat_img = np.concatenate(imgs, axis=1)
            self.writer.add_image('-'.join(labels) + ' ', cat_img.transpose([2, 0, 1]), global_step=step)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message

        for k, loss in losses.items():
            self.writer.add_scalar('loss/' + k, loss, (epoch - 1) * self.datasize + iters)
