import numpy as np
from visdom import Visdom

class Logger(object):
    def __init__(self, vis_screen='draw_lines', visualize=True):
        if visualize:
            self.viz = VisdomPlotter(env_name=vis_screen)
        self.hist_base = []
        self.hist_res = []
        self.hist_aux = []
        self.hist_joint = []

    def log_iteration(self, epoch, batch, loss_base, loss_res, loss_aux, loss_joint):
        print("Epoch: {}, Batch: {}, l_joint= {:0.3f}, l_base= {:0.3f}, l_res= {:0.3f}, l_aux = {:0.3f}".format(
              epoch, batch, loss_joint, loss_base, loss_res, loss_aux))
        self.hist_base.append(loss_base)
        self.hist_res.append(loss_res)
        self.hist_aux.append(loss_aux)
        self.hist_joint.append(loss_joint)

    def plot_epoch(self, epoch):
        self.viz.plot('L_base', 'train', epoch, np.array(self.hist_base).mean())
        self.viz.plot('L_res', 'train', epoch, np.array(self.hist_res).mean())
        self.viz.plot('L_aux', 'train', epoch, np.array(self.hist_aux).mean())
        self.viz.plot('Total', 'train', epoch, np.array(self.hist_joint).mean())
        self.hist_base = []
        self.hist_res = []
        self.hist_joint = []
        self.hist_aux = []

    def draw(self, pred, sketch, corrected, original):
        self.viz.draw('original', original, 'Original Image')
        self.viz.draw('target', sketch, 'Target')
        self.viz.draw('generated', pred, 'Generated Image')
        self.viz.draw('corrected', corrected, 'Correction of the deteriorated image')


class VisdomPlotter(object):

    def __init__(self, env_name='inoue'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, x, y, xlabel='epoch'):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel=xlabel,
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), update="append", 
                env=self.env, win=self.plots[var_name], name=split_name)

    def draw(self, var_name, image, caption):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.image(image, env=self.env,
                opts={"store_history":True, "caption":caption})
        else:
            self.viz.image(image, env=self.env, win=self.plots[var_name],
                opts={"store_history":True, "caption":caption})