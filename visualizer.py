import visdom
import time
import torch
import numpy as np

class Visualizer(visdom.Visdom):
    '''
    封装了visdom的基本操作，但是你仍然可以通过`self.vis.function`
    或者`self.function`调用原生的visdom接口
    比如 
    self.text('hello visdom')
    self.histogram(t.randn(1000))
    self.line(t.arange(0, 10),t.arange(1, 11))
    '''

    def __init__(self, env='default', **kwargs):
        super(Visualizer, self).__init__(env=env, **kwargs)
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        '''
        修改visdom的配置
        '''
        self.__init__(env=env, **kwargs)

    def plot_many(self, d, **kwargs):
        '''
        一次plot多个
        @params d: dict (name, value) i.e. ('loss', 0.11)
        '''
        for k, v in d.items():
            self.plot(k, v, **kwargs)

    def img_many(self, d, **kwargs):
        for k, v in d.items():
            self.img(k, v, **kwargs)

    def plot(self, name, y, x=None, x_start=0, x_step=1, **kwargs):
        '''
        self.plot('loss', 1.00)
        '''
        if isinstance(y, torch.Tensor):
            y = y.cpu().item()

        if x is None:
            x = self.index.get(name, x_start)

        self.line(Y=np.array([y]), X=np.array([x]),
                  win=name,
                  opts=dict(title=name),
                  update=None if x == x_start else 'append',
                  **kwargs)
        self.index[name] = x + x_step

    def img(self, name, img_, **kwargs):
        '''
        self.img('input_img', t.Tensor(64, 64))
        self.img('input_imgs', t.Tensor(3, 64, 64))
        self.img('input_imgs', t.Tensor(100, 1, 64, 64))
        self.img('input_imgs', t.Tensor(100, 3, 64, 64), nrows=10)
        '''
        self.images(img_.cpu().numpy(),
                    win=name,
                    opts=dict(title=name),
                    **kwargs)

    def log(self, info, win='log_text'):
        '''
        self.log({'loss':1, 'lr':0.0001})
        '''
        self.log_text += ('{time} {info} <br>'.format(time=time.strftime('%m%d_%H%M%S'), info=info))
        self.text(self.log_text, win)
