import os
import math
from decimal import Decimal

import utility

import torch
from torch.autograd import Variable
#from tqdm import tqdm
import numpy as np

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.dim = args.dim
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test[0]
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if (self.args.load != '.' or self.args.resume == -1) and os.path.exists(os.path.join(ckp.dir, 'optimizer.pt')):
            print('Loading optimizer from', os.path.join(ckp.dir, 'optimizer.pt'))
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8

    def train(self):
        self.optimizer.step()
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, hr, idx_scale) in enumerate(self.loader_train):
            lr, hr = self.prepare([lr, hr])
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, idx_scale)
            loss = self.loss(sr, hr)
            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        epoch = self.scheduler.last_epoch
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                self.loader_test.dataset.set_scale(idx_scale)
#                tqdm_test = tqdm(self.loader_test, ncols=80)
#                for idx_img, (lr, hr, filename) in enumerate(tqdm_test):
                for idx_img, (lr, hr, filename) in enumerate(self.loader_test):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1) or (self.dim!=2)
                    if not no_eval:
                        lr, hr = self.prepare([lr, hr])
                    else:
                        lr = self.prepare([lr])[0]

                    sr = self.model(lr, idx_scale)
                    if self.dim == 2:
                        sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    if not no_eval:
                        if self.args.testfunc == 'PSNR':
                            eval_results = utility.calc_psnr(
                                sr, hr, scale, self.args.rgb_range,
                                benchmark=self.loader_test.dataset.benchmark
                            )
                        elif self.args.testfunc == 'mse':
                            eval_results = -torch.nn.functional.mse_loss(sr, hr)
                        elif self.args.testfunc == 'accuracy':
                            eval_results = 1-torch.nn.functional.l1_loss((sr>0).float(), hr)
                        else:
                            print('Unknown function for testing', self.args.testfunc)
                            eval_results = 0
                        eval_acc += eval_results
                    if self.dim != 2:
                       eval_acc+=-np.sqrt(np.mean((hr[0].data.cpu().numpy()-sr[0].data.cpu().numpy())**2))
                       #print('debug hr sr', hr, sr)
                       #print(hr-sr) #(sr - hr).pow(2).mean())
                       #eval_acc += (hr-sr).data.cpu().pow(2).mean()  #np.linalg.norm(hr[0].data.cpu().numpy()-sr[0].data.cpu().numpy())
                    save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(filename, save_list, scale)

                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\t{}: {:.4f} (Best: {:.4f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.args.testfunc,
                        self.ckp.log[-1, idx_scale],
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    )
                )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def prepare(self, l, volatile=False):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)
           
        return [_prepare(_l) for _l in l]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch
            return epoch >= self.args.epochs

