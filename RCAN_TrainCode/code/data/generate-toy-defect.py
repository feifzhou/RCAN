# import os

# from data import common

# import numpy as np
#import imageio

#import torch
# import torch.utils.data as data

# class defect():
#     def __init__(self, args, name='defect toy model', train=True, benchmark=False):
#         self.args = args
#         self.name = name
#         self.idx_scale = 0
#         self.train = train
#         self.benchmark = benchmark
#         self.n_in = args.n_colors
#         self.n_out = args.n_colors_out
#         self._cache = None
#         data_range = [list(map(int,r.split('-'))) for r in args.data_range.split('/')]
#         if self.train:
#             self.length=data_range[0][1]-data_range[0][0]
#             f = os.path.join(args.dir_data, 'train.npy')
#             if os.path.exists(f):
#                 self._cache = [(x[:1],x[1:]) for x in np.load(f)]
#         else:
#             self.length=data_range[1][1]-data_range[1][0]
#             f = os.path.join(args.dir_data, 'test.npy')
#             if os.path.exists(f):
#                 self._cache = [(x[:1],x[1:]) for x in np.load(f)]
#             else:
#                 self._cache = [generate(self.n_in) for i in range(self.length)]

#     def __getitem__(self, idx):
#         if not self._cache:
#             hr, lr = generate(self.n_in)
#         else:
#             hr, lr = self._cache[idx]
#         return lr, hr, str(idx)

#     def __len__(self):
#         return self.length

#     def set_scale(self, idx_scale):
#         pass

if __name__ == "__main__":
    import argparse
    import numpy as np
    from sklearn.cluster import DBSCAN
    from sklearn import metrics
    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_size", help="\"Ns Nchannel+1 Nx Ny\"", default="2 7 512 512")
    parser.add_argument("--diameter", help="diameter scaling factor. Default 1 for 512x512 images", type=float, default=1)
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('-o', help='output file', default='output.npy')
    options = parser.parse_args()

    def sigmoid(x):
        return 1/(1 + np.exp(-x)) 

    def f(Nxy, modes, cutoff, cutoff_final):
        xy= [np.arange(Ni) for Ni in Nxy]
        xy_grid = np.array(np.meshgrid(*xy)).transpose(np.roll(np.arange(len(xy)+1),-1))
        # print('debug xy grid', xy_grid.shape)
        vals = [[sigmoid(d-np.linalg.norm(xy_grid-r[None,None,:],axis=(2))) for r, d in mode] for mode in modes]
        # print('debug vals', np.array(vals).shape)
        vals = sigmoid(np.sum(vals, axis=1)-cutoff)
        # print('debug vals', np.array(vals).shape)
        val_out = (np.sign((np.prod(vals, axis=0, keepdims=True)-cutoff_final)*15)+1)/2
        # val_out = sigmoid((np.prod(vals, axis=0, keepdims=True)-cutoff_final)*15)
        return np.concatenate((val_out, vals), axis=0)

    Ninput = list(map(int, options.data_size.split()))
    dim = len(Ninput) - 2
    Ns=Ninput[0]
    Nspace=Ninput[2:]
    alldat=np.zeros(Ninput, dtype=np.float32)
    nchannel = Ninput[1]-1
    for i in range(Ns):
        nmode=np.random.choice(list(range(30, 35)))
        d1=options.diameter*Nspace[0]*0.034
        d2=options.diameter*Nspace[0]*0.089
        modes=[[[np.array([np.random.uniform(0, nx) for nx in Nspace]), np.random.uniform(d1, d2)] for i in range(nmode)] for _ in range(nchannel)]
        # print('debug i', i, modes)
        print(i)
        alldat[i]=f(Nspace, modes, 2.5, 0.002)
        if options.plot:
            for j in range(nchannel+1):
                plt.subplot(1, nchannel+1, j+1)
                plt.imshow(alldat[i, j])
            plt.show()
    np.save(options.o, alldat[...,None])