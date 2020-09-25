##### to use e.g. python generate-toy-defect.py  --data_size "1 5 400 400" --plot
if __name__ == "__main__":
    import argparse
    import numpy as np
    from sklearn.cluster import DBSCAN
    from sklearn import metrics
    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import StandardScaler
    import scipy.ndimage
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_size", help="\"Ns Nchannel+1 Nx Ny\"", default="2 7 512 512")
    parser.add_argument("--diameter", help="diameter scaling factor. Default 1 for 512x512 images", type=float, default=1)
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('-o', help='output file', default='output.npy')
    parser.add_argument('--method', help='pixel=pixel-by-pixel, or dilation', default='pixel')
    options = parser.parse_args()

    def sigmoid(x):
        return 1/(1 + np.exp(-x)) 

    def f(Nxy, modes, cutoff, cutoff_final, method='pixel'):
        xy= [np.arange(Ni) for Ni in Nxy]
        xy_grid = np.array(np.meshgrid(*xy)).transpose(np.roll(np.arange(len(xy)+1),-1))
        # print('debug xy grid', xy_grid.shape)
        vals = [[sigmoid(d-np.linalg.norm(xy_grid-r[None,None,:],axis=(2))) for r, d in mode] for mode in modes]
        # print('debug vals', np.array(vals).shape)
        vals = sigmoid(np.sum(vals, axis=1)-cutoff)
        # print('debug vals', np.array(vals).shape)
        val_out = (np.sign((np.prod(vals, axis=0, keepdims=True)-cutoff_final)*15)+1)/2
        if method == 'pixel':
            pass
        elif method == 'dilation':
            threshold = np.linspace(0.7, 0.44, len(modes))
            mask = np.any(vals>threshold[:,None,None], axis=0)
            val_out = scipy.ndimage.morphology.binary_dilation(val_out[0], iterations=-1, mask=mask)
            val_out = val_out.astype(np.float32)[None,...]
        else:
            raise ValueError('Unknown method ' + method)
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
        alldat[i]=f(Nspace, modes, 2.5, 0.002, method=options.method)
        if options.plot:
            for j in range(nchannel+1):
                plt.subplot(1, nchannel+1, j+1)
                plt.imshow(alldat[i, j])
            plt.show()
    np.save(options.o, alldat)
