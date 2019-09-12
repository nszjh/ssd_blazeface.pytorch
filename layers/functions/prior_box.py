from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                # f_k = self.image_size / self.steps[k]
                # # unit center x,y
                # cx = (j + 0.5) / f_k   # [0, 1]
                # cy = (i + 0.5) / f_k   # [0, 1]

                # # aspect_ratio: 1
                # # rel size: min_size
                # s_k = self.min_sizes[k]/self.image_size
                # mean += [cx, cy, s_k, s_k]

                # # aspect_ratio: 1
                # # rel size: sqrt(s_k * s_(k+1))
                # s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                # mean += [cx, cy, s_k_prime, s_k_prime]

                # print ("prior_box:", i, j, f, cx, cy, f_k, s_k, s_k_prime)

                # # rest of aspect ratios, 
                # for ar in self.aspect_ratios[k]:
                #     if self.version == "CELEBA": # for celeba, we change ratios to 1:1
                #         mean += [cx, cy, s_k*sqrt(ar), s_k*sqrt(ar)]
                #         mean += [cx, cy, s_k*2*sqrt(ar), s_k*2*sqrt(ar)]
                #     else:
                #         mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                #         mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]

                f_k = 1 / f / 2 
                # unit center x,y
                cx = j * f_k * 2 + f_k
                cy = i * f_k * 2 + f_k
                s_k = f_k * 2

                print ("prior_box:", i, j, f, cx, cy, f_k, s_k)

                mean += [cx, cy, s_k, s_k]
                mean += [cx, cy, s_k * 2, s_k * 2]
                # rest of aspect ratios, 
                for ar in self.aspect_ratios[k]:
                    if self.version == "CELEBA": # for celeba, we change ratios to 1:1
                        mean += [cx, cy, s_k*sqrt(ar), s_k*sqrt(ar)]
                        mean += [cx, cy, s_k*2*sqrt(ar), s_k*2*sqrt(ar)]
                    else:
                        mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                        mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]


        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


from config import voc, coco, celeba

if __name__ == "__main__":
    cfg = celeba
    priorbox = PriorBox(cfg)
    print (priorbox.forward())
    print (priorbox.forward().shape)
    

