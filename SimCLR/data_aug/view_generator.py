import numpy as np

np.random.seed(0)


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform1, base_transform2, n_views=2):
        self.base_transform1 = base_transform1[0]
        self.base_transform2 = base_transform2[0]
        self.n_views = n_views

    def __call__(self, x):
        
        t = [self.base_transform1(x), self.base_transform2(x)]
        return t
