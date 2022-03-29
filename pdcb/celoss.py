import numpy as np
import torch
import torch.nn.functional as F

class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: [N, C, H, W], target: [N, H, W]
        # loss = sum(-y_i * log(c_i))
        if logits.dim() > 2:
            logits = logits.view(logits.size(0), logits.size(1), -1)  # [N, C, HW]
            logits = logits.transpose(1, 2)   # [N, HW, C]
            logits = logits.contiguous().view(-1, logits.size(2))    # [NHW, C]
        target = target.view(-1, 1)    # [NHW，1]

        logits = F.log_softmax(logits, 1)
        print(logits.shape)
        for i in range(logits.shape[0]):
            if target[i] != 0:
                logits[i][target[i]] *= 1.5
        logits = logits.gather(1, target)   # [NHW, 1]
        loss = -1 * logits

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

if __name__ == '__main__':
    logits = torch.tensor(np.random.random((1,3,30,30)),dtype=torch.float32)
    label = torch.tensor(np.random.randint(0,2,(1,30,30),dtype=np.int64))

    celoss = CrossEntropyLoss()
    loss = celoss(logits, label)

    print(loss)

