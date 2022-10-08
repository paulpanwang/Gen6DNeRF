import torch
import torch.nn as nn

# vector to skewsym. matrix
def vec2ss_matrix(vector):  
    ss_matrix = torch.zeros((3,3))
    ss_matrix[0, 1] = -vector[2]
    ss_matrix[0, 2] = vector[1]
    ss_matrix[1, 0] = vector[2]
    ss_matrix[1, 2] = -vector[0]
    ss_matrix[2, 0] = -vector[1]
    ss_matrix[2, 1] = vector[0]
    return ss_matrix


class camera_transf(nn.Module):
    def __init__(self):
        super(camera_transf, self).__init__()
        self.w = nn.Parameter(torch.normal(0., 1e-6, size=(3,)) )
        self.v = nn.Parameter(torch.normal(0., 1e-6, size=(3,)) )
        self.theta = nn.Parameter(torch.normal(0., 1e-6, size=()) )

    def forward(self, x):
        exp_i = torch.zeros((4,4)).cuda()
        w_skewsym = vec2ss_matrix(self.w).cuda()
        exp_i[:3, :3] = torch.eye(3).cuda() + torch.sin(self.theta).cuda() * w_skewsym + \
                         (1 - torch.cos(self.theta).cuda()) * torch.matmul(w_skewsym, w_skewsym)
        exp_i[:3, 3] = torch.matmul(torch.eye(3).cuda() * self.theta + (1 - torch.cos(self.theta).cuda()) * w_skewsym \
                     + (self.theta - torch.sin(self.theta)) * torch.matmul(w_skewsym, w_skewsym), self.v)
        exp_i[3, 3] = 1.
        T_i = torch.matmul(exp_i, x)
        return T_i
