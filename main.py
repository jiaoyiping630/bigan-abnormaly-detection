import argparse
from torchvision import datasets, transforms
import torch.optim as optim
from torch.autograd import Variable
import torchvision.utils as vutils
from model import *
import os

batch_size = 100
lr = 1e-4
latent_size = 256
num_epochs = 100
cuda_device = "0"


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | svhn')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--use_cuda', type=boolean_string, default=True)
parser.add_argument('--save_model_dir', required=True)
parser.add_argument('--save_image_dir', required=True)

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
print(opt)

def tocuda(x):
    if opt.use_cuda:
        return x.cuda()
    return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.bias.data.fill_(0)


def log_sum_exp(input):
    m, _ = torch.max(input, dim=1, keepdim=True)
    input0 = input - m
    m.squeeze()
    return m + torch.log(torch.sum(torch.exp(input0), dim=1))

'''
    这个函数的输入是第一批次的样本，是一个(100,3,32,32)的张量
    首先计算在sample上的均值，得到(3,32,32)
    然后将该均值限制在1e-7~1-1e-7之内（一般来讲都在）
    最后返回log(u/(1-u)) 大概是由于generator最后的输出如何转换到真实图像决定的
'''
def get_log_odds(raw_marginals):
    marginals = torch.clamp(raw_marginals.mean(dim=0), 1e-7, 1 - 1e-7)
    return torch.log(marginals / (1 - marginals))


if opt.dataset == 'svhn':
    train_loader = torch.utils.data.DataLoader(
        datasets.SVHN(root=opt.dataroot, split='extra', download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor()
                      ])),
        batch_size=batch_size, shuffle=True)
elif opt.dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=opt.dataroot, train=True, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor()
                      ])),
        batch_size=batch_size, shuffle=True)
else:
    raise NotImplementedError

netE = tocuda(Encoder(latent_size, True))   #   隐向量尺寸，默认256，由于noise设置为了True，因此实际编码尺寸为512，一半给μ，一半给σ
netG = tocuda(Generator(latent_size))
netD = tocuda(Discriminator(latent_size, 0.2, 1))

netE.apply(weights_init)
netG.apply(weights_init)
netD.apply(weights_init)

optimizerG = optim.Adam([{'params' : netE.parameters()},
                         {'params' : netG.parameters()}], lr=lr, betas=(0.5,0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

criterion = nn.BCELoss()

for epoch in range(num_epochs):

    i = 0
    for (data, target) in train_loader:

        '''这里1代表真标签，0代表假标签，batch_size默认是100'''
        real_label = Variable(tocuda(torch.ones(batch_size)))
        fake_label = Variable(tocuda(torch.zeros(batch_size)))

        '''这个噪声会加在送给D的输入上，让D没那么容易训练；随着epoch推进，方差越来越小的，size = (100,3,32,32)'''
        noise1 = Variable(tocuda(torch.Tensor(data.size()).normal_(0, 0.1 * (num_epochs - epoch) / num_epochs)))
        noise2 = Variable(tocuda(torch.Tensor(data.size()).normal_(0, 0.1 * (num_epochs - epoch) / num_epochs)))

        '''在第一个batch中，把样本的均值统计量送给generator，方便其往正确方向上优化'''
        if epoch == 0 and i == 0:
            netG.output_bias.data = get_log_odds(tocuda(data))

        if data.size()[0] != batch_size:
            continue

        d_real = Variable(tocuda(data))                                         #   真实数据

        z_fake = Variable(tocuda(torch.randn(batch_size, latent_size, 1, 1)))   #   假的隐码，(100,256,1,1)的正态随机数
        d_fake = netG(z_fake)                                                   #   假的数据，由假的隐码经过生成器得到的

        z_real, _, _, _ = netE(d_real)                                          #   真的隐码，由真的数据经过编码器得到(100,512,1,1)
        z_real = z_real.view(batch_size, -1)                                    #                                 (100,512)

        mu, log_sigma = z_real[:, :latent_size], z_real[:, latent_size:]        #   把隐码的μ和σ分开
        sigma = torch.exp(log_sigma)
        epsilon = Variable(tocuda(torch.randn(batch_size, latent_size)))

        output_z = mu + epsilon * sigma                                         #   总之，这个才是最后生效的真实隐码 (100,256)

        output_real, _ = netD(d_real + noise1, output_z.view(batch_size, latent_size, 1, 1))    #   判别器对真实样本的响应 (100,)
        output_fake, _ = netD(d_fake + noise2, z_fake)                                          #   判别器对虚假样本的响应 (100,)

        '''注：由于是二分类（真/假），这里的criterion采用的是BCE损失'''
        loss_d = criterion(output_real, real_label) + criterion(output_fake, fake_label)        #   D希望看到真的输出1，看到假的输出0
        loss_g = criterion(output_fake, real_label) + criterion(output_real, fake_label)        #   G希望（判别器）看到假的输出1，看到真的输出0

        if loss_g.data[0] < 3.5:
            optimizerD.zero_grad()
            loss_d.backward(retain_graph=True)  #   因为后面还需要对G进行优化，所以计算图仍然需要保留
            optimizerD.step()

        optimizerG.zero_grad()
        loss_g.backward()
        optimizerG.step()

        if i % 1 == 0:
            print("Epoch :", epoch, "Iter :", i, "D Loss :", loss_d.data[0], "G loss :", loss_g.data[0],
                  "D(x) :", output_real.mean().data[0], "D(G(x)) :", output_fake.mean().data[0])

        if i % 50 == 0:
            vutils.save_image(d_fake.cpu().data[:16, ], './%s/fake.png' % (opt.save_image_dir))
            vutils.save_image(d_real.cpu().data[:16, ], './%s/real.png'% (opt.save_image_dir))

        i += 1

    if epoch % 10 == 0:
        torch.save(netG.state_dict(), './%s/netG_epoch_%d.pth' % (opt.save_model_dir, epoch))
        torch.save(netE.state_dict(), './%s/netE_epoch_%d.pth' % (opt.save_model_dir, epoch))
        torch.save(netD.state_dict(), './%s/netD_epoch_%d.pth' % (opt.save_model_dir, epoch))

        vutils.save_image(d_fake.cpu().data[:16, ], './%s/fake_%d.png' % (opt.save_image_dir, epoch))
