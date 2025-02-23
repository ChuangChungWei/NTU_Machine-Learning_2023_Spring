# %% [markdown]
# # **Homework 8 - Anomaly Detection**
# 
# If there are any questions, please contact mlta-2023-spring@googlegroups.com
# 
# Slide:    [Link](https://docs.google.com/presentation/d/18LkR8qulwSbi3SVoLl1XNNGjQQ_qczs_35lrJWOmHCk/edit?usp=sharing)　Kaggle: [Link](https://www.kaggle.com/t/c76950cc460140eba30a576ca7668d28)

# %% [markdown]
# # Set up the environment
# 

# %% [markdown]
# ## Package installation

# %%
# Training progress bar
#!pip install -q qqdm

# %% [markdown]
# ## Downloading data

# %%
#!git clone https://github.com/chiyuanhsiao/ml2023spring-hw8

# %% [markdown]
# # Import packages

# %%


################Reference about 前處理與fcn架構的想法： r11921091 楊冠彥################################
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.optim import Adam, AdamW
from qqdm import qqdm, format_str
import pandas as pd
from PIL import Image

# %% [markdown]
# # Loading data

# %%

train = np.load('ml2023spring-hw8/trainingset.npy', allow_pickle=True)
test = np.load('ml2023spring-hw8/testingset.npy', allow_pickle=True)

print(train.shape)
print(test.shape)

# %% [markdown]
# ## Random seed
# Set the random seed to a certain value for reproducibility.

# %%
def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seeds(16888)

# %% [markdown]
# # Autoencoder

# %% [markdown]
# # Models & loss

# %%

from torchvision.models import resnet50,resnet18,resnet34,resnet101,regnet_y_1_6gf,convnext_tiny
class fcn_autoencoder(nn.Module):
    def __init__(self):
        super(fcn_autoencoder, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Linear(64* 64 * 3,50),

        )    # Hint: dimension of latent space can be adjusted
        self.decoder = nn.Sequential(

            nn.Linear(50 , 64 * 64 * 3), 
            nn.Tanh()
        )
        '''
        self.encoder1 = nn.Sequential(
            nn.Linear(40 * 40 * 3, 1024),
            #nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),
            
        )    # Hint: dimension of latent space can be adjusted
        '''
        self.encoder2 = nn.Sequential(
            nn.Linear(64 * 64 * 3, 100),

        )    # Hint: dimension of latent space can be adjusted
        self.encoder3 = nn.Sequential(
            nn.Linear(64 * 64 * 3, 80),

        )    # Hint: dimension of latent space can be adjusted
        #self.encoder2=resnet50()
        self.fcn=nn.Sequential(
            nn.Linear(1000,100)
        )
        #self.encoder3=resnet101()
        
        '''
        self.decoder = nn.Sequential(

            nn.Linear(1024, 64 * 64 * 3), 
            nn.Tanh()
        )
        '''

    def forward(self, x):
        x1 = self.encoder1(x)
        #x_view=x.view(-1,3,64,64)
        #x2 = self.encoder2(x)
        #x2=self.fcn(x2)
        #x3 = self.encoder3(x)
        #x=torch.cat((x1,x2,x3),1)
        x=x1
        x = self.decoder(x)
        return x

class resnet(nn.Module):
    def __init__(self,real=True):
        super(resnet, self).__init__()
        self.real=real
        self.encoder4 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 64 * 3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(), 
            nn.Linear(512, 256), 
            nn.ReLU(), 
            nn.Linear(256, 128), 
            nn.ReLU(), 
            nn.Linear(128, 12), 
            nn.ReLU(), 
            nn.Linear(12, 1000)
        )    # Hint: dimension of latent space can be adjusted
        self.encoder1=resnet50()
        self.encoder2=convnext_tiny()
        self.encoder3=resnet101()
        self.fc=nn.Sequential(
            nn.ReLU(), 
            nn.Linear(1000, 100)
        )
        self.decoder = nn.Sequential(
            nn.Linear(100, 12),
            nn.ReLU(), 
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 2048),
            nn.ReLU(), 
            nn.Linear(2048, 64 * 64 * 3), 
            nn.Tanh()
        )
        

    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(x)
        x3 = self.encoder3(x)
        #x4 = self.encoder4(x)
        x=(x1+x2+x3)/3
        #x=torch.cat((x1,x2,x3),1)
        x=self.fc(x)
        x = self.decoder(x)
        x = x.view(-1,3,64,64)
        return x
class resnet_with_classifier(nn.Module):
    def __init__(self):
        super(resnet_with_classifier, self).__init__()
        self.encoder2=torch.nn.Sequential(*(list(resnet50().children())[:-2]))
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),         
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),        
            nn.ReLU(),
			      nn.Conv2d(128, 256, 4, stride=2, padding=1),         
            nn.ReLU(),
                  nn.Conv2d(256, 256, 4, stride=2, padding=1),         
            nn.ReLU(),
                  nn.Conv2d(256, 256, 4, stride=2, padding=1),         
            nn.ReLU(),
                  nn.Conv2d(256, 128, 4, stride=2, padding=1),         
            nn.ReLU(),
        )   # Hint:  dimension of latent space can be adjusted

        self.encoder2=torch.nn.Sequential(*(list(resnet50().children())[:-2]))
        self.conv=nn.Sequential(
            nn.Conv2d(2048, 128, 4, stride=1, padding=1),         
            nn.ReLU(),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(3, 128, 4, stride=2, padding=1),         
            nn.ReLU(),
            nn.Conv2d(128, 128, 4, stride=2, padding=1),        
            nn.ReLU(),
			      nn.Conv2d(128, 256, 4, stride=2, padding=1),         
            nn.ReLU(),
                  nn.Conv2d(256, 256, 4, stride=2, padding=1),         
            nn.ReLU(),
                  nn.Conv2d(256, 256, 4, stride=2, padding=1),         
            nn.ReLU(),
                  nn.Conv2d(256, 128, 4, stride=2, padding=1),         
            nn.ReLU(),
        )   # Hint:  dimension of latent space can be adjusted
        self.decoder = nn.Sequential(
			      nn.ConvTranspose2d(384, 256, 4, stride=2, padding=1),
            nn.ReLU(),
                  nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
                  nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            nn.ReLU(),
                  nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            nn.ReLU(),
			      nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential()
        self.classifier=resnet34()
        self.fc=nn.Sequential(
            nn.Linear(1000, 2),
        )
    def forward(self, x, gt):
        x1 = self.encoder1(x)
        x2 = self.encoder2(x)
        x2=  self.conv(x2)
        x3 = self.encoder3(x)
        x=torch.cat((x1,x2,x3),1)
        noise=torch.rand(x.size()).to('cuda')
        #print("origgt:",gt)
        if gt!="test":
            gt=torch.unsqueeze(gt,1)
            #print(gt.size())
            gt=gt.expand(gt.size(0),x.size(1))
            #print("finalgt:",gt)
            #print("======",noise.size(),gt.size())
            noise=noise*gt
            #noise=torch.mul(noise, gt)
            x=x+noise
        #print(x1.size())
        #print(x2.size())
        #print(x3.size())
        #x=x1
        #x=(x1+x2+x3)/3
        x = self.decoder(x) 
        x = self.classifier(x)
        x = self.fc(x)
        return x


class conv_autoencoder(nn.Module):
    def __init__(self):
        super(conv_autoencoder, self).__init__()
        self.encoder2=torch.nn.Sequential(*(list(resnet50().children())[:-2]))
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 128, 4, stride=2, padding=1),    
            nn.BatchNorm2d(128),    
            nn.ReLU(),
			      nn.Conv2d(128, 256, 4, stride=2, padding=1),    
            nn.BatchNorm2d(256),     
            nn.ReLU(),
                  nn.Conv2d(256, 500, 4, stride=2, padding=1),    
            nn.BatchNorm2d(500),     
            nn.ReLU(),
        )   # Hint:  dimension of latent space can be adjusted

        self.conv=nn.Sequential(
            nn.Conv2d(2048, 512, 4, stride=1, padding=1),         
            nn.ReLU(),
        )
        self.conv=nn.Sequential(
            nn.Conv2d(2048, 512, 4, stride=1, padding=1),         
            nn.ReLU(),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(3, 128, 4, stride=2, padding=1),         
            nn.ReLU(),
            nn.Conv2d(128, 128, 4, stride=2, padding=1),        
            nn.ReLU(),
			      nn.Conv2d(128, 256, 4, stride=2, padding=1),         
            nn.ReLU(),
                  nn.Conv2d(256, 256, 4, stride=2, padding=1),         
            nn.ReLU(),
                  nn.Conv2d(256, 256, 4, stride=2, padding=1),         
            nn.ReLU(),
                  nn.Conv2d(256, 512, 4, stride=2, padding=1),         
            nn.ReLU(),
        )   # Hint:  dimension of latent space can be adjusted
        self.decoder = nn.Sequential(
			      nn.ConvTranspose2d(500, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),    
            nn.ReLU(),
                  nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),    
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )
        #self.encoder1=torch.nn.Sequential(*(list(resnet50().children())[:-2]))
        #self.encoder2=torch.nn.Sequential(*(list(resnet101().children())[:-2]))
        #self.encoder3=torch.nn.Sequential(*(list(resnet50().children())[:-2]))


    def forward(self, x):
        x1 = self.encoder1(x)
        '''
        x1=  self.conv(x1)
        x2 = self.encoder2(x)
        x2=  self.conv(x2)
        x3 = self.encoder3(x)
        
        x=torch.cat((x1,x2,x3),1)
        '''
        x=x1
        #print(x1.size())
        #print(x2.size())
        #print(x3.size())
        #x=x1
        #x=(x1+x2+x3)/3
        x = self.decoder(x) 
        return x


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),    
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),    
            nn.ReLU(),
            nn.Conv2d(48, 384, 4, stride=2, padding=1),    
            nn.ReLU(),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(3, 24, 4, stride=2, padding=1),            
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),    
            nn.ReLU(),
            nn.Conv2d(48, 96, 4, stride=2, padding=1),    
            nn.ReLU(),
            nn.Conv2d(96, 384, 4, stride=2, padding=1),    
            nn.ReLU(),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(3, 48, 4, stride=2, padding=1),            
            nn.ReLU(),
            nn.Conv2d(48, 96, 4, stride=2, padding=1),    
            nn.ReLU(),
            
            nn.Conv2d(96, 192, 4, stride=2, padding=1),    
            nn.ReLU(),
            nn.Conv2d(192, 384, 4, stride=2, padding=1),    
            nn.ReLU(),
        )
        self.enc_out_1 = nn.Sequential(
            nn.Conv2d(384, 32, 4, stride=2, padding=1),  
            nn.ReLU(),
        )
        self.enc_out_2 = nn.Sequential(
            nn.Conv2d(384, 32, 2, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 2, stride=2, padding=1),
            nn.ReLU(),
        )
        # Hint: can add more layers to encoder and decoder
        self.decoder = nn.Sequential(
			      nn.ConvTranspose2d(32, 64, 4, stride=2, padding=1), 
            nn.ReLU(),
                  nn.ConvTranspose2d(64, 128, 4, stride=2, padding=1), 
            nn.ReLU(),
                  nn.ConvTranspose2d(128, 512, 4, stride=2, padding=1), 
            nn.ReLU(),
			      nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1), 
            nn.Tanh(),
        )

    def encode(self, x):
        h1 = self.encoder1(x)
        h2 = self.encoder2(x)
        h3 = self.encoder3(x)
        out1=(self.enc_out_1(h1)+self.enc_out_1(h2)+self.enc_out_1(h3))/3
        out2=(self.enc_out_2(h1)+self.enc_out_2(h2)+self.enc_out_2(h3))/3
        return out1,out2

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_vae(recon_x, x, mu, logvar, criterion):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    mse = criterion(recon_x, x)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return mse + KLD

# %%


# %% [markdown]
# # Dataset module
# 
# Module for obtaining and processing data. The transform function here normalizes image's pixels from [0, 255] to [-1.0, 1.0].
# 

# %%
class CustomTensorDataset(TensorDataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors):
        self.tensors = tensors
        if tensors.shape[-1] == 3:
            self.tensors = tensors.permute(0, 3, 1, 2)
        #self.transform=None
        
        self.transform = transforms.Compose([
          transforms.Lambda(lambda x: x.to(torch.float32)),
          transforms.Lambda(lambda x:  2*x/255.-1 ),
          transforms.CenterCrop(40),
          transforms.Resize((64,64)),
        ])

        
    def __getitem__(self, index):
        x = self.tensors[index]
        
        if self.transform:
            # mapping images to [-1.0, 1.0]
            x = self.transform(x)

        return x

    def __len__(self):
        return len(self.tensors)

# %%


# %% [markdown]
# # Training

# %% [markdown]
# ## Configuration
# 

# %%
# Training hyperparameters
num_epochs = 50
batch_size = 256 # Hint: batch size may be lower
learning_rate = 1e-3

# Build training dataloader
x = torch.from_numpy(train)
train_dataset = CustomTensorDataset(x)

train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

# Model
model_type = 'fcn'   # selecting a model type from {'cnn', 'fcn', 'vae', 'resnet'}
model_classes = {'fcn': fcn_autoencoder(), 'cnn': conv_autoencoder(), 'vae': VAE(),'resnet':resnet(),'resnet_with_classifier':resnet_with_classifier()}
model = model_classes[model_type].cuda()

# Loss and optimizer
criterion = nn.MSELoss()
criterion_test=0

if model_type in ['resnet_with_classifier']:
    criterion = nn.CrossEntropyLoss()
    criterion_test = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# %% [markdown]
# ## Training loop

# %%
from datetime import datetime
import os
import random
t=datetime.now()
os.makedirs("./result" ,exist_ok=True)

f=open("./result/"+str(t)+".txt","a+")
print(model,file=f)
best_loss = np.inf
model.train()

qqdm_train = qqdm(range(num_epochs), desc=format_str('bold', 'Description'))
for epoch in qqdm_train:
    tot_loss = list()
    loss_resnets = list()
    for data in train_dataloader:
        # ===================loading=====================
        rand=0
        img = data.float().cuda()
        gt=0
        if model_type in ['fcn']:
            img = img.view(img.shape[0], -1)
        
        # ===================forward=====================
        
        if model_type in ['vae']:
            output = model(img)
            loss = loss_vae(output[0], img, output[1], output[2], criterion)
        elif model_type in ['resnet_with_classifier']:
            rand=random.random()
            gt=torch.randint(0,2,(batch_size,)).to('cuda').long()     
            '''
            if rand>=0.5:
                gt=torch.ones((batch_size,)).to('cuda').long()                 
                output,output1 = model(img,gt)        
                output=output.to(torch.float32)
                loss_resnet=criterion_test(output1, img)

            else:
                gt=torch.zeros((batch_size,)).to('cuda').long()
                output,output1 = model(img,gt)
                output=output.to(torch.float32)
                loss_resnet=criterion_test(output1, img)
                loss_resnet=loss_resnet
            '''
            output,output1 = model(img,gt)        
            output=output.to(torch.float32)
            loss_resnet=criterion_test(output1, img)
            #loss_resnet=gt_norm*loss_resnet
            loss_resnets.append(loss_resnet.item())
            #print(output1.size(),img.size())
            
            #output=output.type(torch.LongTensor)
            #print(output.size(),gt.size())
            loss = criterion(output, gt)
            '''
            for i,g in enumerate(gt):
                print(output[i],g)
            '''
            
        else:
            output = model(img)
            loss = criterion(output, img)
            
            

        tot_loss.append(loss.item())
        # ===================backward====================
        optimizer.zero_grad()
        if model_type in ['resnet_with_classifier'] :
            loss.backward()
        else:
            loss.backward()
        optimizer.step()
        '''
        if model_type in ['resnet_with_classifier']:
            output,output1 = model(img,gt)
            optimizer.zero_grad()
            
            loss_resnet.backward()
            optimizer.step()
        '''
    # ===================save_best====================
    mean_loss = np.mean(tot_loss)
    mean_loss_resnets=np.mean(loss_resnets)
    if mean_loss < best_loss:
        best_loss = mean_loss
        torch.save(model, 'best_model_{}.pt'.format(model_type))
    # ===================log========================
    f=open("./result/"+str(t)+".txt","a+")
    print("epoch",epoch,":",mean_loss,mean_loss_resnets,file=f)
    qqdm_train.set_infos({
        'epoch': f'{epoch + 1:.0f}/{num_epochs:.0f}',
        'loss': f'{mean_loss:.4f}',
        'resnet_loss': f'{mean_loss_resnets:.4f}',
    })
    # ===================save_last========================
    torch.save(model, 'last_model_{}.pt'.format(model_type))

# %% [markdown]
# # Inference
# Model is loaded and generates its anomaly score predictions.

# %% [markdown]
# ## Initialize
# - dataloader
# - model
# - prediction file

# %%
eval_batch_size = 200

# build testing dataloader
data = torch.tensor(test, dtype=torch.float32)
test_dataset = CustomTensorDataset(data)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=eval_batch_size, num_workers=1)
eval_loss = nn.MSELoss(reduction='none')

# load trained model
checkpoint_path = f'last_model_{model_type}.pt'
model = torch.load(checkpoint_path)
model.eval()

# prediction file 
out_file = 'prediction.csv'

# %%
anomality = list()
with torch.no_grad():
  for i, data in enumerate(test_dataloader):
    img = data.float().cuda()
    #print(img.size())
    if model_type in ['fcn']:
      
      
      '''
      output = img[0]
      tensor = torch.rand(3,300,700)

      transform = transforms.ToPILImage()

      # convert the tensor to PIL image using above transform
      output = transform(output)

      output.save('result_img/'+str(i)+'.png')
      '''
      
      img = img.view(img.shape[0], -1)


    if model_type in ['vae']:
      output = model(img)
      output = output[0]
      loss = eval_loss(output, img).sum([1, 2, 3])
    elif model_type in ['fcn']:
      output = model(img)
      loss = eval_loss(output, img).sum(-1)
      
    elif model_type in ['resnet_with_classifier']:
      #gt=torch.zeros((eval_batch_size,)).to('cuda').long()     
      output,output1=model(img,"test")
      #print(output)
      #print(output.size())
      loss=output[:,1]

      
      #print(loss.size())
      #loss = eval_loss(output, img).sum([1, 2, 3])
    else:
      output = model(img)
      loss = eval_loss(output, img).sum([1, 2, 3])
      
    anomality.append(loss)
#print(anomality.size())
anomality = torch.cat(anomality, axis=0).cpu().numpy()

#anomality = torch.sqrt(anomality).reshape(len(test), 1).cpu().numpy()

df = pd.DataFrame(anomality, columns=['score'])
df.to_csv(out_file, index_label = 'ID')

# %%
print(t)

# %%



