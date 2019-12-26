from MNISTModel import MNISTCNN
import sys

import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import Compose,ToTensor
from torch.utils.data import DataLoader
from torch.optim import Adam

sys.path.insert(0,'../../interface')

from PerceptModule import PerceptModule
class TrainMNIST(PerceptModule):
    def __init__(self,model,model_parameter):
        self.model=model(**model_parameter)
        self.USING_GPU=False
        self.dataloader=None


    def set_gpu(self,num_gpu=1,device_id=None):
        if num_gpu>=1:
            self.model=self.model.cuda()
            self.USING_GPU=True
            return self
        else:
            raise Exception


    def set_cpu(self):
        self.model=self.model.cpu()
        self.USING_GPU=False
        return self


    def set_optimizers(self,
                optimizer=None,
                loss_fn=None,
                LEARNING_RATE=None,
                scheduler=None):
        self.optimizer=optimizer(self.model.parameters(),lr=LEARNING_RATE)
        self.loss_fn=loss_fn
        self.scheduler=scheduler
        return self


    def set_dataloader(self,
                PATH=None,
                BATCH_SIZE=None,
                IS_TRAIN=True,
                transform=None,
                IS_DOWNLOAD=True,
                NUM_WORKERS=1):
        dataset=MNIST(PATH,train=IS_TRAIN,transform=transform,download=IS_DOWNLOAD)
        args={
            'num_workers' : NUM_WORKERS,
            'batch_size' : BATCH_SIZE,
            'shuffle' : True,
        }
        self.dataloader=DataLoader(dataset,**args)
        return self


    def fit(self,NUM_EPOCHES,vervose=False,term=1):
        if self.dataloader is None or self.optimizer is None or self.loss_fn is None:
            raise Exception

        self.model.train()
        for epoch in range(NUM_EPOCHES):
            tot_loss=0.0
            for x,y in self.dataloader:
                self.optimizer.zero_grad()
                if self.USING_GPU:
                    x=x.cuda()
                    y=y.cuda()
                y_=self.model(x)
                loss=self.loss_fn(y_,y)
                loss.backward()
                tot_loss+=loss.item()
                self.optimizer.step()
            if vervose:
                if epoch % term == term-1:
                    print("Epoch {}, Loss(train) : {}".format(epoch+1,tot_loss))
            if self.scheduler is not None:
                self.scheduler.step()


    def predict(self,x):
        self.model.eval()
        if self.USING_GPU:
            x=x.cuda()
        y_=self.model(x)
        return y_


    def get_loss(self,x,y):
        if self.USING_GPU:
            x=x.cuda()
            y=y.cuda()
        y_=self.model(x)
        return self.loss_fn(y_,y)


    def save_model(self,path):
        torch.save(self.model.state_dict(),path)


    def load_model(self,path):
        self.model.load_state_dict(torch.load(path))

if __name__=='__main__':
    testset=MNIST("./",train=False,transform=Compose([ToTensor()]),download=True)
    a=TrainMNIST(MNISTCNN,{'IMG_SIZE':28})
    a.load_model('./temp.pt')
    a.set_gpu().set_optimizers(Adam,nn.CrossEntropyLoss(),0.005)
    a.set_dataloader("./",256,transform=Compose([ToTensor()]))
    a.fit(1,True)
    print(a.predict(testset[0][0].unsqueeze(0)))
    a.save_model('./temp.pt')
