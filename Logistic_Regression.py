import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
from lr_utils import load_dataset
import torch.optim as optim
"""define a class that struct the model"""
class My_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(64*64*3,1),nn.Sigmoid())

    def forwad(self,X):
        logit=self.net(X)
        return logit
class my_data(Dataset):
    def __init__(self,X,Y):
        self.X=X
        self.Y=Y.T
        self.flatten=nn.Flatten()
        X=(torch.from_numpy(self.X)).type(torch.float32)
        Y=torch.from_numpy(self.Y).type(torch.float32)
        X=self.flatten(X)

        X=X/255.#标准化数据对于图片类型数据直接除255就是最好的归一化

        self.X=X
        self.Y=Y
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, item):
        X=self.X[item]
        Y=self.Y[item]
        return X,Y

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()#使用吴恩达老师在coursera上的猫猫数据集
my_datasets_train=my_data(train_set_x_orig,train_set_y)
dataload=DataLoader(my_datasets_train,batch_size=8,shuffle=True)#用于训练的数据集
dataload_train=DataLoader(my_datasets_train,batch_size=209)#用于测试整个训练集性能的数据集
my_datasets_test=my_data(test_set_x_orig,test_set_y)#交叉验证集也就是dev集
dataload_test=DataLoader(my_datasets_test,batch_size=50)#用Dataloader导入


def train(modell,epoch,loss,optim):
    for m in range(epoch):
        for i,(x,y) in enumerate(dataload):
            x,y=x.to(device),y.to(device)
            logit=modell.forwad(x)
            los=loss(logit,y)
            los.backward()
            optim.step()
            optim.zero_grad()

            if i % 10 == 0:
                print(f"loss: {los:>7f}")
"""pred函数将sigmoid的值进行处理如果大于0.5则置1，反之置0"""
def pred(x):
    label=torch.zeros(x.shape)
    for i in range(len(x)):
        if(x[i,0]>0.5):
            label[i,0]=1
        else:
            label[i,0]=0
    return label
def accuracy(y,y_true):
    ac=torch.zeros(y.shape)
    for i in range(len(y)):
        if(y[i,0]==y_true[i,0]):
            ac[i,0]=1
        else:
            ac[i,0]=0
    total=torch.sum(ac,dim=0,keepdim=True)
    total=total/len(ac)
    return str(total)+"%"
def test(dataload,model):
    for i, (x, y) in enumerate(dataload):
        x = x.to(device)
        y_label = my_model.forwad(x)
        y_label = pred(y_label)
        a = accuracy(y_label, y)
        print(a)

device="cuda"
my_model=My_model().to(device)
loss=nn.BCELoss()
opti=optim.Adam(my_model.parameters(),lr=0.001)
#train(my_model,40,loss,opti)用这个来控制是否训练模型
my_model.load_state_dict(torch.load('model.pth'))#从训练好的模型中导入参数
print(test(dataload_train,my_model))
print(test(dataload_test,my_model))
torch.save(my_model.state_dict(), "model.pth")#存储模型
print("Saved PyTorch Model State to model.pth")














