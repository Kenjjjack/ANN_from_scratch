import random
import torch
import matplotlib.pyplot as plt


# def synthetic_data(w,b,num_examples):
#     '''生成y=wx+b噪声'''
#     X=torch.normal(0,1,(num_examples,len(w)))
#     y=torch.matmul(X,w)+b
#     y+=torch.normal(0,0.01,y.shape)
#     return X,y.T

def synthetic_data(a,b,c,num_examples):
    '''生成y=wx+b噪声'''
    X=torch.normal(0,1,(num_examples,len([1])))
    y=a*torch.mul(X,X)+b*X+c
    y+=torch.normal(0,0.01,y.shape)
    return X,y

def data_iter(batch_size,features,labels):
    '''生成数据batch'''
    num_examples=len(features)
    indices=list(range(num_examples))
    #打乱数据顺序
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        batch_indices= torch.tensor(
            indices[i:min(i+batch_size,num_examples)])
        yield features[batch_indices],labels[batch_indices]

class linear:
    '''线形层'''
    def __init__(self,input_shape,output_shape) -> None:
        '''初始化权重'''
        self.input_shape=input_shape
        self.output_shape=output_shape
        self.w=torch.normal(0,0.1,size=(input_shape,output_shape))
        self.b=torch.zeros(1)
        # self.w=torch.tensor([[1.0]])
        # self.b=torch.tensor(0.5)


    def forward(self,X:torch.Tensor):
        '''正向推理'''
        #求一个batch的偏导
        self.dy_dw=X.unsqueeze(-1)  #转置,等价于X.unsqueeze(1).mT
        return torch.mm(X,self.w)+self.b

    def backward(self,dL_dy):
        '''反向传播'''
        #对一个batch中所有样本
        self.w_grad=torch.sum(torch.bmm(self.dy_dw,dL_dy),axis=0)
        self.b_grad=torch.sum(dL_dy)

        dL_dx=torch.matmul(dL_dy,self.w.T)
        return dL_dx
    
    def optimize(self,optimizer):
        '''更新权重'''
        self.w=optimizer(self.w,self.w_grad)
        self.b=optimizer(self.b,self.b_grad)     


class relu:
    def __init__(self) -> None:
        pass

    def forward(self,X:torch.Tensor):
        '''正向推理'''
        #求一个batch的偏导的和
        self.dy_dx=torch.Tensor(torch.where(X>torch.zeros(X.shape) ,torch.ones(X.shape),torch.zeros(X.shape))).unsqueeze(1)
        return torch.max(X,torch.zeros(X.shape))

    def backward(self,dL_dy):
        '''反向传播'''
        # print(dL_dy.shape)
        # print(self.dy_dx.shape)
        dL_dx=torch.mul(dL_dy,self.dy_dx)

        return dL_dx
    
    def optimize(self,optimizer):
        pass


class model:
    def __init__(self) -> None:
        self.lr=0.03
        self.num_epochs=20
        self.batch_size=20
        self.net=[linear(1,30),relu(),linear(30,30),relu(),linear(30,1)]

    def sgd(self,params,grad):
        params -= self.lr * grad / self.batch_size
        return params
    
    def mse(self,y_pred,y):
        return (y_pred-y.reshape(y_pred.shape))**2/2
    
    def dmse_dy(self,y_pred,y):
        return (y_pred-y.reshape(y_pred.shape)).unsqueeze(-1)
    
    def forward(self,X):
        for layer in self.net:
            X=layer.forward(X)
        return X
    
    def backward(self,dL_dy):
        for layer in reversed(self.net):
            dL_dy=layer.backward(dL_dy)

    def optimize(self):
        for layer in reversed(self.net):
            layer.optimize(self.sgd)

    def train(self,features,labels):
        for epoch in range(self.num_epochs):
            for x,y in data_iter(self.batch_size,features,labels):
                y_pred=self.forward(x)
                # print(y_pred)
                # print(y)
                dL_dy=self.dmse_dy(y_pred,y)
                # print(dL_dy)
                #反向传播计算梯度
                self.backward(dL_dy)
                self.optimize()
            #在整个训练集上的loss
            train_l = self.mse(self.forward(features), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')



# true_w=torch.tensor([2,-3.5])
# true_b=4.5
a=1
b=2
c=1
features,labels=synthetic_data(a,b,c,2000)

network=model()
network.train(features,labels)
x=torch.Tensor([[1.0],[0.0],[-1.0],[0.5]])
# # print(x)
y_pred=network.forward(x)
print(y_pred)