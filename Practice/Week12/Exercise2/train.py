import torch
import torch.nn as nn
import torch.optim as optim
import random,tqdm
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torch.autograd import Variable
import sys, os,os,json,csv,pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import glob


class PropNN(nn.Module):

    def __init__(self, hidden_size, propsize):
        super(PropNN, self).__init__()
        self.hidden_size = hidden_size 
                
        self.regression_NN = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 4, propsize)
          )
        self.classifaction_NN = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 4, 33)
          )
        
        self.regression_loss = nn.MSELoss()
        self.classifaction_loss = nn.CrossEntropyLoss(size_average=False)

    def forward(self, fp,regression_label):#, class_label):
        regression_predict = self.regression_NN(fp)
        # classifaction_predict = self.classifaction_NN(fp)
        regression_loss = self.regression_loss(regression_predict,regression_label)
        # classifaction_loss = self.classifaction_loss(classifaction_predict,class_label)
        return regression_loss#,classifaction_loss
    
if __name__ == '__main__':
    for loadfile in glob.glob(os.path.join(os.path.expanduser('~'),'CompChem/Practice/Week12/data/.pkl')):
        files = pickle.load(open(loadfile,'rb'))
        hidden_size = os.path.basename(loadfile).split('.')[0].split('-')[-1]
        test_size = 0.2
        batch_size = 128
        lr = 0.001
        MAX_EPOCH = 60
        PRINT_ITER = 10
        ANNEAL_ITER = 20
        nclips = int(50)
        save_dir = os.path.basename(loadfile).split('.')[0]
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
            os.mkdir(os.path.join(save_dir,'train-error-output'))
            os.mkdir(os.path.join(save_dir,'test-error-output'))
        trainset,testset = train_test_split(files,test_size=test_size, random_state=10001)
        print('\nNum of trainset and testset are %s,%s'%(len(trainset),len(testset)))
        model_prop = PropNN(int(hidden_size), 2)
        model_prop.cuda()
        print ("Model #Params: %dK" % (sum([x.nelement() for x in model_prop.parameters()]) / 1000,))
        lr = float(lr)
        nclips = int(50)
        optimizer = optim.Adam(model_prop.parameters(), lr=lr)
        scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)
        train_error_tot = []
        test_error_tot = []
        train_loss_tot = [] 
        test_loss_tot = []
        total_step = 0
        tot_loss = 0
        for epoch in tqdm.tqdm(range(MAX_EPOCH)):
            random.shuffle(trainset)
            train_batches = [trainset[i: i + batch_size] for i in range(0, len(trainset), batch_size)]
            model_prop.train()
            if len(train_batches[-1]) != batch_size:
                train_batches.pop()
            for train_batch in train_batches:
                total_step += 1
                fp_batch = torch.stack([batch.fp for batch in train_batch])
                regression_label = torch.stack([torch.tensor([batch.TPSA, batch.logp]) for batch in train_batch])
                # classification_label = torch.tensor([batch.rings for batch in train_batch], dtype=torch.long)
                # regression_loss,classifaction_loss = model_prop(fp_batch.cuda(), regression_label.cuda(), classification_label.cuda())
                regression_loss = model_prop(fp_batch.cuda(), regression_label.cuda())
                loss = regression_loss# + classifaction_loss
                loss.backward()
                loss_ = loss.item()
                tot_loss += loss_
                nn.utils.clip_grad_norm_(model_prop.parameters(), nclips)
                optimizer.step()
                if (total_step) % 20 == 0: #Fast annealing
                    train_loss_tot.append([total_step,tot_loss / PRINT_ITER ])
                    tot_loss *= 0
            if epoch % ANNEAL_ITER == 0:
                scheduler.step()
            model_prop.eval()
            train_error = []
            for train_batch in trainset:
                regression_label = torch.tensor([train_batch.TPSA, train_batch.logp])
                #classification_label = torch.tensor(train_batch.rings, dtype=torch.long)
                TPSA_predict,logP_predict = model_prop.regression_NN(train_batch.fp.cuda())
                #ring_predict = model_prop.classifaction_NN(train_batch.fp.cuda())
                #ring_predict = softmaxfun(ring_predict)
                #_, ring_predict = torch.max(ring_predict,0)
                #ring_predict = ring_predict.item()
                train_error.append([TPSA_predict.item(),logP_predict.item(),train_batch.TPSA, train_batch.logp])
            df_train_error = pd.DataFrame(train_error)
            df_train_error.columns = ['TPSA_Predict','LogP_predict','TPSA_True','Logp_True']
            test_error = []
            test_loss = 0
            for test_batch in testset:
                regression_label = torch.tensor([test_batch.TPSA, test_batch.logp])
                #classification_label = torch.tensor(test_batch.rings, dtype=torch.long)
                test_predict = model_prop.regression_NN(test_batch.fp.cuda())
                test_loss += model_prop.regression_loss(test_predict.cuda(),regression_label.cuda()).item()
                TPSA_predict,logP_predict = test_predict
                #ring_predict = model_prop.classifaction_NN(test_batch.fp.cuda())
                #ring_predict = softmaxfun(ring_predict)
                #_, ring_predict = torch.max(ring_predict,0)
                #ring_predict = ring_predict.item()
                test_error.append([TPSA_predict.item(),logP_predict.item(),test_batch.TPSA, test_batch.logp]) 
            test_loss_tot.append([epoch,test_loss / len(testset)])
            df_test_error = pd.DataFrame(test_error)
            df_test_error.columns = ['TPSA_Predict','LogP_predict','TPSA_True','Logp_True']
            if epoch % 5 == 0 :
                torch.save(model_prop.state_dict(), save_dir + "/model.epoch-" + str(epoch))
            df_train_error.to_csv(os.path.join(save_dir,'train-error-output','epoch-%d-train_error.csv' %epoch),index=None)
            df_test_error.to_csv(os.path.join(save_dir,'test-error-output','epoch-%d-test_error.csv' %epoch),index=None)
            
        df_train_loss_tot = pd.DataFrame(train_loss_tot)
        df_train_loss_tot.columns = ['Iteration','Loss']
        df_train_loss_tot.to_csv(os.path.join(save_dir,'train_loss.csv'),index=None)
        df_test_loss_tot = pd.DataFrame(test_loss_tot)
        df_test_loss_tot.columns = ['Iteration','Loss']
        df_test_loss_tot.to_csv(os.path.join(save_dir,'test_loss.csv'),index=None)

