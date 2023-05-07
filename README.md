# IR_for_onlineMarket_w_ResNet18
Image recognation for KazanExpress dataset
Epoch [1/20], Loss (train/test) : 2.6906/2.0638,Acc (train/test): 0.1435/0.0567 /n
Epoch [2/20], Loss (train/test) : 1.8386/1.6680,Acc (train/test): 0.1322/0.0789
Epoch [3/20], Loss (train/test) : 1.4393/0.9365,Acc (train/test): 0.1570/0.0817
Epoch [4/20], Loss (train/test) : 0.6363/0.8044,Acc (train/test): 0.3029/0.0918
Epoch [5/20], Loss (train/test) : 0.4895/0.6002,Acc (train/test): 0.3651/0.1681
Epoch [6/20], Loss (train/test) : 0.4227/0.4727,Acc (train/test): 0.3984/0.1640
Epoch [7/20], Loss (train/test) : 0.3508/0.3352,Acc (train/test): 0.4427/0.1813
Epoch [8/20], Loss (train/test) : 0.2285/0.2190,Acc (train/test): 0.5330/0.2678
Epoch [9/20], Loss (train/test) : 0.1214/0.1484,Acc (train/test): 0.6443/0.2762
Epoch [10/20], Loss (train/test) : 0.0956/0.1138,Acc (train/test): 0.6753/0.4010
Epoch [11/20], Loss (train/test) : 0.0836/0.0825,Acc (train/test): 0.6959/0.4919
Epoch [12/20], Loss (train/test) : 0.0717/0.0648,Acc (train/test): 0.7242/0.5434
Epoch [13/20], Loss (train/test) : 0.0668/0.0597,Acc (train/test): 0.7623/0.5912
Epoch [14/20], Loss (train/test) : 0.0481/0.0522,Acc (train/test): 0.7791/0.6075
Epoch [15/20], Loss (train/test) : 0.0338/0.0458,Acc (train/test): 0.7972/0.6348
Epoch [16/20], Loss (train/test) : 0.0220/0.0224,Acc (train/test): 0.8253/0.6054
Epoch [17/20], Loss (train/test) : 0.0182/0.0190,Acc (train/test): 0.8316/0.6634
Epoch [18/20], Loss (train/test) : 0.0148/0.0165,Acc (train/test): 0.8405/0.6434
Epoch [19/20], Loss (train/test) : 0.0119/0.0162,Acc (train/test): 0.8549/0.7115
Epoch [20/20], Loss (train/test) : 0.0112/0.0138,Acc (train/test): 0.8728/0.7622
pretty good acc for just changing last layer of ResNet18. This code can't show the full tunning process, but this architecture still can be good at classifying stuff like that
![598610](https://user-images.githubusercontent.com/124432421/236702689-1046c983-4402-436e-af59-3308e541ffef.jpg)
or that
![539877](https://user-images.githubusercontent.com/124432421/236702723-a644bbcf-1894-411b-a491-69e1207df6f6.jpg)
![372486](https://user-images.githubusercontent.com/124432421/236702725-14e9a29a-37ec-45ab-9078-c394e769d747.jpg)
# Creating database
Also u can check my Lmdb solution of storing data from images if u have same problems with timeworking of ur hard disk as me
