# Create Model:
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
import torch
from torch import nn

def get_model(num_classes):
    weights = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
    model = deeplabv3_resnet101(weights, progress=True)
    # Same as before just change out_channels
    model.classifier[4] = nn.Conv2d(model.classifier[-1].in_channels, num_classes, 1, 1)
    
    for param in model.parameters():
        param.requires_grad = True
    
    return model


def train_model(model, train_dataloader, test_dataloader, config, optimizer, device):
    epochs = config["epochs"]
    criterion = config["criterion"] 

    
    model.to(device)
    print("Using Device: ", device)

    for epoch in range(epochs):
        # training
        print(f'Current epoch: {epoch+1}')

        vall_loss_list = []
        train_loss_list = []
        
        for batch_i, data in enumerate(train_dataloader):
            model.train()

            x,y = data[0].to(device), data[1].to(device) # x = bs, c, height, widhth
            
            pred_mask = model(x)  
            pred_mask = pred_mask['out'] # model creates dict out and aux we need out 
            
            bs,channels, height, width = y.shape
            #print("y before = ", y.shape) # we expect bs, imheight, imwidth but  we got bs, 1, imheight, imwidth
            y = y.view(bs,height, width)
            loss = criterion(pred_mask, y)

            optimizer.zero_grad()
            loss.backward()
            train_loss_list.append(loss.item())

            optimizer.step()
            
        # validation
        model.eval()
        for batch_i, data in enumerate(test_dataloader):
            x,y = data[0].to(device), data[1].to(device)
            bs,channels, height, width = y.shape
            y = y.view(bs,height, width)

            with torch.no_grad():    
                pred_mask = model(x)  
                pred_mask = pred_mask['out']
            val_loss = criterion(pred_mask, y)
            vall_loss_list.append(val_loss.item())
            
        print("Train_loss: ", sum(train_loss_list)/len(train_loss_list))
        print("Val_loss: ", sum(vall_loss_list)/len(vall_loss_list))
    return model    