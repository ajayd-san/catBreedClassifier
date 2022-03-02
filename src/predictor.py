import joblib
import torchvision
import torch 
from skimage import io
from torch import nn
from torch.nn import functional as F
import albumentations as A
import pandas as pd

class ModifiedPretrained(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.modifiedModel = pretrained_model
        in_features = self.modifiedModel.classifier.in_features
        model = nn.Sequential(
            nn.Linear(in_features, 512), 
            nn.ReLU(),
            nn.Linear(512, 128), 
            nn.ReLU(),
            nn.Linear(128, 12)
        )

        self.modifiedModel.classifier = model

        
    
    def forward(self, input):
        logits = self.modifiedModel(input)
        return logits

densenet = torchvision.models.densenet121() 
model = ModifiedPretrained(densenet)
model.load_state_dict(torch.load('saved_models/iniModel.pth', map_location='cpu'))
label_encoder = joblib.load('saved_models/labelEncoder.joblib')
def predict(img):

    transform = A.Compose([
        A.Resize(224, 224)
    ])

    img = io.imread(img)
    img = transform(image = img)['image']
    img = torch.tensor(img, dtype = torch.float)
    img = torch.permute(img, (2, 0, 1))
    img = torch.unsqueeze(img, 0)
    
    with torch.no_grad():
        model.eval()
        logits = model(img)
        pred = F.softmax(logits, 1) * 100

    pred_df = pd.Series(pred.squeeze(), name = 'proba').to_frame()
    pred_df = pred_df.nlargest(3, 'proba')
    pred_df['encoded_label'] = pred_df.index
    pred_df['label'] = pred_df['encoded_label'].apply(lambda x: label_encoder.inverse_transform([x])[0])
    return pred_df[['label', 'proba']].to_dict('records')
    