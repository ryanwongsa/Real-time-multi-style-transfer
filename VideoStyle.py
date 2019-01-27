import torch
from torchvision import transforms
from inference.Inferencer import Inferencer
from models.PasticheModel import PasticheModel
from PIL import Image
import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_styles = 16
image_size = 512
mean = [0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
model_save_dir = "style16/pastichemodel-FINAL.pth"
transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                             std=std)
    ])

pastichemodel = PasticheModel(num_styles)
inference = Inferencer(pastichemodel,transform,device)
inference.load_model_weights(model_save_dir)

count_total = 0
cap = cv2.VideoCapture('../thousand-miles.mp4')
choice = 0
choice2 = 1
percentage = 0

frame_width = int(912)
frame_height = int(512)
# print(frame_width,frame_height)
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))


while(cap.isOpened()):
    ret, frame = cap.read()
    
    if ret == True:
        print(count_total)
        cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         print(cv2_im.shape)
        pil_im = Image.fromarray(cv2_im)
        res_im = inference.eval_image(pil_im,choice,choice2,percentage)
        open_cv_image = cv2.cvtColor(np.array(res_im), cv2.COLOR_RGB2BGR)
#         print(open_cv_image.shape)
        out.write(open_cv_image)
# #         cv2.imshow('frame',cv2_im)
#         cv2.imshow('frame',frame)
        count_total+=1
        percentage+=0.01
        
        
        if percentage>1:
            percentage=0
            choice2 = choice
            choice = (choice+1)%16

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
out.release()
cap.release()
cv2.destroyAllWindows()