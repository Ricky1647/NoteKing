from .fully_conv_model import cnn_attention_ocr

from .evaluation import wer_eval,preds_to_integer,show,my_collate,AverageMeter

from PIL import Image
import numpy as np 
import string

from skimage.color import gray2rgb
from skimage.transform import resize
import cv2

def make_dict():
    pool = ''
    pool += string.ascii_letters
    pool += "0123456789"
    pool += "!\"#$%&'()*+,-./:;?@[\\]^_`{|}~"
    pool += ' '
    keys = list(pool)
    values = np.array(range(1,len(pool)+1))
    dictionary = dict(zip(keys, values))

    decode_dict=dict((v,k) for k,v in dictionary.items())
    decode_dict.update({93 : "OOK"})
    return decode_dict

def OCR(img, cnn, device, decode_dict):
    img = img.convert('L')
    img=cv2.cvtColor(np.array(img),cv2.COLOR_GRAY2RGB)
    img=img/255
    resize_shape=(32,int(32*img.shape[1]/img.shape[0]))
    img = resize(img,resize_shape,mode="constant")
    img=np.expand_dims(img,0)

    with torch.no_grad():
        img=torch.tensor(img, device=device).float().permute((0,3,1,2))

        log_probs = cnn(img).permute((2,0,1))[:,0,:]
        output = "".join([decode_dict[j] for j in preds_to_integer(log_probs)])
    return output