from PIL import Image

import os

srcpath = r"D:\AI\秀傳提供的bonescan檔案\已處理\彰濱Y(異常)" #目標資料夾
targetpath = r"D:\AI\秀傳提供的bonescan檔案\已處理\彰濱Y(異常)128"#成功資料夾
size = 128#要轉的大小

filelist = os.listdir(srcpath)

def fixed_size(image, size):
    """按照固定尺寸"""
    image = image.resize((size, size), Image.ANTIALIAS)#Image.ANTIALIAS 高质量
    image.save(targetpath + "\\" + file, image.format)#存檔
    
    
def resize_by_percentage(image,percentage):
    '''百分比縮放 80 = 80%'''
    (x, y) = image.size
    per = (percentage/100)
    x_resize = int(x * per)
    y_resize = int(y * per)
    outimage = image.resize((x_resize, y_resize), Image.ANTIALIAS)#Image.ANTIALIAS 高质量
    outimage.save(targetpath + "\\" + file, image.format)#存檔
        

for file in filelist:
    fd_img = open(srcpath + "\\" + file, 'r')#開啟檔案
    print(fd_img.name)
    img = Image.open(fd_img.name)
    #縮到256*256
    resize_by_percentage(img, 80)#縮放圖片的方式 80 = 80%
    fd_img.close()

print("=============FINISHED==============")
