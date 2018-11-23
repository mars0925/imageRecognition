from PIL import Image
from PIL import Image
import os

srcpath = r"D:\AI\檔名處理過的所有圖檔_多LABEL_\圖檔" #目標資料夾
targetpath = r"D:\xray32"#成功資料夾

filelist = os.listdir(srcpath)
for file in filelist:
    fd_img = open(srcpath + "\\" + file, 'r')#開啟檔案
    print(fd_img.name)
    img = Image.open(fd_img.name)
    #縮到256*256
    img = img.resize((32, 32), Image.ANTIALIAS)#Image.ANTIALIAS 高质量
    img.save(targetpath + "\\" + file, img.format)#存檔
    fd_img.close()


