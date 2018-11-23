from PIL import Image

import os

srcpath = r"D:\AI\秀傳提供的bonescan檔案\已處理\彰濱Y(異常)" #目標資料夾
targetpath = r"D:\AI\秀傳提供的bonescan檔案\已處理\彰濱Y(異常)128"#成功資料夾
size = 128#要轉的大小

filelist = os.listdir(srcpath)
for file in filelist:
    fd_img = open(srcpath + "\\" + file, 'r')#開啟檔案
    print(fd_img.name)
    img = Image.open(fd_img.name)
    #縮到256*256
    img = img.resize((size, size), Image.ANTIALIAS)#Image.ANTIALIAS 高质量
    img.save(targetpath + "\\" + file, img.format)#存檔
    fd_img.close()

print("=============FINISHED==============")
