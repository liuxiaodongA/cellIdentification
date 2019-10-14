import cv2
import numpy as np
import os
from xlutils import copy
import xlrd



class Segmenter(object):
   def __init__(self):
      self._mask_32S = None
      self._waterImg = None
# 将掩膜转化为CV_32S
   def setMark(self, mask):
      self._mask_32S = np.int32(mask)
# 进行分水岭操作
   def waterProcess(self, img):
      self._waterImg = cv2.watershed(img, self._mask_32S)
# 获取分割后的8位图像
   def getSegmentationImg(self):
      segmentationImg = np.uint8(self._waterImg)
      return segmentationImg
# 处理分割后图像的边界值
   def getWaterSegmentationImg(self):
      waterSegmentationImg = np.copy(self._waterImg)
      waterSegmentationImg[self._waterImg == -1] = 1
      waterSegmentationImg = np.uint8(waterSegmentationImg)
      return waterSegmentationImg
# 将分水岭算法得到的图像与源图像合并 实现抠图效果
   def mergeSegmentationImg(self,img, waterSegmentationImg, isWhite = False):


      # 计算ret个数6-14,11.39   passing  modify by lxd
      ret, segmentMask = cv2.threshold(waterSegmentationImg, 250, 1, cv2.THRESH_BINARY)
      print("diyiciret",ret)
      segmentMask = cv2.cvtColor(segmentMask, cv2.COLOR_GRAY2BGR)
      mergeImg = cv2.multiply(img, segmentMask)
      if isWhite is True:
         mergeImg[mergeImg == 0] = 255
      return mergeImg

def getBoundingRect(img, pattern):
   _, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   x, y, w, h = cv2.boundingRect(contours[1])
   cv2.rectangle(pattern, (x, y), (x + w, y + h), (0, 0, 200), 2)


def makegray2white(img,num = 75):
   shape_a = img.shape
   height = shape_a[0]
   width = shape_a[1]

   ret,img = cv2.threshold(img,74,255,cv2.THRESH_OTSU)
   for i in range(0,height):
      for j in range(0, width):
         pv = img[i, j]
         if(pv>=num):
            img[i, j] = 255
         else:
            img[i, j] = 0
   cv2.imshow("xiugaihou",img)
   # img = cv2.GaussianBlur(img, (0, 0), 0.3)
   cv2.imwrite("D:/test_picture/train_double/XIUGAIHOU.png", img)
   return img


def makeblack2white(img):
   shape_a = img.shape
   height = shape_a[0]
   width = shape_a[1]

   # ret,img = cv2.threshold(img,74,255,cv2.THRESH_OTSU)
   for i in range(0, height):
      for j in range(0, width):
         pv = img[i, j]
         if (pv == 0):
            img[i, j] = 255
   cv2.imshow("xiugaihou", img)
   # img = cv2.GaussianBlur(img, (0, 0), 0.3)
   cv2.imwrite("D:/test_picture/train_double/aaa.png", img)
   return img

# 高斯降噪
def Gaussian_removenoise(image):
   dst = cv2.GaussianBlur(image, (0, 0), 40)
   cv2.imshow("Gaussian", dst)


# 高斯双边滤波
def bi_fuc(img):
   bil_img = cv2.bilateralFilter(img, 0, 180, 5)
   cv2.imshow("bi_demo", bil_img)
   return bil_img


def erodePIC(img):

   ret, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
   cv2.imshow("222", binary)
   kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
   dst = cv2.erode(binary, kernel)

   cv2.imshow("erode", dst)
   return dst

def wirte_excel_data(excel_path, self, row, col, value_data):
   # 打开文件，并且保留原格式
   self.rbook = xlrd.open_workbook(excel_path, formatting_info=True)
   # 使用xlutils的copy方法使用打开的excel文档创建一个副本
   self.wbook = copy(self.rbook)
   # 使用get_sheet方法获取副本要操作的sheet
   self.w_sheet = self.wbook.get_sheet(0)
   # 写入数据参数包括行号、列号、和值（其中参数不止这些）
   self.w_sheet.write(row, col, value_data)
   # 保存
   self.wbook.save(excel_path)



# def areaCal(contour, excel_path, file, row):
#    area = 0
#    num = 1
#    file_num = file[:4]
#    cell_ty = file[-9:-4]
#
#    excel_path = excel_path.encode("utf-8")
#    rbook = xlrd.open_workbook(excel_path, formatting_info=True)
#    wbook = copy.copy(rbook)
#    w_sheet = wbook.get_sheet(0)
#
#
#    w_sheet.write(row, 1, file_num)
#    w_sheet.write(row, 2, cell_ty)
#
#
#
#    for i in range(len(contour)):
#       area = cv2.contourArea(contour[i])
#       if(area > 100):
#          print(area)
#          num = num + 1
#          w_sheet.write(row, 2 + num, area)
#
#
#       area = 0
#    print("i is %s", num)
#    w_sheet.write(row, 3, num)
#    wbook.save(excel_path)

   # def waterPIC(img, excel_path, file, row):
def waterPIC(img):
   mySegmenter = Segmenter()
   # 获取前景图片
   grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   bil_img = bi_fuc(grayImg)
   grayImg = makegray2white(bil_img)
   # 将颜色除黑色以外变为白色

   blurImg = cv2.blur(grayImg, (3, 3))
   _, binImg = cv2.threshold(blurImg, 30, 255, cv2.THRESH_BINARY_INV)
   kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
   fgImg = cv2.morphologyEx(binImg, cv2.MORPH_CLOSE, kernel1)
   # 获取背景图片
   kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
   dilateImg = cv2.dilate(binImg, kernel2, iterations=4)
   _, bgImg = cv2.threshold(dilateImg, 1, 128, cv2.THRESH_BINARY_INV)
   # 合成掩膜
   maskImg = cv2.add(fgImg, bgImg)
   mySegmenter.setMark(maskImg)
   # 进行分水岭操作 并获得分割图像
   mySegmenter.waterProcess(img)
   waterSegmentationImg = mySegmenter.getWaterSegmentationImg()
   outputImgWhite = mySegmenter.mergeSegmentationImg(img, waterSegmentationImg, True)
   kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
   dilateImg = cv2.dilate(waterSegmentationImg, kernel3)
   _, dilateImg = cv2.threshold(dilateImg, 130, 255, cv2.THRESH_BINARY)
   getBoundingRect(dilateImg, img)



#计算面积
   coimg = makeblack2white(maskImg)
   # cv2.bitwise_not(src, src)
   coimg = makegray2white(coimg, 160)
   cv2.bitwise_not(coimg, coimg)
   coimg = erodePIC(coimg)

   image, contours, hierarchv = cv2.findContours(coimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   cv2.imshow("contours", image)
   # print("counter",np.num(contours))

   color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
   # 将轮廓在Image中生成
   Image = cv2.drawContours(color, contours, -1, (0, 0, 255), 1)
   cv2.imshow("lunkuo", Image)
   # areaCal(contours, excel_path, file,row)

   return dilateImg,outputImgWhite,maskImg,Image


# def readFile(path):




# 读取文件夹内的所有文件
def Filemain(filepath, outpath, excel_path):


   fileNames = os.listdir(filepath)  # 获取当前路径下的文件名，返回List
   num_col = 0
   for file in fileNames:
      if "CELL1" in file:
         num_col = num_col + 1
         newDir = filepath + '/' + file # 将文件命加入到当前文件路径后面
         img = cv2.imread(newDir)
         dilateImg, outputImgWhite, maskImg, img, Image = waterPIC(img, excel_path, file, num_col)

         file = file[:-4]
         cv2.imshow('Contours Image', dilateImg)
         cv2.imwrite(outpath+'/'+file+"Contours.png", dilateImg)

         cv2.imshow('White Image', outputImgWhite)
         cv2.imwrite(outpath+'/'+file+"outputImgWhite.png", outputImgWhite)

         cv2.imshow('Mask Image', maskImg)
         cv2.imwrite(outpath+'/'+file+"maskIMG.png", maskImg)

         cv2.imshow('lunkuo', Image)
         cv2.imwrite(outpath+'/'+file+"lunkuo.png", Image)

         cv2.imshow('Output Image', img)
         cv2.imwrite(outpath + '/' + file + "Output.png", img)



      if "CELL2" in file:
         num_col = num_col + 1
         newDir = filepath + '/' + file # 将文件命加入到当前文件路径后面
         img = cv2.imread(newDir)
         dilateImg, outputImgWhite, maskImg, img, Image = waterPIC(img, excel_path, file, num_col)

         file = file[:-4]
         cv2.imshow('Contours Image', dilateImg)
         cv2.imwrite(outpath+'/'+file+"dilateImg.png", dilateImg)

         cv2.imshow('White Image', outputImgWhite)
         cv2.imwrite(outpath+'/'+file+"outputImgWhite.png", outputImgWhite)

         cv2.imshow('Mask Image', maskImg)
         cv2.imwrite(outpath+'/'+file+"maskIMG.png", maskImg)

         cv2.imshow('Output Image', img)
         cv2.imwrite(outpath+'/'+file+"Output.png", img)

         cv2.imshow('lunkuo', Image)
         cv2.imwrite(outpath + '/' + file + "lunkuo.png", Image)

   return num_col

#
#
# path = "E:/2019cellnum/aaa/" #文件夹目录
# outpayh = "E:/2019cellnum/outpath/"
# excel_path = "E:/2019cellnum/text.xls"
#
# Filemain(path,outpayh, excel_path)
#
#
# cv2.waitKey()
# cv2.destroyAllWindows()