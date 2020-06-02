# coding=utf-8
# 导入一些python包
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
from matplotlib import pyplot as plt
import numpy as np
from keras.models import load_model

# 加载模型
model =load_model('model.h5')

# 改变图片维度
def changeDim(img):

    img = np.expand_dims(img, axis=2)
    img = np.expand_dims(img, axis=0)

    return img

# 读取输入图片
image = cv2.imread("0.jpg")

# 将输入图片裁剪到固定大小
image = imutils.resize(image, height=500)
# 将输入转换为灰度图片
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('1-gray.png', gray)

plt.subplot(121),plt.imshow(image,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(gray,cmap = 'gray')
plt.title('gray Image'), plt.xticks([]), plt.yticks([])
plt.show()

# 进行高斯模糊操作
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imwrite('2-blurred.png', blurred)

plt.subplot(121),plt.imshow(gray,cmap = 'gray')
plt.title('gray Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blurred,cmap = 'gray')
plt.title('blurred Image'), plt.xticks([]), plt.yticks([])
plt.show()

# 执行边缘检测，返回的是二值图，阈值50，阈值200，可选参数255 Sobel算子的大小
edged = cv2.Canny(blurred, 50, 200, 255)
cv2.imwrite('3-edge.png', edged)
# 显示对比图
plt.subplot(121),plt.imshow(blurred,cmap = 'gray')
plt.title('blurred Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edged,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()

# 在边缘检测map中发现轮廓，只检测外轮廓，压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 返回cnts中的countors(轮廓)
cnts = imutils.grab_contours(cnts)
# 根据大小对这些轮廓进行排序，cnts可迭代对象，key进行比较的元素，规则是降序
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
displayCnt = None

# 循环遍历所有的轮廓
for c in cnts:
	# 对轮廓进行近似
	peri = cv2.arcLength(c, True)
	#c输入的点集，0.02指定精度，true近似曲线是闭合的
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)

	# 如果当前的轮廓有4个顶点，我们返回这个结果，即LCD所在的位置
	if len(approx) == 4:
		displayCnt = approx
		break

# 绘制轮廓
img = cv2.drawContours(image.copy(), displayCnt, -1, (255,0,0), 6)
cv2.imwrite('4-Contours.png', img)

plt.subplot(121),plt.imshow(edged,cmap = 'gray')
plt.title('edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img,cmap = 'gray')
plt.title('Contours Image'), plt.xticks([]), plt.yticks([])
plt.show()


# 应用视角变换到LCD屏幕上，gray原始图，轮廓图*缩放比
warped = four_point_transform(gray, displayCnt.reshape(4, 2))
cv2.imwrite('5-warped.png', warped)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Contours Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(warped,cmap = 'gray')
plt.title('warped Image'), plt.xticks([]), plt.yticks([])
plt.show()

output = four_point_transform(image, displayCnt.reshape(4, 2))

# 使用阈值进行二值化
thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cv2.imwrite('6-thresh.png', thresh)

plt.subplot(121),plt.imshow(warped,cmap = 'gray')
plt.title('warped Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(thresh,cmap = 'gray')
plt.title('threshold Image'), plt.xticks([]), plt.yticks([])
plt.show()

# 返回指定形状的结构元素，MORPH_ELLIPSE椭圆形
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5)) 

# 使用形态学操作进行处理，进行开运算，指的是先进行腐蚀操作，再进行膨胀操作
thresh2 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
cv2.imwrite('7-thresh.png', thresh2)

plt.subplot(121),plt.imshow(thresh,cmap = 'gray')
plt.title('threshold Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(thresh2,cmap = 'gray')
plt.title('OPEN Image'), plt.xticks([]), plt.yticks([])
plt.show()


# 在阈值图像中查找轮廓，然后初始化数字轮廓列表
cnts = cv2.findContours(thresh2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# imutils.grab_contours返回cnts中的countors(轮廓)
cnts = imutils.grab_contours(cnts)
digitCnts = []

# 循环遍历所有的候选区域
for c in cnts:
	# 计算轮廓的边界框
	(x, y, w, h) = cv2.boundingRect(c)

	# 如果当前的这个轮廓区域足够大，它一定是一个数字区域
	if w >= 15 and (h >= 30 and h <= 40):
		digitCnts.append(c)


# 从左到右对这些轮廓进行排序
digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
digits = []

# 绘制轮廓
imge = cv2.drawContours(warped.copy(), digitCnts, -1, (255,0,0), 2)
cv2.imwrite('8-Contours.png', imge)

plt.subplot(121),plt.imshow(thresh2,cmap = 'gray')
plt.title('OPEN Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(imge,cmap = 'gray')
plt.title('Contour Image'), plt.xticks([]), plt.yticks([])
plt.show()

# 循环处理每一个数字
i = 0
for c in digitCnts:
	# 获取ROI区域
	(x, y, w, h) = cv2.boundingRect(c)
	width = np.max([h, w])
	if width > 30 and width < 150: 

		# 显示矩形区域 
		# output原图，左上点坐标，右下点坐标，绿色(0, 255, 0)来画出最小的矩形框架，3线宽
		output = cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
		imgTest = thresh2[y:y + h, x:x + w]
		cv2.imwrite('9-roi1.png', imgTest)
		imgTest = cv2.copyMakeBorder(imgTest,0, 0, 8, 8 ,cv2.BORDER_CONSTANT,value=[0,0,0])
		cv2.imwrite('10-roi2.png', imgTest)
		
		if imgTest.shape[0] > 0 and imgTest.shape[1] > 0:
			# 识别字符
			# imgTest原图，(28, 28)大小，INTER_CUBIC - 基于4x4像素邻域的3次插值法
			imgTest = cv2.resize(imgTest, (28, 28), interpolation=cv2.INTER_CUBIC)
			cv2.imwrite('11-imgTest.png', imgTest)
			imgTest = changeDim(imgTest)
			resTest = np.argmax(model.predict(imgTest))
			# 显示识别结果
			font = cv2.FONT_HERSHEY_SIMPLEX
			output =  cv2.putText(output, str(resTest), (x - 10 , y - 10 ), font, 0.65, (0, 255, 0), 1)


plt.subplot(3,3,1),plt.imshow(image,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(3,3,2),plt.imshow(gray,cmap = 'gray')
plt.title('gray Image'), plt.xticks([]), plt.yticks([])

plt.subplot(3,3,3),plt.imshow(blurred,cmap = 'gray')
plt.title('blurred Image'), plt.xticks([]), plt.yticks([])

plt.subplot(3,3,4),plt.imshow(edged,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.subplot(3,3,5),plt.imshow(img,cmap = 'gray')
plt.title('Contours Image'), plt.xticks([]), plt.yticks([])

plt.subplot(3,3,6),plt.imshow(warped,cmap = 'gray')
plt.title('warped Image'), plt.xticks([]), plt.yticks([])

plt.subplot(3,3,7),plt.imshow(thresh,cmap = 'gray')
plt.title('threshold Image'), plt.xticks([]), plt.yticks([])

plt.subplot(3,3,8),plt.imshow(thresh2,cmap = 'gray')
plt.title('OPEN Image'), plt.xticks([]), plt.yticks([])

plt.subplot(3,3,9),plt.imshow(imge,cmap = 'gray')
plt.title('Contour Image'), plt.xticks([]), plt.yticks([])

plt.show()

# 显示最终的输出结果
plt.subplot(121),plt.imshow(image,cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(output,cmap = 'gray')
plt.title('Output Image'), plt.xticks([]), plt.yticks([])
plt.show()

cv2.waitKey(0)