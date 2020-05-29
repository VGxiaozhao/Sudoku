# 数独  
使用了OpenCV+pytorch来处理一张数独图片，使用示例: python3 pipeline.py  

## 前言  
这个是我的第一个图像算法的练手项目，想给大家一些参考，有何不足请批评指正。  
整体的想法主要是分两步，一是数字位置提取，先确定整体数独框后再确定数字位置，提取到数字之后，交给第二部分CNN处理。解数独方面，直接暴力深度搜索即可。  

## 数字提取  
确定框位置：通过opencv findContours找到所有轮廓后，使用arcLength 和 approxPolyDP找到最大的顶角数为4的轮廓即为数独框的位置。  
提取数字位置：这一步比较直接，通过广度搜索将边框线清除，清除完毕之后调用findContours 就能直接得到各个数字的位置了，效果如下。  
![alt 数字提取效果](http://www.srzzc.cn/github_number_extract.png)  

## 数字识别  
这一步比较容易实现，毕竟很多人学习神经网络的第一个项目就是识别mnist手写数字识别。  
但是我们的工程是识别印刷体数字，我实际测试后发现利用mnist的手写数字作为训练集实际效果并不理想。所以我们需要收集相关印刷体数字训练集。  
吐槽一下CSDN的资源分享太恶臭，需要充钱积分才能下载，这还叫分享吗？  
opencv 有一个 putText函数可以生成印刷体数字，函数原型：  
```
putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]]) -> img
    .   @brief Draws a text string.
    .
    .   The function cv::putText renders the specified text string in the image. Symbols that cannot be rendered
    .   using the specified font are replaced by question marks. See #getTextSize for a text rendering code
    .   example.
    .
    .   @param img Image.
    .   @param text Text string to be drawn.
    .   @param org Bottom-left corner of the text string in the image.
    .   @param fontFace Font type, see #HersheyFonts.
    .   @param fontScale Font scale factor that is multiplied by the font-specific base size.
    .   @param color Text color.
    .   @param thickness Thickness of the lines used to draw a text.
    .   @param lineType Line type. See #LineTypes
    .   @param bottomLeftOrigin When true, the image data origin is at the bottom-left corner. Otherwise,
    .   it is at the top-left corner.
```
这个就很不错，参数还指定了字体的大小和粗细，恶心的是字体方框的原点是在左下角，跟图形学是不是有点不符？  
利用这个函数搭配上CV2的getTextSize函数我们可以得到字体的大小，这样就能准确的将数字打印在28*28的图片上每个角落。通过调整数字在图片中的位置，粗细，大小等达到了数据增强的目的，具体可以查看train.py的get_cv_data函数  
注意到putText函数只提供了8种字体，这个是远远不够的。在这里感谢https://github.com/PowerOfDream/digitx 提供的相关字体文件以及truetype生成印刷体数字思路  
使用python的PIL库可以读取一个字体ttf文件生成数据，但是由于不提供实际字体大小的函数，我在这里将图片扩大了一倍（确保数字不越界），然后使用opencv的findContours得到数字的实际大小，就可以生成多张位置不同的数字图了。具体可以查看train.py的get_pil_data函数  
cnn使用AlexNet实现，实际在训练时验证准确率约99.5左右

最终运行效果:  
![alt 运行效果](http://www.srzzc.cn/github_number_result.png)
