import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



val_data = np.load("val20_predict_age_MAE5.37.npy")  # (1034, 1+20) real+20predict
diffT = [2,4,8,10,15,20]

diffT_mae = np.zeros((val_data.shape[0], len(diffT)+1))
diffT_mae[:,0] = np.arange(1, val_data.shape[0]+1)
no = 1

groundTruth = val_data[:,0]
print(groundTruth.shape)
for T in diffT:
	predict_age = val_data[:, 1:(1+T)]
	average = np.mean(predict_age, axis = 1)
	#print(average.shape)
	dy = average - groundTruth   # f(x) - y
	#print(type(dy),dy.shape)
	var =  np.var(predict_age, axis = 1)
	index = np.argsort(var)
	sorted_var = var[index]
	sorted_dy  = dy[index]
	mae = np.zeros(dy.shape[0])
	for i in range(dy.shape[0]):
		mae[i] = np.mean(np.abs(sorted_dy[:(i+1)]))

	var_dy_mae = np.zeros((dy.shape[0], 3))
	var_dy_mae[:,0] = sorted_var
	var_dy_mae[:,1] = sorted_dy
	var_dy_mae[:,2] = mae

	np.savetxt("age_T"+str(T)+ ".csv", var_dy_mae, delimiter=',')    # var dy mae
	np.save("age_T"+str(T)+".npy", var_dy_mae)

	diffT_mae[:,no] = mae
	no = no + 1

np.save("diffT_mae.npy", diffT_mae)
np.savetxt("diffT_mae.csv", diffT_mae, delimiter=',')

exit()


#使用numpy产生数据
x=np.arange(-5,5,0.1)
y=x*3

#创建窗口、子图
#方法1：先创建窗口，再创建子图。（一定绘制）
fig = plt.figure(num=1, figsize=(15, 8),dpi=80)     #开启一个窗口，同时设置大小，分辨率
ax1 = fig.add_subplot(2,1,1)  #通过fig添加子图，参数：行数，列数，第几个。
ax2 = fig.add_subplot(2,1,2)  #通过fig添加子图，参数：行数，列数，第几个。
print(fig,ax1,ax2)

#方法2：一次性创建窗口和一个子图。（空白不绘制）
#ax1 = plt.subplot(1,1,1,facecolor='white')      #开一个新窗口，创建1个子图。(1,1,1)表示1行，1列，第1个子图，facecolor设置背景颜色
#print(ax1)

#获取对窗口的引用
# fig = plt.gcf()   #获得当前figure
# fig=ax1.figure   #获得指定子图所属窗口

#设置子图的基本元素
ax1.set_title('python-drawing')            #设置图体，plt.title
ax1.set_xlabel('x-name')                    #设置x轴名称,plt.xlabel
ax1.set_ylabel('y-name')                    #设置y轴名称,plt.ylabel
plt.axis([-6,6,-10,10])                  #设置横纵坐标轴范围，这个在子图中被分解为下面两个函数
ax1.set_xlim(-5,5)                           #设置横轴范围，会覆盖上面的横坐标,plt.xlim
ax1.set_ylim(-10,10)                         #设置纵轴范围，会覆盖上面的纵坐标,plt.ylim

plot1=ax1.plot(x,y,marker='o',color='g',label='legend1')   #点图：marker图标
plot2=ax1.plot(x,y,linestyle='--',alpha=0.5,color='r',label='legend2')   #线图：linestyle线性，alpha透明度，color颜色，label图例文本

ax1.legend(loc='upper left')            #显示图例,plt.legend()
ax1.text(2.8, 7, r'y=3*x')                #指定位置显示文字,plt.text()
ax1.annotate('important point', xy=(2, 6), xytext=(3, 1.5),  #添加标注，参数：注释文本、指向点、文字位置、箭头属性
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
#显示网格。which参数的值为major(只绘制大刻度)、minor(只绘制小刻度)、both，默认值为major。axis为'x','y','both'
ax1.grid(b=True,which='major',axis='both',alpha= 0.5,color='skyblue',linestyle='--',linewidth=2)

axes1 = plt.axes([.2, .3, .1, .1], facecolor='y')       #在当前窗口添加一个子图，rect=[左, 下, 宽, 高]，是使用的绝对布局，不和以存在窗口挤占空间
axes1.plot(x,y)  #在子图上画图
plt.savefig('aa.jpg',dpi=400,bbox_inches='tight')   #savefig保存图片，dpi分辨率，bbox_inches子图周边白色空间的大小
plt.show()    #打开窗口，对于方法1创建在窗口一定绘制，对于方法2方法3创建的窗口，若坐标系全部空白，则不绘制
