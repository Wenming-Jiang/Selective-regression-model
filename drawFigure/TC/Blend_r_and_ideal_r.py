import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def blend_r_and_ideal_r(inputfile):
    #ordered_var_dy = np.load("./Var2015-2017/15/15_10.45.npy").astype('float32')   #change
    ordered_var_dy = np.load(inputfile).astype('float32')   #change
    #np.save("./Var2015-2017/15/" + "var-dy-random_9_24/" + "shuffleindex.npy", index)
    index = np.load("./Var2015-2017/10/" + "var-dy-random_9_30/" + "shuffleindex.npy")   #unchange
    
    middle = int(ordered_var_dy.shape[0] / 2)
    test_index = index[:middle]
    val_index = index[middle:]
    test_index.sort()
    val_index.sort()
    print(len(test_index), len(val_index))

    test_var_dy = []
    val_var_dy = []
    for i in test_index:
        test_var_dy.append(ordered_var_dy[i,:])
    for i in val_index:
        val_var_dy.append(ordered_var_dy[i, :])
    test_var_dy = np.array(test_var_dy)
    val_var_dy = np.array(val_var_dy)
    print(test_var_dy.shape, val_var_dy.shape)

    test_mse = np.zeros((test_var_dy.shape[0],1))
    test_t   = np.zeros((test_var_dy.shape[0],1))
    test_b   = np.zeros((test_var_dy.shape[0],1))
    for i in range(test_var_dy.shape[0]):
        test_mse[i][0] = np.mean(test_var_dy[:,1][:(i+1)]*test_var_dy[:,1][:(i+1)])
        test_t[i][0]   = 489.55 / np.sqrt(i+1) #* 1.1   #delta = 0.05, b-a = 20*1.1 * 20*1.1 - 0
        test_b[i][0]   = test_mse[i][0] + test_t[i][0]
    test = np.concatenate((test_var_dy, test_mse, test_t, test_b), axis = 1)

    val = test

    var_sorted_test_dy = test_var_dy[:,1]
    sorted_dy = np.sort(np.abs(var_sorted_test_dy))
    ideal_mse = np.zeros((test_var_dy.shape[0],1))
    for i in range(test_var_dy.shape[0]):
        ideal_mse[i][0] = np.mean(sorted_dy[:(i+1)]*sorted_dy[:(i+1)])


    index = np.zeros((test_var_dy.shape[0],1))
    index[:, 0] = np.arange(1, test_var_dy.shape[0]+1)
    testmse_testb_idealmse = np.concatenate((index, test_mse, test_b, ideal_mse), axis = 1)

    np.savetxt("testmse_testb_idealmse.csv", testmse_testb_idealmse, delimiter=',')
    np.save("testmse_testb_idealmse.npy", testmse_testb_idealmse)


blend_r_and_ideal_r("C:/Users/jwm/Desktop/20191229/TC/Var2015-2017/10/10_10.49.npy")

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
