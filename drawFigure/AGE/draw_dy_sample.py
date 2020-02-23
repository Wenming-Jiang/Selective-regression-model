import matplotlib.pyplot as plt
import numpy as np

def draw_dy_sample():
	# 画 f(x)-y VS sample的图
	data = np.load("age_T20.npy").astype("float32") # var dy mae
	dy = data[:, 1]

	ID = np.arange(1, data.shape[0]+1)

	#figsize = 8,6   # 800 * 600 像素
	figsize = 7,5   # 800 * 600 像素
	figure, ax = plt.subplots(figsize=figsize)
	#fig = plt.gcf()
	#fig.set_size_inches(6.4, 4.0)

	plt.xlim(-10, 1080)
	plt.ylim(-30, 42)

	plt.scatter(ID,dy,s=10, c='black')
	# 参数 s：设置散点大小
	# 参数 c：设置散点颜色；常用的'r','b','g','w'...
	# 参数 marker： 设置散点形状；常用的'+', 'o'，'x'...

	#设置坐标刻度值的大小以及刻度值的字体
	plt.tick_params(labelsize=15)

	#设置坐标显示
	#new_ticks = np.linspace(0,1200,5)   #plt.xticks(new_ticks)
	#在对应坐标处更换名称
	plt.xticks([0, 250, 500, 750, 1000])
	plt.yticks([-20, 0, 20, 40])


	font1={'weight':'semibold',
    	'size':20
	}
	#styles=['normal','italic','oblique']
    #weights=['light','normal','medium','semibold','bold','heavy','black']

	plt.xlabel("Sample", fontdict = font1, horizontalalignment='center' )#x轴上的名字
	plt.ylabel("f(x)-y", fontdict = font1, verticalalignment ='center')#y轴上的名字
	plt.tight_layout() #调整整体空白
	plt.savefig('./Figure/age_dy_sampletmp.eps')
	plt.savefig('./Figure/age_dy_sampletmp.png')
	plt.show()


def TC_draw_dy_sample():
	# 画 f(x)-y VS sample的图
	data = np.load("../TC/Var2015-2017/10/var-dy-random_9_30/10_val.npy").astype("float32") # var dy mae t b
	dy = data[:, 1]
	ID = np.arange(1, data.shape[0]+1)
	#figsize = 8,6   # 800 * 600 像素
	figsize = 7,5   # 800 * 600 像素
	figure, ax = plt.subplots(figsize=figsize)
	#fig = plt.gcf()
	#fig.set_size_inches(6.4, 4.0)

	#plt.xlim(-10, 5600)
	#plt.ylim(-75, 50)

	plt.scatter(ID,dy,s=10, c='black')
	# 参数 s：设置散点大小
	# 参数 c：设置散点颜色；常用的'r','b','g','w'...
	# 参数 marker： 设置散点形状；常用的'+', 'o'，'x'...

	#设置坐标刻度值的大小以及刻度值的字体
	plt.tick_params(labelsize=15)

	#设置坐标显示
	#new_ticks = np.linspace(0,1200,5)   #plt.xticks(new_ticks)
	#在对应坐标处更换名称
	#plt.xticks([0, 1000, 2000, 3000, 4000, 5000])
	#plt.yticks([-50, -25, 0, 25, 50])


	font1={'weight':'semibold',
    	'size':20
	}
	#styles=['normal','italic','oblique']
    #weights=['light','normal','medium','semibold','bold','heavy','black']

	plt.xlabel("Sample", fontdict = font1, horizontalalignment='center' )#x轴上的名字
	plt.ylabel("f(x)-y", fontdict = font1, verticalalignment ='center')#y轴上的名字
	plt.tight_layout() #调整整体空白
	plt.savefig('./Figure/TC_dy_sampletmp.eps')
	plt.savefig('./Figure/TC_dy_sampletmp.png')
	plt.show()

def TC_testmse_testb_idealmse(row, col, num, fig):
	# 画 f(x)-y VS sample的图
	data = np.load("../TC/testmse_testb_idealmse.npy").astype("float32") # ID test_mse test_b, ideal_mse
	#ID = data[:,0]
	sumnum = data.shape[0]
	ID = np.linspace(0.0020, 0.9999, 20)
	print(ID.shape)
	print(ID*sumnum)
	test_mse = data[(ID*sumnum).astype(int), 1]
	test_b = data[(ID*sumnum).astype(int), 2]
	ideal_mse = data[(ID*sumnum).astype(int),3]
	figsize = 8,6   # 800 * 600 像素
	ax = fig.add_subplot(row, col, num)#, figsize=figsize)
	#fig = plt.gcf()
	#fig.set_size_inches(6.4, 4.0)

	#plt.xlim(-10, 5600)
	#plt.ylim(-75, 50)
	ax.plot(ID, test_b,    linewidth=2, c='black', marker='*', markersize=8, label=r'Blend-Var $r^\star$')
	ax.plot(ID, test_mse,  linewidth=2, c='b',     marker='^', markersize=8, label=r'Blend-Var $\hat{r}$')
	ax.plot(ID, ideal_mse, linewidth=2, c='r',     marker='.', markersize=8, label=r'Ideal $\hat{r}$')


	ax.axhline(y=data[int(ID[19]*sumnum), 1], ls="--",c="grey", linewidth=1)#添加水平直线
	#设置坐标刻度值的大小以及刻度值的字体
	ax.tick_params(labelsize=15)

	font1={'weight':'semibold',
    	'size':20
	}
	#styles=['normal','italic','oblique']
    #weights=['light','normal','medium','semibold','bold','heavy','black']
	ax.set_xlabel("Coverage", fontdict = font1 )#x轴上的名字
	ax.set_ylabel("Risk (MSE)", fontdict = font1)#y轴上的名字
	#ax.set_title("Risk-Coverage Curve on the Validation Set\n for TC Intensity Estimation.", fontsize = 15)
	#简单的设置legend(设置位置)
	#位置在右上角
	ax.legend(loc = 'upper right', fontsize = 15)
	#plt.savefig('./Figure/TC_testmse_testb_idealmse.eps')
	#plt.savefig('./Figure/TC_testmse_testb_idealmse.png')
	#plt.show()


def AGE_testmse_testb_idealmse(row, col, num, fig):
	# 画 f(x)-y VS sample的图
	data = np.loadtxt("./val_mae_val_b_ideal_mae/valmae_valb_idealmae.csv", delimiter=",", ) # ID val_mae val_b, ideal_mae
	#ID = data[:,0]
	print(data.shape)
	sumnum = data.shape[0]
	ID = np.linspace(0.0005, 0.9999, 20)
	val_mae = data[(ID*sumnum).astype(int), 1]
	val_b = data[(ID*sumnum).astype(int), 2]
	ideal_mae = data[(ID*sumnum).astype(int),3]
	figsize = 8,6   # 800 * 600 像素
	ax = fig.add_subplot(row, col, num)#, figsize=figsize)
	#fig = plt.gcf()
	#fig.set_size_inches(6.4, 4.0)

	#plt.xlim(-10, 5600)
	#plt.ylim(-75, 50)
	ax.plot(ID, val_b,    linewidth=2, c='black', marker='*', markersize=8, label=r'MC-dropout $r^\star$')
	ax.plot(ID, val_mae,  linewidth=2, c='b',     marker='^', markersize=8, label=r'MC-dropout $\hat{r}$')
	ax.plot(ID, ideal_mae, linewidth=2, c='r',     marker='.', markersize=8, label=r'Ideal $\hat{r}$')

	ax.axhline(y=data[int(sumnum)-1,1], ls="--",c="grey", linewidth=1)#添加水平直线
	#设置坐标刻度值的大小以及刻度值的字体
	ax.tick_params(labelsize=15)

	font1={'weight':'semibold',
    	'size':20
	}
	#styles=['normal','italic','oblique']
    #weights=['light','normal','medium','semibold','bold','heavy','black']
	ax.set_xlabel("Coverage", fontdict = font1 )#x轴上的名字
	ax.set_ylabel("Risk (MAE)", fontdict = font1)#y轴上的名字
	#ax.set_title("Risk-Coverage Curve on the Validation Set\n for Apparent Age Estimation.", fontsize = 15)
	#简单的设置legend(设置位置)
	#位置在右上角
	ax.legend(loc = 'lower right', fontsize = 15)
	#plt.savefig('./Figure/AGE_testmse_testb_idealmse.eps')
	#plt.savefig('./Figure/AGE_testmse_testb_idealmse.png')
	#plt.show()


def AGE_diffT(row, col, num, fig):
	# 画 f(x)-y VS sample的图
	data = np.load("./diffT_mae.npy").astype("float32") # ID 02 04 08 10 15 20
	ID = data[:,0]
	sumnum = data.shape[0]
	ID = np.linspace(0.0020, 0.9999, 20)
	print(ID.shape)
	print(ID*sumnum)
	mae_02 = data[(ID*sumnum).astype(int), 1]
	mae_04 = data[(ID*sumnum).astype(int), 2]
	mae_08 = data[(ID*sumnum).astype(int), 3]
	mae_10 = data[(ID*sumnum).astype(int), 4]
	mae_15 = data[(ID*sumnum).astype(int), 5]
	mae_20 = data[(ID*sumnum).astype(int), 6]

	figsize = 8,6   # 800 * 600 像素
	ax = fig.add_subplot(row, col, num)#, figsize=figsize)
	#fig = plt.gcf()
	#fig.set_size_inches(6.4, 4.0)

	#plt.xlim(-10, 5600)
	#plt.ylim(-75, 50)
	ax.plot(ID, mae_02, linewidth=2, marker='.', markersize=8, label='MAE_02')
	ax.plot(ID, mae_04, linewidth=2, marker='*', markersize=8, label='MAE_04')
	ax.plot(ID, mae_08, linewidth=2, marker='x', markersize=8, label='MAE_08')
	ax.plot(ID, mae_10, linewidth=2, marker='^', markersize=8, label='MAE_10')
	#plt.plot(ID, mae_15, linewidth=2, marker='.', markersize=8, label='MAE_15')
	ax.plot(ID, mae_20, linewidth=2, marker='o', markersize=8, label='MAE_20')

	ax.axhline(y=data[int(ID[19]*sumnum), 1], ls="--",c="grey", linewidth=1)#添加水平直线
	#设置坐标刻度值的大小以及刻度值的字体
	ax.tick_params(labelsize=15)

	font1={'weight':'semibold',
    	'size':20
	}
	#styles=['normal','italic','oblique']
    #weights=['light','normal','medium','semibold','bold','heavy','black']
	ax.set_xlabel("Coverage", fontdict = font1)#x轴上的名字
	ax.set_ylabel("Risk (MAE)", fontdict = font1)#y轴上的名字
	#ax.set_title("the Influence of $T$ in MC-dropout.", fontsize = 15)
	#简单的设置legend(设置位置)
	#位置在右上角
	ax.legend(loc = 'lower right', fontsize = 15)
	#plt.savefig('./Figure/AGE_MCdropout_T.eps')
	#plt.savefig('./Figure/AGE_MCdropout_T.png')
	#plt.show()


def TC_diffT(row, col, num, fig):
	# 画 f(x)-y VS sample的图
	data = np.load("../TC/diffT_mse.npy").astype("float32") # ID 02 04 08 10 12 15
	ID = data[:,0]
	sumnum = data.shape[0]
	ID = np.linspace(0.0020, 0.9999, 20)
	print(ID.shape)
	print(ID*sumnum)
	mse_02 = data[(ID*sumnum).astype(int), 1]
	mse_04 = data[(ID*sumnum).astype(int), 2]
	mse_08 = data[(ID*sumnum).astype(int), 3]
	mse_10 = data[(ID*sumnum).astype(int), 4]
	mse_12 = data[(ID*sumnum).astype(int), 5]
	mse_15 = data[(ID*sumnum).astype(int), 6]

	figsize = 8,6   # 800 * 600 像素
	ax = fig.add_subplot(row, col, num)#, figsize=figsize)
	#fig = plt.gcf()
	#fig.set_size_inches(6.4, 4.0)

	#plt.xlim(-10, 5600)
	#plt.ylim(-75, 50)
	ax.plot(ID, mse_02, linewidth=2, marker='.', markersize=8, label='MSE_02')
	ax.plot(ID, mse_04, linewidth=2, marker='*', markersize=8, label='MSE_04')
	ax.plot(ID, mse_08, linewidth=2, marker='x', markersize=8, label='MSE_08')
	ax.plot(ID, mse_10, linewidth=2, marker='^', markersize=8, label='MSE_10')
	#plt.plot(ID, mse_12, linewidth=2, marker='.', markersize=8, label='MSE_12')
	ax.plot(ID, mse_15, linewidth=2, marker='o', markersize=8, label='MSE_15')

	ax.axhline(y=data[int(ID[19]*sumnum), 1], ls="--",c="grey", linewidth=1)#添加水平直线
	#设置坐标刻度值的大小以及刻度值的字体
	ax.tick_params(labelsize=15)

	font1={'weight':'semibold',
    	'size':20
	}
	#styles=['normal','italic','oblique']
    #weights=['light','normal','medium','semibold','bold','heavy','black']
	ax.set_xlabel("Coverage", fontdict = font1)#x轴上的名字
	ax.set_ylabel("Risk (MSE)", fontdict = font1)#y轴上的名字
	#ax.set_title("the Influence of $T$ in Blend-Var.", fontsize = 15)
	#简单的设置legend(设置位置)
	#位置在右上角
	ax.legend(loc = 'upper right', fontsize = 15)
	#plt.savefig('./Figure/TC_BlendVar_T.eps')
	#plt.savefig('./Figure/TC_BlendVar_T.png')
	#plt.show()
if __name__ == '__main__':
	#draw_dy_sample()
	TC_draw_dy_sample()
	
	row = 1
	col = 1
	fig = plt.figure(num=1, figsize=(5.5,5))
	TC_diffT(row, col, 1, fig)
	fig.tight_layout() #调整整体空白
	plt.savefig('./Figure/TC_BlendVar_Ttmp.eps')
	plt.savefig('./Figure/TC_BlendVar_Ttmp.png')
	plt.show()
	
	row = 1
	col = 1
	fig = plt.figure(num=1, figsize=(5.5,5))
	AGE_diffT(row, col, 1, fig)
	fig.tight_layout() #调整整体空白
	plt.savefig('./Figure/AGE_MCdropout_Ttmp.eps')
	plt.savefig('./Figure/AGE_MCdropout_Ttmp.png')
	plt.show()
	
	row = 1
	col = 1
	fig = plt.figure(num=1, figsize=(5.5,5))
	AGE_testmse_testb_idealmse(row, col, 1, fig)
	fig.tight_layout() #调整整体空白
	plt.savefig('./Figure/AGE_testmse_testb_idealmsetmp.eps')
	plt.savefig('./Figure/AGE_testmse_testb_idealmsetmp.png')
	plt.show()
	
	row = 1
	col = 1
	fig = plt.figure(num=1, figsize=(5.5,5))
	TC_testmse_testb_idealmse(row, col, 1, fig)
	fig.tight_layout() #调整整体空白
	plt.savefig('./Figure/TC_testmse_testb_idealmsetmp.eps')
	plt.savefig('./Figure/TC_testmse_testb_idealmsetmp.png')
	plt.show()
	