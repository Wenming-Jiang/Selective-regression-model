import numpy as np
import sys, os
import argparse

parser = argparse.ArgumentParser()
parser.parse_args()
parser.add_argument("-C", "--cafferoot", default="C:/Users/jwm/Desktop/caffe-windows/", help="the caffe root path")
parser.add_argument("-R", "--repeat_times", default=50, help="the times of MC-dropout")
args = parser.parse_args()

#set current dir
#caffe_root = 'C:/Users/jwm/Desktop/caffe-windows/' 
caffe_root = args.caffe_root
sys.path.insert(0, caffe_root + 'python')
import caffe
os.chdir(caffe_root)

net_file = 'data/LAP/moduel/age1.prototxt'  #age_train1.prototxt
caffe_model = 'data/LAP/moduel/dex_chalearn_iccv2015.caffemodel'
mean_proto_file = 'data/LAP/moduel/imagenet_mean.binaryproto'
mean_npy_file = 'data/LAP/moduel/mean.npy'

blob = caffe.proto.caffe_pb2.BlobProto()
data = open(mean_proto_file, 'rb').read()
blob.ParseFromString(data)

array = np.array(caffe.io.blobproto_to_array(blob))
mean_npy = array[0]
np.save(mean_npy_file, mean_npy)

net = caffe.Net(net_file, caffe_model, caffe.TRAIN)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(mean_npy_file).mean(1).mean(1))
transformer.set_raw_scale('data', 255) 
transformer.set_channel_swap('data', (2,1,0))

print("begin to Validation!!!")
path_name = caffe_root+'data/LAP/Validation/' #Test
age = np.array(range(101))

fp = open(caffe_root + 'data/LAP/' + 'val.txt') # test.txt
imageinfos = fp.readlines()
fp.close()
Total_AE = 0
num = 0
predict_age = []
repeat_times = 50

for imageinfo in imageinfos:
	info = imageinfo.split(" ")
	imagename = info[0]
	prensent_age = float(info[1])
	
	print(path_name+imagename)
	if(os.path.exists(path_name+imagename) == False):
		print(path_name+imagename + " not exists!")
		continue
	try:
		im=caffe.io.load_image(path_name+imagename) #test_img
	except ValueError, e:
		print e
	else:
		single_predict_age = [prensent_age]
		this_Ex = 0 
		for i in range(repeat_times):
			net.blobs['data'].data[...] = transformer.preprocess('data',im)
			out = net.forward()
			probablity = net.blobs['prob'].data[0].flatten()	
			Ex = sum(age*probablity)
			single_predict_age.append(Ex)
			this_Ex += Ex
		predict_age.append(single_predict_age)
		num = num + 1
		Total_AE += abs(this_Ex/repeat_times - prensent_age)
		print("tmp MAE: ", Total_AE / num)	

MAE = Total_AE / len(imageinfos)
print("Validation50 set MAE: ", MAE)
predict_age = np.array(predict_age)
print(predict_age.shape)
np.save("val50_predict_age.npy", predict_age)
np.save("val50_predict_age_MAE" + str(MAE) + ".npy", predict_age)



print("begin to Test!!!")
path_name = caffe_root+'data/LAP/Test/' #Test
age = np.array(range(101))

fp = open(caffe_root + 'data/LAP/' + 'test.txt') # test.txt
imageinfos = fp.readlines()
fp.close()
Total_AE = 0
num = 0
predict_age = []
#repeat_times = 50
repeat_times = args.repeat_times

for imageinfo in imageinfos:
	info = imageinfo.split(" ")
	imagename = info[0]
	prensent_age = float(info[1])
	
	print(path_name+imagename)
	if(os.path.exists(path_name+imagename) == False):
		print(path_name+imagename + " not exists!")
		continue
	try:
		im=caffe.io.load_image(path_name+imagename) #test_img
	except ValueError, e:
		print e
	else:
		single_predict_age = [prensent_age]
		this_Ex = 0 
		for i in range(repeat_times):
			net.blobs['data'].data[...] = transformer.preprocess('data',im)
			out = net.forward()
			probablity = net.blobs['prob'].data[0].flatten()	
			Ex = sum(age*probablity)
			single_predict_age.append(Ex)
			this_Ex += Ex
		predict_age.append(single_predict_age)
		num = num + 1
		Total_AE += abs(this_Ex/repeat_times - prensent_age)
		print("tmp MAE: ", Total_AE / num)	

MAE = Total_AE / len(imageinfos)
print("Test50 set MAE: ", MAE)
predict_age = np.array(predict_age)
print(predict_age.shape)
np.save("test50_predict_age.npy", predict_age)
np.save("test50_predict_age_MAE" + str(MAE) + ".npy", predict_age)

