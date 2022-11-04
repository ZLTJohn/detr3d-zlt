filename = 'validation/laser_gt_objects.bin'#path/to/waymo_dataset
# filename = 'results/result_1666096111.0605032.bin'  
# filename = 'data/waymo_v131/waymo_format/gt.bin'
data = open(filename,'rb').read()
from waymo_open_dataset.protos import metrics_pb2
objs = metrics_pb2.Objects()
objs.ParseFromString(data)

objs_out = metrics_pb2.Objects()
for obj in objs.objects:        #2424599
  if obj.object.camera_synced_box.ByteSize() == 0: continue  #1508136
  if obj.object.num_lidar_points_in_box < 1: continue  #1357649
  if obj.object.type == 3: continue #1027302, no influence on result
  objs_out.objects.append(obj)
open('cam_only_val_gt.bin','wb').write(objs_out.SerializeToString())

file3 = 'ap4125_subset.bin'
file4 = 'ap4990_subset.bin'
data3 = open(file3,'rb').read()
data4 = open(file4,'rb').read()
from waymo_open_dataset.protos import metrics_pb2
objs3 = metrics_pb2.Objects()
objs4 = metrics_pb2.Objects()
objs3.ParseFromString(data3)
objs4.ParseFromString(data4)