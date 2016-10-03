from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                                    """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
                                   """Path to the pascalVoc data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                                    """Train the model using fp16.""")

# some default setting
IMAGE_SIZE = 448
NUM_CLASSES = 20
DEFAULT_BOX=2   #this means the number of box which each center predict
DEFAULT_size=7  #this is the output size,default is 7
#NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 
#NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 
COORD=5  #hyperparameter
NOOBJ=0.5

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.


def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var

def _variable_with_weight_decay(name,shape,stddev,wd):
    #name:name of variable
    #shape:list of ints
    #stdeev:stadard deviation of a truncted gausisan
    #wd:a L2loss weight decay multiplied by this float.if none
    #    weight decay is not added
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
            name,
            shape,
            tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def inference(images):
    """build model
        return logits
    """
    with tf.variable_scope('conv1') as scope:
        kernel=_variable_with_weight_decay('weights',
                                            shape=[7,7,3,64],
                                            stddev=5e-2,
                                            wd=0.0005)
        conv=tf.nn.conv2d(images,kernel,[1,2,2,1],padding='SAME')
        biases=_variable_on_cpu('biases',[64],tf.constant_initializer(0.0))
        bias=tf.nn.bias_add(conv,biases)
        conv1=tf.nn.relu(bias,name=scope.name)

    #pool
    pool1=tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool1')

    #conv2
    with tf.variable_scope('conv2') as scope:
        kernel=_variable_with_weight_decay('weights',
                                            shape=[3,3,64,192],
                                            stddev=5e-2,
                                            wd=0.0005)
        conv=tf.nn.conv2d(pool1,kernel,[1,1,1,1],padding='SAME')
        biases=_variable_on_cpu('biases',[192],tf.constant_initializer(0.0))
        bias=tf.nn.bias_add(conv,biases)
        conv2=tf.nn.relu(bias,name=scope.name)

    #pool2
    pool2=tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool2')
    
    #conv3
    with tf.variable_scope('conv3') as scope:
        kernel=_variable_with_weight_decay('weights',
                                            shape=[1,1,192,128],
                                            stddev=5e-2,
                                            wd=0.0005)
        conv=tf.nn.conv2d(pool2,kernel,[1,1,1,1],padding='SAME')
        biases=_variable_on_cpu('biases',[128],tf.constant_initializer(0.0))
        bias=tf.nn.bias_add(conv,biases)
        conv3=tf.nn.relu(bias,name=scope.name)
        
    #conv4
    with tf.variable_scope('conv4') as scope:
        kernel=_variable_with_weight_decay('weights',
                                            shape=[3,3,128,256],
                                            stddev=5e-2,
                                            wd=0.0005)
        conv=tf.nn.conv2d(conv3,kernel,[1,1,1,1],padding='SAME')
        biases=_variable_on_cpu('biases',[256],tf.constant_initializer(0.0))
        bias=tf.nn.bias_add(conv,biases)
        conv4=tf.nn.relu(bias,name=scope.name)
        
    #conv5
    with tf.variable_scope('conv5') as scope:
        kernel=_variable_with_weight_decay('weights',
                                            shape=[1,1,256,256],
                                            stddev=5e-2,
                                            wd=0.0005)
        conv=tf.nn.conv2d(conv4,kernel,[1,1,1,1],padding='SAME')
        biases=_variable_on_cpu('biases',[256],tf.constant_initializer(0.0))
        bias=tf.nn.bias_add(conv,biases)
        conv5=tf.nn.relu(bias,name=scope.name)
   

    #conv6
    with tf.variable_scope('conv6') as scope:
        kernel=_variable_with_weight_decay('weights',
                                            shape=[3,3,256,512],
                                            stddev=5e-2,
                                            wd=0.0005)
        conv=tf.nn.conv2d(conv5,kernel,[1,1,1,1],padding='SAME')
        biases=_variable_on_cpu('biases',[512],tf.constant_initializer(0.0))
        bias=tf.nn.bias_add(conv,biases)
        conv6=tf.nn.relu(bias,name=scope.name)
   
    #pool3
    pool3=tf.nn.max_pool(conv6,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool3')

     #conv7
    with tf.variable_scope('conv7') as scope:
        kernel=_variable_with_weight_decay('weights',
                                            shape=[1,1,512,256],
                                            stddev=5e-2,
                                            wd=0.0005)

        conv=tf.nn.conv2d(pool3,kernel,[1,1,1,1],padding='SAME')
        biases=_variable_on_cpu('biases',[256],tf.constant_initializer(0.0))
        bias=tf.nn.bias_add(conv,biases)
        conv7=tf.nn.relu(bias,name=scope.name)
 
     #conv8
    with tf.variable_scope('conv8') as scope:
        kernel=_variable_with_weight_decay('weights',
                                            shape=[3,3,256,512],
                                            stddev=5e-2,
                                            wd=0.0005)
        conv=tf.nn.conv2d(conv7,kernel,[1,1,1,1],padding='SAME')
        biases=_variable_on_cpu('biases',[512],tf.constant_initializer(0.0))
        bias=tf.nn.bias_add(conv,biases)
        conv8=tf.nn.relu(bias,name=scope.name)
  
     #conv9
    with tf.variable_scope('conv9') as scope:
        kernel=_variable_with_weight_decay('weights',
                                            shape=[1,1,512,256],
                                            stddev=5e-2,
                                            wd=0.0005)
        conv=tf.nn.conv2d(conv8,kernel,[1,1,1,1],padding='SAME')
        biases=_variable_on_cpu('biases',[256],tf.constant_initializer(0.0))
        bias=tf.nn.bias_add(conv,biases)
        conv9=tf.nn.relu(bias,name=scope.name)
   
     #conv10
    with tf.variable_scope('conv10') as scope:
        kernel=_variable_with_weight_decay('weights',
                                            shape=[3,3,256,512],
                                            stddev=5e-2,
                                            wd=0.0005)
        conv=tf.nn.conv2d(conv9,kernel,[1,1,1,1],padding='SAME')
        biases=_variable_on_cpu('biases',[512],tf.constant_initializer(0.0))
        bias=tf.nn.bias_add(conv,biases)
        conv10=tf.nn.relu(bias,name=scope.name)

    #conv11
    with tf.variable_scope('conv11') as scope:
        kernel=_variable_with_weight_decay('weights',
                                            shape=[1,1,512,256],
                                            stddev=5e-2,
                                            wd=0.0005)
        conv=tf.nn.conv2d(conv10,kernel,[1,1,1,1],padding='SAME')
        biases=_variable_on_cpu('biases',[256],tf.constant_initializer(0.0))
        bias=tf.nn.bias_add(conv,biases)
        conv11=tf.nn.relu(bias,name=scope.name)
     
      #conv12
    with tf.variable_scope('conv12') as scope:
        kernel=_variable_with_weight_decay('weights',
                                            shape=[3,3,256,512],
                                            stddev=5e-2,
                                            wd=0.0005)
        conv=tf.nn.conv2d(conv11,kernel,[1,1,1,1],padding='SAME')
        biases=_variable_on_cpu('biases',[512],tf.constant_initializer(0.0))
        bias=tf.nn.bias_add(conv,biases)
        conv12=tf.nn.relu(bias,name=scope.name)

    #conv13
    with tf.variable_scope('conv13') as scope:
        kernel=_variable_with_weight_decay('weights',
                                            shape=[1,1,512,256],
                                            stddev=5e-2,
                                            wd=0.0005)
        conv=tf.nn.conv2d(conv12,kernel,[1,1,1,1],padding='SAME')
        biases=_variable_on_cpu('biases',[256],tf.constant_initializer(0.0))
        bias=tf.nn.bias_add(conv,biases)
        conv13=tf.nn.relu(bias,name=scope.name)
    
    #conv14
    with tf.variable_scope('conv14') as scope:
        kernel=_variable_with_weight_decay('weights',
                                            shape=[3,3,256,512],
                                            stddev=5e-2,
                                            wd=0.0005)
        conv=tf.nn.conv2d(conv13,kernel,[1,1,1,1],padding='SAME')
        biases=_variable_on_cpu('biases',[512],tf.constant_initializer(0.0))
        bias=tf.nn.bias_add(conv,biases)
        conv14=tf.nn.relu(bias,name=scope.name)

     #conv15
    with tf.variable_scope('conv15') as scope:
        kernel=_variable_with_weight_decay('weights',
                                            shape=[1,1,512,512],
                                            stddev=5e-2,
                                            wd=0.0005)
        conv=tf.nn.conv2d(conv14,kernel,[1,1,1,1],padding='SAME')
        biases=_variable_on_cpu('biases',[512],tf.constant_initializer(0.0))
        bias=tf.nn.bias_add(conv,biases)
        conv15=tf.nn.relu(bias,name=scope.name)

      #conv16
    with tf.variable_scope('conv16') as scope:
        kernel=_variable_with_weight_decay('weights',
                                            shape=[3,3,512,1024],
                                            stddev=5e-2,
                                            wd=0.0005)
        conv=tf.nn.conv2d(conv15,kernel,[1,1,1,1],padding='SAME')
        biases=_variable_on_cpu('biases',[1024],tf.constant_initializer(0.0))
        bias=tf.nn.bias_add(conv,biases)
        conv16=tf.nn.relu(bias,name=scope.name)

    #pool4
    pool4=tf.nn.max_pool(conv16,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool4')

    #conv17
    with tf.variable_scope('conv17') as scope:
        kernel=_variable_with_weight_decay('weights',
                                            shape=[1,1,1024,512],
                                            stddev=5e-2,
                                            wd=0.0005)
        conv=tf.nn.conv2d(pool4,kernel,[1,1,1,1],padding='SAME')
        biases=_variable_on_cpu('biases',[512],tf.constant_initializer(0.0))
        bias=tf.nn.bias_add(conv,biases)
        conv17=tf.nn.relu(bias,name=scope.name)

    #conv18
    with tf.variable_scope('conv18') as scope:
        kernel=_variable_with_weight_decay('weights',
                                            shape=[3,3,512,1024],
                                            stddev=5e-2,
                                            wd=0.0005)
        conv=tf.nn.conv2d(conv17,kernel,[1,1,1,1],padding='SAME')
        biases=_variable_on_cpu('biases',[1024],tf.constant_initializer(0.0))
        bias=tf.nn.bias_add(conv,biases)
        conv18=tf.nn.relu(bias,name=scope.name)

    #conv19
    with tf.variable_scope('conv19') as scope:
        kernel=_variable_with_weight_decay('weights',
                                            shape=[1,1,1024,512],
                                            stddev=5e-2,
                                            wd=0.0005)
        conv=tf.nn.conv2d(conv18,kernel,[1,1,1,1],padding='SAME')
        biases=_variable_on_cpu('biases',[512],tf.constant_initializer(0.0))
        bias=tf.nn.bias_add(conv,biases)
        conv19=tf.nn.relu(bias,name=scope.name)

    #conv20
    with tf.variable_scope('conv20') as scope:
        kernel=_variable_with_weight_decay('weights',
                                            shape=[3,3,512,1024],
                                            stddev=5e-2,
                                            wd=0.0005)
        conv=tf.nn.conv2d(conv19,kernel,[1,1,1,1],padding='SAME')
        biases=_variable_on_cpu('biases',[1024],tf.constant_initializer(0.0))
        bias=tf.nn.bias_add(conv,biases)
        conv20=tf.nn.relu(bias,name=scope.name)

    #conv21
    with tf.variable_scope('conv21') as scope:
        kernel=_variable_with_weight_decay('weights',
                                            shape=[3,3,1024,1024],
                                            stddev=5e-2,
                                            wd=0.0005)
        conv=tf.nn.conv2d(conv20,kernel,[1,1,1,1],padding='SAME')
        biases=_variable_on_cpu('biases',[1024],tf.constant_initializer(0.0))
        bias=tf.nn.bias_add(conv,biases)
        conv21=tf.nn.relu(bias,name=scope.name)

    #conv22
    with tf.variable_scope('conv22') as scope:
        kernel=_variable_with_weight_decay('weights',
                                            shape=[3,3,1024,1024],
                                            stddev=5e-2,
                                            wd=0.0005)
        conv=tf.nn.conv2d(conv21,kernel,[1,2,2,1],padding='SAME')
        biases=_variable_on_cpu('biases',[1024],tf.constant_initializer(0.0))
        bias=tf.nn.bias_add(conv,biases)
        conv22=tf.nn.relu(bias,name=scope.name)

    #conv23
    with tf.variable_scope('conv23') as scope:
        kernel=_variable_with_weight_decay('weights',
                                            shape=[3,3,1024,1024],
                                            stddev=5e-2,
                                            wd=0.0005)
        conv=tf.nn.conv2d(conv22,kernel,[1,1,1,1],padding='SAME')
        biases=_variable_on_cpu('biases',[1024],tf.constant_initializer(0.0))
        bias=tf.nn.bias_add(conv,biases)
        conv23=tf.nn.relu(bias,name=scope.name)

    #conv24
    with tf.variable_scope('conv24') as scope:
        kernel=_variable_with_weight_decay('weights',
                                            shape=[3,3,1024,1024],
                                            stddev=5e-2,
                                            wd=0.0005)
        conv=tf.nn.conv2d(conv23,kernel,[1,1,1,1],padding='SAME')
        biases=_variable_on_cpu('biases',[1024],tf.constant_initializer(0.0))
        bias=tf.nn.bias_add(conv,biases)
        conv24=tf.nn.relu(bias,name=scope.name)
    
    #fully connected 1
    with tf.variable_scope('fc1') as scope:
        reshape=tf.reshape(conv24,[FLAGS.batch_size,-1])
        dim=reshape.get_shape()[1].value
        weights=_variable_with_weight_decay('weights',
                                            shape=[dim,4096],
                                            stddev=5e-2,
                                            wd=0.0005)
        biases=_variable_on_cpu('biases',[4096],tf.constant_initializer(0.0))
        fc1=tf.nn.relu(tf.matmul(reshape,weights)+biases,name=scope.name)

    #drop out
    keep_prob = tf.placeholder(tf.float32)
    fc1_drop = tf.nn.dropout(fc1, keep_prob)
    #fully connected 2
    """
    this layer could ensure the output size.
    ouputsize=N*N*(class+5*default_box)
    """
    with tf.variable_scope('fc2') as scope:
        outputsize=DEFAULT_size*DEFAULT_size*(5*DEFAULT_BOX+NUM_CLASSES)
        weights=_variable_with_weight_decay('weights',
                                            shape=[4096,outputsize],
                                            stddev=5e-2,
                                            wd=0.0005)
        biases=_variable_on_cpu('biases',[outputsize],tf.constant_initializer(0.0))
        fc2=tf.nn.relu(tf.matmul(fc1_drop,weights)+biases,name=scope.name)

        
    # a tesor that with[batchsize,outputsize]
    return fc2
    

def loss(fc2,groudtruth):
    #fc2 is a tensor
    #groundtruth is a variable
    predict=tf.reshape(fc2,[FLAGS.batch_size,DEFAULT_size,DEFAULT_size,5*DEFAULT_BOX+NUM_CLASSES])
    #20 means a image have 20 labels,some maybe none ,those boxes with class -1 mean none
    g_box=tf.reshape(groudtruth,[FLAGS.batch_size,20,5])
    #next we should slice the tensor
    batch_class=tf.slice(g_box,[0,0,0],[FLAGS.batch_size,20,1])
    #batch_class is a tensor with[FLAGS.batch_class,20],some maybe none
    
    #next we need to judge the center of groundtruth
    box_size=tf.slice(g_box,[0,0,3],[FLAGS.batch_size,20,2])

    box_center=tf.slice(g_box,[0,0,1],[FLAGS.batch_size,20,2])
    
    size=tf.constant([DEFAULT_size],dtype=tf.int32)
    #now we have the box,next we should ensure which box to be responsible for predictor
    index_side=tf.floordiv(box_center,size)
    index_side=tf.cast(index_side,tf.float32)
    #index_side with shape[FLAGS.batch_size,20,2]
    #now we get every box's index ,next we should match the predicted box
    
    #the index of box should be calculate in format box[0]*DEFAULT_size+box[1] 
    op1=tf.tile([DEFAULT_size,1],[FLAGS.batch_size])
    op2=tf.reshape(op1,[FLAGS.batch_size,2,1])
    index=tf.batch_matmul(tf.cast(index_side, tf.int32),op2)
    #index is a tensor with [batch_size,20] ,this means the box to be responsibel for the prediction        
    #next we should calculate the box regression loss
    regular_size=tf.truediv(box_size,[IMAGE_SIZE])

    # we need get th offset
    cell_len=tf.constant([IMAGE_SIZE/DEFAULT_size*1.0,0,0,IMAGE_SIZE/DEFAULT_size*1.0],dtype=tf.float32)
    op3=tf.tile(cell_len,[FLAGS.batch_size])
    op4=tf.reshape(op3,[FLAGS.batch_size,2,2])
    cell_left=tf.batch_matmul(index_side,op4)
    cell_left=tf.cast(cell_left,tf.int32)
    offset=tf.abs(tf.sub(cell_left,box_center))
    regular_offset=tf.truediv(tf.cast(offset,tf.float32),[IMAGE_SIZE/DEFAULT_size*1.0])
    #regular_offset with the shape[batch_size,20,2]
    #now we should calculate the box regression loss
    # first we should make sure which predict box have the bigger IOU
    iou_pre_gt=IOU(regular_offset,regular_size,batch_class,index_side,predict)
    #now we get the IOU between prediction and gt
    iou_re=tf.reshape(iou_pre_gt,[FLAGS.batch_size,20,DEFAULT_BOX])
    for i in range(FLAGS.batch_size):
        for j in range(20):
            if tf.greater(tf.cast(batch_calss[i,j,0],tf.float32),[-0.5]):
                l_index=tf.argmax(iou_re,2)
                temp_loss_x=tf.square(regular_offset[i,j,0]-predict[i, index_side[i,j,0], index_side[i,j,1],l_index*5])
                temp_loss_y=tf.square(regular_offset[i,j,1]-predict[i, index_side[i,j,0], index_side[i,j,1],l_index*5+1])
                temp_loss_w=tf.square(tf.sqrt(regular_size[i,j,0])-tf.maximum(predict[i, index_side[i,j,0], index_side[i,j,1],l_index*5+2],[0.0]))
                temp_loss_h=tf.square(tf.sqrt(regular_size[i,j,1])-tf.maximum(predict[i, index_side[i,j,0], index_side[i,j,1],l_index*5+3],[0.0]))
                temp_loss_obj_cls=tf.square(1-predict[i, index_side[i,j,0], index_side[i,j,1],l_index*5+4])
                temp_mat=np.zeros([1,NUM_CLASSES])
                temp_mat[0, batch_class[i,j,0]]=1
                temp_loss_each=tf.square(tf.pack(temp_mat)-predict[i, index_side[i,j,0], index_side[i,j,1],5*DEFAULT_BOX:5*DEFAULT_BOX+NUM_CLASSES])
                
                tf.add_to_collection('losses',temp_loss_x*COORD)
                tf.add_to_collection('losses',temp_loss_y*COORD) 
                tf.add_to_collection('losses',temp_loss_w*COORD)
                tf.add_to_collection('losses',temp_loss_h*COORD)
                tf.add_to_collection('losses',temp_loss_obj_cls)
                tf.add_to_collection('losses',temp_loss_each)
    #now we have calculate the cordinates loss ,next cls loss for noobj

    for i in range(FLAGS.batch_size):
        for j in range(20):
            if tf.greater(tf.cast(batch_calss[i,j,0],tf.float32),[-0.5]):
                for t in range(DEFAULT_size):
                    for m in range(DEFAULT_size):
                        for n in range(DEFAULT_BOX):
                            if index_side[i,j,0]==t and index_side[i,j,1]==m and l_index==tf.argmax(iou_re,2):
                                break
                            temp_loss_noob=tf.square(predict[i, t, m, n*5+4]-0)
                            tf.add_to_collection('losses',temp_loss_noob*NOOBJ)

    return tf.add_n(tf.get_collection('losses'),name='total_loss')







def iou(boxs1, boxs2):
    """calculate the iou of boxs1 and boxs2
    Args:
        boxs1: 2-D tensor, shape=[m, 4] (xmin, ymin, xmax, ymax)
        boxs2: 2-D tensor, shape=[n, 4] (xmin, ymin, xmax, ymax)
    Return:
        IOU: 2-D tensor (m, n)
    """

    extend_boxs1 = boxs1
    extend_boxs2 = boxs2

    boxs = tf.pack([extend_boxs1, extend_boxs2])
    
    lr = tf.maximum(boxs[0, :, :, 0:2], boxs[1, :, :, 0:2])
    
    rd = tf.minimum(boxs[0, :, :, 2:], boxs[1, :, :, 2:])
    
    intersection = rd - lr
    inter_square = intersection[:, :, 0] * intersection[:, :, 1]
    
    mask = tf.cast(intersection[:, :, 0] > 0, tf.float32) * tf.cast(intersection[:, :, 1] > 0, tf.float32)
    
    inter_square = mask * inter_square
    
    #calculate the boxs1 square and boxs2 square
    square1 = (boxs1[:, 2] - boxs1[:, 0]) * (boxs1[:, 3] - boxs1[:, 1])
    square2 = (boxs2[:, 2] - boxs2[:, 0]) * (boxs2[:, 3] - boxs2[:, 1])
    
    return inter_square/(square1 + square2 - inter_square)
            
            
    #the number of the box in image
def IOU(regular_offset,regular_size,batch_class,index_side,predict):
    #regular_offset is a tensor with shape(batch_size,20,2)
    #index_side is a tensor with shape[batch_size,20,2]
    #predict is a tensor with shape[batch_size,DEFAULT_size,DEFAULT_size,DEFAULT_BOX*5+class]
    
    #output should be a tensor with shape[batchsize,20,2]
    '''
    means the each box's IOU with the groundtruth.
    '''
    #first we should extract each box in predict
    temp_box=np.array([20*FLAGS.batch_size*DEFAULT_BOX,4])
    for image in range(FLAGS.batch_size):
        for groundtruth in range(20):
            #tf.cond(tf.cast(batch_class[image,groundtruth,0], tf.float32) > [-0.5], )
            #if tf.cast(batch_class[image,groundtruth,0], tf.float32) > [-0.5]:
            if True:
            # enusre the box is a obj
                for box in range(DEFAULT_BOX):
                    box_size=predict[image,index_side[image,groundtruth,box],index_side[image,groundtruth,box],box*5:box*5+4]
                    regular_box=tf.maximum(box_size,[0.0])
                    for i in range(4):
                        temp_box[image*20+groundtruth*DEFAULT_BOX+box,i]=regular_box[0,0,0,i]
            else:
                #this op is for none groudtruth
                for box in range(DEFAULT_BOX):
                    for i in range(4):
                        temp_box[image*20+groundtruth*DEFAULT_BOX+box,i]=0
    regular_pre_box=tf.pack(temp_box)
    #after th op above we get the regular_pre_box
    
    #now we should deal with the groundtruth,convert it's shape as regular_pre_box
    temp_box1=np.array([20*FLAGS.batch_size*DEFAULT_BOX,4])
    for image in range(FLAGS.batch_size):
        for groundtruth in range(20):
            for box in range(DEFAULT_BOX):
            #ground truth only have one box ,we shoulf repeat it DEFAULT_BOX times to match the predict
                for i in range(2):
                    temp_box1[image*20+groundtruth*DEFAULT_BOX+box,i]=regular_offset[image,groundtruth,i]
                for j in range(2):
                    temp_box1[image*20+groundtruth*DEFAULT_BOX+box,j+2]=regular_offset[image,groundtruth,j]
    regular_gr_box=tf.pack(temp_box1)
    
    #next we should adjust it as the laft-top and right-down
    temp_matrix=tf.constant([[1.0,0.0,1.0,0.0], [0.0,1.0,0.0,1.0], [-0.5,0.0,0.5,0.0], [0,-0.5,0.0,0.5]])
    
    ltrd_pre=tf.matmul(regular_pre_box,temp_matrix)
    ltrd_gr=tf.matmul(regular_gr_box,temp_matrix)
    
    output=iou(ltrd_pre,ltrd_gr)
    
    return output

def train(total_loss, global_step):
  """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.scalar_summary('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.histogram_summary(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op
