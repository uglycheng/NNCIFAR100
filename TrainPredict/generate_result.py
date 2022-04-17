from ModelLifeCycle import ModelLifeCycle
from Models import SuperSimpleCNN as Model
# from Models import LinearRegression as Model
import tensorflow as tf
import tensorflow_datasets as tfds
import time
import json
import pandas as pd
import os

def run_different_start_point(tfseed,model_name):
    tf.random.set_seed(tfseed)

    ds, ds_meta = tfds.load('cifar100', with_info=True)
    train = ds['train']
    test = ds['test']


    life_cycle_index = str(int(time.time()))
    model = Model(tfseed)
    optimizer_config = {
        'learning_rate':0.001,
        'beta_1':0.9,
        'beta_2':0.999,
        'epsilon':1e-07,
    }
    optimizer = tf.keras.optimizers.Adam(**optimizer_config)
    def loss(label,pred):
        return tf.nn.softmax_cross_entropy_with_logits(label,pred)

    def acc(label,pred):
        temp = tf.math.argmax(label,axis=-1) == tf.math.argmax(pred,axis=-1)
        return tf.cast(temp, dtype=tf.float32)
    life_cycle_config = {
        'life_cycle_index':life_cycle_index,
        'model':model,
        'optimizer':optimizer,
        'loss_func':loss,
        'acc_func':acc,
        'ckpt_dir':'../CheckPoints/%s/'%model_name,
        'num_max_model_to_keep':5,
        'train_set':train,
        'dev_set':None,
        'test_set':test
    }

    model_life_cycle = ModelLifeCycle(**life_cycle_config)
    train_config = {
        'EPOCH':200,
        'train_shuffle_buffer_size':50000,
        'train_batch_size':32,
        'dev_batch_size':None,
        'train_set_shuffle_seed':tfseed,
        'show_train_info_every':50,
        'save_model_every':100
    }
    train_loss_records, train_acc_records, dev_loss_records, dev_acc_records = model_life_cycle.train(**train_config)
    test_config = {
        'test_batch_size':10000
    }
    pred_results, tlabels, test_acc = model_life_cycle.test(**test_config)
    print("Test Acc: %4f"%test_acc)

    config = {}
    config.update(optimizer_config)
    config.update(life_cycle_config)
    config.update(train_config)
    config.update(test_config)
    config['tf_seed']= tfseed
    excluded_keys = [
        'model',
        'optimizer',
        'loss_func',
        'acc_func',
        'train_set',
        'dev_set',
        'test_set'
    ]
    for key in excluded_keys:
        config.pop(key)

    with open('../Configs/%s/%s.json'%(model_name,life_cycle_index),'w') as config_f:
        config_f.write(json.dumps(config))

    if dev_loss_records:
        train_records = zip(list(range(1,train_config['EPOCH']+1)),train_loss_records, train_acc_records, dev_loss_records, dev_acc_records)
        cols = ['Epoch','Train_Loss','Train_Acc','Dev_Loss','Dev,Acc']
        df_train_log = pd.DataFrame(train_records,columns=cols)
        df_train_log.to_csv('../TrainLogs/%s/%s.csv'%(model_name,life_cycle_index))
    else:
        train_records = zip(list(range(1, train_config['EPOCH'] + 1)), train_loss_records, train_acc_records)
        cols = ['Epoch', 'Train_Loss', 'Train_Acc']
        df_train_log = pd.DataFrame(train_records, columns=cols)
        df_train_log.to_csv('../TrainLogs/%s/%s.csv'%(model_name,life_cycle_index))

    df_result = pd.DataFrame(
        tf.concat([pred_results,tf.cast(tf.expand_dims(tlabels,axis=-1),dtype=tf.float32)],axis=-1).numpy(),
        columns = ['class_%s'%i for i in range(100)]+['tlabels']
    )
    df_result.to_csv('../Results/%s/%s.csv'%(model_name,life_cycle_index))

model_name = 'SuperSimpleCNN_different_start_points'
ckpt_dir_name = '../CheckPoints/%s/'%model_name
if not os.path.isdir(ckpt_dir_name):
    os.mkdir(ckpt_dir_name)
config_dir_name = '../Configs/%s/'%model_name
if not os.path.isdir(config_dir_name):
    os.mkdir(config_dir_name)
train_log_dir_name = '../TrainLogs/%s/'%model_name
if not os.path.isdir(train_log_dir_name):
    os.mkdir(train_log_dir_name)
result_dir_name = '../Results/%s/'%model_name
if not os.path.isdir(result_dir_name):
    os.mkdir(result_dir_name)

for i in range(15):
    run_different_start_point(int(time.time()),model_name)
