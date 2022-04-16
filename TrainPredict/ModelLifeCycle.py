import tensorflow as tf
from data_preprocessing import make_train_batches_cifar100 as make_train_batches
from data_preprocessing import make_test_batches_cifar100 as make_test_batches
import os

class ModelLifeCycle:
    def __init__(self,
                 life_cycle_index,
                 model,
                 optimizer,
                 loss_func,
                 acc_func,
                 ckpt_dir='../CheckPoints/',
                 num_max_model_to_keep=5,
                 train_set=None,
                 dev_set=None,
                 test_set=None):
        self.life_cycle_index = life_cycle_index
        self.model = model
        self.train_set = train_set
        self.dev_set = dev_set
        self.test_set = test_set
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.acc_func = acc_func
        self.ckpt = tf.train.Checkpoint(model=self.model,optimizer=optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, ckpt_dir+str(life_cycle_index), max_to_keep=num_max_model_to_keep)



    def train(self,EPOCH,
              train_shuffle_buffer_size,
              train_batch_size,
              dev_batch_size,
              train_set_shuffle_seed,
              show_train_info_every=5,
              save_model_every=100
              ):
        if self.train_set == None:
            print("No Training Set")
            return

        # Initialize records lists
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_acc = tf.keras.metrics.Mean(name='train_acc')
        train_loss_records = []
        train_acc_records = []
        dev_loss_records = []
        dev_acc_records = []

        # Shuffle Training Set
        train_set = make_train_batches(self.train_set,
                                       shuffle_buffer_size=train_shuffle_buffer_size,
                                       shuffle_seed=train_set_shuffle_seed,
                                       train_batch_size=train_batch_size)

        # Begin Training
        for epoch in range(1,EPOCH+1):
            # Initialize training loss and acc
            train_loss.reset_states()
            train_acc.reset_states()
            # Optimization across batches
            for batch_index,(train_batch,train_label) in enumerate(train_set):
                loss, pred = self.train_step(train_batch,train_label)
                train_loss(loss)
                train_acc(self.acc_func(train_label,pred))
            # Record training loss and acc
            train_loss_records.append(float(train_loss.result()))
            train_acc_records.append(float(train_acc.result()))
            # Validation
            if self.dev_set != None:
                # Record validating loss and acc
                dev_loss, dev_acc = self.validate(dev_batch_size)
                dev_loss_records.append(float(dev_loss))
                dev_acc_records.append(float(dev_acc))
                # Print Info
                if epoch%show_train_info_every ==0:
                    print("Epoch: %s, Train Loss: %4f, Train Acc: %4f, Val Loss: %4f, Val Acc: %4f"%(epoch,
                                                                                                 train_loss.result(),
                                                                                                 train_acc.result(),
                                                                                                 dev_loss,
                                                                                                 dev_acc))
            else:
                # Print Info
                if epoch % show_train_info_every == 0:
                    print("Epoch: %s, Train Loss: %4f, Train Acc: %4f" % (epoch,
                                                                        train_loss.result(),
                                                                        train_acc.result()))
            # Save model
            if epoch % save_model_every ==0:
                self.save_model(str(epoch))

        # Save the final model
        if EPOCH % save_model_every != 0:
            self.save_model(str(EPOCH))
        return train_loss_records, train_acc_records, dev_loss_records, dev_acc_records

    def train_step(self,X,label):
        with tf.GradientTape() as tape:
            pred = self.model(X)
            loss = self.loss_func(label,pred)
        g = tape.gradient(loss,self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(g, self.model.trainable_variables))
        return loss, pred

    def validate(self,dev_batch_size):
        if self.dev_set == None:
            print("Warning: No Dev Set")
            return
        dev_set = make_test_batches(self.dev_set,dev_batch_size)
        dev_loss = tf.keras.metrics.Mean(name='dev_loss')
        dev_acc = tf.keras.metrics.Mean(name='dev_acc')
        for dev_index,(dev_batch,dev_label) in enumerate(dev_set):
            pred = self.model(dev_batch)
            loss = self.loss_func(dev_label,pred)
            dev_loss(loss)
            dev_acc(self.acc_func(dev_label, pred))
        return dev_loss.result(), dev_acc.result()



    def test(self,test_batch_size):
        if self.test_set == None:
            print("Warning: No Test Set")
            return
        test_set = make_test_batches(self.test_set,test_batch_size)
        test_acc = tf.keras.metrics.Mean(name='test_acc')
        predictions = []
        labels = []
        for test_index, (test_batch, test_label) in enumerate(test_set):
            pred_step = self.model(test_batch)
            predictions.append(pred_step)
            labels.append(tf.math.argmax(test_label,axis=-1))
            test_acc(self.acc_func(test_label, pred_step))
        predictions = tf.concat(predictions,axis=0)
        labels = tf.concat(labels, axis=0)
        return predictions, labels, test_acc.result()

    def save_model(self,name):
        self.ckpt_manager.save()
        # model_dir = '../CheckPoints/%s/'%self.life_cycle_index
        # if not os.path.isdir(model_dir):
        #     os.mkdir(model_dir)
        # self.model.save('%s.h5'%name)








