import tensorflow as tf
from keras.models import Sequential, Model
from keras import Input
from keras.layers import LSTM, Dropout, Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam, RMSprop
from keras.losses import sparse_categorical_crossentropy, categorical_crossentropy
from keras.callbacks import TensorBoard, Callback, ModelCheckpoint
import argparse
import logging
import nni
import numpy as np
import os
import shutil
import json
from nni.tuner import Tuner
from nni.experiment import Experiment
from nni.algorithms.hpo.hyperopt_tuner import HyperoptTuner
from nni.tools.nnictl import updater, nnictl_utils


os.environ["CUDA_VISIBLE_DEVICES"] = '1'  
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'  


seed = 0
os.environ['PYTHONHASHSEED'] = str(seed)
tf.random.set_seed(seed)
np.random.seed(seed)
rng = np.random.RandomState(seed)


##### SOME DEFAULT SETTING ###########################################
### These defaults will be replaced
### when calling the function file_settings

network_type = 'cnn'
device = 'watch'
subset = 2
time_window = 2


window_size = int(20*time_window) # 20 Hz sampling times the temporal length of the window

datafile = 'data_'+device+'_subset'+str(subset)+'_'+str(window_size)

searchspace_path = '../searchspaces/nni_SearchSpace_'+network_type+'.json'
######################################################################


eps = 100


def file_settings(args):

    window_size = int(20*args.time_window) # 20 Hz sampling times the temporal length of the window
    datafile = 'data_'+args.device+'_subset'+str(args.subset)+'_'+str(window_size)
    searchspace_path = '../searchspaces/nni_SearchSpace_'+args.network_type+'.json'

    return datafile


def load_wisdm2_data(file_name):
    filepath = os.path.join('../data/',file_name+'.npz')
    a = np.load(filepath)
    return (a['arr_0'], a['arr_1'], a['arr_2'], a['arr_3'], a['arr_4'], a['arr_5'])


def report_result(args, result, result_type):

    if result_type == 'test':
        report_file = out_dir + 'nni_' + args.network_type + '_' + nni.get_experiment_id() + '_' + result_type + '_accs'
    else:
        report_file = out_dir + 'nni_' + args.network_type + '_' + nni.get_experiment_id() + '_' + result_type + '_accs_' + nni.get_trial_id()
    
    with open(report_file, 'a') as f:
        f.write(str(result))
        f.write('\n')
    
    return report_file


class SendMetrics(Callback):
    '''
    Keras callback to send metrics to NNI framework
    '''
    def on_epoch_end(self, epoch, logs={}):
        '''
        Run on end of each epoch
        '''
        LOG.debug(logs)
        
        if 'val_acc' in logs:
            nni.report_intermediate_result(logs['val_acc'])
        else:
            nni.report_intermediate_result(logs['val_accuracy'])


def create_wisdm2_model(args, hyper_params, timesteps, input_dim, n_classes):

    if args.network_type == 'lstm':

        model = Sequential()
    
        model.add(LSTM(hyper_params['nni_network/LSTM_units_1/randint'], return_sequences=True,input_shape=(timesteps, input_dim),name='LSTM_1'))
        model.add(Dropout(hyper_params['nni_network/LSTM_Dropout_1/quniform'],name='Dropout_1'))
        model.add(LSTM(hyper_params['nni_network/LSTM_units_2/randint'],recurrent_regularizer=l2(hyper_params['nni_network/LSTM_l2_2/quniform']),input_shape=(timesteps, input_dim),name='LSTM_2'))
        model.add(Dropout(hyper_params['nni_network/LSTM_Dropout_2/quniform'],name='Dropout_2'))
        model.add(Dense(n_classes, activation='softmax'))

        model.summary()

        optimizer = Adam(lr=hyper_params['nni_network/lr/quniform'])

        model.compile(loss=categorical_crossentropy,
                      optimizer=optimizer,
                      metrics=['accuracy'])
    
    elif args.network_type == 'cnn':

        model = Sequential()
    
        model.add(Conv1D(filters=hyper_params['nni_network/Conv1D_filters_1/randint'], kernel_size=hyper_params['nni_network/Conv1D_kernel_size_1/randint'], activation='relu', kernel_initializer='he_uniform', input_shape=(timesteps,input_dim), name='Conv1D_1'))
        model.add(Conv1D(filters=hyper_params['nni_network/Conv1D_filters_2/randint'], kernel_size=hyper_params['nni_network/Conv1D_kernel_size_2/randint'], activation='relu', kernel_initializer='he_uniform', name='Conv1D_2'))
        model.add(MaxPooling1D(pool_size=2, name='MaxPooling1D'))
        model.add(Flatten())
        model.add(Dense(hyper_params['nni_network/CNN_Dense_1/randint'], activation='relu', name='Dense_1'))
        model.add(Dense(n_classes, activation='softmax', name='Dense_2'))

        model.summary()

        optimizer = Adam(lr=hyper_params['nni_network/lr/quniform'])

        model.compile(loss=categorical_crossentropy,
                      optimizer=optimizer,
                      metrics=['accuracy'])

    return model


def run(args, params):
    
    (x_train, x_val, x_test, y_train, y_val, y_test) = load_wisdm2_data(file_settings(args))
    timesteps = len(x_train[0])
    input_dim = len(x_train[0][0])
    n_classes = len(y_train[0])
    model = create_wisdm2_model(args, params, timesteps, input_dim, n_classes)

    batch_size = params['nni_network/batch_size/randint']

    # training
    history = model.fit(x_train,
                        y_train,
                        validation_data=(x_val, y_val),
                        batch_size=batch_size,
                        epochs=args.epochs,
                        verbose=0,
                        callbacks=[SendMetrics(), TensorBoard(log_dir=TENSORBOARD_DIR), ModelCheckpoint(filepath=out_dir+nni.get_experiment_id()+'_'+nni.get_trial_id()+'_'+'best_train/'+'best_train_'+nni.get_experiment_id()+'_'+nni.get_trial_id(), monitor="val_accuracy", save_best_only=True, save_weights_only=True)])

    # test
    model.load_weights(out_dir+nni.get_experiment_id()+'_'+nni.get_trial_id()+'_'+'best_train/'+'best_train_'+nni.get_experiment_id()+'_'+nni.get_trial_id())
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    
    report_file = report_result(args,acc*100, 'test')
    with open(report_file, 'r') as f:
        if acc*100 >= np.max(np.asarray([(line.strip()) for line in f], dtype=np.float64)):
            model.save_weights(out_dir+'best_test/'+'best_test_'+nni.get_experiment_id())

    LOG.debug('Final result is: %d', acc)
    LOG.debug(print(f"Final validation accuracy: {100 * history.history['val_accuracy'][-1]:.2f}%"))
    LOG.debug(print(f"Best validation accuracy: {100 * np.max(history.history['val_accuracy']):.2f}%"))
    LOG.debug(print(f"Test accuracy from training with best validation accuracy: {100 * acc:.2f}%"))
    
    nni.report_final_result(acc)

    shutil.rmtree(out_dir+nni.get_experiment_id()+'_'+nni.get_trial_id()+'_'+'best_train/')


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--network_type", type=str, default=network_type, help="network type", required=False)
    PARSER.add_argument("--datafile", type=str, default=datafile, help="data file", required=False)
    PARSER.add_argument("--epochs", type=int, default=eps, help="Train epochs", required=False)
    PARSER.add_argument("--filename", type=str, default=searchspace_path, help="File name for search space", required=False)
    PARSER.add_argument("--id", type=str, default=nni.get_experiment_id(), help="Experiment ID", required=False)
    PARSER.add_argument("--device", type=str, default=device, help="From which device the signal is taken", required=False)
    PARSER.add_argument("--subset", type=int, default=subset, help="Activity subset", required=False)
    PARSER.add_argument("--time_window", type=int, default=time_window, help="Length of the time window", required=False)

    ARGS, UNKNOWN = PARSER.parse_known_args()

    datafile = file_settings(ARGS)
    LOG = logging.getLogger('wisdm2_'+ARGS.network_type+'_'+datafile[5:])
    out_dir = '../output/tmp_' + ARGS.network_type + '_' + nni.get_experiment_id() + '_' + datafile[5:] + '/'
    os.environ['NNI_OUTPUT_DIR'] = out_dir
    TENSORBOARD_DIR = os.environ['NNI_OUTPUT_DIR']
    
    try:

        n_tr = 200
        if (nni.get_sequence_id() > 0) & (nni.get_sequence_id()%n_tr == 0):
            updater.update_searchspace(ARGS) # it will use ARGS.filename to update the search space

        PARAMS = nni.get_next_parameter()
        LOG.debug(PARAMS)
        
        run(ARGS, PARAMS)
    
    except Exception as e:
        LOG.exception(e)
        raise
