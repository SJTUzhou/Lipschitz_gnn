import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import seaborn

plot_type = "loss" # "accuracy" # 

'''
# logs_dense=2_diff-cut_class=2_lip=1 PLOT
path_train_with_lip = './logs/logs_dense=2_diff-cut_class=2_lip=1/fit0/model-with-Lip-constr-20210504-131601/train/events.out.tfevents.1620130561.LAPTOP-4NSAND1N.18604.176.v2'
path_val_with_lip = './logs/logs_dense=2_diff-cut_class=2_lip=1/fit0/model-with-Lip-constr-20210504-131601/validation/events.out.tfevents.1620130565.LAPTOP-4NSAND1N.18604.1514.v2'
path_train_without_lip = '.\\logs\\logs_dense=2_diff-cut_class=2_lip=1\\fit0\\model-without-Lip-constr-20210504-132304\\train\\events.out.tfevents.1620130984.LAPTOP-4NSAND1N.18604.70217.v2'
path_val_without_lip = '.\\logs\\logs_dense=2_diff-cut_class=2_lip=1\\fit0\\model-without-Lip-constr-20210504-132304\\validation\\events.out.tfevents.1620130986.LAPTOP-4NSAND1N.18604.71339.v2'
'''

# logs_dense=2_same-cut_class=2_lip=1 PLOT
path_train_with_lip = '.\\logs\\logs_dense=2_same-cut_class=2_lip=1\\fit0\\model-with-Lip-constr-20210513-140736\\train\\events.out.tfevents.1620911256.LAPTOP-4NSAND1N.10740.176.v2'
path_val_with_lip = '.\\logs\\logs_dense=2_same-cut_class=2_lip=1\\fit0\\model-with-Lip-constr-20210513-140736\\validation\\events.out.tfevents.1620911261.LAPTOP-4NSAND1N.10740.1514.v2'
path_train_without_lip = '.\\logs\\logs_dense=2_same-cut_class=2_lip=1\\fit0\\model-without-Lip-constr-20210513-140927\\train\\events.out.tfevents.1620911367.LAPTOP-4NSAND1N.10740.21917.v2'
path_val_without_lip = '.\\logs\\logs_dense=2_same-cut_class=2_lip=1\\fit0\\model-without-Lip-constr-20210513-140927\\validation\\events.out.tfevents.1620911369.LAPTOP-4NSAND1N.10740.23039.v2'


event_train_with_lip = EventAccumulator(path_train_with_lip)
event_train_with_lip.Reload()

event_val_with_lip = EventAccumulator(path_val_with_lip)
event_val_with_lip.Reload()

event_train_without_lip = EventAccumulator(path_train_without_lip)
event_train_without_lip.Reload()

event_val_without_lip = EventAccumulator(path_val_without_lip)
event_val_without_lip.Reload()


var_name = "epoch_{}".format(plot_type)

acc_train_with_lip = []
for scalar in event_train_with_lip.Scalars(var_name):
    acc_train_with_lip.append(scalar.value)
acc_train_with_lip = np.asanyarray(acc_train_with_lip)


acc_val_with_lip = []
for scalar in event_val_with_lip.Scalars(var_name):
    acc_val_with_lip.append(scalar.value)
acc_val_with_lip = np.asanyarray(acc_val_with_lip)

acc_train_without_lip = []
for scalar in event_train_without_lip.Scalars(var_name):
    acc_train_without_lip.append(scalar.value)
acc_train_without_lip = np.asanyarray(acc_train_without_lip)


acc_val_without_lip = []
for scalar in event_val_without_lip.Scalars(var_name):
    acc_val_without_lip.append(scalar.value)
acc_val_without_lip = np.asanyarray(acc_val_without_lip)





x_axis = np.linspace(0, len(acc_train_with_lip), num=len(acc_train_with_lip))

my_signals = [{'name': 'Train {} of model with constraint'.format(plot_type), 'x': x_axis,
    'y': acc_train_with_lip, 'color':'tab:red', 'linewidth':2},
    
    {'name': 'Validation {} of model with constraint'.format(plot_type), 'x': x_axis,
    'y': acc_val_with_lip, 'color':'tab:orange', 'linewidth':2},

    {'name': 'Train {} of model without constraint'.format(plot_type), 'x': x_axis,
    'y': acc_train_without_lip, 'color':'tab:green', 'linewidth':2},
    
    {'name': 'Validation {} of model without constraint'.format(plot_type), 'x': x_axis,
    'y': acc_val_without_lip, 'color':'tab:blue', 'linewidth':2},
    
    
    ]


fig, ax = plt.subplots()

for signal in my_signals:
    ax.plot(signal['x'], signal['y'],
            color=signal['color'],
            linewidth=signal['linewidth'],
            label=signal['name'])

    ax.legend()
    ax.set_title('Train evolution')
    ax.set_xlabel('Iteration')
    ax.set_ylabel(plot_type.capitalize())
    ax.grid(True, which='both')
    seaborn.despine(ax=ax, offset=0)
plt.show()