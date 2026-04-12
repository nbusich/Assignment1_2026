# Function plot training and validation over 
import matplotlib.pyplot as plt

def plot_loss(results,path):
  plt.close('all')

  # get data
  history = results['history']

  epoch_counter = 0 

  epoch_x = [] 
  train_y = []
  dev_y = []
  lr = []

  for epoch in history:
    epoch_x.append(epoch_counter)
    epoch_counter += 1


    train_y.append(epoch['train_loss'])
    dev_y.append(epoch['dev_loss'])
    lr.append(epoch['lr'])

    
  #plot data
  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

  ax1.plot(epoch_x, train_y, label='Train Loss', color='blue')
  ax1.plot(epoch_x, dev_y, label='Dev Loss', color='orange')
  ax1.set_xlabel('Checkpoint')
  ax1.set_ylabel('Loss')
  ax1.set_title('Train vs Dev Loss')
  ax1.legend()
  ax1.grid(True)

  ax2.plot(epoch_x, lr, label='Learning Rate', color='green')
  ax2.set_xlabel('Checkpoint')
  ax2.set_ylabel('Learning Rate')
  ax2.set_title('Learning Rate Schedule')
  ax2.legend()
  ax2.grid(True)

  plt.tight_layout()
  plt.savefig(path)



def plot_f1(results,path):
  plt.close('all')

  # get data
  history = results['history']

  epoch_counter = 0 

  epoch_x = [] 
  train_y = []
  dev_y = []
  lr = []

  for epoch in history:
    epoch_x.append(epoch_counter)
    epoch_counter += 1


    train_y.append(epoch['train_f1'])
    dev_y.append(epoch['dev_f1'])
    lr.append(epoch['lr'])

    
  #plot data
  plt.plot(epoch_x, train_y, label='Train F1', color='blue')
  plt.plot(epoch_x, dev_y, label='Dev F1', color='orange')
  plt.xlabel('Checkpoint')
  plt.ylabel('F1')
  plt.title('Train vs Dev F1')
  plt.legend()
  plt.grid(True)


  plt.tight_layout()
  plt.savefig(path)


def plot_em(results,path):
  plt.close('all')

  # get data
  history = results['history']

  epoch_counter = 0 

  epoch_x = [] 
  train_y = []
  dev_y = []
  lr = []

  for epoch in history:
    epoch_x.append(epoch_counter)
    epoch_counter += 1


    train_y.append(epoch['train_em'])
    dev_y.append(epoch['dev_em'])
    lr.append(epoch['lr'])

    
  #plot data
  plt.plot(epoch_x, train_y, label='Train em', color='blue')
  plt.plot(epoch_x, dev_y, label='Dev em', color='orange')
  plt.xlabel('Checkpoint')
  plt.ylabel('em')
  plt.title('Train vs Dev em')
  plt.legend()
  plt.grid(True)


  plt.tight_layout()
  plt.savefig(path)
