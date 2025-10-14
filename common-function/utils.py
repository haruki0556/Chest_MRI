def eval_loss(loader,device,net,criterion):

    for images,labels in loader:
        break

    inputs = images.to(device)
    labels = labels.to(device)

    outputs = net(inputs)

    loss = criterion(outputs,labels)

    return loss

def fit(net,optimizer,criterion,num_epochs,train_loader,test_loader,device,history):
    from tqdm.notebook import tqdm

    base_epochs = len(history)

    for epoch in range(base_epochs,num_epochs+base_epochs):
        n_train_acc,n_test_acc = 0,0
        train_loss,test_loss = 0,0
        n_train,n_test = 0,0

        net.train()

        for inputs,labels in tqdm(train_loader):
            train_batch_size = len(labels)
            n_train += train_batch_size

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs,labels)

            loss.backward()

            optimizer.step()

            predicted = torch.max(outputs,1)[1]

            train_loss += loss.items() * train_batch_size
            n_train_acc += (predicted == labels).sum().item()


        net.eval()

        for inputs_test,labels_test in test_loader:
            test_batch_size = len(labels_test)

            n_test += test_batch_size

            inputs_test = inputs_test.to(device)
            labels_test = labels_test.to(device)

            outputs_test = net(inputs_test)

            loss_test = criterion(outputs_test,labels_test)

            predicted_test = torch.max(outputs_test,1)[1]

            test_loss += loss_test.item() * test_batch_size
            n_test_acc += (predicted_test == labels_test).sum().item()

        train_acc = n_train_acc / n_train
        test_acc = n_test_acc / n_test

        avg_train_loss = train_loss / n_train
        avg_test_loss = test_loss / n_test

        print(f'Epoch[{epoch+1}/{num_epochs+base_epochs}],loss:{avg_train_loss:.5f} acc:{train_acc:.5f} test_loss:{avg_test_loss} test_acc:{test_acc:.5f}')

        item = np.array([epoch+1, avg_train_loss, train_acc, avg_test_loss, test_acc])
        history = np.vstack((history, item))
    return history

def evaluate_history(history):

    print(f'初期状態: 損失: {history[0,3]:.5f} 精度: {history[0,4]:.5f}') 
    print(f'最終状態: 損失: {history[-1,3]:.5f} 精度: {history[-1,4]:.5f}' )

    num_epochs = len(history)
    unit = num_epochs / 10

    # 学習曲線の表示 (損失)
    plt.figure(figsize=(9,8))
    plt.plot(history[:,0], history[:,1], 'b', label='訓練')
    plt.plot(history[:,0], history[:,3], 'k', label='検証')
    plt.xticks(np.arange(0,num_epochs+1, unit))
    plt.xlabel('繰り返し回数')
    plt.ylabel('損失')
    plt.title('学習曲線(損失)')
    plt.legend()
    plt.show()

    # 学習曲線の表示 (精度)
    plt.figure(figsize=(9,8))
    plt.plot(history[:,0], history[:,2], 'b', label='訓練')
    plt.plot(history[:,0], history[:,4], 'k', label='検証')
    plt.xticks(np.arange(0,num_epochs+1,unit))
    plt.xlabel('繰り返し回数')
    plt.ylabel('精度')
    plt.title('学習曲線(精度)')
    plt.legend()
    plt.show()

# イメージとラベル表示
def show_images_labels(loader, classes, net, device):

    for images, labels in loader:
        break
    
    n_size = min(len(images), 50)

    if net is not None:
      
      inputs = images.to(device)
      labels = labels.to(device)

      
      outputs = net(inputs)
      predicted = torch.max(outputs,1)[1]

    plt.figure(figsize=(20, 15))
    for i in range(n_size):
        ax = plt.subplot(5, 10, i + 1)
        label_name = classes[labels[i]]
        
        if net is not None:
          predicted_name = classes[predicted[i]]
          
          if label_name == predicted_name:
            c = 'k'
          else:
            c = 'b'
          ax.set_title(label_name + ':' + predicted_name, c=c, fontsize=20)
        
        else:
          ax.set_title(label_name, fontsize=20)
        
        image_np = images[i].numpy().copy()

        img = np.transpose(image_np, (1, 2, 0))

        img = (img + 1)/2
    
        plt.imshow(img)
        ax.set_axis_off()
    plt.show()

# PyTorch乱数固定用

def torch_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True