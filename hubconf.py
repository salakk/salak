def get_model(train_data_loader=None, n_epochs=10):
  model = None
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  for X_b, y_b in train_data_loader:
    X = X_b[0]
    (ch, H, W) = X.shape
    break
  
  labels = set()
  for X_b, y_b in train_data_loader:
    labels = labels.union(set(y_b.cpu().detach().numpy()))
    
  num_classes = len(labels)  
  
  
  
class cs21m013(nn.Module):
  def __init__(self, ch, H, W, num_classes):
    super(cs21m013, self).__init__()
    self.conv_layers = nn.Sequential(
        nn.Conv2d(ch, 8, 3, padding='same'),
        nn.Conv2d(8, 16, 3, padding='same'),
        )
    
        in_features = 16 * H * W
        self.fc = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.Linear(2048, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 256),
            nn.Linear(256, num_classes)
            )

  def forward(self, x):
    feat_maps = self.conv_layers(x)
    feats = nn.Flatten(start_dim=1)(feat_maps)
    logits = self.fc(feats)
   
    return logits


model = cs21m013(ch, H, W, num_classes)
model.to(device)
optimizer = optim.Adam(model.parameters())

total_loss = 0.0
correct = 0
for X_b, y_b in tqdm(train_data_loader):
  X_b, y_b = X_b.to(device), y_b.to(device)
  logits = model(X_b)
  loss = nn.CrossEntropyLoss()(logits, y_b)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  
  total_loss += loss.item()
  probs: torch.Tensor = nn.Softmax(dim=1)(logits)
  preds = probs.argmax(dim=1)
  correct += (preds == y_b).sum().item()
    

  print(f"Accuracy of the model: {correct/len(train_data_loader.dataset)}")
  return model



def get_model_advanced(train_data_loader=None, n_epochs=10,lr=1e-4,config=None):
  model = None
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  for X_b, y_b in train_data_loader:
    X = X_b[0]
    (ch, H, W) = X.shape
    break

  labels = set()
  for X_b, y_b in train_data_loader:
    labels = labels.union(set(y_b.cpu().detach().numpy()))
    num_classes = len(labels)


   class cs21m013_model(nn.Module):
     def __init__(self, config, H, W, num_classes):
       super(cs21m013, self).__init__()
       temp = []
       for k in config:
         in_ch, out_ch, kernel_size, stride, padding = k
         temp.append(nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=padding))
            
        self.conv_layers = nn.Sequential(*temp)
        
        def _in_features(H, W, config):
          for k in config:
            in_ch, out_ch, kernel_size, stride, padding = k
            H = H if padding == 'same' else (H - (kernel_size[0]-1) + 2 * padding) / stride
            W = W if padding == 'same' else (W - (kernel_size[1]-1) + 2 * padding) / stride
          return H * W * config[-1][1]
            
        in_features = _in_features(H, W, config)
        self.fc = nn.Sequential(nn.Linear(in_features, 2048),
                                nn.Linear(2048, 1024),nn.Linear(1024, 512)
                                ,nn.Linear(512,  num_classes),)


        def forward(self, x):
          feat_maps = self.conv_layers(x)
          feats = nn.Flatten(start_dim=1)(feat_maps)
          logits = self.fc(feats)
          return logits


    model = cs21m013_model(config, H, W, num_classes)
    model.to(device)
    optimizer = optim.Adam(model.parameters())


    total_loss = 0.0
    correct = 0

    for X_b, y_b in tqdm(train_data_loader):
      X_b, y_b = X_b.to(device), y_b.to(device)
      logits = model(X_b)
      loss = nn.CrossEntropyLoss()(logits, y_b)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
      probs: torch.Tensor = nn.Softmax(dim=1)(logits)
      preds = probs.argmax(dim=1)
      correct += (preds == y_b).sum().item()
    
    
    return model

    print ('Returning model... (cs21m013: xx)')
    print(f"Accuracy of the model: {correct/len(train_data_loader.dataset)}")
    
    return model



def test_model(model1=None, test_data_loader=None):
  accuracy_val, precision_val, recall_val, f1score_val = 0, 0, 0, 0
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
  total_loss = 0.0
  correct = 0
  for X_b, y_b in test_data_loader:
    X_b, y_b = X_b.to(device), y_b.to(device)
    logits = model1(X_b)
    loss = nn.CrossEntropyLoss()(logits, y_b)
    total_loss += loss.item()

    probs = nn.Softmax(dim=1)(logits)
    preds = probs.argmax(dim=1)

    correct = (preds == y_b).sum().item()

    preds = preds.cpu().detach().numpy()
    y_b = y_b.cpu().detach().numpy()


    accuracy_val = correct / len(test_data_loader.dataset)
    precision_val = precision_score(y_b, preds, average='macro')
    recall_val = recall_score(y_b, preds, average='macro')
    f1score_val = f1_score(y_b, preds, average='macro')

    print ('Returning metrics... (cs21m013: xx)')
    return accuracy_val, precision_val, recall_val, f1score_val
