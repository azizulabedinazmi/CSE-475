# 🍃 **Bangladeshi Mango Leaf Classification with Ensemble Methods**

## 📚 Table of Contents
1. [🎯 Introduction](#-introduction)
2. [🛠️ Data Preparation & Preprocessing](#-data-preparation--preprocessing)
3. [🧠 Baseline CNN Model](#-baseline-cnn-model)
4. [🤖 Ensemble Methods](#-ensemble-methods)
    - 🧩 Bagging
    - 🚀 Boosting (AdaBoost)
    - 🏗️ Stacking
5. [📊 Evaluation and Comparison](#-evaluation-and-comparison)
6. [🌀 Explainability with Grad-CAM](#-explainability-with-grad-cam)
7. [💡 Discussion on Ensemble Methods](#-discussion-on-ensemble-methods)
8. [📊 Model Comparison: With Early Stopping vs Without Early Stopping](#model-comparison-with-early-stopping-vs-without-early-stopping)

---

## 🎯 Introduction
This project tackles the problem of classifying Bangladeshi mango leaf images using Convolutional Neural Networks (CNNs) and ensemble methods to improve classification accuracy and robustness.

The following techniques are applied:
- **Baseline CNN model**: A simple CNN for initial results.
- **Ensemble Methods**: Bagging, Boosting (AdaBoost), and Stacking.
- **Grad-CAM**: Visualization to interpret model predictions.

---

## 🛠️ Data Preparation & Preprocessing

```python
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
```

### 🔄 Transformations
- **Train Transform**: Resize, horizontal flip, rotation, color jitter, normalization.
- **Validation/Test Transform**: Resize, normalization (no augmentation).

```python
train_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load full dataset
full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
num_classes = len(full_dataset.classes)
class_names = full_dataset.classes

print("Number of classes:", num_classes)
print("Class names:", class_names)
```
![](img/Number%20of%20classes.png)

### 📦 Dataset Loading
- Dataset is loaded using PyTorch’s `ImageFolder` method.
- The dataset is stratified split into train, validation, and test sets (80-10-10).
- Dataloaders are created for each split.

```python
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
```

### 📊 Visualization
- A function `plot_batch()` visualizes sample images from a batch.

```python
def plot_batch(loader, classes):
    images, labels = next(iter(loader))
    plt.figure(figsize=(12,6))
    for i in range(min(8, len(images))):
        img = images[i].permute(1,2,0).numpy()
        img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # Denormalize
        img = np.clip(img, 0, 1)
        plt.subplot(2,4,i+1)
        plt.imshow(img)
        plt.title(classes[labels[i]])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

plot_batch(train_loader, class_names)
```

---

## 🧠 Baseline CNN Model

### 🏗️ Model Architecture
A custom CNN with three convolutional blocks and a classifier:
```python
class MangoCNN(nn.Module):
    def __init__(self, num_classes):
        super(MangoCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*28*28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```
- **3 Conv Layers**: 32 → 64 → 128 filters.
- **Pooling**: MaxPooling reduces spatial dimensions.
- **Fully Connected**: One hidden layer and dropout.

### 🚀 Training Function
```python
def train_model(model, criterion, optimizer, epochs=50, patience=10):
    best_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        print("-" * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(train_set)
        epoch_acc = correct / total
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)

        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_loss += loss.item() * inputs.size(0)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss / len(val_set)
        val_acc = val_correct / val_total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}\n")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict()  # Save best model state
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                if best_state is not None:
                    model.load_state_dict(best_state)  # Restore best state
                break

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model_with_ES.pth')

    return history

# Initialize model
model = MangoCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train model
history = train_model(model, criterion, optimizer, epochs=50)
```
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: CrossEntropyLoss
- **With Early Stopping**: Triggers if validation loss doesn't improve for 10 epochs
- **Without Early Stopping**: Continues training regardless of validation loss
- **Checkpointing**: Saves the best model to `best_model_with_ES.pth` or `best_model_without_ES.pth`


### 🧪 Evaluation
```python
def evaluate_model(model, loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return all_labels, all_preds

# Load best model
model.load_state_dict(torch.load('best_model_with_ES.pth'))
test_labels, test_preds = evaluate_model(model, test_loader)

# %%
print("Baseline Model Performance:")
print(classification_report(test_labels, test_preds, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.show()
```
- Outputs classification report & confusion matrix.

---

## 🤖 Ensemble Methods

### 🧩 Bagging
```python
class BaggingEnsemble:
    def __init__(self, num_models=3, patience=10):
        self.num_models = num_models
        self.models = [MangoCNN(num_classes).to(device) for _ in range(num_models)]
        self.patience = patience

    def train(self):
        ColorLogger.info("Training Bagging Ensemble\n" + "="*40)
        for i, model in enumerate(self.models):
            model_name = f"MangoCNN_{i+1}"
            ColorLogger.model_status(f"Training Model {i+1}/{self.num_models} ({model_name})")
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            best_val_loss = float('inf')
            patience_counter = 0
            best_state = None

            # Create bootstrap sample with proper worker count
            indices = torch.randint(0, len(train_set), (len(train_set),))
            bootstrap_loader = DataLoader(
                Subset(train_set, indices),
                batch_size=batch_size,
                shuffle=True,
                num_workers=NUM_WORKERS
            )

            # Train model with progress tracking
            model.train()
            for epoch in range(50):
                total_loss = 0
                correct = 0
                total = 0

                for inputs, labels in bootstrap_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

                epoch_acc = correct / total
                ColorLogger.info(f"Model {i+1} ({model_name}) Epoch {epoch+1}: Loss {total_loss/total:.4f} | Acc {epoch_acc:.2%}")

                # Early stopping check
                val_loss = self.evaluate(model, val_loader, criterion)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = model.state_dict()  # Save best model state
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"Early stopping triggered for {model_name} after {epoch+1} epochs")
                        if best_state is not None:
                            model.load_state_dict(best_state)  # Restore best state
                        break

    def evaluate(self, model, loader, criterion):
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        return val_loss / len(loader.dataset)

    def predict(self, loader):
        ColorLogger.info("Making Bagging Predictions")
        all_preds = []

        for model in self.models:
            model.eval()
            model_preds = []
            with torch.no_grad():
                for inputs, _ in loader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    model_preds.extend(preds.cpu().numpy())
            all_preds.append(model_preds)

        # Proper majority voting
        arr = np.array(all_preds)
        majority_vote, _ = mode(arr, axis=0)
        return majority_vote.flatten()  # Fix shape for classification report

# Verify implementation
print("\n" + "="*40)
ColorLogger.success("Running Corrected Bagging Implementation")
bagging = BaggingEnsemble(num_models=3)
bagging.train()

# %%
# Test prediction with shape verification
bagging_preds = bagging.predict(test_loader)
ColorLogger.info(f"Predictions shape: {np.array(bagging_preds).shape}")
ColorLogger.info(f"Test labels shape: {len(test_labels)}")

# %%
# Safe classification report
try:
    bagging_acc = np.mean(bagging_preds == test_labels)
    ColorLogger.success(f"Bagging Accuracy: {bagging_acc:.2%}")
    print(classification_report(test_labels, bagging_preds, target_names=class_names))
except Exception as e:
    ColorLogger.warning(f"Error in classification: {str(e)}")
    print("Predictions:", np.unique(bagging_preds, return_counts=True))
    print("True labels:", np.unique(test_labels, return_counts=True))
```
- Strategy: Trains 3 MangoCNN models instances on bootstrap samples.
- Prediction: **Majority** voting via scipy.stats.mode.

### 🚀 Boosting (AdaBoost)
```python
class AdaBoost:
    def __init__(self, num_models=3, patience=10):
        self.models = []
        self.alphas = []
        self.num_models = num_models
        self.sample_weights = None
        self.patience = patience

    def train(self):
        ColorLogger.info("Training AdaBoost Ensemble\n" + "="*40)
        self.sample_weights = np.ones(len(train_set)) / len(train_set)

        for m in range(self.num_models):
            model_name = f"MangoCNN_{m+1}"
            ColorLogger.model_status(f"Training Booster {m+1}/{self.num_models} ({model_name})")

            # Create weighted dataset
            weighted_indices = np.random.choice(
                range(len(train_set)),
                size=len(train_set),
                p=self.sample_weights,
                replace=True
            )

            # Create model and optimizer
            model = MangoCNN(num_classes).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            best_val_loss = float('inf')
            patience_counter = 0
            best_state = None

            # Create weighted loader
            weighted_loader = DataLoader(
                Subset(train_set, weighted_indices),
                batch_size=batch_size,
                shuffle=True,
                num_workers=NUM_WORKERS
            )

            # Train model
            model.train()
            for epoch in range(50):
                total_loss = 0
                correct = 0
                total = 0

                for inputs, labels in weighted_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

                epoch_acc = correct / total
                ColorLogger.info(f"Booster {m+1} ({model_name}) Epoch {epoch+1}: Loss {total_loss/total:.4f} | Acc {epoch_acc:.2%}")

                # Early stopping check
                val_loss = self.evaluate(model, val_loader, criterion)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = model.state_dict()  # Save best model state
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"Early stopping triggered for {model_name} after {epoch+1} epochs")
                        if best_state is not None:
                            model.load_state_dict(best_state)  # Restore best state
                        break

            ColorLogger.success(f"Booster {m+1} Training Complete")
            self.models.append(model)

    def evaluate(self, model, loader, criterion):
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        return val_loss / len(loader.dataset)

    def predict(self, loader):
        ColorLogger.info("Making AdaBoost Predictions")
        all_preds = []

        for model in self.models:
            model.eval()
            model_preds = []
            with torch.no_grad():
                for inputs, _ in loader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    model_preds.extend(preds.cpu().numpy())
            all_preds.append(model_preds)

        # Weighted majority voting
        arr = np.array(all_preds)
        majority_vote, _ = mode(arr, axis=0)
        return majority_vote.flatten()  # Fix shape for classification report

# Verify AdaBoost implementation
print("\n" + "="*40)
ColorLogger.success("Running Corrected AdaBoost Implementation")
adaboost = AdaBoost(num_models=3)
adaboost.train()

# %%
# Test prediction with verification
adaboost_preds = adaboost.predict(test_loader)
ColorLogger.info(f"Predictions shape: {adaboost_preds.shape}")
ColorLogger.info(f"Test labels shape: {len(test_labels)}")

# %%
# Safe classification report
try:
    adaboost_acc = np.mean(adaboost_preds == test_labels)
    ColorLogger.success(f"AdaBoost Accuracy: {adaboost_acc:.2%}")
    print(classification_report(test_labels, adaboost_preds, target_names=class_names))
except Exception as e:
    ColorLogger.warning(f"Error in classification: {str(e)}")
    print("Predictions:", np.unique(adaboost_preds, return_counts=True))
    print("True labels:", np.unique(test_labels, return_counts=True))
```
- Strategy: **Sequentially** trains models with reweighted samples (misclassified examples get higher weights).
- Prediction: **Weighted majority voting**.

### 🏗️ Stacking
```python
class StackingEnsemble:
    def __init__(self, patience=10):
        self.base_models = [
            ("MangoCNN", MangoCNN(num_classes).to(device)),
            ("VGG16", models.vgg16(pretrained=True).to(device)),
            ("ResNet50", models.resnet50(pretrained=True).to(device))
        ]
        for name, model in self.base_models[1:]:
            if isinstance(model, models.VGG):
                model.classifier[6] = nn.Linear(4096, num_classes).to(device) # Added .to(device)
            elif isinstance(model, models.ResNet):
                model.fc = nn.Linear(model.fc.in_features, num_classes).to(device) # Added .to(device)
        self.meta_model = LogisticRegression(max_iter=1000)
        self.patience = patience

    def train_base_models(self):
        ColorLogger.info("Training Base Models for Stacking\n" + "="*40)
        for idx, (name, model) in enumerate(self.base_models):
            ColorLogger.model_status(f"Training Model {idx+1}/{len(self.base_models)} ({name})")

            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            best_val_loss = float('inf')
            patience_counter = 0
            best_state = None

            # Train model
            model.train()
            for epoch in range(50):
                total_loss = 0
                correct = 0
                total = 0

                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device) # Moved .to(device) here
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

                epoch_acc = correct / total
                ColorLogger.info(f"Model {idx+1} ({name}) Epoch {epoch+1}: Loss {total_loss/total:.4f} | Acc {epoch_acc:.2%}")

                # Early stopping check
                val_loss = self.evaluate(model, val_loader, criterion)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = model.state_dict()  # Save best model state
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"Early stopping triggered for {name} after {epoch+1} epochs")
                        if best_state is not None:
                            model.load_state_dict(best_state)  # Restore best state
                        break

            ColorLogger.success(f"Model {idx+1} ({name}) Training Complete")

    def evaluate(self, model, loader, criterion):
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        return val_loss / len(loader.dataset)

    def train_meta_model(self):
        ColorLogger.info("Generating Meta Features")
        meta_features = []

        for name, model in self.base_models:
            model.eval()
            preds = []
            with torch.no_grad():
                for inputs, _ in val_loader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    probs = torch.softmax(outputs, 1).cpu().numpy()
                    preds.extend(probs)
            meta_features.append(np.array(preds))
            ColorLogger.info(f"Generated meta features from {name}")

        # Ensure equal shape
        min_length = min([arr.shape[0] for arr in meta_features])
        meta_features = [arr[:min_length] for arr in meta_features]

        X_meta = np.hstack(meta_features)
        y_meta = [label for _, label in val_set][:min_length]

        ColorLogger.info(f"Meta dataset shape: {X_meta.shape}")
        self.meta_model.fit(X_meta, y_meta)

    def predict(self, loader):
        ColorLogger.info("Making Stacking Predictions")
        meta_features = []

        for name, model in self.base_models:
            model.eval()
            preds = []
            with torch.no_grad():
                for inputs, _ in loader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    probs = torch.softmax(outputs, 1).cpu().numpy()
                    preds.extend(probs)
            meta_features.append(np.array(preds))
            ColorLogger.info(f"Generated predictions from {name}")

        # Ensure equal shape
        min_length = min([arr.shape[0] for arr in meta_features])
        meta_features = [arr[:min_length] for arr in meta_features]

        X_test_meta = np.hstack(meta_features)
        return self.meta_model.predict(X_test_meta)

# Verify Stacking implementation
print("\n" + "="*40)
ColorLogger.success("Running Corrected Stacking Implementation")
stacking = StackingEnsemble()
stacking.train_base_models()
stacking.train_meta_model()

# %%
# Test prediction with verification
stacking_preds = stacking.predict(test_loader)
ColorLogger.info(f"Predictions shape: {stacking_preds.shape}")
ColorLogger.info(f"Test labels shape: {len(test_labels)}")

# %%
# Safe classification report
try:
    stacking_acc = np.mean(stacking_preds == test_labels[:len(stacking_preds)])
    ColorLogger.success(f"Stacking Accuracy: {stacking_acc:.2%}")
    print(classification_report(
        test_labels[:len(stacking_preds)],
        stacking_preds,
        target_names=class_names
    ))
except Exception as e:
    ColorLogger.warning(f"Error in classification: {str(e)}")
    print("Predictions:", np.unique(stacking_preds, return_counts=True))
    print("True labels:", np.unique(test_labels, return_counts=True))
```
- Base Models:
  - Custom `MangoCNN`
  - Fine-tuned `VGG16` (last layer modified)
  - Fine-tuned `ResNet50` (last layer modified)
-Meta-Model: **Logistic Regression** trained on base models' predictions.

---
## 📊 Evaluation and Comparison

### 📈 Metrics Used
- **Accuracy**
- **Confusion Matrix**
- **Classification Report**

### 📊 Visualization
- Training history graphs.
- Bar plot comparing ensemble accuracies.

```python
# Calculate accuracies
baseline_acc = np.mean(np.array(test_labels) == np.array(test_preds))
bagging_acc = np.mean(np.array(test_labels) == np.array(bagging_preds))
adaboost_acc = np.mean(np.array(test_labels) == np.array(adaboost_preds))
stacking_acc = np.mean(np.array(test_labels) == np.array(stacking_preds))

# Results comparison
results = pd.DataFrame({
    'Model': ['Baseline', 'Bagging', 'AdaBoost', 'Stacking'],
    'Accuracy': [baseline_acc, bagging_acc, adaboost_acc, stacking_acc]
})

plt.figure(figsize=(10,6))
sns.barplot(x='Model', y='Accuracy', data=results, hue='Model', palette='viridis', dodge=False, legend=False)  # Updated line
plt.ylim(0.7, 1.0)
plt.title('Model Comparison', color='blue')  # Set color using matplotlib
plt.show()

ColorLogger.success("Final Results:")
print(results)# Calculate accuracies
baseline_acc = np.mean(np.array(test_labels) == np.array(test_preds))
bagging_acc = np.mean(np.array(test_labels) == np.array(bagging_preds))
adaboost_acc = np.mean(np.array(test_labels) == np.array(adaboost_preds))
stacking_acc = np.mean(np.array(test_labels) == np.array(stacking_preds))

# Results comparison
results = pd.DataFrame({
    'Model': ['Baseline', 'Bagging', 'AdaBoost', 'Stacking'],
    'Accuracy': [baseline_acc, bagging_acc, adaboost_acc, stacking_acc]
})

plt.figure(figsize=(10,6))
sns.barplot(x='Model', y='Accuracy', data=results, hue='Model', palette='viridis', dodge=False, legend=False)  # Updated line
plt.ylim(0.7, 1.0)
plt.title('Model Comparison', color='blue')  # Set color using matplotlib
plt.show()

ColorLogger.success("Final Results:")
print(results)
```

## 🌀 Explainability with Grad-CAM

### 🔍 Grad-CAM Visualization
- Highlights regions that influenced the model’s decision.
- Works with individual models inside ensembles.

```python
def visualize_gradcam(img_path, ensemble, models_list):
    # ensemble.eval()  <- Remove this line as StackingEnsemble doesn't have eval()

    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)

    heatmaps = []
    weights = torch.softmax(torch.tensor([1.0/len(models_list)]*len(models_list)), dim=0).detach().cpu().numpy()

    # Get the target layers for each model type
    target_layers = [
        models_list[0].features[-1],  # MangoCNN
        models_list[1].features[-1],  # Assume adaboost.models[0] is also a custom CNN - Change if necessary
        models_list[2].features[-1]   # MangoCNN
    ]

    for model, layer, w in zip(models_list, target_layers, weights):
        # Set each individual model to eval mode
        model.eval()

        grads, acts = None, None
        def forward_hook(module, inp, out):
            nonlocal acts
            acts = out.detach()
        def backward_hook(module, gin, gout):
            nonlocal grads
            grads = gout[0].detach()

        hook_f = layer.register_forward_hook(forward_hook)
        hook_b = layer.register_backward_hook(backward_hook)

        model.zero_grad()
        out = model(input_tensor)
        out[:, out.argmax(dim=1)].backward()

        pooled_grad = grads.mean([0, 2, 3])
        cam = (acts[0] * pooled_grad[:, None, None]).sum(0).relu()
        cam = cam / cam.max()
        heatmaps.append(w * cam.cpu().numpy())

        hook_f.remove()
        hook_b.remove()

    final_heatmap = np.sum(heatmaps, axis=0)
    final_heatmap = cv2.resize(final_heatmap, (img.width, img.height))
    heatmap = cv2.applyColorMap(np.uint8(255 * final_heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.array(img), 0.6, heatmap, 0.4, 0)

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title('Original')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(overlay)
    plt.title('Grad-CAM')
    plt.axis('off')
    plt.show()

# Example usage:
img_path = '/content/drive/MyDrive/Image_Dataset of_Bangladeshi_Mango_Leaf/Original/Fazli Mango/Fazli_Mango_8.JPG'  # Replace with the path to your image
models_list = [bagging.models[0], adaboost.models[0], stacking.base_models[0][1]]  # List of models used in the ensemble
visualize_gradcam(img_path, stacking, models_list)
```
- Overlays heatmaps onto original images.

---

## 💡 Discussion on Ensemble Methods

| ⚙️ Method   | ✅ Strengths                                | ⚠️ Limitations                             | 🔄 Best Use Case                      |
|------------|---------------------------------------------|-------------------------------------------|--------------------------------------|
| **Bagging**  | Reduces variance, parallelizable            | Might not boost already strong models     | Stable predictions, noisy datasets   |
| **Boosting** | Reduces bias, focuses on hard examples      | Can overfit noisy data, sequential        | Improving weak learner accuracy      |
| **Stacking** | Combines diverse models, maximizes accuracy | Complex, resource-heavy                   | Maximizing accuracy using multiple models |

---

## 🎉 **Summary**
- ✅ **Baseline CNN**: Simple but provides a solid starting point.
- 🧩 **Bagging**: Improves robustness via independent learners.
- 🚀 **Boosting**: Sequentially corrects errors.
- 🏗️ **Stacking**: Combines diverse models for maximum performance.
- 🔍 **Grad-CAM**: Explains "where the model looks" in an image.

---

---

## Model Comparison: With Early Stopping vs Without Early Stopping

| **With Early Stopping** | **Without Early Stopping** |
|-------------------------|----------------------------|
| **Training Output <br> Plots training/validation loss & accuracy over epochs.** <br> ![](img/Training%20Output_With_ES.png) | **Training Output <br> Plots training/validation loss & accuracy over epochs.** <br> ![](img/Training%20Output_Without_ES.png) |
| **Baseline Model Performance** <br> ![](img/Baseline%20Model%20Performance_With_ES.png) | **Baseline Model Performance** <br> ![](img/Baseline%20Model%20Performance_Without_ES.png) |
| **Baseline Model Confusion Matrix** <br> ![](img/Baseline%20Model%20Performance_Confusion%20Matrix_With_ES.png) | **Baseline Model Confusion Matrix** <br> ![](img/Baseline%20Model%20Performance_Confusion%20Matrix_Without_ES.png) |
| **Bagging Ensemble Accuracy** <br> ![](img/Bagging%20Accuracy_With_ES.png) | **Bagging Ensemble Accuracy** <br> ![](img/Bagging%20Accuracy_Without_ES.png) |
| **Bagging Ensemble Confusion Matrix** <br> ![](img/Bagging%20Ensemble%20Confusion%20Matrix_With_ES.png) | **Bagging Ensemble Confusion Matrix** <br> ![](img/Bagging%20Ensemble%20Confusion%20Matrix_Without_ES.png) |
| **AdaBoost Ensemble Accuracy** <br> ![](img/AdaBoost%20Ensemble%20Accuracy_With_ES.png) | **AdaBoost Ensemble Accuracy** <br> ![](img/AdaBoost%20Ensemble%20Accuracy_Without_ES.png) |
| **AdaBoost Ensemble Confusion Matrix** <br> ![](img/AdaBoost%20Ensemble%20Confusion%20Matrix_With_ES.png) | **AdaBoost Ensemble Confusion Matrix** <br> ![](img/AdaBoost%20Ensemble%20Confusion%20Matrix_Without_ES.png) |
| **Stacking Ensemble Accuracy** <br> ![](img/Stacking%20Ensemble%20Accuracy_With_ES.png) | **Stacking Ensemble Accuracy** <br> ![](img/Stacking%20Ensemble%20Accuracy_Without_ES.png) |
| **Stacking Ensemble Confusion Matrix** <br> ![](img/Stacking%20Ensemble%20Confusion%20Matrix_With_ES.png) | **Stacking Ensemble Confusion Matrix** <br> ![](img/Stacking%20Ensemble%20Confusion%20Matrix_Without_ES.png) |
| **Model Comparison** <br> ![](img/Model%20Comparison_With_ES.png) | **Model Comparison** <br> ![](img/Model%20Comparison_Without_ES.png) |
| **Grad-CAM Visualization 1** <br> ![](img/Grad-CAM%20Visualization%201_With_ES.png) | **Grad-CAM Visualization 1** <br> ![](img/Grad-CAM%20Visualization%201_Without_ES.png) |
| **Grad-CAM Visualization 2** <br> ![](img/Grad-CAM%20Visualization%202_With_ES.png) | **Grad-CAM Visualization 2** <br> ![](img/Grad-CAM%20Visualization%202_Without_ES.png) |
| **Grad-CAM Visualization 3** <br> ![](img/Grad-CAM%20Visualization%203_With_ES.png) | **Grad-CAM Visualization 3** <br> ![](img/Grad-CAM%20Visualization%203_Without_ES.png) |


This project is made by [@Azizul Abedin Azmi](https://github.com/azizulabedinazmi) , [@Tanzila Afrin](https://github.com/Tanzila-Afrin) & [@Touhid Limon](https://www.facebook.com/touhid.limon.5)