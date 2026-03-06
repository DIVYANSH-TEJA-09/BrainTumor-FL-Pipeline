# CURSOR AI PROMPTS FOR FL-QPSO PROJECT

**How to use these prompts with Cursor AI:**
1. Open Cursor AI editor
2. Copy the prompt for the component you want to create
3. Paste into Cursor's chat/command interface
4. Review and integrate the generated code
5. Test the code before moving to next component

---

## PROMPT 1: Data Preprocessing Class

```
Create a Python class called BrainTumorPreprocessor for preprocessing brain tumor MRI datasets with the following requirements:

1. Initialize with target_size (default 224x224)
2. Only process 3 valid classes: glioma, meningioma, pituitary (exclude any 'normal' or 'notumor' classes)
3. Create a class_to_idx mapping: glioma=0, meningioma=1, pituitary=2

Methods needed:
- is_valid_class(class_name): Check if class should be included
- load_and_preprocess_image(image_path): 
  * Load image using PIL
  * Convert to RGB if needed
  * Resize to target_size using BILINEAR interpolation
  * Normalize pixel values to [0, 1]
  * Return numpy array shape (H, W, 3)

- process_dataset(dataset_path, client_name, train_ratio=0.7, val_ratio=0.15):
  * Scan dataset_path for class folders
  * Filter only valid classes
  * Load all images and labels
  * Split into train/val/test using stratified sampling
  * Save as numpy arrays: X_train.npy, y_train.npy, etc.
  * Print progress using tqdm
  * Save to /kaggle/working/data/processed/{client_name}/

Include proper error handling for:
- Missing files
- Corrupted images
- Invalid paths

Add docstrings to all methods explaining parameters and return values.
```

---

## PROMPT 2: PyTorch Dataset Class

```
Create a PyTorch Dataset class called BrainTumorDataset for federated learning with these specifications:

Class requirements:
- Inherits from torch.utils.data.Dataset
- __init__ takes: X (numpy array of images), y (numpy array of labels), transform (optional transforms)
- Store X, y, and transform as instance variables
- __len__ returns number of samples
- __getitem__(idx) returns:
  * Convert numpy image to PIL Image (scale back to 0-255)
  * Apply transforms if provided, otherwise use default:
    - Convert to tensor
    - Normalize with ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
  * Return (image_tensor, label_tensor)

Make sure labels are torch.long dtype for CrossEntropyLoss compatibility.

Include complete imports: torch, PIL, torchvision.transforms
Add docstring explaining usage.
```

---

## PROMPT 3: ResNet-18 Model Class

```
Create a PyTorch model class called BrainTumorResNet for 3-class brain tumor classification:

Requirements:
- Inherit from nn.Module
- __init__ takes: num_classes (default 3), pretrained (default True)
- Load pretrained ResNet-18 from torchvision.models
- Replace the final fully connected layer:
  * Get number of input features from original fc layer
  * Create new fc layer: nn.Linear(num_features, num_classes)
- forward(x) passes input through self.model

Also create a factory function:
- create_model(num_classes=3, device='cuda'):
  * Instantiate BrainTumorResNet
  * Move to specified device
  * Return model

Include complete imports: torch, torch.nn, torchvision.models
Print model summary when created (total params, trainable params, model size in MB).
```

---

## PROMPT 4: Federated Client Class

```
Create a FederatedClient class for local training in federated learning:

Constructor parameters:
- client_id: unique identifier string (e.g., 'client1')
- train_loader: PyTorch DataLoader for training
- val_loader: PyTorch DataLoader for validation
- device: 'cuda' or 'cpu'

Instance variables:
- Store all constructor params
- model: will be set later (initially None)
- optimizer: will be set later (initially None)
- criterion: nn.CrossEntropyLoss()
- dataset_size: len(train_loader.dataset)

Methods:

1. set_model(global_model):
   - Deep copy global_model
   - Move to self.device
   - Store as self.model

2. set_optimizer(learning_rate=0.001):
   - Create Adam optimizer for model parameters

3. train_local(epochs=5, verbose=False):
   - Train model for specified epochs
   - For each epoch:
     * Iterate through train_loader with tqdm progress bar
     * Forward pass, compute loss, backward pass, optimizer step
     * Track running loss and accuracy
   - Return: (updated model state_dict, list of epoch losses, list of epoch accuracies)

4. validate():
   - Set model to eval mode
   - Iterate through val_loader without gradients
   - Calculate validation loss and accuracy
   - Return: (val_loss, val_acc)

5. get_dataset_size():
   - Return self.dataset_size

Include complete imports: torch, torch.nn, torch.optim, copy, tqdm
Use proper torch.no_grad() context for validation.
Add descriptive progress bars with loss and accuracy.
```

---

## PROMPT 5: FedAvg Server Class

```
Create a FedAvgServer class implementing Federated Averaging algorithm:

Constructor parameters:
- global_model: the global model to be aggregated
- clients: list of FederatedClient objects
- device: 'cuda' or 'cpu'

Constructor logic:
- Store all parameters
- Calculate total_samples = sum of all client dataset sizes
- Print initialization info (number of clients, total samples)

Methods:

1. aggregate_weights(client_weights):
   - Input: list of tuples (state_dict, dataset_size)
   - Algorithm:
     * Initialize aggregated_weights as copy of first client's weights
     * Set all values to zero
     * For each client:
       - Calculate weight = dataset_size / total_samples
       - Add weighted client weights to aggregated_weights
   - Return: aggregated state_dict

2. evaluate_global_model(test_loader):
   - Set model to eval mode
   - Iterate through test_loader without gradients
   - Calculate test loss (CrossEntropyLoss) and accuracy
   - Return: (accuracy, loss)

3. get_global_model():
   - Return deep copy of global_model

Include complete imports: torch, torch.nn, copy
Use proper tensor operations for aggregation.
Add comments explaining the FedAvg weighted averaging formula.
```

---

## PROMPT 6: QPSO Server Class

```
Create a QPSOServer class implementing Quantum-behaved PSO for federated learning:

Constructor parameters:
- global_model: the global model
- clients: list of FederatedClient objects
- device: 'cuda' or 'cpu'
- beta: contraction-expansion coefficient (default 0.7)

Instance variables:
- Store all constructor params
- personal_best: dict {client_id: state_dict}
- personal_best_scores: dict {client_id: float}
- global_best: state_dict
- global_best_score: float
- mean_best: state_dict

Methods:

1. initialize_particles():
   - Set all personal_bests to current global model
   - Set all scores to 0.0
   - Print confirmation

2. update_personal_best(client_id, client_weights, validation_acc):
   - If validation_acc > personal_best_scores[client_id]:
     * Update personal_best[client_id] = copy of client_weights
     * Update personal_best_scores[client_id] = validation_acc
     * Return True
   - Else return False

3. update_global_best(client_id, validation_acc):
   - If validation_acc > global_best_score:
     * Update global_best = copy of personal_best[client_id]
     * Update global_best_score = validation_acc
     * Return True
   - Else return False

4. calculate_mean_best():
   - Initialize mean_best with zeros
   - Sum all personal_best tensors
   - Divide by number of clients

5. qpso_aggregate(client_weights_list):
   - Input: list of tuples (client_id, state_dict, val_acc)
   - Update personal and global bests for all clients
   - Calculate mean_best
   - For each parameter key:
     * Generate random phi, u
     * Calculate attraction point p = phi * pbest + (1-phi) * gbest
     * Calculate QPSO update:
       - sign = random +1 or -1
       - new_value = p + sign * beta * |mbest - current| * log(1/u)
     * Store in aggregated_weights
   - Return aggregated_weights

6. evaluate_global_model(test_loader):
   - Same as FedAvg

7. get_global_model():
   - Return deep copy of global_model

Include complete imports: torch, torch.nn, copy, random
Add epsilon (1e-8) to u in log calculation to avoid log(0).
Convert tensors to float32 for QPSO calculations, then back to original dtype.
```

---

## PROMPT 7: FedAvg Training Loop Function

```
Create a function train_fedavg for complete training loop:

Parameters:
- server: FedAvgServer object
- clients: list of FederatedClient objects
- global_test_loader: DataLoader for global test set
- num_rounds: number of communication rounds (default 100)
- local_epochs: local training epochs per round (default 5)
- learning_rate: learning rate for local training (default 0.001)
- save_every: save checkpoint every N rounds (default 10)
- verbose: print detailed logs (default True)

Returns:
- history: dict with keys:
  * 'round': list of round numbers
  * 'global_test_acc': list of test accuracies
  * 'global_test_loss': list of test losses
  * 'client1_val_acc', 'client2_val_acc', 'client3_val_acc': per-client validation accuracies
  * 'round_time': time taken per round

Training loop:
For each round from 1 to num_rounds:
  1. Print round header
  2. For each client:
     - Set global model
     - Set optimizer with learning_rate
     - Train locally for local_epochs
     - Validate
     - Collect (state_dict, dataset_size)
  3. Server aggregates using aggregate_weights()
  4. Load aggregated weights into global model
  5. Evaluate global model on global_test_loader
  6. Track all metrics in history
  7. Save best model if accuracy improved
  8. Save checkpoint every save_every rounds
  9. Save history to CSV after each round

Save models to: /kaggle/working/models/fedavg_*.pth
Save metrics to: /kaggle/working/results/fedavg/metrics.csv

Include time tracking with time.time().
Print comprehensive progress information.
Handle best model tracking properly.
```

---

## PROMPT 8: QPSO Training Loop Function

```
Create a function train_qpso for QPSO-FL training loop:

Parameters:
- server: QPSOServer object
- clients: list of FederatedClient objects
- global_test_loader: DataLoader for global test set
- num_rounds: number of communication rounds (default 100)
- local_epochs: local training epochs per round (default 5)
- learning_rate: learning rate (default 0.001)
- save_every: save checkpoint every N rounds (default 10)
- verbose: print detailed logs (default True)

Returns:
- history: dict with keys:
  * 'round', 'global_test_acc', 'global_test_loss'
  * 'global_best_score': QPSO global best score
  * 'client1_val_acc', 'client2_val_acc', 'client3_val_acc'
  * 'client1_pbest_score', 'client2_pbest_score', 'client3_pbest_score'
  * 'round_time'

Training loop:
Initial: Call server.initialize_particles()

For each round:
  1. Print round header
  2. Collect client_weights_list = []
  3. For each client:
     - Set global model
     - Set optimizer
     - Train locally
     - Validate
     - Append (client_id, state_dict, val_acc) to client_weights_list
  4. Server aggregates using qpso_aggregate(client_weights_list)
  5. Load aggregated weights into global model
  6. Track personal best scores and global best score
  7. Evaluate global model
  8. Track all metrics including QPSO-specific ones
  9. Save best model and checkpoints
  10. Save history to CSV

Save models to: /kaggle/working/models/qpso_*.pth
Save metrics to: /kaggle/working/results/qpso/metrics.csv

Print QPSO status (global best score, personal best scores) each round.
Include all time tracking and progress information.
```

---

## PROMPT 9: Data Loader Creation Function

```
Create a function create_data_loaders that sets up all PyTorch DataLoaders:

Parameters:
- batch_size: batch size for training (default 32)
- num_workers: number of workers for data loading (default 2)

Returns:
- loaders: dict with structure:
  {
    'client1': {'train': DataLoader, 'val': DataLoader, 'test': DataLoader, 'train_size': int},
    'client2': {...},
    'client3': {...},
    'global_test': DataLoader
  }

Implementation:
1. Define train_transform with augmentation:
   - RandomHorizontalFlip(0.5)
   - RandomRotation(15)
   - ColorJitter(brightness=0.2, contrast=0.2)
   - ToTensor()
   - Normalize with ImageNet stats

2. Define test_transform (no augmentation):
   - ToTensor()
   - Normalize with ImageNet stats

3. For each client (client1, client2, client3):
   - Load numpy arrays from /kaggle/working/data/processed/{client}/
   - Create BrainTumorDataset for train, val, test
   - Create DataLoaders with:
     * shuffle=True for train, False for val/test
     * pin_memory=True for GPU efficiency
   - Store in loaders dict with train_size

4. Create global_test_loader:
   - Load from /kaggle/working/data/test_set/
   - Create dataset and loader
   - Add to loaders dict

Print summary of dataset sizes for each client.
Use proper error handling for missing files.
```

---

## PROMPT 10: Visualization Functions

```
Create comprehensive visualization functions for FL results comparison:

Function 1: plot_accuracy_comparison(df_fedavg, df_qpso, save_path)
- Create 2x2 subplot grid
- Plot 1 (top-left): Global test accuracy over rounds for both methods
- Plot 2 (top-right): Global test loss over rounds
- Plot 3 (bottom-left): Per-client accuracy for FedAvg
- Plot 4 (bottom-right): Per-client accuracy for QPSO
- Use different colors and markers
- Add legends, grid, labels
- Save to save_path with high DPI (300)

Function 2: plot_confusion_matrix(model_path, test_loader, method_name, device, save_path)
- Load model from model_path
- Generate predictions on test_loader
- Calculate confusion matrix using sklearn
- Plot heatmap using seaborn
- Labels: ['Glioma', 'Meningioma', 'Pituitary']
- Save figure
- Print classification report (precision, recall, f1-score)

Function 3: plot_client_fairness(df_fedavg, df_qpso, save_path)
- Create 1x2 subplot
- Bar plot 1: Final validation accuracy per client for FedAvg
  * Add horizontal line for mean
  * Show std deviation in title
- Bar plot 2: Same for QPSO
- Use colors: ['#FF6B6B', '#4ECDC4', '#45B7D1'] for clients
- Save figure

Function 4: generate_results_summary(df_fedavg, df_qpso)
- Calculate and return dict with:
  * Final accuracies for both methods
  * Best accuracies
  * Rounds to reach 80% accuracy
  * Average round time
  * Total training time
  * Client accuracy std deviation
- Format as pandas DataFrame
- Include improvement percentages

Include all necessary imports: matplotlib, seaborn, sklearn, pandas, numpy
Use proper figure sizing and tight_layout.
```

---

## PROMPT 11: Utility Functions

```
Create utility functions for the FL project:

Function 1: set_seed(seed=42)
- Set random seeds for reproducibility:
  * random.seed(seed)
  * np.random.seed(seed)
  * torch.manual_seed(seed)
  * torch.cuda.manual_seed(seed)
  * torch.cuda.manual_seed_all(seed)
  * torch.backends.cudnn.deterministic = True
  * torch.backends.cudnn.benchmark = False

Function 2: check_gpu()
- Print GPU availability
- Print CUDA version
- Print number of GPUs
- For each GPU:
  * Print name
  * Print memory
- Return device: torch.device('cuda' if available else 'cpu')

Function 3: create_directories()
- Create all necessary directories for the project:
  * /kaggle/working/data/raw/figshare
  * /kaggle/working/data/raw/brisc
  * /kaggle/working/data/raw/kaggle
  * /kaggle/working/data/processed/client1
  * /kaggle/working/data/processed/client2
  * /kaggle/working/data/processed/client3
  * /kaggle/working/data/test_set
  * /kaggle/working/models
  * /kaggle/working/results/fedavg
  * /kaggle/working/results/qpso
  * /kaggle/working/results/plots
  * /kaggle/working/logs
  * /kaggle/working/src
- Use os.makedirs with exist_ok=True
- Print confirmation

Function 4: save_experiment_config(config, method_name)
- Save experiment configuration as JSON
- Path: /kaggle/working/results/{method_name}/config.json
- Include: num_rounds, local_epochs, learning_rate, batch_size, etc.

Function 5: load_checkpoint(model, checkpoint_path)
- Load model state_dict from checkpoint_path
- Handle errors if file doesn't exist
- Print confirmation with checkpoint round number

Include all imports: os, json, random, numpy, torch
```

---

## PROMPT 12: Main Execution Script

```
Create a main execution script that runs the complete FL experiment:

Structure:
1. Initial Setup
   - Set seed for reproducibility
   - Check GPU availability
   - Create directory structure

2. Configuration
   - Define CONFIG dict with:
     * num_rounds = 100
     * local_epochs = 5
     * learning_rate = 0.001
     * batch_size = 32
     * beta = 0.7 (for QPSO)
     * save_every = 10

3. Data Loading
   - Call create_data_loaders()
   - Verify all loaders created
   - Print dataset statistics

4. FedAvg Experiment
   - Print experiment header
   - Create fresh model
   - Create clients list
   - Create FedAvgServer
   - Call train_fedavg()
   - Save final results

5. QPSO Experiment
   - Print experiment header
   - Create fresh model (separate from FedAvg)
   - Create new clients list
   - Create QPSOServer
   - Call train_qpso()
   - Save final results

6. Comparison and Visualization
   - Load results from both experiments
   - Generate comparison plots
   - Generate confusion matrices
   - Generate fairness analysis
   - Save summary statistics

7. Final Report
   - Print final comparison table
   - Print best accuracies
   - Print convergence analysis
   - Save all results to CSV

Include proper error handling at each stage.
Add time tracking for entire experiment.
Print progress messages throughout.
Save intermediate results in case of interruption.
```

---

## PROMPT 13: Debug and Testing Functions

```
Create debugging and testing functions:

Function 1: test_preprocessing(dataset_path, sample_size=5)
- Load sample_size images from dataset_path
- Apply preprocessing
- Visualize before and after preprocessing
- Print shape and dtype information
- Verify normalization range

Function 2: test_model_forward_pass(model, input_shape=(1, 3, 224, 224), device='cuda')
- Create random input tensor
- Pass through model
- Print output shape
- Verify output dimensions
- Test with different batch sizes
- Return True if all tests pass

Function 3: test_client_training(client, num_epochs=2)
- Train client for num_epochs
- Verify loss decreases
- Verify gradients are computed
- Print training metrics
- Return True if training successful

Function 4: test_server_aggregation(server, num_clients=3)
- Create dummy client weights
- Test aggregation function
- Verify output shape matches input
- Verify all parameters updated
- Return True if aggregation works

Function 5: verify_data_integrity()
- Check all preprocessed files exist
- Verify file sizes are reasonable
- Check for NaN or Inf values
- Verify label distribution
- Print summary report

Function 6: quick_test_run(num_rounds=3)
- Run abbreviated training with 3 rounds
- Test both FedAvg and QPSO
- Use small subset of data
- Verify entire pipeline works
- Return True if successful

Include comprehensive error messages.
Add try-except blocks for robust error handling.
Print detailed diagnostic information.
```

---

## PROMPT 14: Results Analysis Functions

```
Create functions for comprehensive results analysis:

Function 1: calculate_convergence_metrics(df_history, target_accuracy=80.0)
- Find round when target_accuracy first reached
- Calculate convergence speed (rounds to target)
- Find final accuracy
- Find best accuracy
- Calculate average improvement per round
- Return dict with all metrics

Function 2: perform_statistical_analysis(df_fedavg, df_qpso)
- Perform paired t-test on accuracies
- Calculate effect size (Cohen's d)
- Test for significant difference in convergence
- Return dict with:
  * p_value
  * is_significant (p < 0.05)
  * effect_size
  * confidence_interval

Function 3: generate_latex_tables(comparison_df)
- Convert comparison DataFrame to LaTeX format
- Format numbers properly (2 decimal places)
- Add table caption
- Save to file
- Return LaTeX string

Function 4: export_results_for_paper()
- Export all key results in publication-ready format:
  * Main comparison table (CSV and LaTeX)
  * Per-round metrics (CSV)
  * Confusion matrices (high-res PNG)
  * All plots (PDF and PNG)
- Create results_for_paper/ directory
- Organize all files
- Print checklist of exported files

Function 5: create_executive_summary(df_fedavg, df_qpso)
- Generate concise summary with:
  * Best accuracy for each method
  * Improvement percentage
  * Convergence comparison
  * Fairness comparison
  * Key findings
- Save as text file and JSON
- Return formatted string

Include imports: pandas, numpy, scipy.stats, json
Format numbers for readability.
Add proper documentation.
```

---

## USAGE INSTRUCTIONS

### Order of Prompts:
1. Start with Prompts 11 (Utilities) - foundation functions
2. Then Prompts 1-2 (Data preprocessing and dataset)
3. Then Prompt 3 (Model)
4. Then Prompt 4 (Client)
5. Then Prompts 5-6 (Servers)
6. Then Prompts 7-8 (Training loops)
7. Then Prompt 9 (Data loaders)
8. Then Prompt 10 (Visualization)
9. Then Prompt 12 (Main script)
10. Finally Prompts 13-14 (Testing and analysis) as needed

### Tips for Using with Cursor:
1. Copy one prompt at a time
2. Review generated code before moving to next
3. Test each component individually
4. Integrate into notebook cell-by-cell
5. Debug any issues before proceeding
6. Save working versions frequently

### Customization:
- Adjust parameters in prompts as needed
- Add specific requirements for your setup
- Modify paths for your environment
- Extend functionality as required

---

**Remember**: These prompts are templates. Feel free to modify them based on your specific needs and Cursor's responses. Always review and test generated code before using in production!
