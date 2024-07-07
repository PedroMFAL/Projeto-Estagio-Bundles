import os

def get_all_folder_names(directory):
    try:
        # List all directories in the given directory
        folders = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
        return folders
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

# Example usage
directory_path = r"C:/Users/35192/Desktop/codigo/codigos.py/projeto-ia-bundle/MultiCBR-main/datasets"
folders = get_all_folder_names(directory_path)

for folder in folders:
    print(f"""
{folder}:
  data_path: './datasets'
  batch_size_train: 2048 # the batch size for training
  batch_size_test: 2048 # the batch size for testing
  topk: [10, 20, 40, 80] # the topks metrics for evaluation
  neg_num: 1 # number of negatives used for BPR loss. All the experiments use 1.

  # the following are the best settings
  aug_type: "Noise" # options: ED, MD, OP, Noise
  ed_interval: 1 # by how many epochs to dropout edge, default is 1
  embedding_sizes: [64] # the embedding size for user, bundle, and item
  num_layerss: [2] # number of layers for the information propagation over all graphs

  # the following dropout rates are with respect to the "aug_type", i.e., if aug_type is ED, the following dropout rates are for ED.
  UB_ratios: [0.0] # the dropout ratio for UB graph
  UI_ratios: [0.0] # the dropout ratio for UI graph
  BI_ratios: [0.2] # the dropout ratio for BI graph

  fusion_weights:
    # fusion weight for representations of different graphs
    # From the first element to the third element are weight for UB, UI, BI graphs correspondingly.
    modal_weight: [0.5, 0.2, 0.3]
    # layer aggregation weights, the i-th element in the list represents the weight of the (i-1)-th propagation layer.
    UB_layer: [0.35, 0.15, 0.5] # layer aggregation coefficients in UB graph
    UI_layer: [0.25, 0.65, 0.1] # layer aggregation coefficients in UI graph
    BI_layer: [0.4, 0.4, 0.2] # layer aggregation coefficients in BI graph

  lrs: [1.0e-3] # learning rate
  l2_regs: [1.0e-5] # the l2 regularization weight: lambda_2
  c_lambdas: [0.05] # the contrastive loss weight: lambda_1
  c_temps: [0.2] # the temperature in the contrastive loss: tau

  epochs: 50 # number of epochs to train
  test_interval: 5 # by how many epochs to run the validation and testing.
    """)