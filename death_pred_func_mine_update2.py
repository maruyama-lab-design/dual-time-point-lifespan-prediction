import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedGroupKFold
import copy
import pandas as pd
import numpy as np
from ipywidgets import IntProgress
from IPython.display import display
import random
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import seaborn as sns
import json
from itertools import product

'''
Functions utilized in the death prediction model

Note that this isn't optimized for CUDA
'''

#additional program with # after context
"""class Model(nn.Module):
  def __init__(self, n_inputs=None, h1=70, h2=70, out_features=1,dropout_rate=0.2): #add dropout_rate
    super().__init__()
    self.fc1 = nn.Linear(n_inputs, h1)
    self.fc2 = nn.Linear(h1, h2)
    self.out = nn.Linear(h2, out_features)

    self.dropout=nn.Dropout(p=dropout_rate) #add
    #self.activation = nn.SiLU()
    self.activation = nn.GELU()

  def forward(self, x):
    #x = F.relu(self.fc1(x))
    x = self.activation(self.fc1(x)) #add
    x = self.dropout(x) #add
    #x = F.relu(self.fc2(x))
    x = self.activation(self.fc2(x)) #add
    x = self.dropout(x) #add
    x = self.out(x)

    return x"""

class Model(nn.Module):
  def __init__(self, n_inputs=None, h1=70, h2=70, out_features=1):
    super().__init__()
    self.fc1 = nn.Linear(n_inputs, h1)
    self.fc2 = nn.Linear(h1, h2)
    self.out = nn.Linear(h2, out_features)


  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.out(x)

    return x


def set_seed(seed):
    '''
    Sets a specific random seed to make results consistent
    '''

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
        
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return


def time_to_death_grouped(data, category):
    '''
    Groups predicted time until death value by selected category
      param data: inputted dataframe, includes predicted time until death and category columns, df
      param category: the category, i.e. column name, to group dataframe by, str
      return: dataframe grouped by category, df
    '''
    
    print(f'Average time to death estimate by {category}:\n')

    # group by category and calculate the mean predicted time until death
    grouped_data = data.groupby(category, as_index=False)['Predicted time until death'].mean(numeric_only=True)

    # sort the grouped data in descending order
    grouped_data = grouped_data.sort_values(by='Predicted time until death', ascending=False)
    
    # print and return the result
    print(grouped_data)
    print('\n')
    
    return grouped_data

def plot_loss(train_losses, val_losses, fold_number=None):
    epochs = range(len(val_losses))
    
    plt.figure(figsize=(10, 6))
    if train_losses:
        plt.plot(epochs, train_losses, label='Training Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='red')
    
    """if train_losses and len(train_losses) != len(val_losses):
        early_stop_epoch = len(val_losses) - 1
        plt.axvline(x=early_stop_epoch, color='gray', linestyle='--', label=f'Early Stop ({early_stop_epoch} epochs)')"""

    plt.title(f'Loss Trajectory - Fold {fold_number}')
    plt.xlabel('Epochs')
    plt.ylabel('MAE') 
    plt.legend()
    plt.grid(True)
    plt.show()

def get_sample_weights(y, bins,normalize = False):
    counts, bin_edges = np.histogram(y, bins=bins)
    #total_count = len(y)

    bin_weights = np.zeros_like(counts, dtype=float)
    bin_weights[counts > 0] = 1.0/counts[counts > 0]

    if np.sum(bin_weights) > 0:
        bin_weights = bin_weights / np.min(bin_weights[bin_weights > 0])
        #bin_weights = bin_weights / np.sum(bin_weights)

    weights_array = np.zeros_like(y, dtype=float)
    for i in range(bins):
        bin_min = bin_edges[i]
        bin_max = bin_edges[i+1]

        if i == bins-1:
            indices  = np.where((y >= bin_min) & (y <= bin_max))[0]
        else:
            indices  = np.where((y >= bin_min) & (y < bin_max))[0]

        weights_array[indices] = bin_weights[i]
    #print(weights_array)
    if normalize and np.sum(weights_array) >0:
        weights_array = weights_array /np.sum(weights_array)
    return weights_array

def cross_validation(X, y, loss_bins, eva_bins, groups,batch_size, n_iterations, loss_mode,test_error_mode,patience, pred_mode, ablation_drop_col = None, scramble_trait=False, remove_trait=False):

  #debug
  #print(f"Initial X columns in cross_validation: {X.columns.tolist()}")
  #print(f"Initial X shape in cross_validation: {X.shape}")

  '''
    runs cross validation to determine loss of neural network model
    param X: all input values, df
    param y: all expected output values, df
    param batch_size: size of each batch to be run by each iteration of the NN, int
    param n_iterations: number of entries within cross validation, int
    param scramble_trait: whether to test the accuracy of the model with scrambled inputs by parameter during testing
    param remove_trait: whether to test the accuracy of the model with removed parameters during training
    return: 3 lists containing all of the models predictions, the actual values, and the loss values
  '''
  #all_approx = []
  #all_actual = []
  all_predictions_df_list = []
  all_losses = []
  all_fold_rmse = []
  all_fold_mae = []
  #all_fold_weighted_rmse = []
  #all_fold_weighted_mae = []
  #all_fold_SLMAE = []
  all_subject_SLMAE = []
  all_fold_SLMAE_std = []
  all_fold_SLRMSE = []
  all_strain_results = []

  strain_cols = ['CD1', 'C57BL6J', 'Sv129Ev']
  X['strain_label'] = X[strain_cols].idxmax(axis=1)

  y_anls=pd.DataFrame(y)
  y_anls.columns = ['death_clock_j']
  y_anls['strain_label'] = X[strain_cols].idxmax(axis=1)
  
  """plt.figure(figsize=(10,6))
  sns.kdeplot(data=y_anls, x= 'death_clock_j', hue = 'strain_label', fill= True, common_norm=False)
  plt.title('Remaining lifespan distribution by strain')
  plt.xlabel('Remaning lifespan')
  plt.ylabel('Density')
  plt.show()"""

  varience_anls=y_anls.groupby('strain_label')['death_clock_j'].std()
  #print('each strain remaining lifespan standard deviation', varience_anls)
  #print(y_anls.groupby('strain_label')['death_clock_j'].describe())
  unique_subjects = X[['ID', 'strain_label']].drop_duplicates()

  # get the data structures to return
  if (scramble_trait or remove_trait):
    trait_loss = {}

  #gkf = GroupKFold(n_splits=n_iterations)
  sgkf = StratifiedGroupKFold(n_splits=n_iterations, shuffle=True, random_state=42)
  groups = X['ID']

  id_to_fold = {}
  for fold_idx, (_, val_idx) in enumerate(sgkf.split(unique_subjects['ID'], unique_subjects['strain_label'],groups=unique_subjects['ID'])):
      for subject_id in unique_subjects.iloc[val_idx]['ID']:
          id_to_fold[subject_id]= fold_idx

  for fold in range(n_iterations):
      print(f"--- Fold {fold +1}/{n_iterations}---")

      test_ids = [sub_id for sub_id, f_idx in id_to_fold.items() if f_idx==fold]
      train_all_ids = [sub_id for sub_id, f_idx in id_to_fold.items() if f_idx!=fold]

      X_train_all = X[X['ID'].isin(train_all_ids)].copy()
      y_train_all = y[X['ID'].isin(train_all_ids)].copy()
      X_test = X[X['ID'].isin(test_ids)].copy()
      y_test = y[X['ID'].isin(test_ids)].copy()

      inner_subjects = X_train_all[['ID', 'strain_label']].drop_duplicates()
      inner_sgkf = StratifiedGroupKFold(n_splits=5,shuffle=True, random_state=42)

      train_sub_indices, val_sub_indices = next(inner_sgkf.split(inner_subjects['ID'], inner_subjects['strain_label'],groups=inner_subjects['ID']))
      val_ids = inner_subjects.iloc[val_sub_indices]['ID'].values
      train_ids = inner_subjects.iloc[train_sub_indices]['ID'].values

      X_train = X_train_all[X_train_all['ID'].isin(train_ids)].copy()
      X_val = X_train_all[X_train_all['ID'].isin(val_ids)].copy()
      y_train = y_train_all[X_train_all['ID'].isin(train_ids)].copy()
      y_val = y_train_all[X_train_all['ID'].isin(val_ids)].copy()

      is_two_timepoint = 'time_point_j' in X_train.columns

      if is_two_timepoint:
        train_weights = calculate_sample_weights(X_train, id_col='ID', tj_col = 'time_point_j')
        tp = 'time_point_j'
      else:
        train_weights = calculate_sample_weights(X_train, id_col='ID', tj_col = 'time_point_in_study_weeks')
        tp = 'time_point_in_study_weeks'

      if 'time_point_in_study_weeks' in X_test.columns:
        non_zero_train = X_train['time_point_in_study_weeks'] != 0
        non_zero_val = X_val['time_point_in_study_weeks'] != 0
        non_zero_test = X_test['time_point_in_study_weeks'] != 0
        X_train = X_train[non_zero_train].copy()
        y_train = y_train[non_zero_train].copy()
        X_val = X_val[non_zero_val].copy()
        y_val = y_val[non_zero_val].copy()
        X_test = X_test[non_zero_test].copy()
        y_test = y_test[non_zero_test].copy()

      #print(X_test['strain_label'].value_counts())
      #print(X_test[['ID', 'strain_label']].drop_duplicates()['strain_label'].value_counts())
      #print(X_test['ID'].nunique())

      if pred_mode == 'NN':
          model,train_losses,val_losses = train_nn(X_train,y_train,X_val,y_val, train_weights, loss_bins, batch_size,loss_mode, patience, ablation_drop_col)
          #plot_loss(train_losses,val_losses, fold_number=fold + 1)
          average_rmse,average_mae,subject_level_MAE,subject_level_MAE_std, subject_level_RMSE,predictions_df_fold, strain_mae_fold = test_nn_new(test_error_mode,model, X_test, y_test,eva_bins, ablation_drop_col)

          #plot_tj_error_analysis(predictions_df_fold)
      elif pred_mode == 'GB':
          #model = train_cat(X_train,y_train,X_val,y_val,loss_mode, patience)
          model,train_losses,val_losses = train_xgb(X_train,y_train,X_val,y_val,loss_bins,loss_mode, patience)
          plot_loss(train_losses,val_losses, loss_mode, fold_number=fold + 1)
          #average_rmse,average_mae,average_wrmse, average_wmae, approx, actual = test_gb(model,test_error_mode, X_test, y_test,loss_bins,eva_bins)
          average_rmse,average_mae,subject_level_MAE,subject_level_MAE_std, subject_level_RMSE,predictions_df_fold,strain_mae_fold = test_gb_linear(model, pred_mode,X_test, y_test,eva_bins,loss_bins, test_error_mode)
      elif pred_mode == 'Ridge' or pred_mode == 'Lasso':
          model, drop_col, val_losses = train_linear(X_train,y_train,X_val,y_val,pred_mode,alpha=1.0)
          average_rmse,average_mae,subject_level_MAE,subject_level_MAE_std, subject_level_RMSE,predictions_df_fold,strain_mae_fold = test_gb_linear(model, pred_mode, X_test, y_test, drop_col,eva_bins,loss_bins, test_error_mode)
      elif pred_mode == 'SVR':
          model, train_losses, test_losses = train_SVR(X_train, y_train, X_val,y_val, C=100.0, gamma='scale', kernel='rbf')
          average_rmse,average_mae,subject_level_MAE,subject_level_MAE_std, subject_level_RMSE,predictions_df_fold,strain_mae_fold = test_gb_linear(model, pred_mode,X_test, y_test,eva_bins,loss_bins, test_error_mode)


      #all_losses.extend(losses)
      all_fold_rmse.append(average_rmse)
      all_fold_mae.append(average_mae)
      #all_fold_SLMAE.append(subject_level_MAE)
      all_subject_SLMAE.extend(subject_level_MAE.tolist())
      all_fold_SLMAE_std.append(subject_level_MAE_std)
      all_fold_SLRMSE.append(subject_level_RMSE)
      all_strain_results.append(strain_mae_fold)
      #all_approx.extend(approx)
      #all_actual.extend(actual)
      all_predictions_df_list.append(predictions_df_fold)
  
  test_predictions_df = pd.concat(all_predictions_df_list,ignore_index=True)
  #plot_tj_error_analysis(test_predictions_df,tp)
  #plot_two_time_point_heatmap(test_predictions_df)
  """test_predictions_df = test_predictions_df.sort_values(
          by=['ID','time_point_j'],
          ascending=[True,True]).reset_index(drop=True)"""
  return test_predictions_df, all_fold_rmse, all_fold_mae, all_subject_SLMAE, all_fold_SLMAE_std, all_fold_SLRMSE, all_strain_results


def linear_drop(X_train,t):
    X_train=pd.DataFrame(X_train)
    corr_matrix = X_train.corr().abs()

    lower = corr_matrix.where(np.tril(np.ones(corr_matrix.shape), k=-1).astype(bool))
    to_drop = []
    for i in range(len(lower.columns)):
        if any(corr_matrix.iloc[i, i+1:] > t):
            to_drop.append(lower.columns[i])
    print(to_drop)
    return to_drop


def calculate_sample_weights(df, id_col='ID', tj_col='tj'):
    #counts = df.groupby([id_col, tj_col])[tj_col].transform('count') # Consider individuals (SLMAE)
    counts = df.groupby(tj_col)[tj_col].transform('count') # Doesn't consider individuals (MAE)
    sample_weights = 1.0/counts
    return sample_weights.values

def plot_tj_error_analysis(predictions_df, tp):
    predictions_df['abs_error'] = (predictions_df['actual'] - predictions_df['approximation']).abs()
    t_stats = predictions_df.groupby(tp)['abs_error'].agg(['mean', 'count']).reset_index()
    t_stats.columns = [tp, 'MAE', 'pair_count']

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_mae = 'tab:blue'
    ax1.set_xlabel(tp, fontsize=12)
    ax1.set_ylabel('MAE', color=color_mae, fontsize=12)
    ax1.plot(t_stats[tp], t_stats['MAE'], marker='o', color=color_mae, linewidth=2, label='MAE per t_j')
    ax1.tick_params(axis='y', labelcolor=color_mae)

    ax2 = ax1.twinx()
    color_count = 'tab:green'
    ax2.set_ylabel('pair count',color=color_count, fontsize=12)
    ax2.bar(t_stats[tp], t_stats['pair_count'], alpha=0.2, color=color_count, label='pair count')
    ax2.tick_params(axis='y', labelcolor=color_count)

    plt.title(f'prediction performance analysis by t_j or t', fontsize=14)
    ax1.grid(True, which ='both', linestyle='--', alpha=0.5)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()

    output_file = "table_analysis.xlsx"

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        t_stats.to_excel(writer, sheet_name='tj_stats', index=False)

    image_file = "plot_analysis.png"
    plt.savefig(image_file, dpi=300)
    print(f"Graph saved as : {image_file}")

    print(t_stats)

    plt.show()

def plot_two_time_point_heatmap(df):
    plot_df = df.copy()
    plot_df['Abs_Error'] = (plot_df['actual']-plot_df['approximation']).abs()

    all_tps = sorted(pd.unique(pd.concat([plot_df['time_point_i'], plot_df['time_point_j']])))

    heatmap_matrix = plot_df.pivot_table(
            index ='time_point_j',
            columns = 'time_point_i',
            values = 'Abs_Error',
            aggfunc='mean'
            )

    heatmap_matrix = heatmap_matrix.reindex(index=all_tps, columns=all_tps)

    mask = np.ones_like(heatmap_matrix, dtype=bool)
    for i, row_val in enumerate(all_tps):
        for j, col_val in enumerate(all_tps):
            if col_val < row_val and not np.isnan(heatmap_matrix.iloc[i,j]):
                mask[i,j] = False

    plt.figure(figsize=(12,9))

    ax= sns.heatmap(
            heatmap_matrix, 
            annot=True,
            fmt=".1f",
            cmap = 'YlOrRd',
            mask=mask,
            cbar_kws={'label': 'Mean Absolute Error(MAE)'},
            linewidths =.5
            )

    plt.title('Error Heatmap for Each Measurement Pair (ti, tj)', fontsize=14)
    plt.xlabel('time-point i',fontsize=12)
    plt.ylabel('time-point j',fontsize=12)

    ax.invert_yaxis()

    plt.tight_layout()
    plt.show()

def train_gb(X_train,y_train,X_val,y_val,loss_mode, patience,n_estimators=500):
    X_train = X_train.values
    y_train = y_train.values.ravel()
    X_val = X_val.values
    y_val = y_val.values.ravel()

    #model initialization
    if loss_mode == 'RMSE':
        loss_function = 'squared_error'
    elif loss_mode == 'MAE':
        loss_function = 'absolute_error'
    else:
        raise ValueError("Invalid loss_mode. Use 'RMSE' or 'MAE'.")

    model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=0.01,
            max_depth=3,
            subsample=0.8,
            loss=loss_function,
            n_iter_no_change=patience,
            validation_fraction=0.1,
            random_state=42
            )
    print("Training Gradient Boosting Regressor...")
    model.fit(X_train,y_train)
    print("Training finished.")

    val_preds = model.predict(X_val)
    if loss_mode == 'RMSE':
        val_loss = np.sqrt(mean_squared_error(y_val,val_preds))
        print(f"Validation RMSE:{val_loss:.4f}")
    elif loss_mode == 'MAE':
        val_loss = mean_absolute_error(y_val, val_preds)
        print(f"Validation MAE:{val_loss:.4f}")

    return model

def train_linear(X_train,y_train,X_val,y_val,model_type,alpha=1.0):
    X_train_fit = X_train.drop(columns=['ID', 'strain_label'])
    X_val_fit = X_val.drop(columns=['ID', 'strain_label'])

    scaler_X = StandardScaler()
    cols_scale = [c for c in X_train_fit.columns if not set(X_train_fit[c].unique()).issubset({0,1}) and 'rank' not in c]
    cols_not_scale = [c for c in X_train_fit.columns if c not in cols_scale]
    X_train_scaled = scaler_X.fit_transform(X_train_fit[cols_scale])
    X_val_scaled = scaler_X.transform(X_val_fit[cols_scale])

    X_train_fit = np.hstack([X_train_scaled, X_train_fit[cols_not_scale].values])
    X_val_fit = np.hstack([X_val_scaled, X_val_fit[cols_not_scale].values])

    #high_corr_remove
    drop_col = linear_drop(X_train_fit,t=0.85)

    X_train_fit = np.delete(X_train_fit, drop_col,axis=1)
    X_val_fit = np.delete(X_val_fit, drop_col,axis=1)


    scaler_y = StandardScaler()
    y_train_fit = scaler_y.fit_transform(y_train.values.reshape(-1,1)).ravel()
    #y_train_fit = y_train.values.ravel()

    is_two_point = 'time_point_j' in X_val.columns

    alpha_list = [0.01,0.1,1,10,100]

    best_model = None
    best_mae = float('inf')

    for alpha in alpha_list:
        if model_type == 'Ridge':
            #print("Training Ridge Regressor")
            model= Ridge(alpha=alpha,random_state=0)
        elif model_type == 'Lasso':
            #print("Traininig Lasso Regressor")
            model= Lasso(alpha=alpha,random_state=0)
        else:
            raise ValueError("Invalid model_type")
        
        model.fit(X_train_fit, y_train_fit)

        #tj-wise MAE
        val_preds = model.predict(X_val_fit)
        val_preds = scaler_y.inverse_transform(val_preds.reshape(-1,1)).ravel()

        val_eval_df = pd.DataFrame({
            'ID':X_val['ID'].values,
            'actual':y_val.values.ravel(),
            'approximation': val_preds
        })

        if is_two_point:
            val_eval_df['time_point_j'] = X_val['time_point_j'].values
            agg_df = val_eval_df.groupby(['ID', 'time_point_j']).agg({
                'actual':'first',
                'approximation': 'mean'
            }).reset_index()

            current_mae = np.mean(np.abs(agg_df['approximation'] - agg_df['actual']))
        else:
            current_mae = np.mean(np.abs(val_eval_df['approximation'] - val_eval_df['actual']))

        #model best
        if current_mae < best_mae:
            best_mae=current_mae
            best_model = model

            best_model.scaler_X = scaler_X
            best_model.scaler_y = scaler_y
            best_model.cols_scale = cols_scale
            best_model.cols_not_scale = cols_not_scale
            print('best alpha update',alpha,best_mae)
    return best_model,drop_col,[]

def train_xgb(X_train,y_train,X_val,y_val,loss_bins,loss_mode, patience,n_estimators=2000):
    X_train_fit = X_train.drop(columns=['ID', 'strain_label']).values
    X_val_fit = X_val.drop(columns=['ID', 'strain_label']).values
    y_train_fit = y_train.values.ravel()
    y_val_fit = y_val.values.ravel()

    if loss_bins !=None:
        sample_weights = get_sample_weights(y_train,loss_bins,normalize=False)

    is_two_point = 'time_point_j' in X_val.columns
    
    #model initialization
    if loss_mode == 'RMSE':
        objective = 'reg:squarederror'
        eval_metric = 'rmse'
    elif loss_mode == 'MAE':
        objective = 'reg:absoluteerror'
        eval_metric = 'mae'
    else:
        raise ValueError("Invalid loss_mode. Use 'RMSE' or 'MAE'.")

    param_grid = {
            'max_depth':[3,4,5,6],
            'learning_rate':[0.03, 0.01]
    }

    best_model = None
    best_mae = float('inf')
    best_val_losses = []

    keys, values = zip(*param_grid.items())

    for v in product(*values):
        params = dict(zip(keys, v))
        print('now params:',params)

        model = XGBRegressor(
            n_estimators=n_estimators,
            #learning_rate=0.01,#0.01
            #max_depth=4, #3
            #subsample=0.8,#0.8
            colsample_bytree=0.8,#0.8
            objective = objective,
            eval_metric=eval_metric,
            early_stopping_rounds = 30,
            random_state=0,
            n_jobs=-1,
            **params
        )
    
        #print("Training Gradient Boosting Regressor...")

        if loss_bins ==None:
            model.fit(X_train_fit,y_train_fit,eval_set=[(X_val_fit,y_val_fit)],verbose=False)
        else:
            model.fit(X_train_fit,y_train_fit,sample_weight=sample_weights, eval_set=[(X_val_fit,y_val_fit)],verbose=False)

        val_preds = model.predict(X_val_fit)
        val_eval_df = pd.DataFrame({
            'ID':X_val['ID'].values,
            'actual': y_val_fit,
            'approximation': val_preds
        })

        if is_two_point:
            val_eval_df['time_point_j'] = X_val['time_point_j'].values
            agg_df = val_eval_df.groupby(['ID', 'time_point_j']).agg({
                'actual':'first',
                'approximation': 'mean'
            }).reset_index()

            current_mae = np.mean(np.abs(agg_df['approximation'] - agg_df['actual']))
        else:
            current_mae = np.mean(np.abs(val_eval_df['approximation'] - val_eval_df['actual']))

        #model best
        if current_mae < best_mae:
            best_mae=current_mae
            best_model = model
            evals_result = model.evals_result()
            best_val_losses = evals_result['validation_0'][eval_metric]
            print('best mae:',best_mae)
    return best_model,[],best_val_losses

def train_SVR(X_train, y_train, X_val, y_val, C=1, gamma='scale', kernel='rbf'):
    X_train_fit = X_train.drop(columns=['ID', 'strain_label'])
    X_val_fit = X_val.drop(columns=['ID', 'strain_label'])

    scaler_X = StandardScaler()
    cols_scale = [c for c in X_train_fit.columns if not set(X_train_fit[c].unique()).issubset({0,1}) and 'rank' not in c]
    cols_not_scale = [c for c in X_train_fit.columns if c not in cols_scale]
    X_train_scaled = scaler_X.fit_transform(X_train_fit[cols_scale])
    X_val_scaled = scaler_X.transform(X_val_fit[cols_scale])

    X_train_fit = np.hstack([X_train_scaled, X_train_fit[cols_not_scale].values])
    X_val_fit = np.hstack([X_val_scaled, X_val_fit[cols_not_scale].values])

    scaler_y = StandardScaler()
    y_train_fit = scaler_y.fit_transform(y_train.values.reshape(-1,1)).ravel()

    is_two_point = 'time_point_j' in X_val.columns

    C_list = [0.01,0.1,1,10]
    #gamma_list = ['scale',0.01,0.001]
    #epsilon_list = [0.1,0.2,0.5]

    best_model = None
    best_mae = float('inf')
    print('Training SVR')

    for C in C_list:
        model = SVR(kernel=kernel, C=C, gamma=gamma)
        model.fit(X_train_fit, y_train_fit)

        #tj-wise MAE
        val_preds_scaled = model.predict(X_val_fit)
        val_preds = scaler_y.inverse_transform(val_preds_scaled.reshape(-1,1)).ravel()

        val_eval_df = pd.DataFrame({
            'ID':X_val['ID'].values,
            'actual':y_val.values.ravel(),
            'approximation': val_preds
        })

        if is_two_point:
            val_eval_df['time_point_j'] = X_val['time_point_j'].values
            agg_df = val_eval_df.groupby(['ID', 'time_point_j']).agg({
                'actual':'first',
                'approximation': 'mean'
            }).reset_index()

            current_mae = np.mean(np.abs(agg_df['approximation'] - agg_df['actual']))
        else:
            current_mae = np.mean(np.abs(val_eval_df['approximation'] - val_eval_df['actual']))

        #model best
        if current_mae < best_mae:
            best_mae=current_mae
            best_model = model

            best_model.scaler_X = scaler_X
            best_model.scaler_y = scaler_y
            best_model.cols_scale = cols_scale
            best_model.cols_not_scale = cols_not_scale
            print('best C update',C,best_mae)
    return best_model,[],[]

def train_cat(X_train,y_train,X_val,y_val,loss_mode, patience, n_estimators=1000):
    X_train = X_train.values
    y_train = y_train.values.ravel()
    X_val = X_val.values
    y_val = y_val.values.ravel()

    #model initialization
    if loss_mode == 'RMSE':
        loss_function = 'RMSE'
    elif loss_mode == 'MAE':
        loss_function = 'MAE'
    else:
        raise ValueError("Invalid loss_mode. Use 'RMSE' or 'MAE'.")

    model = CatBoostRegressor(
            iterations=n_estimators,
            learning_rate=0.03,
            depth=6,
            subsample=1,
            colsample_bylevel=1,
            loss_function=loss_function,
            early_stopping_rounds=patience,
            random_state=0,
            verbose=0,
            eval_metric=loss_function)

    print("Training CatBoost Regressor...")
    model.fit(X_train,y_train,eval_set=(X_val,y_val))
    print("Training finished.")

    val_preds = model.predict(X_val)
    if loss_mode == 'RMSE':
        val_loss = np.sqrt(mean_squared_error(y_val,val_preds))
        print(f"Validation RMSE:{val_loss:.4f}")
    elif loss_mode == 'MAE':
        val_loss = mean_absolute_error(y_val, val_preds)
        print(f"Validation MAE:{val_loss:.4f}")

    return model

def train_nn(X_train, y_train, X_val, y_val, sample_weights, loss_bins, batch_size, loss_mode, patience, ablation_drop_col=None, epochs=500):
  X_train_fit = X_train.drop(columns=['ID', 'strain_label'])
  X_val_fit = X_val.drop(columns=['ID', 'strain_label'])

  if ablation_drop_col!=None:
        X_train_fit = X_train_fit.drop(columns=ablation_drop_col)
        X_val_fit = X_val_fit.drop(columns=ablation_drop_col)

  print(f"Input X columns : {X_train_fit.columns.tolist()}")
  print(f"Input X shape : {X_train_fit.shape[1]}")

  n_inputs = X_train_fit.shape[1]

  scaler_X = StandardScaler()
  cols_scale = [c for c in X_train_fit.columns if not set(X_train_fit[c].unique()).issubset({0,1}) and 'rank' not in c]
  cols_not_scale = [c for c in X_train_fit.columns if c not in cols_scale]
  if len(cols_scale) > 0:
    X_train_scaled = scaler_X.fit_transform(X_train_fit[cols_scale])
    X_val_scaled = scaler_X.transform(X_val_fit[cols_scale])

    if len(cols_not_scale) > 0:
        X_train_fit = np.hstack([X_train_scaled, X_train_fit[cols_not_scale].values])
        X_val_fit = np.hstack([X_val_scaled, X_val_fit[cols_not_scale].values])
    else:
        X_train_fit = X_train_scaled
        X_val_fit = X_val_scaled

  else:
    X_train_fit = X_train_fit.values
    X_val_fit = X_val_fit.values

  scaler_y = StandardScaler()
  y_train_fit = scaler_y.fit_transform(y_train.values.reshape(-1,1)).ravel()

  #cols_drop_time = ['time_point_i', 'time_point_j']
  #X_train_no_time = X_train.drop(columns=[c for c in cols_drop_time if c in X_train.columns])
  #X_val_no_time = X_val.drop(columns=[c for c in cols_drop_time if c in X_val.columns])
  #X_train = X_train_no_time ##
  #X_val= X_val_no_time  ##

  #weights_train = torch.tensor(sample_weights, dtype=torch.float32).view(-1,1)

  X_train_tensor = torch.tensor(X_train_fit, dtype=torch.float32)
  y_train_tensor = torch.tensor(y_train_fit, dtype=torch.float32).view(-1, 1)

  X_val_tensor = torch.tensor(X_val_fit, dtype=torch.float32)
  y_val_scaled = scaler_y.transform(y_val.values.reshape(-1,1)).ravel()
  y_val_scaled_tensor = torch.tensor(y_val_scaled, dtype= torch.float32).view(-1,1)
  y_val = y_val.values.ravel()

  model = Model(n_inputs)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

  if loss_mode == 'RMSE':
      criterion = nn.MSELoss(reduction='none')
  elif loss_mode == 'MAE':
      criterion = nn.L1Loss(reduction='none')

  g = torch.Generator()
  g.manual_seed(42)
  dataset = TensorDataset(X_train_tensor, y_train_tensor)
  dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=True, generator=g)
  #model.train()

  best_mae = float('inf')
  best_model=None

  #training loss memorize
  train_losses = []
  val_losses = []
  epoch_no_improve = 0
  
  is_two_point = 'time_point_j' in X_val.columns

  for epoch in range(epochs):
      model.train()
      current_train_loss=0
      for inputs, targets in dataloader:
          optimizer.zero_grad()

          outputs = model(inputs)
          loss = criterion(outputs, targets).mean()
          #loss = (loss * w).mean()
          loss.backward()
          optimizer.step()
          current_train_loss += loss.item() * len(inputs)
      
      average_train_loss = current_train_loss / len(X_train_tensor)
      train_losses.append(average_train_loss)
          
      # evaluation with validation data
      model.eval()
      with torch.no_grad():
          val_outputs_tensor=model(X_val_tensor)
          val_loss_mem = criterion(val_outputs_tensor, y_val_scaled_tensor).mean().item()
          val_losses.append(val_loss_mem)

          val_outputs = val_outputs_tensor.cpu().numpy()
          val_preds = scaler_y.inverse_transform(val_outputs).ravel()

          val_eval_df = pd.DataFrame({
              'ID':X_val['ID'].values,
              'actual':y_val,
              'approximation': val_preds
          })

          if is_two_point:
              val_eval_df['time_point_j'] = X_val['time_point_j'].values
              agg_df = val_eval_df.groupby(['ID', 'time_point_j']).agg({
                  'actual':'first',
                  'approximation': 'mean'
              }).reset_index()

              current_mae = np.mean(np.abs(agg_df['approximation'] - agg_df['actual']))
          else:
              current_mae = np.mean(np.abs(val_eval_df['approximation'] - val_eval_df['actual']))

          #model best
          if current_mae < best_mae:
              best_mae=current_mae
              best_model = copy.deepcopy(model)
              best_model.scaler_X = scaler_X
              best_model.scaler_y = scaler_y
              best_model.cols_scale = cols_scale
              best_model.cols_not_scale = cols_not_scale
              epoch_no_improve=0
          else:
              epoch_no_improve+=1
              if epoch_no_improve >= patience:
                  print(f"Early stopping at epoch {epoch}")
                  break
  return best_model if best_model else model,train_losses,val_losses

def test_nn_new(loss_mode, model, X_test, y_test, eva_bins, ablation_drop_col=None):
    model.eval()

    IDs = X_test['ID'].values
    strains = X_test['strain_label'].values
    is_two_point = 'time_point_j' in X_test.columns
    if is_two_point:
        tp_i = X_test['time_point_i'].values
        tp_j = X_test['time_point_j'].values 

    X_test_raw = X_test.drop(columns=['ID', 'strain_label'])
    if ablation_drop_col!=None:
        X_test_raw = X_test_raw.drop(columns=ablation_drop_col)

    if len(model.cols_scale) > 0:
        X_test_scaled = model.scaler_X.transform(X_test_raw[model.cols_scale])

        if len(model.cols_not_scale) > 0:
            X_test_np = np.hstack([X_test_scaled, X_test_raw[model.cols_not_scale].values])
        else:
            X_test_np = X_test_scaled

    else:
        X_test_np = X_test_raw.values

    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(X_test_tensor)
        outputs_np = outputs.cpu().flatten().numpy()

        approx_np = model.scaler_y.inverse_transform(outputs_np.reshape(-1,1)).flatten()
        actual_np = y_test.values.ravel()

        absolute_errors = np.abs(approx_np-actual_np)
        squared_errors = (approx_np-actual_np)**2

        evaluation_df = pd.DataFrame({
            'ID':IDs,
            'strain': strains,
            'Absolute_Error': absolute_errors,
            'Squared_Error': squared_errors,
            'actual': actual_np,
            'approximation': approx_np
        })

        if is_two_point:
            evaluation_df['time_point_j'] = tp_j

        #time-point-wise evaluation
        if is_two_point:
            #two time-points
            tp_grouped = evaluation_df.groupby(['ID', 'time_point_j','strain']).agg({
                'actual': 'first',
                'approximation': 'mean'
            }).reset_index()

            tp_grouped['Abs_Err_tp'] = np.abs(tp_grouped['approximation'] - tp_grouped['actual'])
            tp_grouped['Sq_Err_tp'] = (tp_grouped['approximation'] - tp_grouped['actual'])**2

            #time-point-wise MAE/RMSE
            average_mae = tp_grouped['Abs_Err_tp'].mean()
            average_rmse = np.sqrt(tp_grouped['Sq_Err_tp'].mean())

            #each strain MAE
            strain_maes = tp_grouped.groupby('strain')['Abs_Err_tp'].mean()
        else:
            #single time-point
            average_mae = evaluation_df['Absolute_Error'].mean()
            average_rmse = np.sqrt(evaluation_df['Squared_Error'].mean())

            strain_maes = evaluation_df.groupby('strain')['Absolute_Error'].mean()

        #subject_level evaluation
        if is_two_point:
            subject_level_MAEs = tp_grouped.groupby('ID')['Abs_Err_tp'].mean()
            subject_level_MSEs = tp_grouped.groupby('ID')['Sq_Err_tp'].mean()
        else:
            subject_level_MAEs = evaluation_df.groupby('ID')['Absolute_Error'].mean()
            subject_level_MSEs = evaluation_df.groupby('ID')['Squared_Error'].mean()
        
        subject_level_RMSE = np.sqrt(subject_level_MSEs.mean())
        subject_level_MAE_std = subject_level_MAEs.std()
    
        prediction_data = {
            'ID': IDs,
            'strain': strains,
            'actual': actual_np,
            'approximation' : approx_np
        }
        column_order = ['ID', 'strain']

        if is_two_point:
            prediction_data['time_point_j'] = tp_j
            prediction_data['time_point_i'] = tp_i
            column_order.extend(['time_point_j', 'time_point_i'])
        else:
            prediction_data['time_point_in_study_weeks'] = X_test['time_point_in_study_weeks'].values
            column_order.append('time_point_in_study_weeks')

        column_order.extend(['actual','approximation'])
        val_predictions_df = pd.DataFrame(prediction_data, columns=column_order)
        #print(val_predictions_df)
    return average_rmse,average_mae, subject_level_MAEs, subject_level_MAE_std, subject_level_RMSE,val_predictions_df,strain_maes

#def test_gb(model,test_error_mode,X_test,y_test,loss_bins,eva_bins):
def test_gb_linear(model, reg_arch, X_test, y_test, drop_col, eva_bins, loss_mode=None,test_error_mode=None):
    is_two_timepoint =  'time_point_j' in X_test.columns

    IDs = X_test['ID'].values
    strains = X_test['strain_label'].values

    if is_two_timepoint:
        tp_i = X_test['time_point_i'].values
        tp_j = X_test['time_point_j'].values

    X_test = X_test.drop(columns=['ID', 'strain_label'])

    if reg_arch != 'GB':
        X_test_scaled = model.scaler_X.transform(X_test[model.cols_scale])
        X_test_not_scaled = X_test[model.cols_not_scale].values
        X_test_np = np.hstack([X_test_scaled, X_test_not_scaled])
    else:
        X_test_np=X_test

    X_test_np = np.delete(X_test_np, drop_col,axis=1)##

    y_test_np = y_test.values.ravel()

    outputs = model.predict(X_test_np)
    if reg_arch != 'GB':
        approx_np = model.scaler_y.inverse_transform(outputs.reshape(-1,1)).flatten()
    else:
        approx_np = outputs.flatten()
    actual_np = y_test_np.flatten()

    evaluation_df = pd.DataFrame({
        'ID': IDs,
        'strain': strains,
        'actual': actual_np,
        'approximation': approx_np
    })

    if is_two_timepoint:
        evaluation_df['time_point_j'] = tp_j
        tp_grouped = evaluation_df.groupby(['ID', 'time_point_j', 'strain']).agg({
            'actual': 'first',
            'approximation': 'mean'
        }).reset_index()

        tp_grouped['Abs_Err_tp'] = np.abs(tp_grouped['approximation']- tp_grouped['actual'])
        tp_grouped['Sq_Err_tp'] = (tp_grouped['approximation']- tp_grouped['actual'])**2

        #tj-wise MAE/RMSE
        average_mae =  tp_grouped['Abs_Err_tp'].mean()
        average_rmse = np.sqrt(tp_grouped['Sq_Err_tp'].mean())

        #tj-wise each strain MAE
        strain_maes = tp_grouped.groupby('strain')['Abs_Err_tp'].mean()

    else:
        evaluation_df['Abs_Err'] = np.abs(evaluation_df['approximation'] - evaluation_df['actual'])
        evaluation_df['Sq_Err'] = (evaluation_df['approximation'] - evaluation_df['actual'])**2

        average_mae = evaluation_df['Abs_Err'].mean()
        average_rmse = np.sqrt(evaluation_df['Sq_Err'].mean())

        strain_maes = evaluation_df.groupby('strain')['Abs_Err'].mean()

    #Subject Level Error
    if is_two_timepoint:
        subject_level_MAEs = tp_grouped.groupby('ID')['Abs_Err_tp'].mean()
        subject_level_MSEs = tp_grouped.groupby('ID')['Sq_Err_tp'].mean()
    else:
        #evaluation_df['Abs_Err'] = np.abs(evaluation_df['approximation'] - evaluation_df['actual'])
        #evaluation_df['Sq_Err'] = (evaluation_df['approximation'] - evaluation_df['actual'])**2
        subject_level_MAEs = evaluation_df.groupby('ID')['Abs_Err'].mean()
        subject_level_MSEs = evaluation_df.groupby('ID')['Sq_Err'].mean()

    subject_level_RMSE = np.sqrt(subject_level_MSEs.mean())
    subject_level_MAE_std = subject_level_MAEs.std()

    prediction_data ={
            'ID':IDs,
            'strain': strains,
            'actual':actual_np,
            'approximation': approx_np
    }
    column_order = ['ID']

    if is_two_timepoint:
        prediction_data['time_point_i']=tp_i
        prediction_data['time_point_j']=tp_j
        column_order.append('time_point_i')
        column_order.append('time_point_j')
    else:
        tp_col = 'time_point_in_study_weeks'
        prediction_data[tp_col] = X_test[tp_col].values
        column_order.append(tp_col)

    column_order.extend(['actual','approximation'])
    val_predictions_df = pd.DataFrame(prediction_data,columns=column_order)

    #return rmse_average_loss,mae_average_loss, weighted_rmse_loss,  weighted_mae_loss,approx,actual
    return average_rmse, average_mae, subject_level_MAEs, subject_level_MAE_std, subject_level_RMSE, val_predictions_df, strain_maes

def get_loso_data(X, y, test_strain, min_id_count):
    test_mask = (X['strain_label'] == test_strain)
    test_X = X[test_mask].copy()
    test_y = y[test_mask].copy()

    train_full_X = X[~test_mask].copy()
    train_strains = train_full_X['strain_label'].unique()

    id_counts= train_full_X.groupby('strain_label')['ID'].nunique()
    total_ids = id_counts.sum()

    selected_indices = []
    for s in train_strains:
        s_id_size = int(np.floor((id_counts[s] / total_ids) * min_id_count))
        all_ids_in_s = train_full_X[train_full_X['strain_label'] == s]['ID'].unique()
        np.random.seed(42)
        selected_ids = np.random.choice(all_ids_in_s, size = s_id_size, replace=False)

        idx = train_full_X[train_full_X['ID'].isin(selected_ids)].index
        selected_indices.extend(idx.tolist())
    train_X = X.loc[selected_indices].copy()
    train_y = y.loc[selected_indices].copy()
    return train_X, train_y, test_X, test_y

def loso(X, y, batch_size, epochs=1000):
    all_strain_rmse = []
    all_strain_mae = []
    all_subject_SLMAE = []
    all_subject_SLRMSE = []
    all_subject_SLMAE_std = []

    strain_cols = ['CD1', 'C57BL6J', 'Sv129Ev']
    X['strain_label'] = X[strain_cols].idxmax(axis=1)

    strains = X['strain_label'].unique()
    id_counts_per_strain = X.groupby('strain_label')['ID'].nunique()
    id_combinations_size = []
    for ts in strains:
        id_combinations_size.append(id_counts_per_strain[strains != ts].sum())
    min_id_count = min(id_combinations_size)
    print(f"minimum trainning ID counts: {min_id_count}")

    for test_strain in strains:
        print(f"\n Testing on strain: {test_strain}")

        X_train_raw, y_train_raw, X_test, y_test = get_loso_data(X, y, test_strain, min_id_count)
        #学習データの個体数確認
        print(X_train_raw['strain_label'].value_counts())
        print("The number of traning subjects each strain", X_train_raw[['ID', 'strain_label']].drop_duplicates()['strain_label'].value_counts())

        train_idx_list = []
        val_idx_list = []
        inner_strains = X_train_raw['strain_label'].unique()

        for s in inner_strains:
            s_ids = X_train_raw[X_train_raw['strain_label'] == s]['ID'].unique()
            np.random.seed(42)
            np.random.shuffle(s_ids)

            split_point = int(len(s_ids)*0.75)
            s_train_ids = s_ids[:split_point]
            s_val_ids = s_ids[split_point :]

            train_idx_list.extend(X_train_raw[X_train_raw['ID'].isin(s_train_ids)].index.tolist())
            val_idx_list.extend(X_train_raw[X_train_raw['ID'].isin(s_val_ids)].index.tolist())

        X_train = X_train_raw.loc[train_idx_list]
        X_val = X_train_raw.loc[val_idx_list]
        y_train = y_train_raw.loc[train_idx_list]
        y_val = y_train_raw.loc[val_idx_list]

        #single time-point tp=0 remove
        is_two_point = 'time_point_j' in X_train.columns
        is_single_point = 'time_point_in_study_weeks' in X_train.columns
        if is_single_point:
            non_zero_train = X_train['time_point_in_study_weeks'] != 0
            non_zero_val = X_val['time_point_in_study_weeks'] != 0
            non_zero_test = X_test['time_point_in_study_weeks'] != 0
            X_train = X_train[non_zero_train].copy()
            X_val = X_val[non_zero_val].copy()
            X_test = X_test[non_zero_test].copy()
            y_train = y_train[non_zero_train].copy()
            y_val = y_val[non_zero_val].copy()
            y_test = y_test[non_zero_test].copy()
        elif is_two_point:
            test_tp_i = X_test['time_point_i'].values
            test_tp_j = X_test['time_point_j'].values

        test_IDs = X_test['ID'].values

        #ID, strain_label remove
        X_train_fit = X_train.drop(columns = ['ID', 'strain_label', 'CD1', 'Sv129Ev', 'C57BL6J'])
        X_val_fit = X_val.drop(columns = ['ID', 'strain_label', 'CD1', 'Sv129Ev', 'C57BL6J'])
        X_test_raw = X_test.drop(columns = ['ID', 'strain_label', 'CD1', 'Sv129Ev', 'C57BL6J'])

        #scaling feature
        cols_scale = [c for c in X_train_fit.columns if not set(X_train_fit[c].unique()).issubset({0,1}) and 'rank' not in c]
        cols_not_scale = [c for c in X_train_fit.columns if c not in cols_scale]

        scaler_X = StandardScaler()
        X_train_fit_scaled = scaler_X.fit_transform(X_train_fit[cols_scale])
        X_val_fit_scaled = scaler_X.transform(X_val_fit[cols_scale])
        X_test_raw_scaled = scaler_X.transform(X_test_raw[cols_scale])

        #inputs feature
        X_train_fit = np.hstack([X_train_fit_scaled, X_train_fit[cols_not_scale].values])
        X_val_fit = np.hstack([X_val_fit_scaled, X_val_fit[cols_not_scale].values])
        X_test_raw = np.hstack([X_test_raw_scaled, X_test_raw[cols_not_scale].values])

        #scaling remaining lifespan
        scaler_y = StandardScaler()
        y_train_fit = scaler_y.fit_transform(y_train.values.reshape(-1,1)).ravel()
        y_val_fit = scaler_y.transform(y_val.values.reshape(-1,1)).ravel()
        y_test_raw = y_test.values.ravel()

        #Tensor化
        X_train_tensor = torch.FloatTensor(X_train_fit)
        X_val_tensor = torch.FloatTensor(X_val_fit)
        X_test_tensor = torch.FloatTensor(X_test_raw)
        y_train_tensor = torch.FloatTensor(y_train_fit).view(-1,1)
        y_val_tensor = torch.FloatTensor(y_val_fit).view(-1,1)
        #y_test_tensor = torch.FloatTensor(y_test_fit).view(-1,1)

        #NN training
        model = Model(n_inputs = X_train_fit.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
        criterion = nn.L1Loss(reduction='none')

        g = torch.Generator()
        g.manual_seed(42)

        dataset =TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)

        best_val_loss = float('inf')
        best_model = None
        patience = 100
        stop_counter = 0

        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            model.train()
            current_train_loss = 0
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets).mean()
                loss.backward()
                optimizer.step()
                current_train_loss += loss.item()*len(inputs)
            average_train_loss = current_train_loss / len(X_train_tensor)
            train_losses.append(average_train_loss)

            #validation
            model.eval()
            with torch.no_grad():
                val_loss_mem = criterion(model(X_val_tensor), y_val_tensor).mean().item()
                val_losses.append(val_loss_mem)

                val_preds = scaler_y.inverse_transform(model(X_val_tensor).cpu().numpy()).ravel()
                val_eval_df = pd.DataFrame({
                    'ID': X_val['ID'].values,
                    'actual': y_val,
                    'approximation':val_preds
                })

                if is_two_point:
                    val_eval_df['time_point_j'] = X_val['time_point_j'].values
                    agg_df = val_eval_df.groupby(['ID', 'time_point_j']).agg({
                        'actual': 'first',
                        'approximation': 'mean'
                    }).reset_index()

                    current_val_mae = np.mean(np.abs(agg_df['approximation'] - agg_df['actual']))
                else:
                    current_val_mae = np.mean(np.abs(val_eval_df['approximation'] - val_eval_df['actual']))

                # best model
                if current_val_mae < best_val_loss:
                    best_val_loss = current_val_mae
                    torch.save(model.state_dict(), f'model_{test_strain}.pth')
                    stop_counter = 0
                else:
                    stop_counter += 1
                    if stop_counter >= patience: break
        plot_loss(train_losses,val_losses, fold_number=test_strain)
        #Testing
        model.load_state_dict(torch.load(f'model_{test_strain}.pth'))
        model.eval()
        with torch.no_grad():
            outputs_np = model(X_test_tensor).cpu().flatten().numpy()
            approx_np = scaler_y.inverse_transform(outputs_np.reshape(-1,1)).flatten()
            actual_np = y_test_raw.flatten()

            evaluation_df = pd.DataFrame({
                'ID': test_IDs,
                'actual': actual_np,
                'approximation': approx_np
            })

            if is_two_point:
                evaluation_df['time_point_j'] = test_tp_j

                tp_grouped = evaluation_df.groupby(['ID', 'time_point_j']).agg({
                    'actual':'first',
                    'approximation': 'mean'
                }).reset_index()

                tp_grouped['Abs_Err_tp'] = np.abs(tp_grouped['approximation'] - tp_grouped['actual'])
                tp_grouped['Sq_Err_tp'] = (tp_grouped['approximation'] - tp_grouped['actual'])**2

                average_mae = tp_grouped['Abs_Err_tp'].mean()
                average_rmse = np.sqrt(tp_grouped['Sq_Err_tp'].mean())

                subject_level_MAEs = tp_grouped.groupby('ID')['Abs_Err_tp'].mean()
                subject_level_MSEs = tp_grouped.groupby('ID')['Sq_Err_tp'].mean()
            else:
                evaluation_df['Abs_Err'] = np.abs(evaluation_df['approximation'] - evaluation_df['actual'])
                evaluation_df['Sq_Err'] = (evaluation_df['approximation'] - evaluation_df['actual'])**2

                average_mae = evaluation_df['Abs_Err'].mean()
                average_rmse = np.sqrt(evaluation_df['Sq_Err'].mean())

                subject_level_MAEs = evaluation_df.groupby('ID')['Abs_Err'].mean()
                subject_level_MSEs = evaluation_df.groupby('ID')['Sq_Err'].mean()

            subject_level_RMSE = np.sqrt(subject_level_MSEs.mean())
            subject_level_MAE_std = subject_level_MAEs.std()

            prediction_data = {
                'ID': test_IDs,
                'actual': actual_np,
                'approximation': approx_np
            }
            column_order = ['ID']

            if is_two_point:
                prediction_data['time_point_i'] = test_tp_i
                prediction_data['time_point_j'] = test_tp_j
                column_order.extend(['time_point_i', 'time_point_j'])
            else:
                tp_col = 'time_point_in_study_weeks'
                prediction_data[tp_col] = X_test[tp_col].values
                column_order.append(tp_col)

            column_order.extend(['actual', 'approximation'])

            all_strain_mae.append(average_mae)
            all_strain_rmse.append(average_rmse)
            all_subject_SLMAE.append(subject_level_MAEs)
            all_subject_SLRMSE.append(subject_level_RMSE)
            all_subject_SLMAE_std.append(subject_level_MAE_std)

            print(f"Result for {test_strain}: MAE = {average_mae:.4f}")
    return all_strain_mae, all_strain_rmse, all_subject_SLMAE, all_subject_SLRMSE, all_subject_SLMAE_std

def generate_nn_pred(model, X):
  '''
  gets predictions for input values using trained NN

    param model: trained NN, torch object
    param X: input values, df

    returns: n predictions, numpy array
  '''
  model.eval()

  X = X.values
  X = torch.tensor(X, dtype=torch.float32)

  with torch.no_grad():
     outputs = model(X)
  outputs = outputs.detach().numpy() 

  return outputs   
