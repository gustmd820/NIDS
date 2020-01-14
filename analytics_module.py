import pandas as pd
import numpy as np

def next_batch(num, data):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]

    return np.asarray(data_shuffle)

def error_statistics_check(error):
    mu = np.mean(np.matrix(error))
    sigma = np.std(np.matrix(error))
    
    print("mu=%f, sigma=%f"%(mu, sigma))
    return mu, sigma

def get_df_performance(input_data, output_data, total_data):
    error = np.sqrt(np.sum(np.square(output_data - input_data), axis=1))
    total_data = np.append(total_data, error.reshape(-1,1), axis=1)
    col_nums = len(total_data[0])

    df = pd.DataFrame(total_data).iloc[:,[-2, -1]].rename(columns={col_nums-2:'attack_types', col_nums-1:'error'})
    
    return df


# performance check 함수: threshold와 데이터 프레임을 입력하여, 성능을 출력 
def performance_check(threshold, df):
    tp = len(df[(df['error'] >= threshold) & (df['attack_types'] == 'abnormal')])
    fp = len(df[(df['error'] >= threshold) & (df['attack_types'] == 'normal')])
    tn = len(df[(df['error'] <= threshold) & (df['attack_types'] == 'normal')])
    fn = len(df[(df['error'] <= threshold) & (df['attack_types'] == 'abnormal')])
    
    accuracy = (tp + tn) / (tn + fp + fn + tp)
    sensitivity = tp / (fn + tp) 
    specificity = tn / (tn + fp)
    precision = tp / (fp + tp)
    recall = tp / (tp + fn)
    f1_score = 2 / ((1 / recall) + (1 / precision))
    
    print("tn:%.5f, fp:%.5f, fn:%.5f, tp:%.5f, total:.%5f" % (tn, fp, fn, tp, tn+fp+fn+tp))
    print("accuracy: %.5f, f1_score: %.5f" % (accuracy, f1_score))
    print("sensitivity : %.5f, specificity : %.5f" % (sensitivity, specificity))
    print("precision : %.5f, recall : %.5f" % (precision, recall))   