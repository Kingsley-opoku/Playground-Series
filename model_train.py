
import pandas as pd
import time
import seaborn as sns
from sklearn.model_selection        import train_test_split
from sklearn.metrics                import accuracy_score, balanced_accuracy_score
                    

class TrainModel:
    def __init__(self, classifier) -> any:
       self.classifer=classifier

    def train_pred(self, x, y) -> pd.DataFrame:
        """"""
        x_train, x_val, y_train, y_val= train_test_split(x, y, random_state=0,test_size=0.2, stratify=y) 
        results = pd.DataFrame({'Model': [], 'Accuracy': [], 'Bal Acc.': [], 'Time': []})
        
        for model_name, model in self.classifer.items():
    
            start_time = time.time()
            model.fit(x_train, y_train)
            total_time = time.time() - start_time
        
            pred = model.predict(x_val)
        results = results.append({"Model":    model_name,
                              "Accuracy": accuracy_score(y_val, pred)*100,
                              "Bal Acc.": balanced_accuracy_score(y_val, pred)*100,
                              "Time":     total_time},
                              ignore_index=True)
           
        return results


        

    