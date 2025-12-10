from xgboost import XGBClassifier

def generate_xgboost(x_train, y_train):
    xgboost_model = XGBClassifier(
        n_estimators=200,           
        max_depth=5,                
        learning_rate=0.1,          
        scale_pos_weight=20,        
        eval_metric='logloss',
        random_state=42
    )

    xgboost_model.fit(x_train, y_train)
    return xgboost_model
