from sklearn.ensemble import RandomForestClassifier

def generate_rf(x_train, y_train):
    random_forest = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    random_forest.fit(x_train, y_train)
    return random_forest
