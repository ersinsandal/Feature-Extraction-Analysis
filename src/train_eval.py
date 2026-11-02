from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def train_svm(X_train, y_train):
    """
    SVM modelini eğitir.
    Bu projede SVM sabit karar yöntemi olarak kullanılır.
    Böylece karşılaştırmada değişen şey yalnızca feature extraction yöntemi olur.
    """
    model = SVC(kernel="linear")
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Modeli test eder ve temel performans metriklerini döndürür.
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    return accuracy, precision, recall, macro_f1, cm