from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

wine = load_wine()
x, y = wine.data, wine.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
x_train_s = scaler.fit_transform(x_train)
x_test_s  = scaler.transform(x_test)
