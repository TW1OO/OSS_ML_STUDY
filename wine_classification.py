!apt-get install -y -q fonts-nanum

import matplotlib as mpl
import matplotlib.font_manager as fm

fm.fontManager.__init__()
nanum_fonts = [
    f for f in fm.findSystemFonts()
    if 'Nanum' in f and 'Gothic' in f and 'Bold' not in f and 'Extra' not in f
]
if nanum_fonts:
    fe = fm.FontEntry(fname=nanum_fonts[0], name='NanumGothic')
    fm.fontManager.ttflist.insert(0, fe)
    mpl.rcParams['font.family'] = 'NanumGothic'
mpl.rcParams['axes.unicode_minus'] = False

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

wine = load_wine()
X, y = wine.data, wine.target
print(f"데이터 shape: X={X.shape}, y={y.shape}")
print(f"클래스: {wine.target_names}")

colors = ['#E74C3C', '#2ECC71', '#3498DB']

pca_raw = PCA(n_components=2)
X_raw_2d = pca_raw.fit_transform(X)
var_raw = pca_raw.explained_variance_ratio_

for cls in range(3):
    mask = y == cls
    plt.scatter(X_raw_2d[mask, 0], X_raw_2d[mask, 1],
                c=colors[cls], marker='o', s=40,
                edgecolors='k', linewidths=0.5, label=wine.target_names[cls])
plt.xlabel(f'PC1 ({var_raw[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({var_raw[1]*100:.1f}%)')
plt.title('스케일링 이전 데이터 분포 (PCA 2D)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

k_range = range(1, 21)
accuracies = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_s, y_train)
    accuracies.append(accuracy_score(y_test, knn.predict(X_test_s)))

best_k = np.argmax(accuracies) + 1
best_acc = max(accuracies)
print(f"\n최적 K: {best_k}, 최고 정확도: {best_acc:.4f}")

model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=wine.target_names))

pca = PCA(n_components=2)
X_train_2d = pca.fit_transform(X_train_s)
X_test_2d  = pca.transform(X_test_s)
var_ratio   = pca.explained_variance_ratio_

colors = ['#E74C3C', '#2ECC71', '#3498DB']   # class_0, class_1, class_2
markers_train = 'o'
markers_test  = '^'

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f'KNN Wine Classification  |  Best K={best_k}  |  Accuracy={best_acc:.4f}',
             fontsize=14, fontweight='bold', y=1.02)
ax0 = axes[0]
ax0.plot(k_range, accuracies, marker='o', color='steelblue', linewidth=2)
ax0.axvline(best_k, color='red', linestyle='--', label=f'Best K={best_k}')
ax0.scatter([best_k], [best_acc], color='red', s=100, zorder=5)
ax0.set_xlabel('K (이웃 수)', fontsize=11)
ax0.set_ylabel('Accuracy', fontsize=11)
ax0.set_title('K에 따른 정확도 변화', fontsize=12)
ax0.set_xticks(list(k_range))
ax0.legend()
ax0.grid(True, alpha=0.3)
ax1 = axes[1]
for cls in range(3):
    # Train 데이터 (원형)
    mask_tr = y_train == cls
    ax1.scatter(X_train_2d[mask_tr, 0], X_train_2d[mask_tr, 1],
                c=colors[cls], marker='o', s=60, alpha=0.6, edgecolors='white', linewidths=0.5)
    # Test 데이터 (삼각형)
    mask_te = y_test == cls
    ax1.scatter(X_test_2d[mask_te, 0], X_test_2d[mask_te, 1],
                c=colors[cls], marker='^', s=100, alpha=1.0, edgecolors='black', linewidths=0.8)

# 범례: 클래스 색상
class_patches = [mpatches.Patch(color=colors[i], label=wine.target_names[i]) for i in range(3)]
# 범례: 마커 종류
train_handle = plt.Line2D([0], [0], marker='o', color='gray', label='Train', markersize=8,
                           markerfacecolor='gray', linestyle='None')
test_handle  = plt.Line2D([0], [0], marker='^', color='gray', label='Test',  markersize=8,
                           markerfacecolor='gray', linestyle='None')
ax1.legend(handles=class_patches + [train_handle, test_handle], fontsize=9)
ax1.set_xlabel(f'PC1 ({var_ratio[0]*100:.1f}%)', fontsize=11)
ax1.set_ylabel(f'PC2 ({var_ratio[1]*100:.1f}%)', fontsize=11)
ax1.set_title('PCA 2D 산점도 (Train ● / Test ▲)', fontsize=12)
ax1.grid(True, alpha=0.3)

# ── 7-3. KNN 결정 경계 + 테스트 데이터 ────────────
ax2 = axes[2]

# PCA 공간에서 KNN 재학습 (결정 경계용)
knn_2d = KNeighborsClassifier(n_neighbors=best_k)
knn_2d.fit(X_train_2d, y_train)

# 메쉬 그리드
x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# 결정 경계 배경
from matplotlib.colors import ListedColormap
bg_cmap = ListedColormap(['#FADBD8', '#D5F5E3', '#D6EAF8'])
ax2.contourf(xx, yy, Z, cmap=bg_cmap, alpha=0.6)
ax2.contour(xx, yy, Z, colors='gray', linewidths=0.8, alpha=0.5)

# 테스트 데이터 점 (정답/오답 구분)
for cls in range(3):
    mask = y_test == cls
    correct   = mask & (y_test == y_pred)
    incorrect = mask & (y_test != y_pred)
    ax2.scatter(X_test_2d[correct,   0], X_test_2d[correct,   1],
                c=colors[cls], marker='^', s=120, edgecolors='black', linewidths=0.8, zorder=4)
    ax2.scatter(X_test_2d[incorrect, 0], X_test_2d[incorrect, 1],
                c=colors[cls], marker='X', s=150, edgecolors='red',   linewidths=1.2, zorder=5)

class_patches2 = [mpatches.Patch(color=colors[i], label=wine.target_names[i]) for i in range(3)]
ok_handle   = plt.Line2D([0], [0], marker='^', color='gray', label='정답', markersize=8,
                          markerfacecolor='gray', linestyle='None')
err_handle  = plt.Line2D([0], [0], marker='X', color='gray', label='오답', markersize=9,
                          markerfacecolor='gray', markeredgecolor='red', linestyle='None')
ax2.legend(handles=class_patches2 + [ok_handle, err_handle], fontsize=9)
ax2.set_xlabel(f'PC1 ({var_ratio[0]*100:.1f}%)', fontsize=11)
ax2.set_ylabel(f'PC2 ({var_ratio[1]*100:.1f}%)', fontsize=11)
ax2.set_title('결정 경계 + 테스트 데이터 (▲정답 / ✕오답)', fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
