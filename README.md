# MNIST Activation Function Lab

**Bilingual README · 雙語 README**

[English](#english) · [繁體中文](#繁體中文)

# English

This reproducible MNIST lab investigates a central question from neural-network activation functions: **which activation should a hidden layer use, and why is Softmax normally used at the output of a ten-class classifier?**

It is an original, modular implementation inspired by the learning objective of Section 5.4.2. It does not reproduce a textbook program; instead, it adds command-line controls, result charts, reproducibility settings, and VS Code debugging tasks.

## What this project investigates

1. With all other conditions held fixed, compare `sigmoid`, `tanh`, `relu`, and `softplus` in the hidden layers on MNIST loss and accuracy.
2. Explain why `softmax` is the standard output activation for one-of-ten digit classification.
3. Provide a `legacy` suite that assigns the same activation to every layer. This is useful for interpreting a textbook-style comparison, but is **not** recommended as a production model design.
4. Split MNIST's official training data deterministically into fitting and validation subsets; keep the official test set untouched until final evaluation.

### How should the textbook-style activation table be interpreted?

The reference experiment used the same activation in two hidden layers and the output layer, together with **MSE, SGD, and 20 epochs**. Its reported test accuracies were:

| Hidden layer 1 | Hidden layer 2 | Output layer | Test accuracy |
| --- | --- | --- | ---: |
| Softmax | Softmax | Softmax | 0.1135 |
| ReLU | ReLU | ReLU | 0.9202 |
| Softplus | Softplus | Softplus | 0.8136 |
| tanh | tanh | tanh | 0.9030 |
| Sigmoid | Sigmoid | Sigmoid | 0.2195 |
| ReLU | ReLU | Softmax | 0.8617 |

These are observations from one specific configuration, not a universal activation-function ranking. In particular, ReLU or tanh at the output changes the meaning of the model output. A high accuracy does **not** make it a well-calibrated probability model. The main suite therefore fixes the output layer to Softmax and changes only the hidden-layer activation.

## Core concepts

### 1. MNIST and supervised classification

- Each MNIST image is a `28 × 28` grayscale image with pixel values from `0` to `255`.
- A fully connected layer consumes one feature vector, so every image is flattened to `784 = 28 × 28` values and normalized to `[0, 1]` by dividing by `255`.
- Digits `0…9` are converted to one-hot labels. For example, `3 → [0,0,0,1,0,0,0,0,0,0]`.
- Training data updates model weights. With the default `--validation-fraction 0.1`, 54,000 of MNIST's official 60,000 training examples are used for fitting and 6,000 for validation. The official 10,000-image test set remains independent and is used only for final `evaluate()` calls.

### 2. Network architecture

```text
784 normalized pixels
        │
Dense(hidden_units, activation = selected activation)   ← hidden_1
        │
Dense(hidden_units, activation = selected activation)   ← hidden_2
        │
Dense(10, activation = softmax)                         ← digit_probability
        │
P(digit = 0), …, P(digit = 9)
```

The default width is 256 units per hidden layer so the experiment runs on ordinary computers. Use `--hidden-units 784` for a wider textbook-style setting.

### 3. Roles of the activation functions

| Function | Definition / range | Strength | Interpretation in this lab |
| --- | --- | --- | --- |
| Sigmoid | `σ(x)=1/(1+e^{-x})`, `(0,1)` | Has a probability interpretation | Saturates near both extremes, so gradients can vanish in deeper networks; usually a weak default for modern MLP hidden layers. |
| tanh | `tanh(x)=(e^x-e^{-x})/(e^x+e^{-x})`, `(-1,1)` | Zero-centered; often easier than Sigmoid | Still saturates; a valid comparison when inputs are normalized. |
| ReLU | `max(0,x)`, `[0,∞)` | Fast and retains a gradient of 1 for positive inputs | A strong hidden-layer baseline; watch for dead ReLUs. |
| Softplus | `ln(1+e^x)`, `(0,∞)` | Smooth version of ReLU and differentiable everywhere | Useful comparison, but normally less preferred than ReLU for this task. |
| Softmax | `p_k=e^{z_k}/Σe^{z_i}`, sum is 1 | Converts ten scores into a mutually exclusive probability distribution | **Standard output activation** for ten-class MNIST, normally paired with categorical cross-entropy. |

### 4. Why should Softmax stay at the output?

Classification accuracy only checks whether the largest output matches the correct class. That means a non-Softmax output can occasionally score well while still being a poor probability model:

- ReLU and Softplus outputs do not necessarily sum to one.
- `tanh` can output negative values, which cannot be probabilities.
- Sigmoid treats every class independently; it is more natural for multi-label classification than for one mutually exclusive digit.
- Softmax makes the classes compete and gives categorical cross-entropy a valid probability interpretation.

The main experiment therefore fixes **Softmax at the output** and changes only the hidden activation. The `legacy` suite exists solely for textbook-style comparison.

## What do loss, accuracy, epoch, and batch mean?

- **Loss (categorical cross-entropy):** measures how much probability the model assigns to the correct class. If digit 7 receives probability `0.20` and later `0.90`, the loss improves strongly even when both predictions have the correct argmax.
- **Accuracy:** the fraction for which `argmax(prediction) == argmax(one-hot target)`. It is intuitive, but does not show confidence or calibration.
- **Epoch:** one complete pass through the training set. With 60,000 images and `batch_size=128`, one epoch performs about `ceil(60000/128)=469` weight updates.
- **Batch:** a small set of samples used to compute one gradient update; this is mini-batch gradient descent.
- **Overfitting signal:** training loss keeps decreasing while validation loss rises. Do not judge a model by training accuracy alone.

## Quick start in VS Code

```bash
git clone <your-repository-url>
cd mnist-activation-function-lab
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Run a small smoke test first:

```bash
python run_experiment.py --suite hidden --epochs 2 --train-limit 10000 --test-limit 2000 --validation-fraction 0.1
```

Run the recommended hidden-layer comparison:

```bash
python run_experiment.py --suite hidden --epochs 20 --hidden-units 784
```

Run the textbook-style all-layer comparison:

```bash
python run_experiment.py --suite legacy --epochs 20 --hidden-units 784 --loss mse --optimizer sgd --learning-rate 0.01
```

The `results/` directory will contain:

- `metrics.csv`: final training, validation, and independent-test metrics plus running time for every experiment.
- `summary_validation_accuracy.png`: candidate ranking by validation accuracy. Select an activation from this measure, not from the test score.
- `history_*.png`: training and validation loss/accuracy curves.
- `predictions_*.png`: sixteen test images with true and predicted labels.
- `run_config.json`: the parameters needed to reproduce the run.

Press `F5` in VS Code and select **Quick MNIST activation check**, or use **Terminal → Run Task** and choose `Run quick activation check`.

## Correct experimental practice

The important decision process is:

1. Choose the output activation from the task semantics: use Softmax for single-label multiclass MNIST.
2. Compare hidden activations according to gradient flow, convergence, and preprocessing: ReLU is a strong first choice; tanh and Softplus are meaningful alternatives; Sigmoid is a useful baseline.
3. Choose among activation candidates using the validation metrics. Treat the separate test metric as a final unbiased estimate after the comparison design is locked.
4. Change one variable at a time. Keep the split, seed, architecture, epochs, batch size, loss, and optimizer fixed before interpreting a result.

## Verification status

The automated tests cover configuration validation, deterministic/disjoint train-validation splitting, and the CLI validation-split option. The current code environment has not yet completed a TensorFlow MNIST training run, so this repository deliberately does **not** claim any new measured activation-function accuracy in the README. Run the quick command above after installing `requirements.txt`; it will write the measured values to `results/metrics.csv` together with the exact run configuration.

### Small-scale verification run (2026-10-16)

The implementation was smoke-tested with TensorFlow 2.19.1 using `--epochs 1 --hidden-units 64 --batch-size 128 --train-limit 2000 --test-limit 500 --validation-fraction 0.1 --seed 42`. This creates 1,800 fitting examples, 200 validation examples, and an independent 500-example test subset. The six automated tests passed. These one-epoch, small-sample results verify the pipeline only; they are **not** a final MNIST performance claim.

| Hidden activation | Validation accuracy | Independent test accuracy |
| --- | ---: | ---: |
| tanh | 0.6900 | 0.7320 |
| ReLU | 0.6600 | 0.6640 |
| Softplus | 0.4800 | 0.4660 |
| Sigmoid | 0.3200 | 0.3840 |

## Project structure

```text
mnist-activation-function-lab/
├── .vscode/                 # F5 launch configuration and tasks
├── results/                 # generated CSV and PNG files (not committed)
├── src/
│   ├── config.py             # experiment settings and reproducibility
│   ├── data.py               # loading, flattening, normalization, one-hot labels
│   ├── models.py             # controlled MLP factory
│   └── experiment.py         # training loop, plots, and result table
├── tests/test_config.py
├── run_experiment.py         # CLI entry point
├── requirements.txt
└── README.md
```

## Extensions

1. Replace Adam with SGD (with and without momentum) while holding everything else fixed.
2. Keep ReLU + Softmax and compare `hidden_units` values of 64, 256, and 784.
3. Add `EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)` and inspect overfitting.
4. Compare this MLP with a CNN, which uses the local spatial structure of images.

## License

MIT.

# 繁體中文

以 MNIST 手寫數字辨識，實作並檢驗「神經網路中常用激活函數」的第一個核心問題：**隱藏層該選什麼激活函數？輸出層為何通常必須選 Softmax？**

這是依照第 5.4.2 節的實驗精神重新整理的原創、可重現專案；它不複製教材程式碼，而是將流程模組化、加上結果圖、命令列參數與 VS Code 偵錯設定。

## 本專案回答什麼

1. 在其餘條件相同時，將隱藏層分別設為 `sigmoid`、`tanh`、`relu`、`softplus`，測試 MNIST 的 loss 與 accuracy。
2. 說清楚 `Softmax` 應放在十分類輸出層的原因。
3. 另外保留 `legacy` 模式，模擬「每一層都採相同激活函數」的教材式對照；這適合理解表格，不應視為正式模型設計。
4. 將 MNIST 官方訓練集以固定 seed 切成訓練／驗證資料，官方測試集則完全保留到最後評估。

### 教材頁面中表 5-4 的結果，該怎麼看？

你提供的頁面是以兩個隱藏層與輸出層採用相同激活函數、搭配 **MSE 與 SGD、訓練 20 輪** 的單次實驗。表中的測試準確率如下：

| 隱藏層 1 | 隱藏層 2 | 輸出層 | 測試準確率 |
| --- | --- | --- | ---: |
| Softmax | Softmax | Softmax | 0.1135 |
| ReLU | ReLU | ReLU | 0.9202 |
| Softplus | Softplus | Softplus | 0.8136 |
| tanh | tanh | tanh | 0.9030 |
| Sigmoid | Sigmoid | Sigmoid | 0.2195 |
| ReLU | ReLU | Softmax | 0.8617 |

這些數字是當時那組設定下的觀察結果，不是所有資料集、初始化與套件版本都會重現的「函數總排名」。特別是把 ReLU、tanh 等放進輸出層會改變輸出表示方式，因此**不能單純把 0.9202 解讀成「ReLU 輸出層一定優於 Softmax」**。本專案把這種組合獨立為 `legacy`，並把較嚴謹的「固定 Softmax、只比較隱藏層」設為主實驗。

## 對應的觀念

### 1. MNIST 資料與監督式分類

- 每張圖片是 `28 × 28` 灰階像素，灰階值本來在 `0…255`。
- 全連接層一次接收一條特徵向量，所以先攤平成 `784 = 28 × 28` 維；再除以 `255` 正規化到 `[0, 1]`。
- 類別是數字 `0…9`。本專案將整數標籤轉為 one-hot，例如 `3 → [0,0,0,1,0,0,0,0,0,0]`。
- 訓練集才用來更新權重。預設 `--validation-fraction 0.1` 時，官方 60,000 筆訓練資料中有 54,000 筆用於訓練、6,000 筆用於驗證；官方 10,000 筆測試集不參與每個 epoch 的監看，只在最後以 `evaluate()` 獨立評估。

### 2. 網路架構

```text
784 normalized pixels
        │
Dense(hidden_units, activation = selected activation)   ← hidden_1
        │
Dense(hidden_units, activation = selected activation)   ← hidden_2
        │
Dense(10, activation = softmax)                         ← digit_probability
        │
P(digit = 0), …, P(digit = 9)
```

預設每個隱藏層為 256 個神經元，讓一般電腦較容易完成實驗。想貼近教材示例的層寬，可改用 `--hidden-units 784`。

### 3. 五個函數的任務分工

| 函數 | 定義／值域 | 優點 | 在這個 MNIST 實驗的判讀 |
| --- | --- | --- | --- |
| Sigmoid | `σ(x)=1/(1+e^{-x})`，`(0,1)` | 有明確機率意義 | 深層時在兩側飽和，導數接近 0，容易梯度消失；不宜作現代 MLP 的預設隱藏層。 |
| tanh | `tanh(x)=(e^x-e^{-x})/(e^x+e^{-x})`，`(-1,1)` | 零中心，通常比 Sigmoid 好訓練 | 仍會飽和；小型、正規化良好的網路可用。 |
| ReLU | `max(0,x)`，`[0,∞)` | 正區間導數為 1，計算快、梯度較能傳遞 | 常見的隱藏層起點；要留意「死亡 ReLU」。 |
| Softplus | `ln(1+e^x)`，`(0,∞)` | 是平滑的 ReLU，處處可微 | 平滑但正值區較慢；可作對照，不通常優先於 ReLU。 |
| Softmax | `p_k=e^{z_k}/Σe^{z_i}`，總和為 1 | 將 10 個分數轉成互斥類別的機率分布 | **十分類輸出層標準選擇**，通常與 categorical cross-entropy 配對。 |

### 4. 為什麼不要把 ReLU、tanh、Sigmoid、Softplus 直接當十分類輸出？

「分類對不對」只需最大值所在的類別正確；因此某些非 Softmax 輸出仍可能得到不低的 accuracy。但這不等於它是合理的機率模型：

- ReLU 與 Softplus 的十個輸出不保證加總為 1。
- tanh 可輸出負值，不能表示機率。
- Sigmoid 對每一類各自獨立，適合 **多標籤** 任務，不自然表達「一張圖只是一個數字」的互斥關係。
- Softmax 明確強迫十類互相競爭，並讓 `categorical_crossentropy` 有正確的機率意義。

所以主要實驗固定 **Softmax 輸出層**，只改隱藏層；`legacy` 僅用於重看教材式組合為何會造成明顯差異。

## Loss、accuracy、epoch 到底在量什麼？

- **Loss（categorical cross-entropy）**：量化「模型給正確類別的機率是否夠高」。正確答案是 7，但模型預測 7 的機率從 `0.20` 升到 `0.90`，即使兩次 argmax 都是 7，loss 仍會大幅下降；因此它比 accuracy 更細緻，是反向傳播實際要最小化的目標。
- **Accuracy**：`argmax(預測機率) == argmax(one-hot 真實標籤)` 的比例。它直觀，但無法反映信心高低。
- **Epoch**：模型完整看完一次訓練集並更新多次權重，稱為一個 epoch。若 `batch_size=128`、有 60,000 筆訓練資料，一個 epoch 約有 `ceil(60000/128)=469` 次更新。
- **Batch**：每次送入 128 筆資料，計算梯度後更新一次權重；這是 mini-batch gradient descent。
- **過擬合訊號**：訓練 loss 繼續下降、但 validation loss 反而上升；此時不應只看訓練 accuracy。

## 快速開始（VS Code）

```bash
git clone <your-repository-url>
cd mnist-activation-function-lab
python -m venv .venv
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux：

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

先跑小資料集確認環境：

```bash
python run_experiment.py --suite hidden --epochs 2 --train-limit 10000 --test-limit 2000 --validation-fraction 0.1
```

正式比較四個隱藏層激活函數（建議的現代多分類設定）：

```bash
python run_experiment.py --suite hidden --epochs 20 --hidden-units 784
```

教材式「所有層同一函數」對照（表 5-4 的條件）：

```bash
python run_experiment.py --suite legacy --epochs 20 --hidden-units 784 --loss mse --optimizer sgd --learning-rate 0.01
```

執行後會在 `results/` 產生：

- `metrics.csv`：各實驗最終的訓練、驗證、獨立測試指標與時間。
- `summary_validation_accuracy.png`：依驗證正確率產生的候選模型排名；選擇激活函數時應看這張圖，不應以測試分數挑選。
- `history_*.png`：每個實驗的 train / validation loss 與 accuracy 曲線。
- `predictions_*.png`：16 張測試影像的真實類別與預測類別。
- `run_config.json`：這次實驗的可重現參數。

在 VS Code 直接按 `F5`，選 **Quick MNIST activation check** 即可；或從 **Terminal → Run Task** 選 `Run quick activation check`。

## 與教材表格的正確讀法

教材示例顯示，同一資料集、相同網路與訓練輪數下，激活函數組合會讓測試正確率產生很大落差；尤其若使輸出不再是合理的十類機率分布，結果可能很差。不要把單次表格數字當成「永遠的排名」：它還受初始化、層寬、epoch、學習率、optimizer、套件版本及硬體影響。

你真正要帶走的模型選擇準則是：

1. 先由**任務輸出語意**決定輸出層：MNIST 這種單標籤多分類選 Softmax。
2. 再由梯度傳遞、收斂速度、資料正規化情況比較隱藏層：ReLU 是很合理的起點，tanh / Softplus 是可驗證的替代方案，Sigmoid 通常是較弱基線。
3. 以驗證集的結果挑選激活函數；當比較設計已固定後，再將獨立測試分數當成最終泛化能力估計。
4. 每一次只改一個因素，固定資料切分、seed、架構、epoch、batch size、optimizer，才可以解釋結果。

## 驗證狀態

自動化測試涵蓋：設定參數驗證、訓練／驗證資料切分的可重現性與互斥性、CLI 的驗證集比例選項，以及訓練流程是否確實以驗證集監看、以獨立測試集最終評估。安裝 `requirements.txt` 後執行前述快速指令，實測結果與完整設定會輸出到 `results/metrics.csv`。

### 小規模實測（2026-10-16）

本次以 TensorFlow 2.19.1 執行 `--epochs 1 --hidden-units 64 --batch-size 128 --train-limit 2000 --test-limit 500 --validation-fraction 0.1 --seed 42` 作為煙霧測試。此設定將資料分為 1,800 筆訓練、200 筆驗證，以及獨立 500 筆測試；六項自動化測試均通過。以下僅驗證管線可執行，**不是**完整 MNIST 訓練後可用來宣稱的最終排名。

| 隱藏層激活函數 | 驗證正確率 | 獨立測試正確率 |
| --- | ---: | ---: |
| tanh | 0.6900 | 0.7320 |
| ReLU | 0.6600 | 0.6640 |
| Softplus | 0.4800 | 0.4660 |
| Sigmoid | 0.3200 | 0.3840 |

## 專案結構

```text
mnist-activation-function-lab/
├── .vscode/                 # F5 偵錯與工作任務
├── results/                 # 執行後的 CSV 與圖（不提交）
├── src/
│   ├── config.py             # 參數與可重現設定
│   ├── data.py               # MNIST 載入、攤平、正規化、one-hot
│   ├── models.py             # 固定架構的 MLP 工廠
│   └── experiment.py         # 實驗迴圈、圖表、結果表
├── tests/test_config.py
├── run_experiment.py         # CLI 入口
├── requirements.txt
└── README.md
```

## 延伸練習

1. 將 Adam 改為 SGD（有/無 momentum），其餘不變，觀察收斂速度。
2. 固定 ReLU + Softmax，將 `hidden_units` 比較 64、256、784。
3. 加入 `EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)`，觀察過擬合時的差異。
4. 將這個 MLP 與 CNN 比較；CNN 利用影像的局部空間結構，通常會有更好的 MNIST 效果。

## License

MIT.
