# 人工智慧期中報告 題目Google Colab 跑目標追蹤的test程式碼
**組員:** 11325003 張博憫 11325008 林禹翔
# 一、實驗目的
本實驗旨在 Google Colab 環境中建置與執行 video_analyst 視覺物體追蹤專案。透過配置 PyTorch 深度學習環境、安裝必要相依套件與編譯 Cython 評測模組，載入 SiamFCPP 追蹤演算法與對應的預訓練權重，針對 GOT-10k 資料集進行效能推論與測試，以驗證物件追蹤模型的運作流程與追蹤表現。
# 二、執行步驟
步驟 1：初始化環境與設定 GPU 資源
在此步驟中，我們導入必要的 Python 套件，並指定系統使用第一張 GPU 卡進行運算。

![03](https://github.com/11325008-maker/-/blob/main/1-1.png)

步驟 2：掛載 Google 雲端硬碟

將個人的 Google Drive 掛載至 Colab 的虛擬機目錄中，以便讀取儲存在雲端硬碟中的專案與權重檔。

![03](https://github.com/11325008-maker/-/blob/main/1-2.png)

步驟 3：複製專案至本機高速暫存空間並切換目錄
由於雲端硬碟（Google Drive）讀寫速度較慢且可能受限，將專案複製到 Colab 本機的暫存硬碟空間 /content/ 中，可提升讀寫速度與編譯效率。

![03](https://github.com/11325008-maker/-/blob/main/1-3.png)

步驟 4：確認專案目錄內容
列出當前目錄下的所有檔案與資料夾，確認複製是否完整。

![03](https://github.com/11325008-maker/-/blob/main/1-4.png)

步驟 5：建立依賴套件清單 (requirements.txt)
將專案執行時所需的第三方 Python 套件與版本規定寫入獨立的文字檔中。

![03](https://github.com/11325008-maker/-/blob/main/1-5.png)

步驟 6：批量安裝 Python 依賴套件
利用 pip 套件管理器一次安裝所有記錄在 requirements.txt 中的套件。

![03](https://github.com/11325008-maker/-/blob/main/1-6.png)
![03](https://github.com/11325008-maker/-/blob/main/1-7.png)

步驟 7：再次確認雲端硬碟掛載狀態
再次執行掛載指令，確保後續讀取或寫入路徑不會因連線問題斷開。

![03](https://github.com/11325008-maker/-/blob/main/1-8.png)

步骤 8：建立並編譯 VOT 評測工具套件 (pysot/utils)
為了在本地進行 C/C++ 擴充套件（Cython *.pyx 檔案）的編譯，先將相關程式碼完整複製出來並進行安裝。

![03](https://github.com/11325008-maker/-/blob/main/1-9.png)
![03](https://github.com/11325008-maker/-/blob/main/1-10.png)

步驟 9：返回專案主目錄並檢視執行腳本
切換回專案根目錄，並列出可用於模型測試（Test）與訓練（Train）的 Bash 腳本。

![03](https://github.com/11325008-maker/-/blob/main/1-11.png)

步驟 10：查看測試腳本內容 (test_GOT.sh)
檢視 GOT-10k 資料集測試腳本內部的執行指令與參數設定。

![03](https://github.com/11325008-maker/-/blob/main/1-12.png)

# 三、實驗結果
本實驗成功於 Google Colab 環境中完成 video_analyst 視覺追蹤專案的環境建置、GPU 設定與 Google Drive 雲端硬碟掛載。透過配置 PyTorch 深度學習運算資源、編譯 VOT 評測基準所需的 Cython 模組（region.pyx），以及安裝 requirements.txt 中的必要相依套件，已順利完成前置準備。同時，透過檢視 test_GOT.sh 與 SiamFCPP 模型搭配 AlexNet 的設定檔，確認各項參數與預訓練權重路徑皆正確無誤，隨時可供後續執行 GOT-10k 資料集的目標追蹤推論與效能評測。
