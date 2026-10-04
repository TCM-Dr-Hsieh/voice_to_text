# 語音書寫

這是 Windows 本機語音書寫工具。頁面只有文章編輯器：把游標放在插入位置後，可以用本機麥克風、電腦播放聲音、遠端瀏覽器音訊，或上傳音訊檔產生文字。Qwen3-ASR 與 ForcedAligner 在本機背景程序執行；校稿呼叫設定的 OpenAI 相容聊天服務（預設 LM Studio）。

## 安裝（全新電腦）

**不需要預先安裝 conda 或任何 Python 環境。** 在全新的 Windows 電腦上，雙擊 `install.cmd`（或在 PowerShell 執行 `.\setup.ps1`）即可完成全部安裝：

| 步驟 | 內容 | 下載量 |
|---|---|---|
| 預檢 | Windows 64 位元、磁碟空間（≥ 20 GB）、NVIDIA 顯卡與驅動（≥ 570）、網路 | — |
| Python | 自動找 Python 3.12；找不到時**經你同意**用 winget 安裝（只裝給目前使用者，不需系統管理員） | 約 25 MB |
| `.venv` | UI 用的獨立環境（NiceGUI、httpx、SoundCard、PyAV…），版本固定在 `constraints.txt` | 約 250 MB |
| `.venv-asr` | **獨立的** ASR 環境，含 `torch 2.7.1+cu128` 與 Qwen3-ASR 相依套件（94 個固定版本，見 `requirements-asr.txt`，不借用任何其他環境） | 約 4 GB（其中 torch 3.3 GB） |
| 模型 | 從 Hugging Face 下載固定版本的 `Qwen3-ASR-1.7B` 與 `Qwen3-ForcedAligner-0.6B` 到 `models\` | 約 6.1 GB |
| 設定 | 寫入 `data\settings.json`（模型路徑；可順便設定校稿 LLM） | — |
| 健檢 | `tools\doctor.py`：環境、GPU、模型、麥克風、LLM 連線，並實際載入 ASR 辨識一次 | — |

**需要自備**：NVIDIA 顯卡（語音辨識在 CPU 上太慢，不建議）與驅動、網路、以及一個 OpenAI 相容的 LLM 伺服器（本腳本不安裝 LLM；在「模型設定」填入網址與模型，或安裝時加參數）。

```powershell
.\setup.ps1                                  # 全部自動；可中斷後重跑，已完成的步驟會略過、下載會續傳
.\setup.ps1 -LlmUrl http://192.168.1.10:8080/v1 -LlmModel my-model     # 順便設定校稿 LLM
.\setup.ps1 -HfEndpoint https://hf-mirror.com                           # 網路受限時用鏡像下載模型
.\setup.ps1 -ModelsDir E:\models                                        # 模型放在別的磁碟
.\setup.ps1 -SkipModels                      # 模型已另外放好：之後在「模型設定」指定資料夾
.\setup.ps1 -Force                           # 重建 .venv 與 .venv-asr
.\setup.ps1 -Dev                             # 另外安裝 pytest（開發用）
```

若直接執行 `.\setup.ps1` 被「執行原則」擋住，改用 `powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1`（`install.cmd` 已經這樣做）。專案路徑請盡量短（例如 `C:\voice-writing`），避免 Windows 260 字元路徑上限。其他參數見 `Get-Help .\setup.ps1 -Detailed`。安裝記錄在 `setup.log`；任何時候都可以用 `.venv\Scripts\python.exe tools\doctor.py [--smoke]` 重新健檢。

- 既有的 `.venv`、`.venv-asr` 若是疊加在其他 Python／conda 環境上，或基底 Python 已不存在，會被自動偵測並改建成獨立環境；改建前請先關閉語音書寫。
- 已有模型時（例如 `D:\models`）：把路徑填在「模型設定」（即 `data\settings.json`），安裝腳本會沿用，不重複下載。若同時指定 `-ModelsDir`，則以 `-ModelsDir` 為準。模型資料夾可用絕對路徑，也可用相對於本專案的路徑（預設 `models\Qwen3-ASR-1.7B`），後者讓整個資料夾可以複製到別台電腦。
- 模型版本固定在 `tools\download_models.py` 的 `MODELS`（commit SHA），逐檔大小與內容雜湊（權重 SHA-256、其餘檔案 git blob 雜湊）記在 `tools\model_manifest.json`；升級時改 `MODELS` 後執行 `.venv\Scripts\python.exe tools\update_manifest.py` 重新產生。
- 沿用的既有模型若與固定版本不同（例如別的版本或檔案損毀），安裝與健檢會**警告**但不強制改用；`download_models.py --check` 會以非 0 結束。預設只比對檔案大小；要核對**每個檔案**的內容雜湊：`.venv-asr\Scripts\python.exe tools\download_models.py --check --verify`，或 `.venv\Scripts\python.exe tools\doctor.py --verify-models`。

## 啟動

在此資料夾雙擊 start.cmd，或執行：

    .\.venv\Scripts\python.exe app.py --open

頁面預設為 http://127.0.0.1:2020/，若該埠已被其他程式佔用，可用環境變數 VOICE_APP_PORT 改埠。`.venv` 和 `.venv-asr` 位於本資料夾；校稿服務需另外啟動。

## 書寫與音訊來源

文章只存在目前瀏覽器頁面，程式不會自動儲存。工具列可開啟、儲存、另存 TXT，並在未儲存時提醒。開始音訊前先把游標放到插入位置；若選取文字，本次成功辨識的語音會取代所選範圍。音訊處理中文章暫停手動編輯，畫面在固定插入點預覽校稿；完成後一次寫入，取消則還原文章。沒有成功定稿文字時，原文章與選取範圍保持不變。

選「音訊檔」後，上傳 WAV、MP3、M4A、FLAC 或 OGG，再按「開始轉錄」。它使用與即時音訊相同的固定視窗、重疊裁切和校稿流程，完成後插入文章。停止可處理已讀取的音訊；未讀取部分不再處理。音訊檔不另存錄音 WAV。

遠端麥克風與遠端電腦音訊由瀏覽器擷取並經同源 WebSocket 傳回主機；其他電腦使用時需要 HTTPS 及瀏覽器音訊授權。VOICE_APP_SHARED=1 會隱藏主機裝置和設定入口，只留下遠端來源與音訊檔。分享模式沒有登入驗證，入口仍須限制可進入的人員。

同一個瀏覽器分頁在遠端音訊連線短暫中斷時，會繼續擷取並暫存在該分頁。瀏覽器每 5 秒檢查連線；15 秒沒有伺服器回應時，即使 WebSocket 尚未通知關閉，也會嘗試重連。伺服器 20 秒沒有收到音訊或心跳，也會將舊連線視為失效。同一段錄音的新連線可安全接管舊連線，並在偵測斷線後 30 秒內補傳。伺服器用封包序號確認進度，重傳不會把同一段聲音重複寫入；錄音仍保存為同一個 WAV。按「停止並完成」會等待暫存音訊送完。若超過重連期限、暫存超過 32 MiB，或分頁被關閉／重新整理，音訊可能不完整；已收到的部分會標示為未完成。這項補傳只適用同一分頁及同一網址；若 cloudflared 重啟後換成新網址，必須重新開啟頁面。

首次載入 ASR 時，伺服器會定期回報準備中；尚未開始錄音就斷線時，同一分頁會用相同識別碼重送啟動要求。若伺服器連續 15 秒沒有回應，瀏覽器會嘗試重連；準備超過約 6 分鐘仍未完成時會停止等待，並解除文章編輯鎖定。

## 固定視窗辨識與三段校稿

設音訊視窗長度為 L、左側參考重疊為 Z。第 n 段音訊範圍是：

    [(n−1)(L−Z), (n−1)(L−Z)+L]

例如 L=9、Z=4 時，前三段為 0–9、5–14、10–19 秒。Overlap 沒有固定四秒上限，但須符合 0 ≤ Z ≤ L−3，讓每段至少有三秒新音訊。每隔 L−Z 秒產生一段新 ASR。停止或音訊檔結束時，如果還有新音訊，會辨識不足 L 的最後尾段；剛好停在完整視窗末端時不再重送只有 overlap 的尾段。停頓不觸發切段，沒有定時初稿，也沒有保留兩秒右側尾端。

每段 ASR 的完整重疊視窗和時間對齊保存在紀錄中。組成文章時，前一視窗已涵蓋的音訊歸前一段；程式用原始 ASR 的 ForcedAligner 時間，裁掉新視窗左側重複文字。校稿後文字不沿用 ASR 的逐字時間戳。

每收到一段新 ASR，LLM 參考已鎖定前文，校正最多三段：「前兩段可修訂文字＋最新 ASR 新增文字」。第 1、2 段到達時也立即校稿並預覽。從第 3 段起，每輪回傳三段後，最舊的一段鎖定，剩下兩段可於下輪再校正；停止或檔案結束後全部鎖定。符合段號格式且沒有把非空段刪空的校稿結果會直接套用；回應格式錯誤、非空段回空白或請求失敗時會重試，仍失敗則保留目前文字並顯示警告。文章內容可在完成後直接編輯。

## 設定與資料

data/settings.json 保存視窗長度、Overlap、模型路徑和校稿連線資訊。讀取舊設定檔時會忽略已移除的初稿間隔、停頓門檻、音量門檻和校稿最大改動比例。API Key 以本機明文儲存，不包含於語音紀錄。

「儲存錄音」只適用即時來源，會把連續 16 kHz WAV 和同名完整紀錄 JSON 寫入 data/recordings/。音訊檔來源和未開啟儲存錄音的來源可在頁面匯出本次 JSON；書寫文章仍需另外儲存 TXT。辨識或對齊失敗時保留片段供重試；可略過對齊失敗的片段，紀錄中保留可能缺漏的時間範圍與原始診斷。

## 驗證

    .\.venv\Scripts\python.exe -m pytest -q                      # 需要以 setup.ps1 -Dev 安裝 pytest
    .\.venv\Scripts\python.exe tools\doctor.py --smoke           # 環境、模型、麥克風、LLM 健檢，並實際辨識一次
    .\.venv\Scripts\python.exe tools\check_pipeline.py path\to\sample.wav

自動測試涵蓋視窗邊界、左側重疊裁切、三段滾動校稿、檔案與遠端音訊、取消／重試、設定、安裝工具和瀏覽器書寫邏輯。已用公開音檔在 CUDA 上驗證 Qwen3-ASR 與 ForcedAligner；LM Studio 及實際麥克風流程仍需自行驗收。

## 授權

本專案以 [Apache License 2.0](LICENSE) 授權，版權與第三方元件的授權資訊見 [NOTICE](NOTICE)。內建的 OpenCC-JS 瀏覽器版（繁簡轉換）及其字典資料的授權檔在 `voice_app/OPENCC_*`。安裝時下載的 Python 套件與 Qwen3-ASR／Qwen3-ForcedAligner 模型不包含在本倉庫內，各自適用其原始授權（皆為 Apache-2.0 或寬鬆授權，詳見 NOTICE）。
