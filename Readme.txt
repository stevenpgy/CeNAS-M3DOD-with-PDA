資料夾 Multi
	configs:
	    damoyolo_tinynasL45_L:設定主要參數，和模型架構
	damo:
	    apis:這個資料夾包含訓練和推論的API
		detector_trainer.py：訓練邏輯主控（包含資料讀取、訓練流程、optimizer、checkpoint、distill、EMA 等）
		detector_inference.py：基於 PyTorch 的模型推論與評估功能
		detector_inference_trt.py：使用 TensorRT 引擎加速模型推論

	    augmentations:data augmation的一些方法，讓bounding box畫的比較準
		box_level_augs:
		    box_level_augs.py:主模組，包含Box_augs類別。負責根據不同物件大小與訓練進度，挑選對應的資料增強策略，並應用於指定的 bounding boxes。支援 color 與 geometric 類型的增強，並透過高斯圖平滑混合轉換結果。
		    color_augs.py`:實作多種針對影像顏色的增強方式
		    geometric_augs.py:提供框內物件的幾何變換操作
		    gaussian_maps.py:用來產生高斯遮罩
		    __init__.py:初始化模組載入設定
		
		scale_aware_aug.py:
		    本模組實作了基於物件尺度感知scale-aware的自動資料增強策略

	    base_models:模型架構的資料夾，主要放模型架構包含back bone，頸部結構，程式函數設定
		backbones:
		    nas_backbones:這些 .txt檔案描述了各種不同的TinyNAS 模型結構

		tinynas_csp.py、tinynas_mob.py、tiny_res.py:
		    三種不同風格的TinyNAS主幹網Back bone實作，每個模組提供不同架構設計以因應不同設備需求與效能取捨

	        core:
		    atts_assigner.py:實作 ATSS演算法，根據預測框與 ground truth 的距離與 IoU 分數，自動選出正樣本，提升 anchor-based 方法的分配效率
		    bbox_calculator.py:負責各種 bounding box 幾何計算，包括 IoU、面積、中心點、邊界轉換等，為損失函數與樣本分配等模組提供支援
		    end2end.py:支援模型導出時的推論流程，包含 Non-Maximum Suppression、分數篩選與 bounding box 後處理，用於部署 ONNX 或 TensorRT 模型的預測後處理
		    ops.py:定義多種基礎模組
		    ota_assigner.py:實作 OTA，基於 optimal transport cost matrix 計算正負樣本分配關係，能進一步優化偵測器在複雜背景下的配對策略
		    utils.py:提供多用途小工具
		    weight_init.py:定義多種神經網路初始化方法

		hesds:
		    zero_head.py:實作ZeroHead模型
		
		losses:
		    gfocal_loss.py:實作QualityFocalLoss、DistributionFocalLoss、GIoULoss
		    distill_loss.py:定義用於知識蒸餾的特徵對齊損失函數

		necks:
		    giraffe_fpn_btn.py:實作GiraffeNeckV2類別，這是一種結合CSPStage模組、上採樣、下採樣與殘差連接的強化型 FPN
		
		config:
		    augmentations.py:定義訓練與測試的資料增強規則
		    base.py:主設定模組，包含完整的訓練與測試流程設定
		    paths_catalog.py:集中式資料集路徑設定

	    dataset:負責處理加載數據的模組

	    detector:
		detector.py:物件偵測主模型

	    structures:這個資料夾通常定義基本資料結構，支撐 bounding box、image tensor 等處理流程
		bounding_box.py:定義邊界框的
		boxlist_ops.py:定義多種運算函式
		image_list.py:是一種將多張不同尺寸的影像組成一個 tensor 的資料結構

	    utils:不是結構的程式碼，主要是輔助的工具 
		boxes.py:邊界框處理工具集
		checkpoint.py:模型存取工具
		debug_utils.py:提供訓練資料可視化功能，將圖像與其標註bounding boxes 和 label標示後儲存，用於偵錯模型輸入資料是否正確。
		demo_utils.py:包含模型推論時的後處理
		dist.py:分散式訓練工具
		imports.py:動態匯入模組的工具函式，用於支援自訂配置檔案載入
		logger.py:設定與管理訓練過程的日誌輸出
		metric.py:評估與訓練過程追蹤
		model_utils.py:模型所用的工具，像是參數和flops分析
		timer.py:時間記錄工具
		visualize.py:提供模型推論後的視覺化工具，將偵測結果的bounding boxes、分類標籤、信心分數畫在圖片上
		
	datasets:
	    放置資料的地方
	tools:
	    資料夾包含訓練、測試與結果分析相關的執行腳本，提供模型訓練train.py、驗證與測試eval.py、以及訓練過程與預測結果視覺化如 plot.py等功能

檔案 requirements:主要的環境套件

訓練指令
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 29501 train.py -f configs/damoyolo_tinynasL45_L.py

eval指令
python -m torch.distributed.launch --nproc_per_node=1 Multi/tools/eval.py -f configs/damoyolo_tinynasL45_L.py --ckpt workdirs/damoyolo_tinynasL45_L/0725/epoch_600_ckpt.pth