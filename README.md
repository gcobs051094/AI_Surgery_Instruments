# 手術器械分類
29類台灣常見手術器械分類
關鍵字: image segmentation、surgery instruments


## 成果
1. 當前類別29類
    *    類別詳情請見 utils.py
    *    訓練資料從roborflow抓取(抓處後記得刪除954.jpg)


## 自建資料庫
[手術器械_Segmentation_特定資料+Ori ](https://universe.roboflow.com/doccamsurgerytool/-_segmentation_-ori)
[手術器械_DSLR_Segmentation](https://universe.roboflow.com/doccamsurgerytool/-_dslr_segmentation)


## 問題
1. 目前偵測穩定度不足、缺乏多樣化資料(角度、亮度、背景場景)
2. 詳細可見手術器械demo.pptx

## 未來執行
1. 日後Model要重新訓練(增加padding保持比例) 
2. 補足問題第1點