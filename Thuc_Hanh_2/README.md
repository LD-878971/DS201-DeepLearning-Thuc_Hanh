# DS201-DeepLearning
# DS201 · Lab 2 · Convolutional Neural Network (VinaFood21)

Huấn luyện và đánh giá 4 mô hình trên bộ dữ liệu **VinaFood21** (21 lớp món ăn Việt Nam):

| Bài | Model              | Ghi chú                          |
|-----|--------------------|----------------------------------|
| 1   | LeNet-5            | Từ đầu                           |
| 2   | GoogLeNet          | Inception module                 |
| 3   | ResNet-18          | Residual connection              |
| 4   | ResNet-50 pretrained | Fine-tune từ ImageNet (HuggingFace / torchvision) |

Optimizer: **Adam**  
Độ đo chính: **Accuracy, Precision-macro, Recall-macro, F1-macro**

## Cấu trúc thư mục
.
├── data.py              # transforms + DataLoader   
├── engine.py            # train_one_epoch + evaluate   
├── models.py            # LeNet5, GoogLeNet, ResNet18, PretrainedResNet50   
├── train.py             # script huấn luyện (chọn model bằng --model)   
├── evaluate.py          # script đánh giá  
├── requirements.txt  
└── README.md  

## Huấn luyện
- Bài 1 – LeNet-5   
python train.py --model lenet --data_root VinaFood21 --out_dir outputs_lenet   
- Bài 2 – GoogLeNet  
python train.py --model googlenet --data_root VinaFood21 --out_dir outputs_googlenet   
- Bài 3 – ResNet-18  
python train.py --model resnet18 --data_root VinaFood21 --out_dir outputs_resnet18   
- Bài 4 – ResNet-50 pretrained (nên dùng ảnh 224 và lr nhỏ hơn)   
python train.py --model resnet50 --data_root VinaFood21 \ --img_size 224 --lr 1e-4 --epochs 10 --out_dir outputs_resnet50   

## Đánh giá
python evaluate.py --model lenet     --data_root VinaFood21 --ckpt outputs_lenet/best.pt   
python evaluate.py --model googlenet --data_root VinaFood21 --ckpt outputs_googlenet/best.pt   
python evaluate.py --model resnet18  --data_root VinaFood21 --ckpt outputs_resnet18/best.pt    
python evaluate.py --model resnet50  --data_root VinaFood21 --ckpt outputs_resnet50/best.pt   
