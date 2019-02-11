#res
#sudo python evaluate.py
#Rank@1:0.879454 Rank@5:0.956651 Rank@10:0.971793 mAP:0.707328
#python evaluate_rerank.py
#top1:0.907957 top5:0.948337 top10:0.961995 mAP:0.855202

#pcb:
#sudo python evaluate_gpu.py
#Rank@1:0.894596 Rank@5:0.959620 Rank@10:0.974762 mAP:0.713481
#sudo python evaluate_rerank.py
#top1:0.916568 top5:0.953088 top10:0.964667 mAP:0.858806

#dense-erasing=0.5
#sudo python evaluate_gpu.py
#Rank@1:0.896081 Rank@5:0.960808 Rank@10:0.974169 mAP:0.720840
#sudo python evaluate_rerank.py
#top1:0.919834 top5:0.958135 top10:0.969715 mAP:0.868861

#pcb-erasing=0.5
#sudo python evaluate_gpu.py
#Rank@1:0.924584 Rank@5:0.970012 Rank@10:0.979810 mAP:0.774712
#sudo python evaluate_rerank.py
#top1:0.944181 top5:0.966152 top10:0.973872 mAP:0.901758

#res-erasing=0.5
#sudo python evaluate_gpu.py
#Rank@1:0.886876 Rank@5:0.956948 Rank@10:0.976247 mAP:0.714406
#sudo python evaluate_rerank.py
#top1:0.911817 top5:0.951603 top10:0.964074 mAP:0.863305

#dense
#sudo python evaluate_gpu.py
#Rank@1:0.879454 Rank@5:0.956948 Rank@10:0.972387 mAP:0.715160
# sudo python evaluate_rerank.py
#top1:0.904691 top5:0.948931 top10:0.963480 mAP:0.852221

#res -eraseing=0.5 color_jitter
#sudo python evaluate_gpu.py
#Rank@1:0.854810 Rank@5:0.949822 Rank@10:0.966746 mAP:0.686222
#sudo python evaluate_rerank.py
#top1:0.887470 top5:0.937648 top10:0.952791 mAP:0.846748

#res color_jitter ps:这块是不能放在一起比较的，因为是两种完全不同的数据增强的方式
#sudo python evaluate_gpu.py
#Rank@1:0.848575 Rank@5:0.945071 Rank@10:0.963183 mAP:0.643818
#sudo python evaluate_rerank.py
#top1:0.876781 top5:0.934382 top10:0.951010 mAP:0.823771

#pcb part=4 
#sudo python train.py --gpu_ids 0 --name PCB_part_4__16 --train_all --batchsize 16 --PCB
#sudo python evaluate_gpu.py
#0.889846 Rank@5:0.956057 Rank@10:0.970012 mAP:0.681990
#sudo python evaluate_rerank.py
#top1:0.910333 top5:0.949822 top10:0.963183 mAP:0.843482

#pcb_part_6_concat
#sudo python evaluate_gpu.py
#Rank@1:0.835808 Rank@5:0.933789 Rank@10:0.959323 mAP:0.595832
#sudo python evaluate_rerank.py
#top1:0.873219 top5:0.930226 top10:0.948931 mAP:0.784707

#pytorch_result_RES_concat_16.mat
#sudo python evaluate_gpu.py
#Rank@1:0.826900 Rank@5:0.938242 Rank@10:0.964074 mAP:0.632857
#sudo python evaluate_rerank.py
#top1:0.865796 top5:0.924881 top10:0.942993 mAP:0.811186

#batch-hard triplet loss – original semi-hard triplet loss的改进版
#softmax loss
#fa、fp、fn -- 分别是从anchor、positive、negative样本中拿到的特征；
#将softmax loss应用在降维之前的2048维特征（LG/softmax、LP2/sotfmax、LP3/softmax），以及应用在Part-2和Part-3分支降维之后的256维局部特征（LP2/#softmax0、LP2/softmax1和LP3/softmax0、LP3/softmax1、LP3/softmax2）。对所有分支降维成256之后（非局部），也计算batch-hard triplet loss（LG/triplet、LP2/triplet、LP3/triplet）。
#此外，不会在局部特征上使用triplet loss。 由于不对齐或其他问题，局部特征的内容可能会发生巨大变化，这使得triplet loss往往会在训练期间破坏模型。
#在本文中，我们提出了多粒度网络（MGN），这是一种新颖的多分支深度网络，用于学习行人重识别任务中的判别表示。 MGN中的每个分支都用特定的粒度分区来学习全局或
#局部表示。该方法直接在水平分割的特征条上学习局部特征，是完全端到端的，并且不引入区域建议或姿势估计等局部定位操作。
#在测试时，将256维的所有特征串联作为最终特征，无需使用2048维的特征，使用欧氏距离作为两个行人相似度的度量。
#三个分支最后一层特征都会进行一次全局MaxPooling操作global max-pooling（GMP），然后再将特征由2048维降为256维。最后256维特征同时用于Softmax Loss与#Triplet Loss计算。另外，作者在2048维的地方添加一个额外的全局Softmax Loss，该任务将帮助网络更全面学习图片全局特征。

#pytorch_result_RES_concat_16.mat +BN
#sudo python evaluate_gpu.py
#Rank@1:0.864311 Rank@5:0.947150 Rank@10:0.966746 mAP:0.663378
#sudo python evaluate_rerank.py
#top1:0.890736 top5:0.938836 top10:0.952494 mAP:0.825902

#attention + res
#sudo python evaluate_gpu.py
#Rank@1:0.886888 Rank@5:0.965962 Rank@10:0.973183 mAP:0.727323
#sudo python evaluate_rerank.py
#top1:0.924893 top5:0.948836 top10:0.965166 mAP:0.868608

#channel attention +res
##sudo python evaluate_gpu.py
#Rank@1:0.873812 Rank@5:0.955760 Rank@10:0.968824 mAP:0.687599
#sudo python evaluate_rerank.py
#top1:0.899347 top5:0.948040 top10:0.961401 mAP:0.850413

#res
#sudo python evaluate_gpu.py
#Rank@1:0.862827 Rank@5:0.950416 Rank@10:0.967933 mAP:0.674622
#sudo python evaluate_rerank.py
#top1:0.891330 top5:0.939727 top10:0.955166 mAP:0.840004
