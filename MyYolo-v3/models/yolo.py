from torch import nn
from .backbone import DarkNet53
from .bottleneck import SppNet
from .header import Detector

# XXX:可使用yaml文件配置网络结构
# note:该类封装YOLOSpp主体网络, 定义forward方法实现特征提取
class YOLOSpp(nn.Module):
    def __init__(self, channel, num_classes, num_anchor) -> None:
        """初始化完整的YOLOSpp网络

        Parameters
        ----------
        channel : int
            输入图像通道数，一般为3
        num_classes : int
            待检测物体类别数
        num_anchor : int
            锚框数
        """
        super().__init__()
        self.yolo_body = DarkNet53(channel) # 骨干网络
        self.yolo_neck = SppNet() # 颈部网络
        self.yolo_head = Detector(num_classes, num_anchor) # 探测头

    def forward(self, x):
        """特征提取

        Parameters
        ----------
        x : tensor
            图片tensor格式

        Returns
        -------
        tuple
            small, middle, large尺度特征
        """
        backbone_feature = self.yolo_body(x) # 骨干网络下采样8, 16, 32倍, 输出tuple(small, middle, large)
        neck_feature = self.yolo_neck(*backbone_feature) # 颈部网络融合骨干网络特征, 输出tuple(small, middle, large)
        scal_small, scal_middle, scal_large = self.yolo_head(*neck_feature) # 探测头融合颈部网络特征
        return scal_small, scal_middle, scal_large # 返回融合特征