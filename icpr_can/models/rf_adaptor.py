import torch.nn as nn

class S2VAdaptor(nn.Module):
    """ Semantic to Visual adaptation module"""
    def __init__(self, in_channels=512):
        """RF-Learning s2v adaptor
        Args:
            in_channels (int): input channels
        """
        super(S2VAdaptor, self).__init__()

        self.in_channels = in_channels  # 512

        # feature strengthen module, channel attention
        self.channel_inter = nn.Linear(self.in_channels, self.in_channels, bias=False)
        self.channel_bn = nn.BatchNorm1d(self.in_channels)
        self.channel_act = nn.ReLU(inplace=True)

    def forward(self, semantic):
        """
        Args:
            semantic (Torch.Tensor): recognition feature
        Returns:
            Torch.Tensor: strengthened recognition feature
        """
        semantic_source = semantic  # batch, channel, height, width

        # feature transformation
        semantic = semantic.squeeze(2).permute(0, 2, 1)  # batch, width, channel
        channel_att = self.channel_inter(semantic)       # batch, width, channel
        channel_att = channel_att.permute(0, 2, 1)       # batch, channel, width
        channel_bn = self.channel_bn(channel_att)        # batch, channel, width
        channel_att = self.channel_act(channel_bn)       # batch, channel, width

        # Feature enhancement
        channel_output = semantic_source * channel_att.unsqueeze(-2)  # batch, channel, 1, width

        return channel_output


class V2SAdaptor(nn.Module):
    """ Visual to Semantic adaptation module"""
    def __init__(self, in_channels=512, return_mask=False):
        """
        RF-Learning v2s adaptor
        Args:
            in_channels (Tensor): input channels
            return_mask (bool): whether to return attention mask
        """
        super(V2SAdaptor, self).__init__()

        # parameter initialization
        self.in_channels = in_channels
        self.return_mask = return_mask

        # output transformation
        self.channel_inter = nn.Linear(self.in_channels, self.in_channels, bias=False)
        self.channel_bn = nn.BatchNorm1d(self.in_channels)
        self.channel_act = nn.ReLU(inplace=True)

    def forward(self, visual):
        """
        Args:
            visual (Torch.Tensor): visual counting feature
        Returns:
            Torch.Tensor: strengthened visual counting feature
        """
        
        # Feature enhancement
        visual = visual.squeeze(2).permute(0, 2, 1)  # batch, width, channel
        channel_att = self.channel_inter(visual)     # batch, width, channel
        channel_att = channel_att.permute(0, 2, 1)   # batch, channel, width
        channel_bn = self.channel_bn(channel_att)    # batch, channel, width
        channel_att = self.channel_act(channel_bn)   # batch, channel, width
        # size alignment
        channel_output = channel_att.unsqueeze(-2)   # batch, width, channel

        if self.return_mask:
            return channel_output, channel_att
        return channel_output


class BasicBlock(nn.Module):
    """Res-net Basic Block"""
    expansion = 1

    def __init__(self, inplanes, planes,
                 stride=1, downsample=None,
                 norm_type='BN', **kwargs):
        """
        Args:
            inplanes (int): input channel
            planes (int): channels of the middle feature
            stride (int): stride of the convolution
            downsample (int): type of the down_sample
            norm_type (str): type of the normalization
            **kwargs (None): backup parameter
        """
        super(BasicBlock, self).__init__()
        self.conv1 = self._conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = self._conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def _conv3x3(self, in_planes, out_planes, stride=1):
        """
        Args:
            in_planes (int): input channel
            out_planes (int): channels of the middle feature
            stride (int): stride of the convolution
        Returns:
            nn.Module: Conv2d with kernel = 3
        """

        return nn.Conv2d(in_planes, out_planes,
                         kernel_size=3, stride=stride,
                         padding=1, bias=False)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input feature
        Returns:
            torch.Tensor: output feature of the BasicBlock
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

class RFAdaptor(nn.Module):
    def __init__(self,params):
        super().__init__()
        self.params = params
        self.with_neck_v2s = self.params['rfadaptor']['with_neck_v2s']
        self.with_neck_s2v = self.params['rfadaptor']['with_neck_s2v']
        
        self.out_channel = self.params['counting_decoder']['in_channel'] * 2
        self.output_channel_block = [int(self.out_channel / 4), int(self.out_channel / 2),
                                     self.out_channel, int(self.out_channel / 2)]
        
        
        block = BasicBlock
        layers = [1, 2, 5, 3]
        self.inplanes = int(self.out_channel // 2)

        self.relu = nn.ReLU(inplace=True)

        self.layer3 = self._make_layer(block, self.output_channel_block[2], layers[2], stride=1)
        self.conv3 = nn.Conv2d(self.output_channel_block[2],
                               self.output_channel_block[2],
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.output_channel_block[2])

        self.layer4 = self._make_layer(block, self.output_channel_block[3], layers[3], stride=1)
        self.conv4_1 = nn.Conv2d(self.output_channel_block[3],
                                 self.output_channel_block[3],
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(self.output_channel_block[3])
        self.conv4_2 = nn.Conv2d(self.output_channel_block[3],
                                 self.output_channel_block[3],
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4_2 = nn.BatchNorm2d(self.output_channel_block[3])

        self.inplanes = int(self.out_channel // 2)

        self.v_layer3 = self._make_layer(block, self.output_channel_block[2], layers[2], stride=1)
        self.v_conv3 = nn.Conv2d(self.output_channel_block[2],
                                 self.output_channel_block[2],
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.v_bn3 = nn.BatchNorm2d(self.output_channel_block[2])

        self.v_layer4 = self._make_layer(block, self.output_channel_block[3], layers[3], stride=1)
        self.v_conv4_1 = nn.Conv2d(self.output_channel_block[3],
                                   self.output_channel_block[3],
                                   kernel_size=3, stride=1, padding=1, bias=False)
        self.v_bn4_1 = nn.BatchNorm2d(self.output_channel_block[3])
        self.v_conv4_2 = nn.Conv2d(self.output_channel_block[3],
                                   self.output_channel_block[3],
                                   kernel_size=3, stride=1, padding=1, bias=False)
        self.v_bn4_2 = nn.BatchNorm2d(self.output_channel_block[3])
        
        
        self.neck_v2s = V2SAdaptor(in_channels=self.output_channel_block[3])
        self.neck_s2v = S2VAdaptor(in_channels=self.output_channel_block[3])
        
        # init model
        self.apply(self._init_weights)
        
    def _init_weights(self,m):
        # weight initialization
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
    def _make_layer(self, block, planes, blocks, stride=1):
        """
        Args:
            block (block): convolution block
            planes (int): input channels
            blocks (list): layers of the block
            stride (int): stride of the convolution
        Returns:
            nn.Sequential: the combination of the convolution block
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
        
    def forward(self, cnn_features):
        x_1 = cnn_features
        
        # ====== visual branch =====
        # visual stage 1
        v_x = self.v_layer3(x_1)
        v_x = self.v_conv3(v_x)
        v_x = self.v_bn3(v_x)
        v_x = self.relu(v_x)

        # visual stage 2
        v_x = self.v_layer4(v_x)
        v_x = self.v_conv4_1(v_x)
        v_x = self.v_bn4_1(v_x)
        v_x = self.relu(v_x)
        v_x = self.v_conv4_2(v_x)
        v_x = self.v_bn4_2(v_x)
        v_x = self.relu(v_x)

        # ====== semantic branch =====
        x = self.layer3(x_1)
        x = self.conv3(x)
        x = self.bn3(x)
        x_2 = self.relu(x)

        x = self.layer4(x_2)
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.relu(x)
        
        # neck network
        batch, source_channels, v_source_height, v_source_width = v_x.size()
        visual_feature = v_x.view(batch, source_channels, 1, v_source_height * v_source_width)
        rcg_feature = x.view(batch, source_channels, 1, v_source_height * v_source_width)
       
        if self.with_neck_v2s:
            v2s = self.neck_v2s(visual_feature).view(batch, source_channels, v_source_height , v_source_width)
            v_rcg_feature = x * v2s
        else:
            v_rcg_feature = x
        
        if self.with_neck_s2v:
            s2v = self.neck_s2v(rcg_feature).view(batch, source_channels, v_source_height , v_source_width)
            v_visual_feature = v_x + s2v
        else:
            v_visual_feature = v_x

        return v_rcg_feature, v_visual_feature