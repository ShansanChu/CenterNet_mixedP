from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

try:
  from external.nms import soft_nms_39
except:
  print('NMS not imported! If you need it,'
        ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
from models.decode import multi_pose_decode
from models.utils import flip_tensor, flip_lr_off, flip_lr
from utils.image import get_affine_transform
from utils.post_process import multi_pose_post_process
from utils.debugger import Debugger
from torch import finfo
from .base_detector import BaseDetector

class MultiPoseDetector(BaseDetector):
  def __init__(self, opt):
    super(MultiPoseDetector, self).__init__(opt)
    self.flip_idx = opt.flip_idx
    self.model.apply(self.custm_half)
    print('Before fusion model is {}'.format(self.model))
    fusion_list=self.fusion_pattern()
    #fusion model bn with conv
    self.model=torch.quantization.fuse_modules(self.model,fusion_list)
    print('MODEL is {}'.format(self.model))
    #get quantization config
    config=torch.quantization.get_default_qconfig("fbgemm")
    self.model.qconfig=config
    #prepare model for quantization
    torch.quantization.prepare_qat(self.model,inplace=True)
    self.add_output_check_hooks()
    self.model.eval()
    #self.model.apply(custm_half)

  def qat_train(self):
      """
      train process calibration
      """
  def fusion_pattern(self):
      """
      fusion patterns detecter with module name
      """
      fusion_list=[]
      #two passes method is used
      map_bn_conv=dict()
      for name,module in self.model.named_modules():
          if 'conv' in name:
              map_bn_conv[name]=None
          elif 'bn' in name and name.replace('bn','conv') in map_bn_conv:
              map_bn_conv[name.replace('bn','conv')]=name
              if isinstance(module,torch.nn.modules.container.Sequential):
                  print(name,name.replace('bn','conv'))
                  print('*'*20)
                  print(module)
                  print('='*20)
              else:fusion_list.append([name.replace('bn','conv'),name])
      print('fusion list is ',fusion_list)
      return fusion_list
  def custm_half(self,module):
      """
      customerize clamp fp32 weight as well as buffer to fp16.
      """
      for para in module.named_parameters():
          para[1].data=torch.clamp(para[1],finfo(torch.float16).min,finfo(torch.float16).max)
      for buff in module.named_buffers():
          buff[1].data=torch.clamp(buff[1],finfo(torch.float16).min,finfo(torch.float16).max)

      

  def pre_hook(self,module,input):
      #print(input[0])
      #print(type(input[0]))
      if isinstance(input,tuple):
          result=[]
          for i in range(len(input)):
              result.append(input[i].half())
          return tuple(result)
      return input[0].half()
  def tuple_range(self,tuples):
      outR=False
      for i in range(len(tuples)):
          outR=outR and torch.any(torch.gt(tuples[i],finfo(torch.float16).max))
          outR=outR and torch.any(torch.lt(tuples[i],finfo(torch.float16).min))
      return outR
  def forward_hooks_cast(self,module,input,output):
      if isinstance(input,tuple):
          if self.tuple_range(input):
              print('non_bn!!!!!ATTENTION!!!!!current layer input out range of fp16')
      elif torch.any(torch.gt(input,finfo(torch.float16).max)) or torch.any(torch.lt(input,finfo(torch.float16).min)):
          print('non_bn!!!!!ATTENTION!!!!!current layer input out range of fp16')
      if torch.any(torch.gt(output,finfo(torch.float16).max)) or torch.any(torch.lt(output,finfo(torch.float16).min)):
          print('non_bn!!!!!!ATTENTION!!!!!!current layer output is out range of fp16')
      return output.float()
  def forward_hooks_checkR(self,module,input,output):
      if isinstance(input,tuple):
          if self.tuple_range(input):
              print('non_bn!!!!!ATTENTION!!!!!current layer input out range of fp16')
      elif torch.any(torch.gt(input,finfo(torch.float16).max)) or torch.any(torch.lt(input,finfo(torch.float16).min)):
          print('!!!!!ATTENTION!!!!!current layer input out range of fp16')
      if torch.any(torch.gt(output,finfo(torch.float16).max)) or torch.any(torch.lt(output,finfo(torch.float16).min)):
          print('!!!!!!ATTENTION!!!!!!current layer output is out range of fp16')
      return output

  def check_nan_inf(self,module,input,output):
      """
      check if nan or inf present in output
      """
      non_num=False
      if not isinstance(output,tuple):
          non_num=torch.any(torch.isinf(output)) or torch.any(torch.isnan(output))
      else:
          print(output[1])
      if non_num:
          print('ATTENTION!! NAN or inf present')
  def add_output_check_hooks(self):
    # check for the root modules
    print('ATTENTION!!!')
    for name,module in self.model.named_modules():
        if not list(module.named_children()): #and (isinstance(module,nn.Conv2d) or isinstance(module,nn.Linear)):
            print(name)
            print([(name_para,para.shape) for name_para,para in module.named_parameters()])
            print('==========================================')
            #if True:
            if name not in ['cnvs.0.bn','cnvs.1.bn']:
            #if not isinstance(module,torch.nn.BatchNorm2d):
                #isinstance(module,torch.nn.Conv2d) or isinstance(module,torch.nn.ReLU):# isinstance(module,torch.nn.BatchNorm2d) :
                print('**',name)
                module.half()
                module.register_forward_pre_hook(self.pre_hook)
                module.register_forward_hook(self.forward_hooks_cast)
            #module.register_forward_hook(self.check_nan_inf)#self.min_forward_hooks)
            else:
                #module.register_forward_hook(self.forward_hooks_checkR)
                print(module)
                if not isinstance(module,torch.nn.BatchNorm2d):continue
                print(module.running_mean)
                print(module.running_var)
                print('^'*5,name)
                if torch.any(torch.gt(module.running_var,torch.finfo(torch.float16).max)):
                    print('ATTENTION!!!!')
                module.half()
                module.register_forward_hook(self.forward_hooks_cast)
                module.register_forward_pre_hook(self.pre_hook)

  def process(self, images, return_time=False):
    with torch.no_grad():
      torch.cuda.synchronize()
      output = self.model(images)[-1]
      output['hm'] = output['hm'].sigmoid_()
      if self.opt.hm_hp and not self.opt.mse_loss:
        output['hm_hp'] = output['hm_hp'].sigmoid_()

      reg = output['reg'] if self.opt.reg_offset else None
      hm_hp = output['hm_hp'] if self.opt.hm_hp else None
      hp_offset = output['hp_offset'] if self.opt.reg_hp_offset else None
      torch.cuda.synchronize()
      forward_time = time.time()
      
      if self.opt.flip_test:
        output['hm'] = (output['hm'][0:1] + flip_tensor(output['hm'][1:2])) / 2
        output['wh'] = (output['wh'][0:1] + flip_tensor(output['wh'][1:2])) / 2
        output['hps'] = (output['hps'][0:1] + 
          flip_lr_off(output['hps'][1:2], self.flip_idx)) / 2
        hm_hp = (hm_hp[0:1] + flip_lr(hm_hp[1:2], self.flip_idx)) / 2 \
                if hm_hp is not None else None
        reg = reg[0:1] if reg is not None else None
        hp_offset = hp_offset[0:1] if hp_offset is not None else None
      
      dets = multi_pose_decode(
        output['hm'], output['wh'], output['hps'],
        reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K)

    if return_time:
      return output, dets, forward_time
    else:
      return output, dets

  def post_process(self, dets, meta, scale=1):
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets = multi_pose_post_process(
      dets.copy(), [meta['c']], [meta['s']],
      meta['out_height'], meta['out_width'])
    for j in range(1, self.num_classes + 1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 39)
      # import pdb; pdb.set_trace()
      dets[0][j][:, :4] /= scale
      dets[0][j][:, 5:] /= scale
    return dets[0]

  def merge_outputs(self, detections):
    results = {}
    results[1] = np.concatenate(
        [detection[1] for detection in detections], axis=0).astype(np.float32)
    if self.opt.nms or len(self.opt.test_scales) > 1:
      soft_nms_39(results[1], Nt=0.5, method=2)
    results[1] = results[1].tolist()
    return results

  def debug(self, debugger, images, dets, output, scale=1):
    dets = dets.detach().cpu().numpy().copy()
    dets[:, :, :4] *= self.opt.down_ratio
    dets[:, :, 5:39] *= self.opt.down_ratio
    img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(((
      img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
    pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
    debugger.add_blend_img(img, pred, 'pred_hm')
    if self.opt.hm_hp:
      pred = debugger.gen_colormap_hp(
        output['hm_hp'][0].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hmhp')
  
  def show_results(self, debugger, image, results):
    debugger.add_img(image, img_id='multi_pose')
    for bbox in results[1]:
      if bbox[4] > self.opt.vis_thresh:
        debugger.add_coco_bbox(bbox[:4], 0, bbox[4], img_id='multi_pose')
        debugger.add_coco_hp(bbox[5:39], img_id='multi_pose')
    debugger.show_all_imgs(pause=self.pause)
