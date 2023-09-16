# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from typing import Optional, Sequence

import mmcv
import numpy as np
from mmengine.fileio import get
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.utils import mkdir_or_exist
from mmengine.visualization import Visualizer

from mmdet3d.registry import HOOKS
from mmdet3d.structures import Det3DDataSample


@HOOKS.register_module()
class Det3DVisualizationHook(Hook):
    """Detection Visualization Hook. Used to visualize validation and testing
    process prediction results.

    In the testing phase:

    1. If ``show`` is True, it means that only the prediction results are
        visualized without storing data, so ``vis_backends`` needs to
        be excluded.
    2. If ``test_out_dir`` is specified, it means that the prediction results
        need to be saved to ``test_out_dir``. In order to avoid vis_backends
        also storing data, so ``vis_backends`` needs to be excluded.
    3. ``vis_backends`` takes effect if the user does not specify ``show``
        and `test_out_dir``. You can set ``vis_backends`` to WandbVisBackend or
        TensorboardVisBackend to store the prediction result in Wandb or
        Tensorboard.

    Args:
        draw (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        interval (int): The interval of visualization. Defaults to 50.
        score_thr (float): The threshold to visualize the bboxes
            and masks. Defaults to 0.3.
        show (bool): Whether to display the drawn image. Default to False.
        vis_task (str): Visualization task. Defaults to 'mono_det'.
        wait_time (float): The interval of show (s). Defaults to 0.
        test_out_dir (str, optional): directory where painted images
            will be saved in testing process.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(self,
                 draw: bool = False,
                 interval: int = 50,
                 score_thr: float = 0.3,
                 show: bool = False,
                 vis_task: str = 'mono_det',
                 wait_time: float = 0.,
                 test_out_dir: Optional[str] = None,
                 backend_args: Optional[dict] = None,
                 draw_interval: Optional[int] = 200,
                 draw_stop_idx: Optional[int] = 6000,
                 log_table : Optional[bool] = False
                 ):
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.interval = interval
        self.score_thr = score_thr
        self.show = show
        if self.show:
            # No need to think about vis backends.
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')
        self.vis_task = vis_task

        self.wait_time = wait_time
        self.backend_args = backend_args
        self.draw = draw
        self.test_out_dir = test_out_dir
        self._test_index = 0
        self.draw_interval = draw_interval
        self.draw_stop_idx = draw_stop_idx
        self.log_table = log_table

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[Det3DDataSample]) -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`DetDataSample`]]): A batch of data samples
                that contain annotations and predictions.
        """
        if self.draw is False:
            return

        # There is no guarantee that the same batch of images
        # is visualized for each evaluation.
        total_curr_iter = runner.iter + batch_idx

        data_input = dict()

        # Visualize only the first data
        if 'img_path' in outputs[0]:
            img_path = outputs[0].img_path
            img_bytes = get(img_path, backend_args=self.backend_args)
            img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
            data_input['img'] = img

        if 'lidar_path' in outputs[0]:
            lidar_path = outputs[0].lidar_path
            num_pts_feats = outputs[0].num_pts_feats
            pts_bytes = get(lidar_path, backend_args=self.backend_args)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
            points = points.reshape(-1, num_pts_feats)
            data_input['points'] = points

        if total_curr_iter % self.interval == 0:
            self._visualizer.add_datasample(
                f'vis_bbox_3d_idx={outputs[0].sample_idx}',
                data_input,
                data_sample=outputs[0],
                show=self.show,
                vis_task=self.vis_task,
                wait_time=self.wait_time,
                pred_score_thr=self.score_thr,
                step=total_curr_iter)

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[Det3DDataSample]) -> None:
        """Run after every testing iterations.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`DetDataSample`]): A batch of data samples
                that contain annotations and predictions.
        """
        if self.draw is False:
            return

        if self.test_out_dir is not None:
            self.test_out_dir = osp.join(runner.work_dir, runner.timestamp,
                                         self.test_out_dir)
            mkdir_or_exist(self.test_out_dir)

        for data_sample in outputs:
            self._test_index += 1

        # Modify by Yuxi Qian, add mutli-camera visualization and log into wandb self.table
        import wandb

        self.table = None
        
        if all(data_sample.sample_idx % self.draw_interval != 0 for data_sample in outputs) and self.log_table:
            return

        out_file = None
        if self.test_out_dir is not None:
            out_file = osp.basename(img_path)
            out_file = osp.join(self.test_out_dir, out_file)
        
        for data_sample in outputs:
            self._test_index += 1
            if data_sample.sample_idx % self.draw_interval != 0:
                continue
            # Dictionary to store wandb image lists with camera_key as keys
            camera_key_vis_images_dict = {}
            idxs = ''
            # Multi-camera scenario
            
            if ('img_path' in data_sample and isinstance(data_sample.img_path, list)) or ('lidar_path' in data_sample and isinstance(data_sample.lidar_path, list)):  
                if not ('img_path' in data_sample):
                    assert False
                for i, single_img_path in enumerate(data_sample.img_path):
                    data_input = dict()

                    img_bytes = get(single_img_path, backend_args=self.backend_args)
                    img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                    data_input['img'] = img

                    # Handling lidar_path for both list and single scenario
                    if 'lidar_path' in data_sample:
                        if isinstance(data_sample.lidar_path, list):
                            lidar_path = data_sample.lidar_path[i]
                        else:
                            lidar_path = data_sample.lidar_path

                        num_pts_feats = data_sample.num_pts_feats
                        pts_bytes = get(lidar_path, backend_args=self.backend_args)
                        points = np.frombuffer(pts_bytes, dtype=np.float32)
                        points = points.reshape(-1, num_pts_feats)
                        data_input['points'] = points

                    camera_key = [part for part in single_img_path.split('/') if "CAM" in part][0]

                    # Check for the camera fields
                    for field in ['lidar2cam', 'cam2img', 'lidar2img']:
                        if hasattr(data_sample, field):
                            value = data_sample.get(field)
                            if isinstance(value, list):
                                data_input[field] = value[i]

                            else:
                                assert False
                                data_input[field] = value
                    gt, pred = self._visualizer.add_datasample(
                        f'vis_bbox_3d_idx={data_sample.sample_idx}_{camera_key}',
                        data_input,
                        data_sample=data_sample,
                        show=self.show,
                        vis_task=self.vis_task,
                        wait_time=self.wait_time,
                        pred_score_thr=self.score_thr,
                        out_file=out_file,
                        step=self._test_index)
                    # Get the camera key from the img_path

                    # Convert np arrays to wandb images and add to the list
                    camera_key_visualization_list = [
                        wandb.Image(gt['img'], caption=f"{camera_key}_bbox_3d_gt"),
                        wandb.Image(pred['img'], caption=f"{camera_key}_bbox_3d_pred")
                    ]
                    
                    # Store the wandb images list in the dictionary
                    camera_key_vis_images_dict[camera_key] = camera_key_visualization_list
                    
                # If camera_key_vis_images_dict isn't empty, proceed
                if camera_key_vis_images_dict:
                    # If self.table hasn't been created yet
                    if not self.table:
                        # Base columns for self.table
                        columns = ["sample_idx"] + list(camera_key_vis_images_dict.keys()) + ["paths_to_images"]
                        self.table = wandb.Table(columns=columns)

                    # Create row data
                    row_data = [data_sample.sample_idx]
                    for key in self.table.columns:
                        if key == "sample_idx":
                            continue
                        elif key == "paths_to_images":
                            row_data.append([osp.basename(path) for path in data_sample.img_path])
                        else:
                            row_data.append(camera_key_vis_images_dict.get(key, None))

                # Add the row data to the self.table
                idxs += str(data_sample.sample_idx) + ','
                
                print(f'current_sample_idx={data_sample.sample_idx}')
                if data_sample.sample_idx >= 6000:
                    self.log_table = True
                    
                self.table.add_data(*row_data)
                                  
            # Single camera scenario
            elif ('img_path' in data_sample and isinstance(data_sample.img_path, str)) or ('lidar_path' in data_sample and isinstance(data_sample.lidar_path, str)):
                data_input = dict()
                
                if 'img_path' in data_sample:
                    img_path = data_sample.img_path
                    img_bytes = get(img_path, backend_args=self.backend_args)
                    img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                    data_input['img'] = img

                if 'lidar_path' in data_sample:
                    lidar_path = data_sample.lidar_path
                    num_pts_feats = data_sample.num_pts_feats
                    pts_bytes = get(lidar_path, backend_args=self.backend_args)
                    points = np.frombuffer(pts_bytes, dtype=np.float32)
                    points = points.reshape(-1, num_pts_feats)
                    data_input['points'] = points


                self._visualizer.add_datasample(
                    f'test_vis_bbox_3d_idx={data_sample.sample_idx}',
                    data_input,
                    data_sample=data_sample,
                    show=self.show,
                    vis_task=self.vis_task,
                    wait_time=self.wait_time,
                    pred_score_thr=self.score_thr,
                    out_file=out_file,
                    step=self._test_index)
        
        if self.table and self.log_table:
            name = 'test_visualization_batch' # _idx=' +  idxs.rstrip(',')
            print('log wandb test visualization table')
            wandb.log({name: self.table})
            # End of modification
