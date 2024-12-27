import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from configs.state_vec import STATE_VEC_IDX_MAPPING
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from models.multimodal_encoder.t5_encoder import T5Embedder
from models.rdt_runner import RDTRunner


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

sys.path.append(PARENT_DIR)
print(f"current_dir: {CURRENT_DIR}")
print(f"parent_dir: {PARENT_DIR}")

from utils import (
    convert_rotation_matrix_to_euler,
    compute_rotation_matrix_from_ortho6d,
    convert_euler_to_rotation_matrix,
    compute_ortho6d_from_rotation_matrix,
)

# The indices that the raw vector should be mapped to in the unified action vector
UR5_STATE_INDICES = (
    [STATE_VEC_IDX_MAPPING["gripper_open"]]
    + [STATE_VEC_IDX_MAPPING["eef_pos_x"]]
    + [STATE_VEC_IDX_MAPPING["eef_pos_y"]]
    + [STATE_VEC_IDX_MAPPING["eef_pos_z"]]
    + [STATE_VEC_IDX_MAPPING["eef_angle_0"]]
    + [STATE_VEC_IDX_MAPPING["eef_angle_1"]]
    + [STATE_VEC_IDX_MAPPING["eef_angle_2"]]
    + [STATE_VEC_IDX_MAPPING["eef_angle_3"]]
    + [STATE_VEC_IDX_MAPPING["eef_angle_4"]]
    + [STATE_VEC_IDX_MAPPING["eef_angle_5"]]
)
print(f"UR5_STATE_INDICES: {UR5_STATE_INDICES}")


# Create the RDT model
def create_model(args, **kwargs):
    model = RoboticDiffusionTransformerModel(args, **kwargs)
    pretrained = kwargs.get("pretrained", None)
    if pretrained is not None and os.path.isfile(pretrained):
        model.load_pretrained_weights(pretrained)
    return model


class RoboticDiffusionTransformerModel(object):
    """A wrapper for the RDT model, which handles
    1. Model initialization
    2. Encodings of instructions
    3. Model inference
    """

    def __init__(
        self,
        args,
        device_1 ="cuda",
        device_2 = None,
        dtype=torch.bfloat16,
        image_size=None,
        control_frequency=30,
        pretrained=None,
        pretrained_text_encoder_name_or_path=None,
        pretrained_vision_encoder_name_or_path=None,
    ):
        self.args = args
        self.dtype = dtype
        self.image_size = image_size
        self.device_1 = device_1
        self.device_2 = device_2
        self.control_frequency = control_frequency
        # We do not use the text encoder due to limited GPU memory
        self.text_tokenizer, self.text_model = self.get_text_encoder(
            pretrained_text_encoder_name_or_path
        )
        self.image_processor, self.vision_model = self.get_vision_encoder(
            pretrained_vision_encoder_name_or_path
        )
        self.policy = self.get_policy(pretrained)

        self.reset()

    def get_policy(self, pretrained):
        """Initialize the model."""
        # Initialize model with arguments
        if pretrained is None or os.path.isfile(pretrained):
            img_cond_len = (
                self.args["common"]["img_history_size"]
                * self.args["common"]["num_cameras"]
                * self.vision_model.num_patches
            )

            _model = RDTRunner(
                action_dim=self.args["common"]["state_dim"],
                pred_horizon=self.args["common"]["action_chunk_size"],
                config=self.args["model"],
                lang_token_dim=self.args["model"]["lang_token_dim"],
                img_token_dim=self.args["model"]["img_token_dim"],
                state_token_dim=self.args["model"]["state_token_dim"],
                max_lang_cond_len=self.args["dataset"]["tokenizer_max_length"],
                img_cond_len=img_cond_len,
                img_pos_embed_config=[
                    # No initial pos embed in the last grid size
                    # since we've already done in ViT
                    (
                        "image",
                        (
                            self.args["common"]["img_history_size"],
                            self.args["common"]["num_cameras"],
                            -self.vision_model.num_patches,
                        ),
                    ),
                ],
                lang_pos_embed_config=[
                    # Similarly, no initial pos embed for language
                    ("lang", -self.args["dataset"]["tokenizer_max_length"]),
                ],
                dtype=self.dtype,
            )
        else:
            _model = RDTRunner.from_pretrained(pretrained)

        return _model

    def get_text_encoder(self, pretrained_text_encoder_name_or_path):
        text_embedder = T5Embedder(
            from_pretrained=pretrained_text_encoder_name_or_path,
            model_max_length=self.args["dataset"]["tokenizer_max_length"],
            device_1 = self.device_1,
            device_2 = self.device_2
        )
        tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model
        return tokenizer, text_encoder

    def get_vision_encoder(self, pretrained_vision_encoder_name_or_path):
        vision_encoder = SiglipVisionTower(
            vision_tower=pretrained_vision_encoder_name_or_path, args=None
        )
        image_processor = vision_encoder.image_processor
        return image_processor, vision_encoder

    def reset(self):
        """Set model to evaluation mode."""
        device_1 = self.device_1
        device_2 = self.device_2
        weight_dtype = self.dtype
        self.policy.eval()
        self.text_model.eval()
        self.vision_model.eval()

        self.policy = self.policy.to(device_2, dtype=weight_dtype)
        # self.text_model = self.text_model.to(device_1, dtype=weight_dtype)
        self.vision_model = self.vision_model.to(device_1, dtype=weight_dtype)

    def load_pretrained_weights(self, pretrained=None):
        if pretrained is None:
            return
        print(f"Loading weights from {pretrained}")
        filename = os.path.basename(pretrained)
        if filename.endswith(".pt"):
            checkpoint = torch.load(pretrained)
            self.policy.load_state_dict(checkpoint["module"])
        elif filename.endswith(".safetensors"):
            from safetensors.torch import load_model
            load_model(self.policy, pretrained)
        else:
            raise NotImplementedError(f"Unknown checkpoint format: {pretrained}")

    def encode_instruction(self, instruction):
        """Encode string instruction to latent embeddings.

        Args:
            instruction: a string of instruction
            device: a string of device
        
        Returns:
            pred: a tensor of latent embeddings of shape (text_max_length, 512)
        """
        tokens = self.text_tokenizer(
            instruction, return_tensors="pt",
            padding="longest",
            truncation=True
        )["input_ids"]

        tokens = tokens.view(1, -1)
        print(f"Input tokens device: {tokens.device}")
        tokens = tokens.to(self.text_model.device)
        print(f"Input tokens send to model device: {tokens.device}")
        print(f"Model parameters device: {self.text_model.device}")

        with torch.no_grad():
            pred = self.text_model(tokens).last_hidden_state.detach()
        print(f"out device: {pred.device}")
        return pred

    
    def _convert_ee_pose_to_6d(self, input):
        # Ensure the input tensor is in float32 before converting to numpy
        input = input.to(torch.float32).cpu().numpy()
    
        eef_pos = input[:, :, :3]
        eef_ang = convert_euler_to_rotation_matrix(input[:, :, 3:6].reshape(-1, 3))
        eef_ang = compute_ortho6d_from_rotation_matrix(eef_ang)
        eef_ang = eef_ang.reshape(1, 1, -1)
        gripper_open = (input[:, :, 12] + 1) / 2
        gripper_open = gripper_open[..., np.newaxis]
    
        output = np.concatenate([gripper_open, eef_pos, eef_ang], axis=2)
        output = torch.from_numpy(output).to(torch.bfloat16).cuda()
        return output
    
    def _convert_6d_to_ee_pose(self, input):
        d = input.device
        input = input.float().cpu().numpy()
        # print(f"action input shape: {input.shape}")
        gripper_open = input[:, :, 0] * 2 - 1
        gripper_open = gripper_open[..., np.newaxis]
        eef_pos = input[:, :, 1:4]
        eef_ang = compute_rotation_matrix_from_ortho6d(np.squeeze(input[:, :, 4:10]))
        eef_ang = convert_rotation_matrix_to_euler(eef_ang)
        eef_ang = np.expand_dims(eef_ang, axis=0)
        # print(
        #     f"eef_pos shape: {eef_pos.shape}, \
        #     eef_ang shape: {eef_ang.shape}, \
        #     gripper_open shape: {gripper_open.shape}"
        # )
        output = np.concatenate([eef_ang,eef_pos, gripper_open], axis=2)
        output = torch.from_numpy(output).float().to(d)
        return output

    def _format_obs_to_state(self, obs):
        """
        Format the end-effector pose into the unified action vector.

        Args:
            ee_pose (torch.Tensor): The end-effector pose to be formatted.
                qpos ([B, N, (7+1)*2]).

        Returns:
            state (torch.Tensor): The formatted vector for RDT ([B, N, 128]).
        """

        B, N, _ = obs.shape
        # print(f"obs shape: {obs.shape}")
        state = torch.zeros(
            (B, N, self.args["model"]["state_token_dim"]),
            device=obs.device,
            dtype=obs.dtype,
        )
        obs = self._convert_ee_pose_to_6d(obs).to(device=state.device)
        print(f"state device: {state.device}")
        print(f"obs device: {obs.device}")
        # Fill into the unified state vector
        state[:, :, UR5_STATE_INDICES] = obs
        # Assemble the mask indicating each dimension's availability
        state_elem_mask = torch.zeros(
            (B, self.args["model"]["state_token_dim"]),
            device=obs.device,
            dtype=obs.dtype,
        )
        state_elem_mask[:, UR5_STATE_INDICES] = 1
        return state, state_elem_mask

    def _unformat_state_to_action(self, action):
        """
        Unformat the unified action vector into the eef pose to be executed.

        Args:
            action (torch.Tensor): The unified action vector to be unformatted.
                ([B, N, 128])

        Returns:
            ee_pose (torch.Tensor): The unformatted robot eef pose.
                ([B, N, (7+1)*2]).
        """
        # print(f"[raw] unformat action: {action[:, 0, :]}")
        action_indices = UR5_STATE_INDICES
        ee_pose = action[:, :, action_indices]
        # print(f"[before] unformat action to joints: {ee_pose[:, 0, :]}")

        ee_pose = self._convert_6d_to_ee_pose(ee_pose)
        # print(f"[after] action: {ee_pose[:, 0, :]}")

        return ee_pose


    @torch.no_grad()
    def step(self, proprio, images, text_embeds):
        """
        Predict the next action chunk given the
        proprioceptive states, images, and instruction embeddings.

        Args:
            proprio: proprioceptive states
            images: RGB images, the order should be
                [ext_{t-1}, right_wrist_{t-1}, left_wrist_{t-1},
                ext_{t}, right_wrist_{t}, left_wrist_{t}]
            text_embeds: instruction embeddings

        Returns:
            action: predicted action
        """
        device = self.device_2
        dtype = self.dtype

        # The background image used for padding
        background_color = np.array(
            [int(x * 255) for x in self.image_processor.image_mean], dtype=np.uint8
        ).reshape(1, 1, 3)
        background_image = (
            np.ones(
                (
                    self.image_processor.size["height"],
                    self.image_processor.size["width"],
                    3,
                ),
                dtype=np.uint8,
            )
            * background_color
        )

        # Preprocess the images by order and encode them
        image_tensor_list = []
        for image in images:
            if image is None:
                # Replace it with the background image
                image = Image.fromarray(background_image)

            if self.image_size is not None:
                image = transforms.Resize(self.data_args.image_size)(image)

            if self.args["dataset"].get("auto_adjust_image_brightness", False):
                pixel_values = list(image.getdata())
                average_brightness = sum(sum(pixel) for pixel in pixel_values) / (
                    len(pixel_values) * 255.0 * 3
                )
                if average_brightness <= 0.15:
                    image = transforms.ColorJitter(brightness=(1.75, 1.75))(image)

            if self.args["dataset"].get("image_aspect_ratio", "pad") == "pad":

                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(
                            pil_img.mode, (width, width), background_color
                        )
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(
                            pil_img.mode, (height, height), background_color
                        )
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result

                image = expand2square(
                    image, tuple(int(x * 255) for x in self.image_processor.image_mean)
                )
            image = self.image_processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ][0]
            image_tensor_list.append(image)

        image_tensor = torch.stack(image_tensor_list, dim=0).to(device, dtype=dtype)

        image_embeds = self.vision_model(image_tensor).detach()
        image_embeds = image_embeds.reshape(
            -1, self.vision_model.hidden_size
        ).unsqueeze(0).to(device)
        print(f"image embeds device: {image_embeds.device}")
        # Prepare the proprioception states and the control frequency
        joints = proprio.to(device).unsqueeze(0)  # (1, 1, 14)
        states, state_elem_mask = self._format_obs_to_state(
            joints
        )  # (1, 1, 128), (1, 128)
        states, state_elem_mask = states.to(device, dtype=dtype), state_elem_mask.to(
            device, dtype=dtype
        )
        states = states[:, -1:, :]  # (1, 1, 128)
        ctrl_freqs = torch.tensor([self.control_frequency]).to(device)

        text_embeds = text_embeds.to(device, dtype=dtype)

        # Predict the next action chunk given the inputs
        trajectory = self.policy.predict_action(
            lang_tokens=text_embeds,
            lang_attn_mask=torch.ones(
                text_embeds.shape[:2], dtype=torch.bool, device=text_embeds.device
            ),
            img_tokens=image_embeds,
            state_tokens=states,
            action_mask=state_elem_mask.unsqueeze(1),
            ctrl_freqs=ctrl_freqs,
        )
        trajectory = self._unformat_state_to_action(trajectory).to(torch.float32)

        return trajectory
