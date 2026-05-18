# Extract and save only the necessary weights for the e2e model
# python tools/convert_to_e2e.py --checkpoint_path /path/to/convert.pth --save_path /path/to/save.pth

import torch
import argparse

def main(args):
    checkpoint_path = args.checkpoint_path
    save_path = args.save_path

    model_state_dict = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in model_state_dict:
        model_state_dict = model_state_dict['model_state_dict']

    model_state_dict_e2e = {
        k: v for k, v in model_state_dict.items() if
        k.startswith("fmt") or k.startswith("motion_autoencoder") \
            or k.startswith("audio_encoder.audio_projection") \
            or k == "audio_encoder.wav2vec2.masked_spec_embed"
    }
    torch.save(model_state_dict_e2e, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()
    main(args)